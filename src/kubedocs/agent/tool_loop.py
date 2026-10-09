"""真正的 function calling 工具循环：**模型自己选工具、自己决定调几次**。

## 为什么必须有一个循环，而不是一串回调

仓库里已经有 `ToolSpec` / `ToolRegistry`（声明）与 `rag_tools` / `clinic.tools`（执行体），
但控制流一直是写死的回调序列：代码决定第几步做什么，模型只在"抽症状""生成回答"两个点上
被调用。那是一台**带图外壳的流水线**，不是 agent——模型没有机会说"我得先查号源再回答"，
也没有机会在工具报错之后换一个工具重试。

`run_tool_loop` 补上这一层：把 `spec.schema()` 作为 `tools=` 交给 LLM，读回 `tool_calls`，
执行，把结果按 `role="tool"` 回灌，再让模型决定下一步。**没有关键词匹配、没有硬编码的
工具顺序**——哪天模型换了个工具名，这个文件一行都不用改。

## 为什么错误要"回灌"而不是抛出

工具报错（名字不存在、参数不合法、业务失败）在 agent 循环里是**正常的一步**，不是异常
终止：把错误文本当作该步的工具结果交回模型，它下一轮就有机会改参数、换工具、或换个说法
回答。直接抛出等于剥夺它自我纠正的机会，也让整轮问答一起失败。代价是模型可能固执地重试
同一个错名字，所以 `max_steps` 是硬上限，且同一工具名连续两次不存在会记进 `notes`
（这类记录通常意味着提示词或 schema 在误导模型，值得人去看一眼）。

## 为什么 side_effect 工具没有 confirm 就拒绝执行

"挂号"会真实占用号源且不可逆。循环在**执行之前**检查 `confirm` 回调；没有回调就拒绝执行，
把"需要用户确认"回灌给模型，让它先去征求用户同意。这是安全属性：默认不执行不可逆操作，
而不是默认执行、出了事再回滚。

## 多轮上下文（`history`）怎么进来

`history` 插在 **system 之后、本轮 `question` 之前**，顺序原样保留：

    [system?] + history... + [本轮 user]

只接受 `role` 为 `user` / `assistant` 的条目，其余**直接报错**（见 `_history_messages`）。
这个参数是补上一次真实缺陷加上的：调用方（`clinic_react.turn`）把 history 拼进了一个
本地 `messages` 列表，却**从来没把它交给循环**——而 `messages` 被 `extend` 过，
所以连 F841 都抓不到。表现是"`/chat` 多轮对话没有上下文"，而日志里看不出任何异常。
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from kubedocs.agent.tools import ToolError, ToolSpec

TOOL_RESULT_LIMIT = 4000
"""单条工具结果回灌给模型的最大字符数。

检索类工具一次能返回几十 KB，不截断就会把对话历史挤出上下文窗口。**错误文本不截断**：
它本来很短，而被截掉的错误会让模型无从改起。
"""

ConfirmFn = Callable[["ToolCall"], bool]
"""副作用工具的执行确认：返回 True 才真的执行。没有它，`side_effect` 工具一律拒绝。"""

HISTORY_ROLES = ("user", "assistant")
"""`history` 里**允许**出现的角色，只有这两种。

- 不许 `system`：系统提示由调用方的 `system=` 参数独占。允许外部再塞一条，就会出现
  两个会各自漂移的提示词源（`build_system_prompt` 的 docstring 里同一条理由）。
- 不许 `tool`：`role="tool"` 的消息要靠 `tool_call_id` 与某条 `assistant.tool_calls`
  配对。循环自己生成的那些在本轮内部才成立；外部塞进来的 `tool` 消息没有归属，
  模型会读到一段不知道从哪来的结果——比不传更糟。
"""


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用的完整记录：模型要的参数 + 实际拿到结果或错误。"""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str = ""

    @property
    def ok(self) -> bool:
        """这一步是否成功。失败原因一律在 `error` 里（含参数校验、副作用未确认）。"""
        return not self.error


@dataclass
class AgentTurn:
    """一轮问答的完整轨迹：回答 + 工具调用 + 预算消耗。"""

    question: str
    answer: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def used_tools(self) -> list[str]:
        """模型请求过的工具名，去重保序。

        含"不存在的工具名"：它记录的是**模型想用什么**（用于发现提示词 / schema 的问题），
        而不是"成功执行了什么"；后者看 `tool_calls` 里每项的 `error`。
        """
        names: list[str] = []
        for call in self.tool_calls:
            if call.name not in names:
                names.append(call.name)
        return names


class ToolCaller(Protocol):
    """LLM 接口。实现方负责绑 LiteLLM 的 tools=/tool_choice=。"""

    def choose(self, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> dict[str, Any]:
        """返回 {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}"""


@dataclass(frozen=True)
class _ParsedCall:
    """从 LLM 响应里解析出来的一次工具调用（还没执行）。"""

    call_id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str = ""

    def as_message(self) -> dict[str, Any]:
        """转成 assistant 消息里的 tool_calls 项（OpenAI 形状）。"""
        return {"id": self.call_id, "type": "function",
                "function": {"name": self.name, "arguments": _json_text(self.arguments)}}


def _history_messages(history: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """把调用方给的多轮上下文**逐条校验**后转成消息（顺序原样保留）。

    非法形状一律 `ValueError`，**不静默丢弃也不静默改写**：丢一条 history 在日志里
    与"模型没记住"长得一模一样，是查不出来的那类 bug；把非法的 `role` 硬塞进请求
    则会让供应商报一个与真实原因无关的错。两种静默都不可接受。

    只保留 `role` / `content` 两个键（重建字典，而不是原样透传）：调用方在条目里多带的
    `tool_calls` 之类字段，会与它自己并不存在的 `tool` 消息配不上对，构造一个"半个
    工具调用"的上下文。
    """
    if not history:
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            raise ValueError(f"history[{index}] 不是对象（{type(item).__name__}）："
                             "需要 {'role': ..., 'content': ...}")
        role = item.get("role")
        if role not in HISTORY_ROLES:
            raise ValueError(
                f"history[{index}] 的 role={role!r} 不合法：只允许 {list(HISTORY_ROLES)}"
                "（system 由 system= 参数独占；tool 消息必须与本轮的 assistant.tool_calls 配对）")
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError(f"history[{index}] 的 content 必须是字符串，"
                             f"收到 {type(content).__name__}")
        out.append({"role": role, "content": content})
    return out


def run_tool_loop(question: str, *, tools: Sequence[ToolSpec], llm: ToolCaller,
                  system: str = "", max_steps: int = 5,
                  confirm: ConfirmFn | None = None,
                  history: Sequence[dict[str, Any]] | None = None) -> AgentTurn:
    """跑一轮"模型选工具 -> 执行 -> 回灌 -> 再选"的循环，直到模型给出最终回答。

    `max_steps` 是 **LLM 轮次**的硬上限（不是工具执行次数）：一轮里模型可以要求调多个
    工具，它们都算同一步。达到上限仍未收敛就停下——宁可留下"没答完"的记录，也不要无界地
    烧预算。

    `confirm` 只对 `side_effect=True` 的工具生效；为 `None` 时这类工具一律拒绝执行。

    `history` 是**此前几轮**的对话（`{"role", "content"}`，role 只允许 user/assistant），
    会被插在 system 与本轮 `question` 之间。默认 `None` 的行为与没有这个参数时**逐字相同**：
    只有 `system`（若有）与本轮 `question` 两条。这里**不做长度/字符截断**——上下文窗口
    的预算属于调用方的职责（`api/app.py` 用 `[-8:]` 保留最近 4 轮），循环自己不猜一个
    没验证过的截断策略。
    """
    if max_steps < 1:
        raise ValueError(f"max_steps 必须 >= 1（收到 {max_steps}）：0 步的循环没有意义")

    specs = list(tools)
    registry: dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in registry:
            # 同名工具无法判断该调哪一个，静默让后者覆盖前者会变成难以排查的错调。
            raise ValueError(f"工具名 {spec.name!r} 重复：分发时无法判断该调用哪一个")
        registry[spec.name] = spec
    schemas = [spec.schema() for spec in specs]

    turn = AgentTurn(question=question)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    # 顺序固定为 system -> history（原序）-> 本轮 user。history 里的角色原样保留，
    # 不做合并/重排：调用方给什么顺序就是什么顺序，改了就等于替它猜意图。
    messages.extend(_history_messages(history))
    messages.append({"role": "user", "content": question})

    unknown_name = ""
    unknown_streak = 0

    for _ in range(max_steps):
        turn.steps += 1
        try:
            turn.llm_calls += 1
            response = llm.choose(messages, schemas)
        except Exception as e:  # noqa: BLE001 - LLM 是外部服务，任何异常都不该让 agent 崩掉
            turn.notes.append(
                f"LLM 调用失败（{type(e).__name__}: {e}）：已用已有 {len(turn.tool_calls)} 次"
                "工具结果收尾（提交给调用方的 answer 为空，证据保留在 tool_calls 里）；"
                "这里不重试，避免供应商故障时把账单放大"
            )
            return turn

        if not isinstance(response, dict):
            turn.notes.append(
                f"LLM 返回了非 dict 的响应（{type(response).__name__}）：无法解析工具调用，已停止"
            )
            return turn

        turn.cost_usd += _as_float(response.get("cost_usd"))
        content = _as_text(response.get("content"))
        calls = _parse_calls(response.get("tool_calls"), step=turn.steps)

        if not calls:
            turn.answer = content
            return turn

        # 把模型这一步的要求原样记回历史：不回灌"它自己说过要调什么"，后面的 role="tool"
        # 消息就没有归属，模型看不懂工具结果是从哪来的。
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [c.as_message() for c in calls]})

        for parsed in calls:
            call = _execute(parsed, registry=registry, confirm=confirm, notes=turn.notes)
            turn.tool_calls.append(call)
            messages.append({"role": "tool", "tool_call_id": parsed.call_id,
                             "name": parsed.name or "unknown", "content": _tool_message(call)})

            if parsed.name and parsed.name not in registry:
                unknown_streak = unknown_streak + 1 if parsed.name == unknown_name else 1
                unknown_name = parsed.name
                if unknown_streak == 2:
                    turn.notes.append(
                        f"工具 {parsed.name!r} 连续 {unknown_streak} 次不存在：模型没有按回灌的"
                        "错误改选工具，通常是提示词或工具 schema 在误导它"
                    )
            else:
                unknown_name, unknown_streak = "", 0

    turn.notes.append(
        f"达到步数上限（max_steps={max_steps}）：模型在 {turn.steps} 步内没有给出最终回答，"
        f"已停止并保留 {len(turn.tool_calls)} 次工具调用（answer 为空）"
    )
    return turn


def _execute(parsed: _ParsedCall, *, registry: dict[str, ToolSpec], confirm: ConfirmFn | None,
             notes: list[str]) -> ToolCall:
    """执行一次已解析的调用，**任何失败都变成 `ToolCall.error`**。

    这里刻意不抛异常：一次工具失败的语义是"这一步没成功"，不是"整轮失败"。只有把失败文本
    回灌给模型，它才有机会改参数或换工具。
    """
    if parsed.parse_error:
        return ToolCall(name=parsed.name, arguments=parsed.arguments, error=parsed.parse_error)

    try:
        spec = _lookup(parsed.name, registry)
    except ToolError as e:
        return ToolCall(name=parsed.name, arguments=parsed.arguments, error=str(e))

    if spec.handler is None:
        return ToolCall(name=spec.name, arguments=parsed.arguments,
                        error=f"工具 {spec.name!r} 只声明了 schema、没有执行体：无法执行")

    # 先校验参数、再要确认：让用户去确认一个本来就跑不通的调用，是在浪费他的注意力。
    try:
        validated = spec.params.model_validate(parsed.arguments)
    except ValidationError as e:
        return ToolCall(name=spec.name, arguments=parsed.arguments,
                        error=f"参数校验未通过（{spec.params.__name__}）：{e}")

    if spec.side_effect:
        denial = _confirm_side_effect(spec, validated, confirm, notes)
        if denial:
            return ToolCall(name=spec.name, arguments=parsed.arguments, error=denial)

    try:
        result = _invoke(spec, validated)
    except ToolError as e:
        # 可预期的业务失败（号源已满、检索器连不上）：一句人话就够，不要 traceback。
        return ToolCall(name=spec.name, arguments=parsed.arguments, error=f"工具报告失败：{e}")
    except Exception as e:  # noqa: BLE001 - 工具的编程错误同样不该让整轮 traceback
        return ToolCall(name=spec.name, arguments=parsed.arguments,
                        error=f"工具抛出未预期异常（{type(e).__name__}: {e}）")
    return ToolCall(name=spec.name, arguments=parsed.arguments, result=result)


def _lookup(name: str, registry: dict[str, ToolSpec]) -> ToolSpec:
    """按名取工具；取不到抛 `ToolError`（与 `ToolRegistry.get` 同一句话术）。

    错误信息里带上可用工具列表：模型下一轮才有可能改成正确的名字。只说"不存在"，
    等于让它猜。
    """
    spec = registry.get(name)
    if spec is None:
        raise ToolError(f"没有名为 {name!r} 的工具；可用：{sorted(registry)}")
    return spec


def _confirm_side_effect(spec: ToolSpec, validated: BaseModel, confirm: ConfirmFn | None,
                         notes: list[str]) -> str:
    """副作用工具的执行闸门：**默认拒绝**，只有 `confirm` 明确点头才放行。

    返回空串表示放行，否则返回要回灌给模型的拒绝理由。
    """
    if confirm is None:
        notes.append(f"工具 {spec.name!r} 有副作用但没有提供 confirm 回调：已拒绝执行，"
                     "把“需要用户确认”回灌给模型")
        return (f"拒绝执行 {spec.name!r}：它会产生不可逆的副作用，必须先得到用户确认。"
                "请先向用户说明将要执行什么并征求同意，用户同意后再重新调用。")

    # 确认回调看到的是**校验后的完整参数**（含默认值），而不是模型省略了字段的原始 JSON：
    # 用户要确认的，是他真正会执行的那一次调用。
    proposal = ToolCall(name=spec.name, arguments=validated.model_dump())
    try:
        allowed = bool(confirm(proposal))
    except Exception as e:  # noqa: BLE001 - 确认回调可能接 UI / 图，失败一律按"未确认"处理
        notes.append(f"confirm 回调对 {spec.name!r} 抛异常（{type(e).__name__}: {e}）：按未确认处理")
        return f"未获得用户确认（确认回调失败：{type(e).__name__}）：{spec.name!r} 未执行"
    if not allowed:
        return (f"用户拒绝了 {spec.name!r}：不要重复调用它；如需继续，先说明理由或改问用户"
                "别的信息。")
    return ""


def _invoke(spec: ToolSpec, validated: BaseModel) -> Any:
    """按执行体的签名，把**校验后的参数**交给它。

    仓库里有两种执行体形状，两种都要支持：`rag_tools` 是 `handler(params)`（收一个 pydantic
    模型），`clinic.tools` 是 `handler(**kwargs)`。用签名区分，而不是要求调用方再声明一次——
    "声明的东西就是能跑的东西"这条不变量，不该因为执行体的形状不同而破功。

    注意两件事：非法参数在上一道闸门就被拦下了；到这里交给执行体的是 `model_dump()` 的结果
    （默认值补齐、类型已 coerce），不是模型给的原始 JSON。
    """
    handler = spec.handler
    if handler is None:  # 只为类型收窄；真正拦住"只声明不实现"的是调用方
        raise ToolError(f"工具 {spec.name!r} 没有执行体")
    signature = inspect.signature(handler)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in signature.parameters.values())
    if accepts_kwargs or not _takes_params_model(signature):
        return handler(**validated.model_dump())
    return handler(validated)


def _takes_params_model(signature: inspect.Signature) -> bool:
    """判断执行体是不是 `handler(params: SomeModel)` 形状。

    按名字（`params`）判断是主力手段：本仓库的模块都开了 `from __future__ import annotations`，
    注解在运行时是**字符串**，靠 `issubclass` 认不出来。类型注解那条只是给没开 future 的模块兜底。
    """
    values = list(signature.parameters.values())
    if len(values) != 1:
        return False
    param = values[0]
    if param.kind not in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return False
    if param.name in {"params", "p"}:
        return True
    annotation = param.annotation
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _parse_calls(raw: object, *, step: int) -> list[_ParsedCall]:
    """解析 LLM 返回的 `tool_calls`。

    形状不对**不抛异常**，而是记成 `parse_error` 回灌——模型吐出畸形 JSON 是外部输入的常态，
    该由它自己改，而不是让整轮问答 traceback。`call_id` 带上步号，保证同一轮对话里不重复。
    """
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    parsed: list[_ParsedCall] = []
    for index, item in enumerate(items):
        name, arguments, error = _parse_one(item)
        parsed.append(_ParsedCall(call_id=f"call_{step}_{index + 1}", name=name,
                                  arguments=arguments, parse_error=error))
    return parsed


def _parse_one(item: object) -> tuple[str, dict[str, Any], str]:
    """解析单个 tool_call，返回 `(name, arguments, 解析错误)`。"""
    if not isinstance(item, dict):
        return "", {}, f"tool_call 不是对象（{type(item).__name__}）：无法执行"
    # 兼容两种形状：适配器归一化过的 {"name", "arguments"}，以及 LiteLLM / OpenAI 原样的
    # {"function": {"name", "arguments": "<JSON 字符串>"}}。
    function = item.get("function")
    source = function if isinstance(function, dict) else item
    raw_name = source.get("name")
    name = raw_name if isinstance(raw_name, str) else ""
    raw_arguments = source.get("arguments")
    if raw_arguments is None or raw_arguments == "":
        return name, {}, ""
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as e:
            return name, {}, f"tool_call 的 arguments 不是合法 JSON（{e}）：请重新给出参数"
    if not isinstance(raw_arguments, dict):
        return name, {}, (f"tool_call 的 arguments 必须是对象，收到 "
                          f"{type(raw_arguments).__name__}：请按 schema 重新给出参数")
    return name, raw_arguments, ""


def _tool_message(call: ToolCall) -> str:
    """把一步的结果压成回灌给模型的文本。

    失败用 `{"error": ...}`、成功用结果本身：模型更容易读懂工具"原始"的返回，而错误必须
    显眼。错误不截断（它短），成功结果按 `TOOL_RESULT_LIMIT` 截断（检索类工具能把上下文撑爆）。
    """
    if call.error:
        return _json_text({"error": call.error})
    text = _json_text(call.result)
    if len(text) <= TOOL_RESULT_LIMIT:
        return text
    return f"{text[:TOOL_RESULT_LIMIT]}…（结果过长，已截断）"


def _json_text(value: object) -> str:
    """序列化成 JSON 文本。

    `ensure_ascii=False` 保留中文；`default=str` 兜住任意返回对象（clinic 的取号单就是一个
    自定义对象）——工具返回什么形状都不该让循环崩掉。
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _as_text(value: object) -> str:
    """把 content 安全地取成 str；`None` 变空串，避免把 `"None"` 当成回答。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_float(value: object) -> float:
    """把可选的 `cost_usd` 安全地转成 float。

    供应商适配器不一定上报成本，上报了也不一定给对，还可能给成 JSON 字符串（`"0.0012"`）。
    能转就转、转不动按 0 处理：成本统计不该因为一个脏字段把整轮问答搞崩，也不该悄悄丢掉
    一个本来能用的数字。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
