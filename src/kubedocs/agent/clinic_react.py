"""LLM 驱动的 clinic agent：`/chat` 的 agent 化路径。

控制流交给 LLM：由 function calling 选工具、定步数，因此"先查号源再问用户"这类顺序可以
当场决定；`ClinicAgent`（确定性图）则把顺序写死在代码里，两者共用同一批工具与红线不变量。

红线**不能**做成工具：急诊识别是安全属性，模型某次"觉得不用查"就会漏报，代价不可接受。
所以红线在**进入循环之前**用确定性规则判定（`RiskGuard`），命中即返回急诊引导，
LLM 没有任何机会参与这个决定。

挂号的确认策略：**模型只能提议**（`propose_booking` 是只读的），真正取号/退号由患者
在界面上点按钮，走 `/consult/booking/confirm` 与 `/consult/booking/cancel` 两个端点。
所以这条路径**不给**工具循环任何 `confirm` 回调——`book_appointment` 标着 `side_effect`，
没有回调就一律拒绝执行。这与 `/consult` 完全同一套流程：提议在服务端留下一次性单号，
确认请求只回传那个单号，科室/日期/费用都取自服务端那张单子（模型与前端都改不动）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kubedocs.agent.bootstrap import register_domain_tools
from kubedocs.agent.tool_loop import AgentTurn, run_tool_loop
from kubedocs.agent.tools import ToolRegistry, ToolSpec
from kubedocs.config import Settings
from kubedocs.domains.clinic.redline import RiskGuard, emergency_advice, emergency_menu

SYSTEM_TEMPLATE = """你是医院预问诊助手，帮助患者判断该挂哪个科、并协助挂号。

规则：
1. **只做导诊与挂号协助，不下诊断结论**，不推荐具体用药与剂量。
2. 需要事实资料（科室名、号源、医学常识）时**先用工具查证**，不要凭记忆编造。
3. 你**只有**下列工具可用。不要调用没有列出的工具，也不要凭印象编一个工具名——
   调用不存在的工具只会拿到一句错误，然后白白多走一步：
{tool_list}
4. 有副作用的工具（例如挂号）只有在**用户明确要求**时才调；调用前先用查询类工具
   确认有号，并把科室名说清楚。
5. 症状信息不足时就问一个最关键的问题，一次只问一件事。
6. 出现胸痛、呼吸困难、意识不清、大出血等紧急情况，提示立即拨打 120 或去急诊。
7. 用简体中文，简洁分点。"""

TOOL_LIST_HEADER = "可用工具清单（只能调用这些）："
"""工具清单那一行的**机器可读标记**：测试靠它把「提示承诺的工具集合」抠出来，与
`ToolRegistry` 里实际注册的集合逐名比对。改这句话要同步改测试。"""


def build_system_prompt(tools: Sequence[ToolSpec]) -> str:
    """按**实际注册的工具**生成系统提示。

    提示必须动态生成：工具集合随装配而变（例如不传 retriever 时 clinic 域没有
    `search_docs`），写死的提示词会承诺一个不存在的工具，模型每次调用都只拿回一句
    "没有这个工具"，白烧一轮预算还拖长响应。

    只列工具名、不复述每个工具的说明：说明与 `when_not` 已随 function calling 的 schema
    一起交给模型（见 `ToolSpec.schema()`），再抄一遍等于有两个会各自漂移的描述源；
    而"只能调这些"这条**约束**是 schema 表达不了的，才是提示词该管的事。

    ## 不变量

    提示词承诺的工具集合 == 实际注册的工具集合（`tests/unit/test_prompt_tool_parity.py`
    钉住了这一点）。多做会让模型空转，少做会让工具白注册——两个方向都要防。
    """
    names = sorted(t.name for t in tools)
    # 空清单那行**故意不以 “- ” 开头**：`- ` 前缀是“这是一个工具名”的约定，让占位文案
    # 也用这个前缀，会把“没有工具”读成“有一个叫（当前没有任何工具可用）的工具”。
    block = (
        "\n".join([TOOL_LIST_HEADER, *(f"- {name}" for name in names)])
        if names
        else f"{TOOL_LIST_HEADER}\n（当前没有任何工具可用：不要声称自己查过任何资料）"
    )
    # 用 replace 而不是 str.format：提示词里出现花括号（例如 JSON 示例）时不会炸。
    return SYSTEM_TEMPLATE.replace("{tool_list}", block)


@dataclass
class ClinicTurn:
    """一轮问诊的结果：供 SSE 发 tool 事件与最终文本。"""

    text: str
    tool_calls: list[dict[str, Any]]
    emergency: bool = False
    steps: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    """其中**算不出价**（litellm 价格表里没有这个模型）的调用次数。

    透传到 HTTP 层，让"未知成本"在**响应里**也能看见，而不是只躺在账本文件里。
    """
    notes: list[str] | None = None
    pending_action: dict[str, Any] | None = None
    """这一轮模型提议的**待确认挂号单**（`propose_booking` 的产出）。

    前端拿它渲染"确认挂号"卡片。它是工具的结构化结果，不是模型那段文字——
    科室、日期、时段、挂号费都来自服务端写下的那张单子。
    """


class ClinicReactRuntime:
    """把「注册工具 + 红线预检 + 工具循环」装配成一个可复用对象。

    持有 `guard` 与 `service`：红线判定与号源库都是**有状态/需复用**的，每轮新建会丢掉
    号源库存（那会变成"挂完号又有了"）。
    """

    def __init__(
        self,
        settings: Settings,
        service: Any,
        retriever: Any | None = None,
        pending: Any | None = None,
    ) -> None:
        self.settings = settings
        self.service = service
        self.retriever = retriever
        self.guard = RiskGuard()
        # 预算闸门与 CLI `ask` / `POST /ask` **共用同一个花费文件**：两条路径算的是同一个
        # 钱包，各记各的账等于给了两份额度。路径来自 `settings.llm_spend_path`，
        # 是**全局一份**、不随域目录切换。
        # 这是长寿命对象（`api/app.py` 缓存它），所以闸门状态跨请求有效。
        from kubedocs.generate import DailyBudget

        self.budget = DailyBudget(
            settings.llm_spend_path,
            settings.llm_daily_budget_usd,
            unpriced_usd=settings.llm_unpriced_call_usd,
        )
        # 注册工具是"接线"动作：不做它，run_tool_loop 拿到的 tools 会是空的。
        # `pending` 必须与 `/consult` 是**同一份**（`app.state.clinic_pending`），
        # 否则在问诊页提议的单子在对话页确认不了。
        register_domain_tools("clinic", service=service, retriever=retriever, pending=pending)

    @property
    def tools(self) -> list[Any]:
        return list(ToolRegistry.for_domain("clinic"))

    @property
    def system_prompt(self) -> str:
        """这一轮实际发给模型的系统提示：按**刚才注册进去的工具**生成。

        每次取都重新生成（不在 `__init__` 里缓存）：`ToolRegistry` 是全局的，幂等装配
        随时可能把它换掉，缓存的提示词会与真实可调用的工具脱节，而"提示承诺 == 实际注册"
        这条不变量正是这个属性存在的理由。
        """
        return build_system_prompt(self.tools)

    def has_llm(self) -> bool:
        from kubedocs.generate import has_api_key

        return has_api_key(self.settings.llm_model)

    def turn(
        self,
        message: str,
        *,
        history: list[dict[str, Any]] | None = None,
        patient_id: str = "anon",
    ) -> ClinicTurn:
        """跑一轮。**红线先判**，命中则不进循环。

        `patient_id` 由**调用方**给（HTTP 层从请求里取），经 `context=` 注入到工具执行体。
        它不进工具 schema，所以模型无法伪造"这是谁"。
        """
        hit = self.guard.check(message)
        if hit is not None:
            # 红线命中：确定性返回，LLM 无权参与
            text = emergency_menu(hit.word)
            return ClinicTurn(
                text=text,
                tool_calls=[
                    {
                        "tool": "check_red_line",
                        "word": hit.word,
                        "status": hit.status,
                        "experiencer": hit.experiencer,
                        "temporality": hit.temporality,
                    }
                ],
                emergency=True,
                notes=["红线命中（确定性规则，LLM 不可绕过）"],
            )

        if not self.has_llm():
            return ClinicTurn(
                text=(
                    "当前没有配置大模型密钥，无法进行对话式问诊。"
                    "如需挂号请直接说明科室；紧急情况请立即拨打 120。"
                ),
                tool_calls=[],
                notes=["未配置 LLM 密钥"],
            )

        from kubedocs.agent.llm_tool_caller import LiteLLMToolCaller

        caller = LiteLLMToolCaller(self.settings)
        prompt = self.system_prompt

        # history 只走 `run_tool_loop(history=...)` 这一条路：消息顺序（system → history
        # → 本轮 user）与角色校验都在循环里做，本模块不自己拼消息列表。
        #
        # **不传 `confirm`**：确认的主体是患者的一次点击，不是"这句话里有没有挂号两个字"。
        # 没有回调 ⇒ 标了 `side_effect` 的 `book_appointment` 一律被拒，模型只能提议；
        # 患者在卡片上点确认后才由 `/consult/booking/confirm` 真正取号。
        turn: AgentTurn = run_tool_loop(
            message,
            tools=self.tools,
            llm=caller,
            system=prompt,
            max_steps=5,
            history=history,
            budget=self.budget,
            context={"patient_id": patient_id},
        )
        return ClinicTurn(
            text=turn.answer or "（模型没有给出内容，请再说一次或换个说法）",
            tool_calls=[
                {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "error": tc.error or None,
                    "result": None if tc.error else tc.result,
                }
                for tc in turn.tool_calls
            ],
            steps=turn.steps,
            llm_calls=turn.llm_calls,
            notes=list(turn.notes),
            cost_usd=turn.cost_usd,
            unpriced_calls=turn.unpriced_calls,
            pending_action=_pending_action(turn),
        )


def emergency_text_for(message: str) -> str | None:
    """分离出来的红线判定，便于单测与复用。"""
    hit = RiskGuard().check(message)
    return emergency_advice(hit.word) if hit else None


def _pending_action(turn: AgentTurn) -> dict[str, Any] | None:
    """从这一轮的工具调用里挑出待确认挂号单（最后一张生效）。

    形状与 `/consult` 的 `pending_action` **逐字段一致**：两条路径的确认按钮打的是同一个
    端点、传的是同一个 `proposal_id`，前端只需要一条渲染分支。
    """
    found: dict[str, Any] | None = None
    for call in turn.tool_calls:
        if call.error or call.name != "propose_booking":
            continue
        result = call.result
        if isinstance(result, dict) and isinstance(result.get("proposal"), dict):
            found = dict(result["proposal"])
    return found
