"""LLM 驱动的 clinic agent：`/chat` 的 agent 化路径。

控制流交给 LLM：由 function calling 选工具、定步数，因此"先查号源再问用户"这类顺序可以
当场决定；`ClinicAgent`（确定性图）把顺序写死在代码里，它现在是**对照臂**（不在请求路径上，
见 `api/app.py` 的 `_get_clinic_agent`），两者共用同一批工具、同一份参考标准面板。

## 判断权全在模型，规则资产只提供线索

红线词表 + 三维门控、症状词表、症状→科室票数**都不再产出结论**：它们由
`domains.clinic.reference.clinic_reference()` 装配成一份「参考标准」面板，随系统提示一起
交给模型，由模型判断是否急症、该挂哪个科。本模块**没有**"命中即短路"的分支——那正是以前
的形态（规则抢在模型前面下结论，模型连患者原话的上下文都看不到）。

## `emergency` 是展示层检测，不是医学判断

急诊判断写在模型这一轮的正文里。`ClinicTurn.emergency` 只回答"前端要不要弹急诊横幅"，
判据是正文里有没有出现「120 / 急诊」。它是**展示逻辑**：不参与医学推理、不改动回答，
也不是"模型判了急症"的结构化字段（那种字段属于 `/consult` 的输出契约）。

## 没有大模型密钥时只回一句话

没有判断者就没有结论：不检索、不装配面板、不进工具循环，只回 `NO_LLM_MESSAGE`
（文案的唯一来源是 `llm_status.py`）。模型调用失败（超时/限流/输出不可解析/步数耗尽）
同样不产出任何兜底内容，只回 `LLM_FAILED_MESSAGE`。

## 挂号的确认策略

**模型只能提议**（`propose_booking` 是只读的），真正取号/退号由患者在界面上点按钮，走
`/consult/booking/confirm` 与 `/consult/booking/cancel` 两个端点。所以这条路径**不给**工具
循环任何 `confirm` 回调——`book_appointment` 标着 `side_effect`，没有回调就一律拒绝执行。
这与 `/consult` 完全同一套流程：提议在服务端留下一次性单号，确认请求只回传那个单号，
科室/日期/费用都取自服务端那张单子（模型与前端都改不动）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from kubedocs.agent.bootstrap import register_domain_tools
from kubedocs.agent.tool_loop import AgentTurn, run_tool_loop
from kubedocs.agent.tools import ToolRegistry, ToolSpec
from kubedocs.clinical.reference import BookingFact
from kubedocs.config import Settings
from kubedocs.domains.clinic.data.departments import DEPARTMENTS
from kubedocs.domains.clinic.redline import RiskGuard
from kubedocs.domains.clinic.reference import clinic_reference
from kubedocs.llm_status import LLM_FAILED_MESSAGE, NO_LLM_MESSAGE

logger = logging.getLogger(__name__)

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


def build_system_prompt(tools: Sequence[ToolSpec], *, reference: str = "") -> str:
    """按**实际注册的工具**生成系统提示；`reference` 非空时把参考标准面板追加在末尾。

    提示必须动态生成：工具集合随装配而变（例如不传 retriever 时 clinic 域没有
    `search_docs`），写死的提示词会承诺一个不存在的工具，模型每次调用都只拿回一句
    "没有这个工具"，白烧一轮预算还拖长响应。

    只列工具名、不复述每个工具的说明：说明与 `when_not` 已随 function calling 的 schema
    一起交给模型（见 `ToolSpec.schema()`），再抄一遍等于有两个会各自漂移的描述源；
    而"只能调这些"这条**约束**是 schema 表达不了的，才是提示词该管的事。

    ## 面板为什么拼进同一条 system 消息，而不是另起一条

    `run_tool_loop` 的消息序列不变量是 `system?(单条) -> history -> 本轮 user`，且 `history`
    里的 `system` 角色会被 `ValueError` 拒掉（见 `tool_loop._history_messages`）。多插一条
    system 会把这个不变量变成"两条 system 谁优先"的解释题，还要改动既有断言与日志解析。
    拼在**同一条消息的末尾**还有第二个好处：面板紧挨着本轮提问，模型读到的顺序与它要
    判断的问题最近。

    ## 不变量

    提示词承诺的工具集合 == 实际注册的工具集合（`tests/unit/test_prompt_tool_parity.py`
    钉住了这一点）。多做会让模型空转，少做会让工具白注册——两个方向都要防。
    面板追加在工具清单**之后**，而清单由 `TOOL_LIST_HEADER` + 连续的 `- 名字` 行构成、
    后面紧跟非 `- ` 开头的正文，所以追加不会污染那段机器可读标记的解析。
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
    prompt = SYSTEM_TEMPLATE.replace("{tool_list}", block)
    return f"{prompt}\n\n{reference}" if reference else prompt


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
    reference: dict[str, Any] | None = None
    """这一轮交给模型的**参考标准面板**（`FactSet.to_payload()`）。

    与 `/consult` 的 `reference` 是同一形状、同一装配来源：规则资产说了什么、原话怎么被
    归一、服务端有哪些挂号事实，都可以逐轮对账。**它不是结论**——结论在 `text` 里，
    由模型给出；`reference` 是"依据"的留档。
    """


BOOKING_HISTORY_LIMIT = 5
"""面板里最多回看几条服务端挂号记录（新到旧）。

只取最近几条：面板的作用是让模型**有事实可对照**（尤其别凭空说"已挂上"），不是让患者在
这里看完整就诊史（那有专门的接口）。
"""

_CANCELLED_STATUSES = frozenset({"cancelled", "canceled", "已退号"})
"""号源库里表示"这个号已经退掉"的状态取值。

`cancelled` 是 `pg_provider.STATUS_CANCELLED` 与 `mock._STATUS_CANCELLED` 的同一取值；
另外两个只是宽容写法——把"已退号"读成"已确认"会让模型对着一个不存在的号说话。
"""


@lru_cache(maxsize=1)
def _triage_links() -> dict[str, tuple[str, ...]]:
    """本域的「症状 → 候选科室」关联表。**进程内只读一次**。

    关联表来自知识卡片语料（几万张卡，解析一次是秒级），而 `turn()` 每个请求都要它：
    不缓存等于每个请求都重读一遍语料。它与请求无关，`maxsize=1` 就够；测试要换一张
    关联表时记得 `_triage_links.cache_clear()`（或直接替换这个名字）。

    读取失败（语料缺失 / 格式坏）返回**空表**而不是抛异常：面板会如实写"关联表为空、
    科室候选线索缺失，请仅依据知识与你的判断"，模型照样能给出判断——比让整个 `/chat`
    因为缺一份语料而 500 好得多。写法与 `domains/clinic/pack.link_vocabulary()` 一致。
    """
    try:
        from kubedocs.domains.clinic.domain import ClinicPack

        return {str(k): tuple(str(v) for v in values) for k, values in ClinicPack().triage_links().items()}
    except Exception as e:  # noqa: BLE001 - 缺语料只是"这一层没有关联表"，不是服务故障
        logger.warning("症状关联表读取失败，面板里的科室候选线索会缺失：%s: %s", type(e).__name__, e)
        return {}


def _as_int(value: object) -> int:
    """面板里的数值字段（费用/队列号）：脏值按 0 处理，不让一个字段把整轮问诊打断。

    0 会被 `FactSet.known_numbers()` 过滤掉，所以它不会被当成一条"真实读到的数值"。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _booking_facts(service: Any, patient_id: str) -> tuple[list[BookingFact], list[str]]:
    """把**服务端已经发生过**的挂号/退号读成面板条目，返回 `(事实, 装配说明)`。

    这一条是给模型"别凭空说已挂上"用的：面板里没有对应条目时，模型就不该宣称取号成功
    （`clinical/verify.py` 的 `fabricated_booking` 也会据此把它拦下）。**没有已确认的挂号
    时返回空列表**——`FactSet.render()` 会打印"本轮服务端没有任何已确认的挂号/退号动作"
    那句明确的否定，而这正是"模型看到空白就按对话语气补一句已为您挂上"的解毒剂。

    读取失败不让整轮崩，但**也不装作"没有动作"**：把问题作为装配说明回传，面板会写明
    "这次业务事实可能不完整"。静默返回空列表会把"查不到"说成"没发生过"，那是最危险的一种
    沉默——模型会据此否认患者真实存在的号。
    """
    try:
        rows = service.my_appointments(patient_id, limit=BOOKING_HISTORY_LIMIT)
    except Exception as e:  # noqa: BLE001 - 号源是外部服务，读不到不该让问诊中断
        reason = f"{type(e).__name__}: {e}"
        logger.warning("读挂号记录失败，本轮面板的业务事实不完整：%s", reason)
        return [], [f"挂号记录读取失败（{reason}）：本轮业务事实可能不完整，不要据此否认已有挂号"]

    facts: list[BookingFact] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "") or "")
        facts.append(
            BookingFact(
                # 号源只把"已退号"标出来；其余（paid / pending_payment）都是服务端真的记了一笔。
                kind="cancelled" if status in _CANCELLED_STATUSES else "confirmed",
                department=str(row.get("department", "") or ""),
                date=str(row.get("date", "") or ""),
                # `time` 是号源契约里的时段键（库里叫 period），不要在这里改名。
                time_slot=str(row.get("time", "") or ""),
                fee=_as_int(row.get("fee")),
                ticket_no=str(row.get("ticket_no", "") or ""),
                queue_no=_as_int(row.get("queue_no")),
                status=status,
            )
        )
    return facts, []


class ClinicReactRuntime:
    """把「注册工具 + 参考标准装配 + 工具循环」装配成一个可复用对象。

    持有 `guard` 与 `service`：红线线索与号源库都是**有状态/需复用**的，每轮新建会丢掉
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

        这里**不含参考标准面板**：面板是按每一轮的原话现装配的（`turn()` 传
        `reference=`），而本属性回答的是"工具清单是什么"。
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
        """跑一轮。**判断权全在模型**：规则资产只进参考标准面板。

        这一轮发生的事：

        1. 没有密钥 ⇒ 直接回 `NO_LLM_MESSAGE`（不检索、不装配面板、不进循环）；
        2. 装配**参考标准面板**——红线线索（关键词表 + 三维门控）、症状归一线索、
           科室候选票数、服务端已确认的挂号事实，全部作为"线索"交给模型；
        3. 把面板拼进系统提示，跑工具循环，由模型自己决定调什么工具、判不判急症。

        `patient_id` 由**调用方**给（HTTP 层从请求里取），经 `context=` 注入到工具执行体。
        它不进工具 schema，所以模型无法伪造"这是谁"；面板里的挂号事实也按它读。
        """
        if not self.has_llm():
            # 没有判断者就没有结论：这里什么都不查、什么都不兜底，只回一句话。
            return ClinicTurn(text=NO_LLM_MESSAGE, tool_calls=[], notes=["未连接大模型"])

        booking, booking_problems = _booking_facts(self.service, patient_id)
        facts = clinic_reference(
            message,
            links=_triage_links(),
            guard=self.guard,
            known_departments=DEPARTMENTS,
            booking=booking,
            notes=booking_problems,
        )

        from kubedocs.agent.llm_tool_caller import LiteLLMToolCaller

        caller = LiteLLMToolCaller(self.settings)
        # 面板拼进**同一条** system 消息（理由见 `build_system_prompt`）。
        prompt = build_system_prompt(self.tools, reference=facts.render())

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
        notes = list(turn.notes)
        if facts.red_line_is_reportable():
            notes.append("红线线索已随参考标准面板交给模型判断（本路径不做规则短路）")
        if not turn.answer:
            # 工具循环把"调用失败/输出不可解析/步数耗尽"都收成空 answer + 一句 note，
            # 所以这里统一回失败提示：不产出任何兜底内容（没有判断者就没有结论）。
            notes.append("模型没有产出可用回答：只回失败提示，不产出兜底内容")
        answer = turn.answer or LLM_FAILED_MESSAGE
        return ClinicTurn(
            text=answer,
            tool_calls=[
                {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "error": tc.error or None,
                    "result": None if tc.error else tc.result,
                }
                for tc in turn.tool_calls
            ],
            # **展示层检测**：判断权已经交给模型（结论写在正文里），这里只决定前端要不要
            # 弹急诊横幅。所以它只看模型这段文字里有没有出现「120 / 急诊」——
            # 这是展示逻辑，**不是医学判断**，也不改变回答内容。
            emergency=("120" in answer or "急诊" in answer),
            steps=turn.steps,
            llm_calls=turn.llm_calls,
            notes=notes,
            cost_usd=turn.cost_usd,
            unpriced_calls=turn.unpriced_calls,
            pending_action=_pending_action(turn),
            reference=facts.to_payload(),
        )


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
