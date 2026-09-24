"""LLM 驱动的 clinic agent：`/chat` 的 agent 化路径。

## 与 `ClinicAgent`（确定性图）的区别

| | ClinicAgent（图） | 本模块（工具循环） |
|---|---|---|
| 控制流由谁定 | **代码**（8 个硬编码回调 + 确定性边） | **LLM**（function calling 选工具、定步数） |
| 能自己决定"先查号源再问用户"吗 | 不能，顺序写死了 | 能 |

在此之前 `/chat` 走的是图。那张图有 checkpointer、有 interrupt、有红线强制边，
**但没有工具调用** —— 所以它是一台"带图外壳的流水线"，与 medical-agent 的
orchestrator 在"控制流由谁决定"这一点上并无本质区别。本模块补上这一条。

## 红线为什么**不能**做成工具

急诊识别是安全属性。若把它做成一个工具交给 LLM 选择，模型某次"觉得不用查"
就会漏掉急诊——**漏报急诊的代价不可接受**。所以红线在**进入循环之前**用确定性
规则判定：命中就直接返回急诊引导，LLM 没有任何机会参与这个决定。
这与图版本里"红线走 LLM 不可达的强制边"是同一个不变量，只是实现位置不同。

## 挂号的确认策略

`book_appointment` 有副作用（占用号源、不可撤销）。HTTP 上没法中断等确认，
所以策略是：**用户这一轮的原文里明确要求了挂号，才算确认**。
否则工具循环会拒绝执行并把"需要用户确认"回灌给模型。
`when_not` 也写了"用户没明确要求时绝对不要调"——两道防线。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kubedocs.agent.bootstrap import register_domain_tools
from kubedocs.agent.tool_loop import AgentTurn, run_tool_loop
from kubedocs.agent.tools import ToolRegistry
from kubedocs.config import Settings
from kubedocs.domains.clinic.pack import wants_booking
from kubedocs.domains.clinic.redline import RiskGuard, emergency_advice, emergency_menu

SYSTEM = """你是医院预问诊助手，帮助患者判断该挂哪个科、并协助挂号。

规则：
1. **只做导诊与挂号协助，不下诊断结论**，不推荐具体用药与剂量。
2. 需要资料时用工具查（科室列表、号源余量、知识库检索），不要凭记忆编造科室名或号源。
3. 挂号（book_appointment）只有在**用户明确要求挂号**时才调；调用前先用
   get_availability 确认有号，并把科室名说清楚。
4. 症状信息不足时就问一个最关键的问题，一次只问一件事。
5. 出现胸痛、呼吸困难、意识不清、大出血等紧急情况，提示立即拨打 120 或去急诊。
6. 用简体中文，简洁分点。"""


@dataclass
class ClinicTurn:
    """一轮问诊的结果：供 SSE 发 tool 事件与最终文本。"""

    text: str
    tool_calls: list[dict[str, Any]]
    emergency: bool = False
    steps: int = 0
    llm_calls: int = 0
    notes: list[str] | None = None


class ClinicReactRuntime:
    """把「注册工具 + 红线预检 + 工具循环」装配成一个可复用对象。

    为什么持有 `guard` 与 `service`：红线判定与号源库都是**有状态/需复用**的，
    每轮新建会丢掉号源库存（那会变成"挂完号又有了"的荒谬现象）。
    """

    def __init__(self, settings: Settings, service: Any,
                 retriever: Any | None = None) -> None:
        self.settings = settings
        self.service = service
        self.retriever = retriever
        self.guard = RiskGuard()
        # 注册工具是"接线"动作：不做它，run_tool_loop 拿到的 tools 会是空的
        register_domain_tools("clinic", service=service, retriever=retriever)

    @property
    def tools(self) -> list[Any]:
        return list(ToolRegistry.for_domain("clinic"))

    def has_llm(self) -> bool:
        from kubedocs.generate import has_api_key

        return has_api_key(self.settings.llm_model)

    def turn(self, message: str, *, history: list[dict[str, Any]] | None = None) -> ClinicTurn:
        """跑一轮。**红线先判**，命中则不进循环。"""
        hit = self.guard.check(message)
        if hit is not None:
            # 红线命中：确定性返回，LLM 无权参与
            text = emergency_menu(hit.word)
            return ClinicTurn(text=text, tool_calls=[
                {"tool": "check_red_line", "word": hit.word, "status": hit.status,
                 "experiencer": hit.experiencer, "temporality": hit.temporality}],
                emergency=True, notes=["红线命中（确定性规则，LLM 不可绕过）"])

        if not self.has_llm():
            return ClinicTurn(
                text=("当前没有配置大模型密钥，无法进行对话式问诊。"
                      "如需挂号请直接说明科室；紧急情况请立即拨打 120。"),
                tool_calls=[], notes=["未配置 LLM 密钥"])

        from kubedocs.agent.llm_tool_caller import LiteLLMToolCaller

        caller = LiteLLMToolCaller(self.settings)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": message})

        # 用户这一轮明确要求挂号 = 对他的取号确认；否则拒绝执行有副作用的工具
        user_asked_to_book = wants_booking(message)
        turn: AgentTurn = run_tool_loop(
            message, tools=self.tools, llm=caller, system=SYSTEM, max_steps=5,
            confirm=(lambda _call: True) if user_asked_to_book else None,
        )
        return ClinicTurn(
            text=turn.answer or "（模型没有给出内容，请再说一次或换个说法）",
            tool_calls=[{"tool": tc.name, "arguments": tc.arguments,
                         "error": tc.error or None,
                         "result": None if tc.error else tc.result}
                        for tc in turn.tool_calls],
            steps=turn.steps, llm_calls=turn.llm_calls, notes=list(turn.notes))


def emergency_text_for(message: str) -> str | None:
    """分离出来的红线判定，便于单测与复用。"""
    hit = RiskGuard().check(message)
    return emergency_advice(hit.word) if hit else None
