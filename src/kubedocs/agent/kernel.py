"""Agent 内核：图 + 持久化 + 中断 + 有界追问。

## 图的形状（clinic 域）

```
START → redline(确定性红线) ─命中→ emergency(给建议) → END
                │不命中
                ↓
            observe(抽证据 + 候选打分) → decide(停/问/挂号/确认)
                          ┌────────────┴─────────────┐
                          ↓                          ↓
                    ask(追问, interrupt 挂起)   confirm(挂号确认, interrupt 挂起)
                          │用户回答                  │用户回答
                          └──────→ redline ↺          ├─同意→ book → END
                                                     └─不同意→ answer → END
```

**每一轮的输入都要重新过 `redline`**（首轮是 `question`，续答轮是 `resume_text`）：
只在首轮判红线，会让"用户在回答追问时才说出胸痛/呼吸困难"绕过急诊分支——
那是安全属性上的漏判，不是体验问题。同理，`book` 之前必须有 `confirm`：
**一个关键词命中就占用号源，不是"用户要求挂号"的充分证据**（"我不想挂号"也含关键词）。

## 三个刻意的设计

1. **红线走 LLM 不可达的强制边**：`redline` 节点之后由**确定性函数**决定是否转急诊，
   没有任何工具或模型输出能让它改道。急诊识别是安全属性，不是可协商的判断。
   该节点在**每一轮**都会被进入，所以"续答轮说到红线症状"同样会被拦下。

2. **追问用 `interrupt()` 真挂起**，而不是"在内存里记个标志位等下一轮"。
   区别在于：挂起的状态由 checkpointer 保存，换实例/进程重启后能原地续跑；
   标志位方案一旦进程重启就丢上下文（源项目就是标志位方案）。

3. **轮次有硬上限**：`decide` 里除了"候选领先度达标"还有 `max_turns` 兜底。
   无上限的追问会变成一个不肯结束的对话——这是 agent 最常见的失败模式之一。

## 依赖注入

节点用闭包捕获 `ClinicDeps`，所以测试可以注入打桩的红线/号源/检索，
不需要起模型也不需要起库。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kubedocs.agent.state import (
    ClinicState,
    SymptomEvidence,
    empty_state,
)

if TYPE_CHECKING:
    # 只在类型检查时导入：langgraph 的导入代价不该由 `import kubedocs.agent.kernel` 来付
    from langgraph.checkpoint.base import BaseCheckpointSaver

# 追问上限的默认值。3 轮以内信息基本到手，超过 6 轮用户会烦——这既是体验也是成本约束。
DEFAULT_MAX_TURNS = 6


@dataclass
class ClinicDeps:
    """内核需要的一切外部能力，全部可替换（测试注入打桩）。"""

    observe: Callable[[str, ClinicState], list[SymptomEvidence]] = field(default=lambda text, state: [])
    """从用户这句话里抽症状证据（含来源归因）。"""

    score: Callable[[Sequence[SymptomEvidence]], dict[str, float]] = field(default=lambda ev: {})
    """按证据给候选打分。"""

    should_stop: Callable[[dict[str, float], int], tuple[bool, str]] = field(
        default=lambda scores, turn: (turn >= DEFAULT_MAX_TURNS, "达到轮次上限")
    )
    """是否结束追问，返回 (是否结束, 原因)。"""

    next_question: Callable[[dict[str, float], ClinicState], str | None] = field(
        default=lambda scores, state: None
    )
    """挑下一个问题（可能带试探性假设）；None 表示没问题可问了。"""

    check_red_line: Callable[[str], Any | None] = field(default=lambda text: None)
    """急诊红线检查；返回非 None 即命中。"""

    answer: Callable[[str, ClinicState], dict[str, Any]] = field(
        default=lambda q, state: {"text": "", "citations": []}
    )
    """生成带引用的回答（self-ask 走这里）。"""

    book: Callable[[ClinicState], dict[str, Any]] = field(default=lambda state: {})
    """挂号（有副作用，仅在确认后调用）。"""

    wants_booking: Callable[[str], bool] = field(default=lambda text: False)
    """用户是否要求挂号。**必须能识别否定**："我不想挂号"不是挂号请求。"""

    confirm_booking: Callable[[str], bool] = field(default=lambda text: False)
    """用户对"确认挂号吗？"的回答是否表示同意。

    默认 `False` 是刻意的：**没接确认回调 = 不挂号**。
    副作用工具的默认行为必须是"拒绝"，而不是"先做了再说"。
    """

    wants_emergency_care: Callable[[str], bool] = field(default=lambda text: False)
    """用户在急诊二选一里是否选择立即就医。"""

    emergency_advice: Callable[[str], str] = field(default=lambda word: "")
    emergency_menu: Callable[[str], str] = field(default=lambda word: "")


# --------------------------------------------------------------------------- 节点
def _make_nodes(deps: ClinicDeps, max_turns: int) -> dict[str, Callable[..., Any]]:
    # 在这个作用域导入：节点是这里的闭包，名字必须在这里可见。
    # 放模块顶层会让 import kubedocs.agent.kernel 就付出 langgraph 的导入代价。
    from langgraph.types import interrupt

    def node_redline(state: ClinicState) -> dict[str, Any]:
        # **每一轮**的输入都要过红线：首轮是 question，续答轮是 resume_text。
        # 只看 question 会让"用户在回答追问时才说出胸痛"绕过急诊分支。
        text = str(state.get("resume_text") or state.get("question", ""))
        hit = deps.check_red_line(text)
        if hit is None:
            return {"diagnostics": ["红线未命中（确定性规则）"]}
        word = getattr(hit, "word", str(hit))
        # 红线命中也要把已说症状留下：不能因为转急诊就丢掉病历
        evidence = deps.observe(text, state)
        return {
            "red_line": {
                "word": word,
                "status": getattr(hit, "status", ""),
                "experiencer": getattr(hit, "experiencer", ""),
                "temporality": getattr(hit, "temporality", ""),
                "advice_given": False,
            },
            "evidence": evidence,
            "status": "emergency",
            "diagnostics": [f"红线命中「{word}」（确定性规则，LLM 不可绕过）"],
        }

    def node_emergency(state: ClinicState) -> dict[str, Any]:
        red = state.get("red_line") or {}
        word = str(red.get("word", ""))
        first = not red.get("advice_given")
        text = deps.emergency_menu(word) if first else deps.emergency_advice(word)
        red = dict(red)
        red["advice_given"] = True
        return {"red_line": red, "status": "emergency", "answer": {"text": text, "citations": []}}

    def node_observe(state: ClinicState) -> dict[str, Any]:
        # resume_text 有值说明这是"用户回答了追问"的那一轮；否则是首轮提问。
        text = state.get("resume_text") or state.get("question", "")
        evidence = deps.observe(str(text), state)
        merged = list(state.get("evidence", [])) + evidence
        scores = deps.score(merged)
        turn = int(state.get("turn", 0)) + 1
        return {
            "evidence": evidence,
            "candidates": scores,
            "turn": turn,
            "diagnostics": [f"第 {turn} 轮：新增证据 {len(evidence)} 条"],
        }

    def node_decide(state: ClinicState) -> dict[str, Any]:
        scores = dict(state.get("candidates", {}))
        turn = int(state.get("turn", 0))
        text = str(state.get("resume_text") or state.get("question", ""))
        if deps.wants_booking(text):
            return {"status": "booking", "stop_reason": "用户要求挂号"}
        stop, reason = deps.should_stop(scores, turn)
        if stop:
            return {"status": "answering", "stop_reason": reason}
        if turn >= max_turns:
            # 硬上限兜底：无上限的追问会变成不肯结束的对话（agent 最常见的失败模式之一）
            return {"status": "answering", "stop_reason": f"达到轮次上限 {max_turns}"}
        return {"status": "asking"}

    def node_ask(state: ClinicState) -> dict[str, Any]:
        question = deps.next_question(dict(state.get("candidates", {})), state)
        if not question:
            return {"status": "answering", "stop_reason": "没有更有鉴别力的问题可问"}
        # 真挂起：状态由 checkpointer 落盘，进程重启后可原地续跑
        answer = interrupt({"question": question, "turn": state.get("turn", 0)})
        # asked 用 operator.add 累加去重交给 next_question；这里只负责追加本次问题
        return {"last_question": question, "asked": [question], "resume_text": str(answer)}

    def node_answer(state: ClinicState) -> dict[str, Any]:
        result = deps.answer(state.get("question", ""), state)
        return {"answer": result, "status": "done", "stop_reason": state.get("stop_reason", "")}

    def node_book(state: ClinicState) -> dict[str, Any]:
        ticket = deps.book(state)
        return {"ticket": ticket, "status": "done", "stop_reason": "已挂号"}

    def node_confirm(state: ClinicState) -> dict[str, Any]:
        """挂号前的**显式确认**：真挂起，等用户回答。

        为什么要这一步：`wants_booking` 是关键词判定，它证明的是"用户提到了挂号"，
        不是"用户同意占用一个号源"。占号是**有副作用**的动作（会扣掉别人的号），
        所以必须有一次独立的确认，而不是靠一次关键词命中就下单。
        与工具循环路径的 `_confirm_side_effect` 是同一原则，只是这里确定性实现。
        """
        reply = interrupt(
            {
                "question": "确认要挂号吗？回复「确认」我就为您取号。",
                "confirm": "booking",
            }
        )
        text = str(reply)
        return {"resume_text": text, "booking_confirmed": bool(deps.confirm_booking(text))}

    return {
        "redline": node_redline,
        "emergency": node_emergency,
        "observe": node_observe,
        "decide": node_decide,
        "ask": node_ask,
        "confirm": node_confirm,
        "answer": node_answer,
        "book": node_book,
    }


# --------------------------------------------------------------------------- 路由（确定性）
def _after_redline(state: ClinicState) -> str:
    """红线之后的强制分支：**只看状态，不看模型输出**。"""
    return "emergency" if state.get("red_line") else "observe"


def _after_decide(state: ClinicState) -> str:
    status = state.get("status", "asking")
    if status == "booking":
        # 挂号意图成立**不代表可以下单**：先去 confirm 拿一次明确同意。
        return "confirm"
    if status == "asking":
        return "ask"
    return "answer"


def _after_confirm(state: ClinicState) -> str:
    """确认之后：同意才占号，否则回去作答（不产生任何副作用）。"""
    return "book" if state.get("booking_confirmed") else "answer"


def _after_ask(state: ClinicState) -> str:
    """追问之后分两路。

    - `answering`：**没有更有鉴别力的问题可问了** —— 必须直接去作答。
      若这里无条件回到 `observe`，就会形成一个"问不出来→再看一遍→还是问不出来"
      的空转循环，最终靠递归上限才停下（错误信息还很难懂）。
    - 其余（含 `interrupt` 被 resume 回来的情况）：**回 `redline`**，把用户的回答
      先过一遍确定性红线，再变成新证据。
      回到 `observe` 也"能跑"，但那会让续答轮绕开急诊判定——
      而用户完全可能在回答追问时才说出胸痛/呼吸困难。
    """
    return "answer" if state.get("status") == "answering" else "redline"


# --------------------------------------------------------------------------- 装配
def build_clinic_graph(
    deps: ClinicDeps,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    recursion_limit: int = 25,
) -> Any:
    """装配并编译 clinic 域的 agent 图。

    `checkpointer=None` 时用 `InMemorySaver`（测试/无盘场景）；
    传入 `SqliteSaver` 则状态落 SQLite，可断点续跑。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    nodes = _make_nodes(deps, max_turns)
    g = StateGraph(ClinicState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "redline")
    g.add_conditional_edges("redline", _after_redline, {"emergency": "emergency", "observe": "observe"})
    g.add_edge("emergency", END)
    g.add_edge("observe", "decide")
    g.add_conditional_edges("decide", _after_decide, {"ask": "ask", "answer": "answer", "confirm": "confirm"})
    # 续答轮先回 redline（每一轮都过确定性红线），而不是直接回 observe。
    g.add_conditional_edges("ask", _after_ask, {"redline": "redline", "answer": "answer"})
    g.add_conditional_edges("confirm", _after_confirm, {"book": "book", "answer": "answer"})
    g.add_edge("answer", END)
    g.add_edge("book", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class ClinicAgent:
    """对图的薄封装：把"启动 / 续跑 / 查状态"变成三个方法。

    存在意义是**不让调用方碰 LangGraph 的细节**（config、thread_id、Command），
    CLI 与 HTTP 层只认这三个方法。
    """

    def __init__(
        self,
        deps: ClinicDeps,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self.graph = build_clinic_graph(deps, max_turns=max_turns, checkpointer=checkpointer)
        self.max_turns = max_turns

    @staticmethod
    def config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(self, question: str, *, thread_id: str, patient_id: str = "anon") -> ClinicState:
        state = empty_state(question, patient_id=patient_id)
        return self.graph.invoke(state, self.config(thread_id))  # type: ignore[no-any-return]

    def resume(self, text: str, *, thread_id: str) -> ClinicState:
        """把用户对追问的回答送回图里，从挂起点续跑。"""
        from langgraph.types import Command

        return self.graph.invoke(Command(resume=text), self.config(thread_id))  # type: ignore[no-any-return]

    def snapshot(self, *, thread_id: str) -> ClinicState:
        got = self.graph.get_state(self.config(thread_id))
        values = getattr(got, "values", {})
        return dict(values)  # type: ignore[return-value]

    def pending_question(self, *, thread_id: str) -> str | None:
        """当前是否停在某个追问上（true = 在等用户回答）。"""
        got = self.graph.get_state(self.config(thread_id))
        for task in getattr(got, "tasks", ()) or ():
            for itr in getattr(task, "interrupts", ()) or ():
                value = getattr(itr, "value", None)
                if isinstance(value, dict) and value.get("question"):
                    return str(value["question"])
        return None
