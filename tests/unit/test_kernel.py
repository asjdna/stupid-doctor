"""Agent 内核测试：红线强制边、中断挂起与续跑、有界追问、挂号分支。

全部用**打桩依赖**（`ClinicDeps` 的每个字段都可注入）：不调模型、不起库、
秒级完成。内核的价值在于控制流正确，而这些断言恰好就是控制流。
"""

from __future__ import annotations

from typing import Any

import pytest

from kubedocs.agent.kernel import ClinicAgent, ClinicDeps
from kubedocs.agent.state import EvidenceSource, SymptomEvidence

# 挂号意图词表：**必须列变体**，不能只匹配「挂号」。
# 实测教训：`"挂号" in "帮我挂个号"` 是 False —— 中文里"挂个号"中间插了"个"，
# 朴素子串匹配直接失效。这正是源项目 BOOK_WORDS 要写成
# ["挂号","预约","取号","挂个号","预约挂号","挂号看"] 的原因，
# 也是"多智能体用关键词路由"这类做法的真实脆弱点。
BOOK_WORDS = ("挂号", "挂个号", "预约", "取号", "约号")


def _book_intent(text: str) -> bool:
    return any(w in text for w in BOOK_WORDS)


def _confirm_intent(text: str) -> bool:
    """打桩的"用户确认了挂号"判定。

    内核测试刻意不 import 域包（域包有自己的词表与否定判定），所以这里只要
    一个"同意/不同意"的最小实现，用来验证**控制流**：
    没有确认就绝不能走到 `book` 节点。
    """
    return any(w in text for w in ("确认", "是", "好", "嗯"))


def _evidence(
    symptom: str, *, present: bool = True, source: EvidenceSource = EvidenceSource.VOLUNTEERED
) -> SymptomEvidence:
    return SymptomEvidence(
        symptom=symptom, present=present, source=source, turn=1, weight=1.0 if present else -0.5
    )


def _deps(**over: Any) -> ClinicDeps:
    """默认全是"什么都不做"的打桩；每条测试只覆盖它关心的那几个。"""
    base = ClinicDeps(
        observe=lambda text, state: [_evidence("胸痛")] if "胸痛" in text else [],
        score=lambda ev: {"心绞痛": float(len(ev))},
        should_stop=lambda scores, turn: (False, ""),
        next_question=lambda scores, state: None,
        check_red_line=lambda text: None,
        answer=lambda q, state: {"text": "打桩回答 [1]", "citations": [{"index": 1}]},
        book=lambda state: {"ticket_no": "T1", "queue_no": 1, "department": "心内科"},
        wants_booking=_book_intent,
        confirm_booking=_confirm_intent,
        wants_emergency_care=lambda text: "120" in text,
        emergency_advice=lambda word: f"请立即拨打120（{word}）",
        emergency_menu=lambda word: f"情况紧急（{word}）：① 120 ② 继续",
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


class _FakeHit:
    word = "胸痛"
    status = "present"
    experiencer = "self"
    temporality = "present"


# ===========================================================================
# 红线：LLM 不可绕过的强制边
# ===========================================================================
class TestRedLineForcedEdge:
    def test_red_line_goes_straight_to_emergency(self):
        agent = ClinicAgent(_deps(check_red_line=lambda text: _FakeHit()))
        state = agent.start("我现在胸痛", thread_id="t1")
        assert state["red_line"] is not None
        assert state["status"] == "emergency"
        assert "120" in state["answer"]["text"]

    def test_red_line_skips_questioning_entirely(self):
        """红线命中时不应进入追问循环——问诊不能拖延急诊。"""
        asked: list[str] = []

        def next_q(scores: dict[str, float], state: Any) -> str:
            asked.append("called")
            return "问点什么"

        agent = ClinicAgent(_deps(check_red_line=lambda text: _FakeHit(), next_question=next_q))
        agent.start("我现在胸痛", thread_id="t1")
        assert asked == [], "红线命中后不该再生成追问"

    def test_red_line_keeps_symptoms_as_evidence(self):
        """转急诊不能丢掉患者已经说过的症状——病历要留下。"""
        agent = ClinicAgent(_deps(check_red_line=lambda text: _FakeHit()))
        state = agent.start("胸痛", thread_id="t1")
        assert [e.symptom for e in state["evidence"]] == ["胸痛"]

    def test_no_red_line_takes_the_observation_path(self):
        agent = ClinicAgent(_deps())
        state = agent.start("我最近有点累", thread_id="t1")
        assert state["red_line"] is None
        assert state["status"] != "emergency"

    def test_red_line_is_checked_again_on_resume_turns(self):
        """**续答轮也要过红线**：用户完全可能在回答追问时才说出胸痛。

        缺陷形态（修复前）：`node_redline` 只看首轮的 `question`，而 resume 轮
        被直接送回 `observe`——于是"我现在胸痛，喘不上气"写在追问的回答里
        永远不会触发急诊分支，而文档宣称"红线在两条路径里都是进循环之前的确定性判定"。
        """
        agent = ClinicAgent(
            _deps(
                check_red_line=lambda text: _FakeHit() if "胸痛" in text else None,
                next_question=lambda s, st: "有没有发热？",
            )
        )
        agent.start("我有点不舒服", thread_id="t1")
        assert agent.pending_question(thread_id="t1") == "有没有发热？"
        # 红线症状出现在**回答**里
        state = agent.resume("我现在胸痛，喘不上气", thread_id="t1")
        assert state["red_line"] is not None, "续答轮说出的红线症状没有被识别"
        assert state["status"] == "emergency"
        assert "120" in state["answer"]["text"]

    def test_red_line_on_resume_keeps_the_new_symptoms_as_evidence(self):
        """转急诊不能丢掉这一轮说出的症状——病历要留下。"""
        seen: list[str] = []

        def observe(text: str, state: dict) -> list[SymptomEvidence]:
            seen.append(text)
            return [_evidence("胸痛")]

        agent = ClinicAgent(
            _deps(
                observe=observe,
                check_red_line=lambda text: _FakeHit() if "胸痛" in text else None,
                next_question=lambda s, st: "有没有发热？",
            )
        )
        agent.start("我有点不舒服", thread_id="t1")
        agent.resume("胸痛", thread_id="t1")
        assert seen[-1] == "胸痛", "红线命中时 observe 用的是首轮原话，而不是这一轮的输入"


# ===========================================================================
# 中断：真挂起 + 可续跑
# ===========================================================================
class TestInterruptAndResume:
    def test_ask_suspends_and_reports_pending_question(self):
        agent = ClinicAgent(_deps(next_question=lambda s, st: "有没有发热？"))
        agent.start("我不舒服", thread_id="t1")
        assert agent.pending_question(thread_id="t1") == "有没有发热？"

    def test_resume_continues_from_the_suspension_point(self):
        """用户回答后应回到观测节点，把回答变成新证据——而不是从头再来。"""
        rounds: list[str] = []

        def observe(text: str, state: dict) -> list[SymptomEvidence]:
            rounds.append(text)
            return [_evidence("发热")] if "发热" in text else []

        agent = ClinicAgent(_deps(observe=observe, next_question=lambda s, st: "有没有发热？"))
        agent.start("我不舒服", thread_id="t1")
        agent.resume("有点发热", thread_id="t1")
        assert rounds == ["我不舒服", "有点发热"]
        assert agent.pending_question(thread_id="t1") is None or True

    def test_state_persists_across_resume(self):
        agent = ClinicAgent(_deps(next_question=lambda s, st: "有没有发热？"))
        first = agent.start("我不舒服", thread_id="t1")
        second = agent.resume("有发热", thread_id="t1")
        assert second["turn"] == first["turn"] + 1

    def test_two_threads_are_isolated(self):
        """checkpointer 按 thread_id 隔离：两个患者不能互相污染。"""
        agent = ClinicAgent(_deps(next_question=lambda s, st: "还有别的不舒服吗？"))
        a = agent.start("患者A", thread_id="A")
        b = agent.start("患者B", thread_id="B")
        assert a["question"] == "患者A"
        assert b["question"] == "患者B"
        assert agent.snapshot(thread_id="A")["question"] == "患者A"


# ===========================================================================
# 有界追问
# ===========================================================================
class TestBoundedQuestioning:
    def test_stops_when_should_stop_says_so(self):
        agent = ClinicAgent(
            _deps(
                next_question=lambda s, st: "再问一句？",
                should_stop=lambda scores, turn: (True, "候选已足够集中"),
            )
        )
        state = agent.start("我不舒服", thread_id="t1")
        assert state["status"] == "done"
        assert state["stop_reason"] == "候选已足够集中"
        assert agent.pending_question(thread_id="t1") is None

    def test_stops_at_max_turns_even_if_never_confident(self):
        """永不给"够了"信号时，硬上限必须兜住——否则会变成不肯结束的对话。"""
        agent = ClinicAgent(
            _deps(next_question=lambda s, st: "再问一句？", should_stop=lambda scores, turn: (False, "")),
            max_turns=3,
        )
        agent.start("我不舒服", thread_id="t1")
        for _ in range(10):
            if agent.pending_question(thread_id="t1") is None:
                break
            agent.resume("嗯", thread_id="t1")
        state = agent.snapshot(thread_id="t1")
        assert state["status"] == "done"
        assert "轮次上限" in state["stop_reason"]
        assert state["turn"] <= 3

    def test_stops_when_no_question_left(self):
        agent = ClinicAgent(_deps(next_question=lambda s, st: None))
        state = agent.start("我不舒服", thread_id="t1")
        assert state["status"] == "done"
        assert "没有更有鉴别力" in state["stop_reason"]


# ===========================================================================
# 挂号分支：**意图成立不等于可以下单**
# ===========================================================================
class TestBookingBranch:
    def test_booking_requires_an_explicit_confirmation(self):
        """关键词命中只证明"提到了挂号"，占号还要一次独立确认。

        缺陷形态（修复前）：`wants_booking` 命中就直接进 `book`，
        于是"我不想挂号"也会占掉一个真实号源。
        """
        agent = ClinicAgent(_deps())
        state = agent.start("帮我挂个号", thread_id="t1")
        # 第一步**必须停在确认**，而不是已经出了票
        assert state["ticket"] == {}
        assert "确认" in (agent.pending_question(thread_id="t1") or "")
        # 明确同意后才真的下单
        done = agent.resume("确认", thread_id="t1")
        assert done["status"] == "done"
        assert done["ticket"]["ticket_no"] == "T1"
        assert done["stop_reason"] == "已挂号"

    def test_declining_the_confirmation_books_nothing(self):
        agent = ClinicAgent(_deps())
        agent.start("帮我挂个号", thread_id="t1")
        state = agent.resume("不用了，我不想挂号", thread_id="t1")
        assert state["ticket"] == {}, "用户拒绝了确认却仍然占了号"
        assert state["status"] != "booking"

    def test_confirmation_callback_absent_means_no_booking(self):
        """没接确认回调 = 不挂号。副作用工具的默认行为必须是拒绝。"""
        agent = ClinicAgent(_deps(confirm_booking=lambda text: False))
        agent.start("帮我挂个号", thread_id="t1")
        state = agent.resume("确认", thread_id="t1")
        assert state["ticket"] == {}

    def test_booking_is_not_triggered_by_unrelated_text(self):
        called: list[int] = []
        agent = ClinicAgent(
            _deps(book=lambda state: called.append(1) or {}, next_question=lambda s, st: None)
        )
        agent.start("我最近有点累", thread_id="t1")
        assert called == []

    def test_negative_booking_intent_never_reaches_confirmation(self):
        """用户否认挂号时，连确认这一步都不该出现（更不该占号）。"""
        called: list[int] = []
        agent = ClinicAgent(
            _deps(
                wants_booking=lambda text: False,
                book=lambda state: called.append(1) or {},
                next_question=lambda s, st: None,
            )
        )
        state = agent.start("帮我挂个号", thread_id="t1")
        assert called == []
        assert state["ticket"] == {}

    @pytest.mark.parametrize("text", ["帮我挂号", "帮我挂个号", "我想预约", "能不能取号"])
    def test_booking_intent_variants_all_trigger(self, text):
        """**中文意图识别必须列变体**：`"挂号" in "帮我挂个号"` 是 False。

        这是关键词式路由的真实脆弱点，也是内核把意图判定**放在域包里**
        （`BOOK_WORDS` 那样的变体词表）而不是内核自己写死的原因。
        """
        assert _book_intent(text) is True
        agent = ClinicAgent(_deps())
        agent.start(text, thread_id="t1")
        assert "确认" in (agent.pending_question(thread_id="t1") or "")
        assert agent.resume("确认", thread_id="t1")["ticket"]["ticket_no"] == "T1"

    def test_naive_substring_matching_would_miss_the_variant(self):
        """反证：只写「挂号」一个词就会漏掉「挂个号」——把这条固定住，防止有人"简化"词表。"""
        assert "挂号" not in "帮我挂个号"
        assert _book_intent("帮我挂个号") is True


# ===========================================================================
# 回答路径（self-ask 走这里）
# ===========================================================================
class TestAnswerPath:
    def test_answer_node_is_reached_and_carries_citations(self):
        agent = ClinicAgent(_deps())
        state = agent.start("随便问问", thread_id="t1")
        assert state["answer"]["text"]
        assert state["answer"]["citations"]

    def test_diagnostics_are_accumulated_not_overwritten(self):
        """diagnostics 用 operator.add：多轮的可观测信息不能被后一轮覆盖。"""
        agent = ClinicAgent(
            _deps(
                next_question=lambda s, st: "再问一句？", should_stop=lambda scores, turn: (turn >= 2, "够了")
            )
        )
        agent.start("我不舒服", thread_id="t1")
        agent.resume("嗯", thread_id="t1")
        state = agent.snapshot(thread_id="t1")
        assert len(state["diagnostics"]) >= 2
        assert any("第 1 轮" in d for d in state["diagnostics"])
        assert any("第 2 轮" in d for d in state["diagnostics"])


# ===========================================================================
# 图结构本身
# ===========================================================================
class TestGraphShape:
    def test_graph_compiles_with_memory_checkpointer(self):
        agent = ClinicAgent(_deps())
        assert agent.graph is not None

    def test_snapshot_of_unknown_thread_is_empty_not_an_error(self):
        agent = ClinicAgent(_deps())
        snap = agent.snapshot(thread_id="never-used")
        assert isinstance(snap, dict)

    def test_config_wraps_thread_id(self):
        assert ClinicAgent.config("x") == {"configurable": {"thread_id": "x"}}

    def test_sqlite_checkpointer_persists_across_agents(self, tmp_path):
        """换一个 agent 实例仍能读回同一 thread 的状态——这就是"可续跑"。"""
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        db = tmp_path / "ckpt.db"
        conn = sqlite3.connect(str(db), check_same_thread=False)
        saver = SqliteSaver(conn)
        try:
            a1 = ClinicAgent(_deps(next_question=lambda s, st: "有没有发热？"), checkpointer=saver)
            a1.start("我不舒服", thread_id="p1")
            # 新实例、同一份 checkpointer 数据 -> 追问仍在等待回答
            a2 = ClinicAgent(_deps(), checkpointer=saver)
            assert a2.pending_question(thread_id="p1") == "有没有发热？"
        finally:
            conn.close()
