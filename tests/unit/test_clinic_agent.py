"""集成测试：把域资产与 agent 内核接起来，跑通问诊 / 挂号 / 紧急三条流程。

与 `test_kernel.py` 测控制流不同，这里测**装配**：真实用上症状词典、三维极性判定、
红线、号源库与假设打分器，接口对不上、来源归因写反、词表漏变体都会在这里露出来。

前置条件：问诊流程一律用非红线症状（咳嗽 / 发热 / 头痛…）。「胸痛」是急诊红线词，
一出现就走 emergency 分支、观测节点不执行、候选自然为空——那是红线在正常工作，
所以只有紧急流程用红线词。
"""

from __future__ import annotations

import pytest

from kubedocs.agent.kernel import ClinicAgent
from kubedocs.agent.state import EvidenceSource
from kubedocs.domains.clinic.mock import MockAppointmentProvider
from kubedocs.domains.clinic.pack import (
    BOOK_WORDS,
    build_clinic_deps,
    extract_evidence,
    invert_links,
    wants_booking,
)

# 症状 -> 候选（疾病 + 科室）。真实语料缺失也能测——这正是"规则轨必需、检索轨可选"的设计目标。
LINKS = {
    "咳嗽": ["支气管炎", "呼吸内科"],
    "发热": ["支气管炎", "感染科"],
    "头痛": ["偏头痛", "神经内科"],
    "胸闷": ["心绞痛", "心内科"],
}


def _agent(**kw):
    provider = kw.pop("provider", None) or MockAppointmentProvider()
    wiring = build_clinic_deps(provider, links=LINKS, **kw)
    return ClinicAgent(wiring.deps, max_turns=kw.get("max_turns", 4)), wiring


# 证据抽取与来源归因
class TestExtractEvidence:
    def test_volunteered_symptom_gets_full_weight(self):
        ev = extract_evidence("我最近咳嗽", 1)
        assert ev[0].symptom == "咳嗽"
        assert ev[0].source is EvidenceSource.VOLUNTEERED
        assert ev[0].weight == pytest.approx(1.0)

    def test_confirmed_after_suggestion_is_discounted(self):
        """系统建议后用户确认 → 权重减半。这是防建议性偏倚的核心机制。"""
        ev = extract_evidence("是的，有咳嗽", 2, suggested=["咳嗽"])
        assert ev[0].source is EvidenceSource.CONFIRMED
        assert ev[0].weight == pytest.approx(0.5)

    def test_negated_symptom_is_recorded_as_negative(self):
        ev = extract_evidence("我没有咳嗽", 1)
        assert ev and ev[0].present is False
        assert ev[0].source is EvidenceSource.NEGATED
        assert ev[0].weight < 0

    def test_negated_after_suggestion_is_even_weaker(self):
        ev = extract_evidence("没有咳嗽", 2, suggested=["咳嗽"])
        assert ev[0].source is EvidenceSource.NEGATED_AFTER_SUGGESTION
        assert abs(ev[0].weight) < 0.5

    def test_symptom_free_text_yields_nothing(self):
        assert extract_evidence("今天天气不错", 1) == []


class TestLinkInversion:
    def test_invert_groups_symptoms_by_candidate(self):
        inv = invert_links(LINKS)
        assert inv["呼吸内科"] == ["咳嗽"]
        assert set(inv["支气管炎"]) == {"咳嗽", "发热"}

    def test_empty_links_is_fine(self):
        assert invert_links({}) == {}


# 装配完整性
class TestWiring:
    def test_missing_corpus_is_reported_not_fatal(self):
        """语料缺失是**正常情况**（受限语料不入库），必须只记一条说明而不报错。"""
        _a, wiring = _agent()
        assert wiring.links == LINKS
        w2 = build_clinic_deps(MockAppointmentProvider(), links={})
        assert w2.notes and "语料缺失" in w2.notes[0]

    def test_wants_booking_handles_chinese_variants(self):
        """中文意图词表必须列变体：「挂号」不是「帮我挂个号」的子串。"""
        assert "挂号" not in "帮我挂个号"
        for text in ("帮我挂号", "帮我挂个号", "我想预约", "能不能取号"):
            assert wants_booking(text) is True
        assert wants_booking("我最近有点累") is False
        assert len(BOOK_WORDS) >= 3

    @pytest.mark.parametrize(
        "text",
        [
            "我不想挂号",
            "不要预约",
            "不用取号",
            "先别帮我挂号",
            "别挂号",
            "别帮我预约",
            "不要帮我挂号",
        ],
    )
    def test_negated_booking_intent_is_not_a_request(self, text):
        """挂号意图必须判否定，否则"我不想挂号"会命中「挂号」并占掉一个号源。"""
        assert wants_booking(text) is False, f"{text!r} 被判成了挂号请求"

    @pytest.mark.parametrize("text", ["特别疼想挂号", "别的科能不能挂号"])
    def test_imperative_negation_does_not_break_normal_phrases(self, text):
        """祈使否定「别」不能误伤复合词：「特别」「别的」不是否定。

        反例必须真的含挂号词：`别的科的号能挂吗` 里没有「挂号」，判 False 是对的。
        """
        assert wants_booking(text) is True, f"{text!r} 被误判为否定"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("确认", True),
            ("是的", True),
            ("好的", True),
            ("嗯", True),
            ("确认挂号", True),
            ("不用了", False),
            ("先不挂", False),
            ("取消", False),
            ("别确认", False),
        ],
    )
    def test_confirm_booking_reads_consent(self, text, expected):
        """确认判定同样要能识别否定：「不用了」不是同意。"""
        from kubedocs.domains.clinic.pack import confirm_booking

        assert confirm_booking(text) is expected


# 紧急流程
class TestEmergencyFlowEndToEnd:
    def test_red_line_forces_emergency_without_questioning(self):
        agent, _ = _agent()
        state = agent.start("我现在胸痛", thread_id="e1")
        assert state["red_line"] is not None
        assert state["status"] == "emergency"
        assert "120" in state["answer"]["text"]
        assert agent.pending_question(thread_id="e1") is None

    @pytest.mark.parametrize(
        "text,why",
        [
            ("我老公胸痛，我没事", "家属"),
            ("我去年胸痛过一次，现在好了", "既往"),
            ("我没有胸痛", "否认"),
            ("如果胸痛怎么办", "假设"),
        ],
    )
    def test_no_false_emergency(self, text, why):
        """装配后仍然不误报——三维门控在真实链路上生效的证明。"""
        agent, _ = _agent()
        assert agent.start(text, thread_id="e2")["red_line"] is None, f"误报急诊（{why}）"

    def test_emergency_path_records_evidence(self):
        agent, _ = _agent()
        assert [e.symptom for e in agent.start("胸痛", thread_id="e3")["evidence"]] == ["胸痛"]

    def test_red_line_word_is_never_suggested_as_a_hypothesis(self):
        from kubedocs.agent import hypothesis
        from kubedocs.domains.clinic.redline import EMERGENCY_RED_LINE

        for word in EMERGENCY_RED_LINE:
            assert hypothesis.is_safe_to_suggest(word) is False


# 问诊流程
class TestConsultationFlowEndToEnd:
    def test_symptoms_accumulate_across_turns(self):
        agent, _ = _agent()
        agent.start("我咳嗽", thread_id="c1")
        agent.resume("还有点发热", thread_id="c1")
        symptoms = {e.symptom for e in agent.snapshot(thread_id="c1")["evidence"]}
        assert {"咳嗽", "发热"} <= symptoms

    def test_candidates_are_scored_from_evidence(self):
        agent, _ = _agent()
        state = agent.start("我咳嗽而且发热", thread_id="c2")
        assert state["candidates"], "有证据却没有候选分数"
        assert state["candidates"].get("支气管炎", 0) > 0

    def test_negated_symptom_lowers_its_candidate(self):
        """否定证据必须真的影响排序——否则鉴别诊断无从收敛。"""
        agent, _ = _agent()
        scores = agent.start("我咳嗽，但没有发热", thread_id="c3")["candidates"]
        assert scores.get("支气管炎", 0) > scores.get("感染科", 0)

    def test_questioning_is_bounded(self):
        agent, _ = _agent(max_turns=3)
        agent.start("我不舒服", thread_id="c4")
        for _ in range(8):
            if agent.pending_question(thread_id="c4") is None:
                break
            agent.resume("嗯，有点", thread_id="c4")
        snap = agent.snapshot(thread_id="c4")
        assert snap["status"] == "done"
        assert snap["turn"] <= 3

    def test_conservative_answer_when_no_candidates(self):
        agent, _ = _agent()
        state = agent.start("今天天气不错", thread_id="c5")
        assert state["status"] == "done"
        assert "就医" in state["answer"]["text"] or "120" in state["answer"]["text"]

    def test_answer_lists_directions_not_a_diagnosis(self):
        """只给方向、不给诊断结论——这是医疗域刻意的边界。

        `max_turns=1` 强制第一轮就作答；有候选时 agent 默认先追问，不能直接断言 answer。
        """
        agent, _ = _agent(max_turns=1)
        state = agent.start("我咳嗽", thread_id="c6")
        text = state["answer"]["text"]
        assert "不能替代医生诊断" in text


# 挂号流程
class TestBookingFlowEndToEnd:
    def test_booking_picks_a_real_department_not_a_disease(self):
        """候选集里疾病名与科室名混在一起，挂号必须挑出真的可挂号科室。

        直接取 Top1 会拿到「支气管炎」这种疾病名，`resolve_department` 会失败。
        """
        agent, wiring = _agent()
        agent.start("我咳嗽", thread_id="b1")
        # 第一轮回答"帮我挂号" = 表达**意图** -> 停在确认；第二轮明确确认才下单
        agent.resume("帮我挂号", thread_id="b1")
        state = agent.resume("确认", thread_id="b1")
        ticket = state["ticket"]
        known = {d["name"] for d in wiring.service.departments()}  # type: ignore[union-attr]
        assert ticket.get("department") in known, f"挂到了非科室：{ticket}"

    def test_ticket_is_persisted_to_the_provider(self):
        agent, wiring = _agent()
        agent.start("我咳嗽", thread_id="b2")
        agent.resume("帮我挂号", thread_id="b2")
        agent.resume("确认", thread_id="b2")
        assert wiring.service is not None
        assert wiring.service.my_appointments("anon") != []

    def test_booking_without_any_candidate_reports_human_readable_error(self):
        """没有候选就挂不了号：必须给一句人话 + 可选科室，而不是抛异常。"""
        agent, _ = _agent()
        agent.start("帮我挂号", thread_id="b3")
        state = agent.resume("确认", thread_id="b3")
        ticket = state["ticket"]
        assert "error" in ticket
        assert ticket["error"], "错误信息不能为空"

    def test_declining_booking_intent_books_nothing(self):
        """否认挂号的话不得占号（"我不想挂号"这类否定意图不是下单）。"""
        agent, wiring = _agent()
        agent.start("我咳嗽", thread_id="b4")
        state = agent.resume("我不想挂号", thread_id="b4")
        assert state["ticket"] == {}, "用户否认挂号却仍然占了号"
        assert wiring.service is not None
        assert wiring.service.my_appointments("anon") == []
