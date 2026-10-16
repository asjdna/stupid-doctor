"""输出约束层（`clinical/verify.py`）的契约测试：**模型说的每一句都要能被事实校验**。

这一组用例存在的理由只有一个：把判断权交给大模型之后，最危险的不再是"规则太笨"，
而是**模型编造一件服务端没做过的事**。所以每个用例都对应一条"会让患者吃亏"的幻觉，
而不是在为某个函数凑覆盖率。
"""

from __future__ import annotations

from kubedocs.clinical.contracts import (
    ClinicalAssessment,
    DepartmentAdvice,
    ExamAdvice,
    RedFlag,
)
from kubedocs.clinical.evidence import (
    EvidenceBundle,
    EvidenceItem,
    RedLineSignal,
)
from kubedocs.clinical.reference import (
    BookingFact,
    DepartmentSignal,
    FactSet,
    SymptomSignal,
    build_facts,
)
from kubedocs.clinical.verify import (
    NO_EVIDENCE_ANSWER,
    facts_only_answer,
    strip_ungrounded,
    verify,
)

DEPARTMENTS = ("呼吸内科", "心内科", "神经内科", "消化内科", "皮肤科", "耳鼻喉科")


def _evidence(*ids: str) -> EvidenceBundle:
    return EvidenceBundle(
        items=[EvidenceItem(id=i, kind="knowledge", text=f"卡片 {i} 的正文", score=1.0) for i in ids]
    )


def _facts(*, evidence: EvidenceBundle | None = None, **kw: object) -> FactSet:
    base: dict[str, object] = {
        "evidence": evidence if evidence is not None else _evidence("K1"),
        "known_departments": DEPARTMENTS,
    }
    base.update(kw)
    return build_facts(**base)  # type: ignore[arg-type]


def _assessment(
    *,
    summary: str = "建议到呼吸内科就诊。",
    urgency: str = "routine",
    departments: list[str] | None = None,
    exams: list[ExamAdvice] | None = None,
    red_flags: list[RedFlag] | None = None,
) -> ClinicalAssessment:
    return ClinicalAssessment(
        urgency=urgency,  # type: ignore[arg-type]
        departments=[DepartmentAdvice(name=d) for d in (departments or ["呼吸内科"])],
        patient_summary=summary,
        exams=list(exams or []),
        red_flags=list(red_flags or []),
    )


class TestFabricatedBooking:
    """**本文件的核心用例**：服务端没挂上号，模型不许宣布挂上了。

    这类输出对患者是不可挽回的误导——他会以为号已经到手，不去复诊、不查状态。
    """

    def test_claiming_a_booking_that_never_happened_is_blocked(self):
        facts = _facts()
        prose = "已经为您挂上了呼吸内科的号，凭据号 SNP39TAF，请按时就诊。"
        v = verify(_assessment(summary=prose), prose, facts)
        assert not v.ok()
        assert "fabricated_booking" in v.kinds()

    def test_the_blocked_prose_never_reaches_the_patient(self):
        """拦下之后患者看到的只能是事实回答，**不能**再看到那句幻觉。"""
        facts = _facts()
        prose = "已经为您挂上了呼吸内科的号。"
        v = verify(_assessment(summary=prose), prose, facts)
        fallback = facts_only_answer(facts, reason="；".join(x.kind for x in v.blocking()))
        assert "挂上" not in fallback
        assert "本轮没有产生任何挂号或退号动作" in fallback

    def test_a_server_confirmed_booking_may_be_described(self):
        """反向：服务端真的确认了，模型就可以说——不能把正确的话一起拦掉。"""
        facts = _facts(
            booking=[
                BookingFact(
                    kind="confirmed",
                    department="呼吸内科",
                    date="2026-10-17",
                    time_slot="上午",
                    fee=35,
                    ticket_no="SNP39TAF",
                    queue_no=1,
                    status="paid",
                )
            ]
        )
        prose = "已为您取号：呼吸内科 2026-10-17 上午，凭据号 SNP39TAF，挂号费 35 元。"
        assert verify(_assessment(summary=prose), prose, facts).ok()

    def test_a_cancelled_booking_cannot_be_described_as_successful(self):
        facts = _facts(booking=[BookingFact(kind="cancelled", ticket_no="SNP39TAF", department="呼吸内科")])
        prose = "已经为您挂上了呼吸内科的号。"
        assert not verify(_assessment(summary=prose), prose, facts).ok()

    def test_a_fabricated_ticket_number_is_blocked(self):
        prose = "已取号，凭据号 ABCD2345，请按时候诊。"
        v = verify(_assessment(summary=prose), prose, _facts())
        assert "unknown_ticket" in v.kinds()
        assert not v.ok()


class TestUnknownWhenNothingRetrieved:
    """检索为空 ⇒ 只能如实说不知道，不许拿通用医学知识补。"""

    def test_medical_content_without_evidence_is_blocked(self):
        facts = _facts(evidence=EvidenceBundle())
        prose = "您的情况可能是急性支气管炎，建议做胸部 CT，可以先吃阿莫西林。"
        v = verify(
            _assessment(summary=prose, exams=[ExamAdvice(name="胸部 CT", evidence_ids=[])]),
            prose,
            facts,
        )
        assert "medical_without_evidence" in v.kinds()
        assert not v.ok()
        assert NO_EVIDENCE_ANSWER in v.feedback() or "没有检索到" in v.feedback()

    def test_a_plain_unknown_answer_passes(self):
        facts = _facts(evidence=EvidenceBundle())
        prose = NO_EVIDENCE_ANSWER
        assert verify(_assessment(summary=prose, departments=["呼吸内科"]), prose, facts).ok()

    def test_emergency_guidance_is_allowed_without_evidence(self):
        """安全例外：建议就医/拨 120 是**安全动作**，不是事实断言，检索为空时照样要给。

        否则"检索不到就说不知道"会把该报警的急症一起拦下来——拿安全换整洁。
        """
        facts = _facts(evidence=EvidenceBundle())
        prose = "请立即拨打 120 或前往最近的急诊科，不要等待在线建议。"
        assessment = _assessment(
            summary=prose,
            urgency="emergency",
            red_flags=[RedFlag(condition="急性冠脉综合征", severity="critical", why="胸痛伴大汗")],
        )
        assert verify(assessment, prose, facts).ok()


class TestGroundingChecks:
    def test_a_department_outside_the_table_is_flagged(self):
        facts = _facts()
        prose = "建议到呼吸内科就诊。"
        assessment = _assessment(departments=["呼吸内科", "火星科"])
        v = verify(assessment, prose, facts)
        assert "department_not_grounded" in v.kinds()
        cleaned = strip_ungrounded(assessment, v, facts)
        assert [d.name for d in cleaned.departments] == ["呼吸内科"]

    def test_stripping_never_leaves_zero_departments(self):
        """全都不接地时保留原样：否则"科室名不接地"会被掩盖成"模型没给科室"。"""
        facts = _facts()
        assessment = _assessment(departments=["火星科"])
        v = verify(assessment, "建议到火星科就诊。", facts)
        assert [d.name for d in strip_ungrounded(assessment, v, facts).departments] == ["火星科"]

    def test_a_cited_id_that_does_not_exist_is_flagged_and_stripped(self):
        facts = _facts(evidence=_evidence("K1"))
        assessment = _assessment(exams=[ExamAdvice(name="胸部 CT", evidence_ids=["K9"])])
        v = verify(assessment, "建议做胸部 CT。", facts)
        assert "fabricated_evidence" in v.kinds()
        assert strip_ungrounded(assessment, v, facts).exams == []

    def test_a_cited_id_that_exists_survives(self):
        facts = _facts(evidence=_evidence("K1"))
        assessment = _assessment(exams=[ExamAdvice(name="胸部 CT", evidence_ids=["K1"])])
        v = verify(assessment, "建议做胸部 CT。", facts)
        assert v.ok()
        assert len(strip_ungrounded(assessment, v, facts).exams) == 1

    def test_an_amount_the_server_never_gave_is_a_warning(self):
        facts = _facts()
        v = verify(_assessment(summary="挂号费 88 元。"), "挂号费 88 元。", facts)
        assert "unknown_amount" in v.kinds()
        assert v.ok(), "金额只是一条提示，不该把整段回答拦掉"

    def test_a_server_given_amount_passes(self):
        facts = _facts(
            booking=[
                BookingFact(
                    kind="available",
                    department="呼吸内科",
                    date="2026-10-17",
                    time_slot="上午",
                    fee=35,
                    remaining=9,
                )
            ]
        )
        assert verify(_assessment(summary="挂号费 35 元。"), "挂号费 35 元。", facts).ok()


class TestRedLineIsAClueNotAVerdict:
    """关键词命中**不再是强制边**：它作为线索进面板，判断权仍在模型。

    但"系统担心过这件事"不能被抹掉——模型不提，就留一条 warn。
    """

    def test_a_reportable_hit_that_the_model_ignores_is_recorded(self):
        bundle = EvidenceBundle(
            items=[EvidenceItem(id="K1", kind="knowledge", text="胸痛相关")],
            red_line=RedLineSignal(
                word="胸痛", assertion="present", experiencer="self", temporality="present"
            ),
        )
        facts = _facts(evidence=bundle)
        v = verify(_assessment(summary="建议到心内科就诊。"), "建议到心内科就诊。", facts)
        assert "red_line_unexplained" in v.kinds()
        assert v.ok(), "判断权在模型：这条只留痕，不拦截"

    def test_explaining_the_hit_clears_the_warning(self):
        bundle = EvidenceBundle(
            items=[EvidenceItem(id="K1", kind="knowledge", text="胸痛相关")],
            red_line=RedLineSignal(
                word="胸痛", assertion="present", experiencer="self", temporality="present"
            ),
        )
        facts = _facts(evidence=bundle)
        prose = "您提到的胸痛目前看不像急性冠脉事件，但若加重请立即就医。"
        assert "red_line_unexplained" not in verify(_assessment(summary=prose), prose, facts).kinds()

    def test_a_family_members_symptom_is_not_the_patients_complaint(self):
        """三维门控说"不是本人当前主诉"时，模型判 routine 不该被追着要解释。"""
        bundle = EvidenceBundle(
            items=[EvidenceItem(id="K1", kind="knowledge", text="胸痛相关")],
            red_line=RedLineSignal(
                word="胸痛", assertion="present", experiencer="other", temporality="present", reportable=False
            ),
        )
        facts = _facts(evidence=bundle)
        assert facts.red_line_is_reportable() is False
        assert "red_line_unexplained" not in verify(_assessment(), "建议到心内科就诊。", facts).kinds()


class TestReferencePanel:
    """面板是模型的依据，所以它自己不能说错话。"""

    def test_the_panel_states_there_was_no_booking_action(self):
        """这条**必须显式出现**：模型看到空白会按语气补一句"已挂上"。"""
        text = _facts().render()
        assert "没有" in text and "挂号" in text

    def test_the_panel_renders_every_clue_with_provenance(self):
        facts = _facts(
            symptoms=[SymptomSignal(matched="一直咳", canonical="咳嗽", corpus_keys=["持续性咳嗽"])],
            departments=[DepartmentSignal(name="呼吸内科", score=3.0, source="症状关联票数")],
            booking=[
                BookingFact(
                    kind="available",
                    department="呼吸内科",
                    date="2026-10-17",
                    time_slot="上午",
                    fee=35,
                    remaining=9,
                )
            ],
            notes=["检索式分诊不可用：索引未建"],
        )
        text = facts.render()
        for need in ("参考标准", "不是结论", "持续性咳嗽", "呼吸内科", "35", "余 9", "索引未建"):
            assert need in text, f"面板缺少 {need}"

    def test_the_panel_says_so_when_the_red_line_table_missed(self):
        assert "未命中" in _facts().render()

    def test_the_payload_is_machine_checkable(self):
        facts = _facts(symptoms=[SymptomSignal(matched="一直咳", canonical="咳嗽")])
        payload = facts.to_payload()
        assert payload["has_evidence"] is True
        assert payload["evidence_ids"] == ["K1"]
        assert payload["symptoms"][0]["canonical"] == "咳嗽"
        assert payload["known_departments"] == list(DEPARTMENTS)
