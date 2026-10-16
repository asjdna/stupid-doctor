"""临床管线的**输出约束**行为：判断交给模型之后，事实校验必须真的在拦人。

这一组用例只关心一件事：**服务端事实与模型输出冲突时，患者看到的是什么**。
其中最要紧的一条是"没挂上却说挂上"——那类输出会让患者以为号已到手，
不去复诊、不查状态，属于不可挽回的误导。

管线在这里被当成一个整体来测：给一个会吐特定 JSON 的假工具循环，
看它最终交给调用方的 `message` / `assessment` / `contract_violations`。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem
from kubedocs.clinical.pipeline import ClinicalPipeline
from kubedocs.llm_status import LLM_FAILED_MESSAGE, NO_LLM_MESSAGE


class _Backend:
    """检索后端替身：**不返回任何知识**（要测"检索为空"那条路时就靠它）。"""

    name = "stub"

    def __init__(self, hits: list[Any] | None = None) -> None:
        self._hits = hits or []

    def search(self, query: str, *, top_k: int = 10) -> list[Any]:
        return list(self._hits)

    def health(self) -> Any:
        return SimpleNamespace(ok=True, name=self.name, detail="")


def _payload(
    *,
    summary: str,
    urgency: str = "routine",
    departments: list[str] | None = None,
    exams: list[dict[str, Any]] | None = None,
    followups: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "urgency": urgency,
            "red_flags": [],
            "departments": [{"name": d, "confidence": 0.5} for d in (departments or ["呼吸内科"])],
            "differentials": [],
            "exams": exams or [],
            "medications": [],
            "patient_summary": summary,
            "followup_questions": followups or [],
            "citations": [],
        },
        ensure_ascii=False,
    )


class _Loop:
    """假工具循环：按脚本依次返回结果，并记录被调用了几次。"""

    def __init__(self, *texts: str, evidence: EvidenceBundle | None = None) -> None:
        self._texts = list(texts)
        self._evidence = evidence
        self.calls: list[str] = []

    def run(self, *, system: str, question: str, **_: Any) -> Any:
        self.calls.append(system)
        text = self._texts[min(len(self.calls) - 1, len(self._texts) - 1)]
        return SimpleNamespace(final_text=text, calls=[], evidence=self._evidence or EvidenceBundle())


def _pipeline(
    loop: Any,
    *,
    has_llm: Any = None,
    hits: list[Any] | None = None,
    reference_builder: Any = None,
) -> ClinicalPipeline:
    return ClinicalPipeline(
        backend=_Backend(hits),
        reasoner=SimpleNamespace(),
        tool_loop=loop,
        has_llm=has_llm,
        reference_builder=reference_builder,
    )


class TestNoModelNoContent:
    """没有判断者就没有结论：**不检索、不装配面板、不给任何兜底内容**。"""

    def test_without_a_model_key_the_only_reply_is_the_connection_message(self):
        loop = _Loop(_payload(summary="本该被忽略"))
        turn = _pipeline(loop, has_llm=lambda: False).advance("我咳嗽")
        assert turn.message == NO_LLM_MESSAGE
        assert turn.assessment is None
        assert loop.calls == [], "没有模型时不该再去调它"

    def test_a_failed_model_call_does_not_produce_clinical_content(self):
        """调用失败**不再**"按现有证据给出有限结论"——那种结论没有判断者。"""

        class _Boom:
            def run(self, **_: Any) -> Any:
                raise RuntimeError("upstream timeout")

        turn = _pipeline(_Boom()).advance("我咳嗽")
        assert turn.message == LLM_FAILED_MESSAGE
        assert turn.assessment is None, "没有判断者时不该留下任何结论"

    def test_an_unparseable_answer_does_not_produce_clinical_content(self):
        turn = _pipeline(_Loop("这不是 JSON")).advance("我咳嗽")
        assert turn.message == LLM_FAILED_MESSAGE
        assert turn.assessment is None


class TestFabricatedBookingIsBlocked:
    """**核心用例**：服务端没有挂号事实，模型不许宣布挂上了。"""

    def test_the_patient_never_sees_the_fabricated_sentence(self):
        loop = _Loop(_payload(summary="已经为您挂上呼吸内科的号，凭据号 ABCD2345。"))
        turn = _pipeline(loop).advance("帮我挂个号")
        assert "挂上" not in turn.message, "被拦下的那句话不许上屏"
        assert "已经为您" not in turn.message
        kinds = {v["kind"] for v in turn.contract_violations}
        assert "fabricated_booking" in kinds

    def test_the_retry_gets_the_violation_as_feedback(self):
        """第①级：带违规原因重生成一次。第二次的系统提示里必须带上那条反馈。"""
        loop = _Loop(
            _payload(summary="已经为您挂上了。"),
            _payload(summary="已生成待确认的挂号单，需要您点确认后才生效。"),
        )
        turn = _pipeline(loop).advance("帮我挂个号")
        assert len(loop.calls) == 2, "违规后应当重生成一次"
        assert "完成态说法" in loop.calls[1], "重生成时要把违规原因回灌给模型"
        assert "点确认后才生效" in turn.message
        # 第一轮的违规仍然留档：拦下的东西必须可追溯，不能因为第二轮回正了就当没发生
        assert {v["kind"] for v in turn.contract_violations} == {"fabricated_booking"}

    def test_a_second_violation_falls_back_to_a_facts_only_answer(self):
        loop = _Loop(
            _payload(summary="已经为您挂上了。"),
            _payload(summary="已经为您挂上了，凭据号 ABCD2345。"),
        )
        turn = _pipeline(loop).advance("帮我挂个号")
        assert len(loop.calls) == 2
        assert "挂上" not in turn.message
        assert "本轮没有产生任何挂号或退号动作" in turn.message

    def test_a_server_confirmed_booking_may_be_told_to_the_patient(self):
        """反向：服务端**确实**确认了，模型就可以说——不能把正确的话一起拦掉。"""
        from kubedocs.clinical.reference import BookingFact, build_facts

        facts = build_facts(
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
        loop = _Loop(
            _payload(summary="已为您取号：呼吸内科 2026-10-17 上午，凭据号 SNP39TAF，挂号费 35 元。")
        )
        turn = _pipeline(loop, reference_builder=lambda *_: facts).advance("帮我挂个号")
        assert "SNP39TAF" in turn.message
        assert turn.contract_violations == []
        assert len(loop.calls) == 1, "没有违规就不该重生成"


class TestNoEvidenceMeansUnknown:
    def test_medical_content_without_retrieved_evidence_is_blocked(self):
        loop = _Loop(
            _payload(
                summary="您可能是支气管炎，建议做胸部 CT。",
                exams=[{"name": "胸部 CT", "why": "排除肺炎", "evidence_ids": ["K1"]}],
            )
        )
        turn = _pipeline(loop).advance("我咳嗽")
        kinds = {v["kind"] for v in turn.contract_violations}
        assert "medical_without_evidence" in kinds
        assert "可能是支气管炎" not in turn.message
        assert "没有检索到相关资料" in turn.message

    def test_emergency_guidance_survives_without_evidence(self):
        """安全例外：拨 120 是安全动作，不是事实断言——检索为空也要给。"""
        loop = _Loop(
            _payload(
                summary="请立即拨打 120 或前往最近医院急诊科，不要等待在线建议。",
                urgency="emergency",
            )
        )
        turn = _pipeline(loop).advance("我胸痛")
        assert "120" in turn.message
        assert turn.action == "emergency"
        assert turn.contract_violations == []

    def test_with_retrieved_evidence_the_medical_content_passes(self):
        evidence = EvidenceBundle(
            items=[EvidenceItem(id="K1", kind="knowledge", text="咳嗽常见于上呼吸道感染")]
        )
        loop = _Loop(
            _payload(
                summary="结合资料看，可能是上呼吸道感染，建议到呼吸内科就诊。",
                exams=[{"name": "血常规", "why": "看有无细菌感染", "evidence_ids": ["K1"]}],
            ),
            evidence=evidence,
        )
        turn = _pipeline(loop).advance("我咳嗽")
        assert turn.contract_violations == []
        assert len(turn.assessment.exams) == 1


class TestReferenceIsCarriedOut:
    """面板与违规必须成对回传，否则事后无法区分"依据不够"与"模型没守规矩"。"""

    def test_the_reference_payload_is_exposed(self):
        loop = _Loop(_payload(summary="建议到呼吸内科就诊。"))
        turn = _pipeline(loop).advance("我咳嗽")
        assert turn.reference is not None
        assert "has_evidence" in turn.reference
        assert turn.reference["evidence_ids"] == []

    def test_the_custom_builder_is_used(self):
        seen: list[str] = []

        def builder(question: str, evidence: Any, patient_id: str) -> Any:
            seen.append(patient_id)
            from kubedocs.clinical.reference import build_facts

            return build_facts(notes=["域层装配生效"])

        loop = _Loop(_payload(summary="建议到呼吸内科就诊。"))
        turn = _pipeline(loop, reference_builder=builder).advance("我咳嗽", patient_id="p9")
        assert seen == ["p9"], "面板装配要拿到患者标识（号源事实是按人查的）"
        assert turn.reference is not None and "域层装配生效" in turn.reference["notes"]
