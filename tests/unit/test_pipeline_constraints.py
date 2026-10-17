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
    intent: str = "medical",
) -> str:
    return json.dumps(
        {
            "urgency": urgency,
            "intent": intent,
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
        # 这是**按设计的降级**，不是系统故障：服务端确实给出了（只含事实的）回答。
        assert turn.action == "answer"

    def test_booking_intent_drops_unrequested_medical_content_without_blocking(self):
        """事务性需求（挂号/号源）不需要检查用药：删掉那些条目，**但回答照常给**。

        证据是齐的，所以不该走 block（那样会把"呼吸内科今天有号"这句有用的话一起丢掉）。
        """
        loop = _Loop(
            _payload(
                summary="呼吸内科今天上午还有号，可以直接挂。",
                intent="booking",
                exams=[{"name": "血常规", "why": "看看感染方向", "evidence_ids": ["K1"]}],
            ),
            evidence=EvidenceBundle(
                items=[EvidenceItem(id="K1", kind="knowledge", text="血常规可提示感染方向", score=1.0)]
            ),
        )
        turn = _pipeline(loop).advance("我想挂呼吸内科")
        assert turn.action != "error"
        assert turn.message == "呼吸内科今天上午还有号，可以直接挂。"
        assert [v["kind"] for v in turn.contract_violations] == ["medical_content_not_requested"]
        assert turn.assessment is not None
        assert turn.assessment.exams == []
        assert len(loop.calls) == 1  # strip 级不重生成，不额外花钱

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
        """域层建造器被调用，且**拿到患者标识**（号源事实是按人查的）。

        它会被调用**两次**：循环之前装配"线索版"（进提示词）、循环之后装配"带证据版"
        （做校验与留档）。两次都必须拿到 `patient_id`——少了后一次，校验就不知道
        这个人到底有没有挂号事实。
        """
        seen: list[str] = []

        def builder(question: str, evidence: Any, patient_id: str) -> Any:
            seen.append(patient_id)
            from kubedocs.clinical.reference import build_facts

            return build_facts(notes=["域层装配生效"])

        loop = _Loop(_payload(summary="建议到呼吸内科就诊。"))
        turn = _pipeline(loop, reference_builder=builder).advance("我咳嗽", patient_id="p9")
        assert seen and set(seen) == {"p9"}, seen
        assert turn.reference is not None and "域层装配生效" in turn.reference["notes"]


class TestThePanelReachesTheModel:
    """**面板必须进提示词**，否则"规则资产作为线索、由模型判断"这条架构没有生效。

    这条是实测抓出来的：面板原先只在工具循环**跑完之后**装配，只用于校验与响应留档，
    而模型那一侧一条线索都看不到（它只看得到 system.md 与自己检索到的东西）。
    """

    def test_the_clue_panel_is_in_the_system_prompt(self):
        loop = _Loop(_payload(summary="建议到呼吸内科就诊。"))
        _pipeline(loop).advance("我咳嗽")
        assert loop.calls, "循环必须被调用"
        assert "参考标准" in loop.calls[0], "线索面板没进系统提示"

    def test_the_pre_loop_panel_defers_retrieval_instead_of_claiming_nothing_was_found(self):
        """循环前那一份不能给出"知识依据为空"的结论——那是循环**之后**的状态。

        混起来会让模型以为系统查过了、确实没有，于是既不检索也不再作答。
        （注意：`system.md` 的 R6 里本来就有"没有检索到任何资料"这句话，所以这里断言的是
        **面板那一节**的措辞，不是整段系统提示里有没有这几个字。）
        """
        loop = _Loop(_payload(summary="建议到呼吸内科就诊。"))
        _pipeline(loop).advance("我咳嗽")
        assert "由你自己决定" in loop.calls[0]
        assert "四、知识依据：**不在这一段里**" in loop.calls[0]

    def test_the_retry_prompt_also_carries_the_panel(self):
        loop = _Loop(
            _payload(summary="已经为您挂上了。"),
            _payload(summary="已生成待确认的挂号单，需要您点确认后才生效。"),
        )
        _pipeline(loop).advance("帮我挂个号")
        assert len(loop.calls) == 2
        assert "参考标准" in loop.calls[1], "重生成时把面板摘掉，等于换个更差的环境要求它做对"
        assert "完成态说法" in loop.calls[1]


def _payload_with_signal(
    *, summary: str, likelihood: str, severity: str = "severe", ruled_out_by: str = ""
) -> str:
    """带一条危险信号的模型输出（`angle` 必填，否则契约会记一条）。"""
    return json.dumps(
        {
            "urgency": "urgent",
            "red_flags": [
                {
                    "condition": "急性冠脉综合征",
                    "severity": severity,
                    "why": "胸痛伴气短",
                    "angle": "组合共现",
                    "likelihood": likelihood,
                    "discriminator": "",
                    "ruled_out_by": ruled_out_by,
                    "evidence_ids": [],
                }
            ],
            "departments": [{"name": "心内科", "confidence": 0.6}],
            "differentials": [],
            "exams": [],
            "medications": [],
            "patient_summary": summary,
            "followup_questions": [],
            "information_gaps": [],
            "confidence": 0.7,
            "citations": [],
        },
        ensure_ascii=False,
    )


class TestActiveInquiryLayer:
    """**第二层：主动问诊**。未排除的危险信号必须有下文，否则服务端补一句追问。"""

    def test_unresolved_signal_gets_a_deterministic_fallback_question(self):
        loop = _Loop(_payload_with_signal(summary="建议尽快就诊，注意休息。", likelihood="possible"))
        turn = _pipeline(loop).advance("我胸口有点闷")

        assert safety_outcome(turn)["followup"], "未排除的信号没有下文时必须补追问"
        assert "急性冠脉综合征" in turn.message
        assert turn.action == "ask", "补了追问就该是 ask，否则前端不会把它当成待回答的问题"

    def test_the_model_discriminator_is_used_when_present(self):
        """服务端优先用模型写的那句"能区分要急救/不用急救"的问题。"""
        payload = json.loads(_payload_with_signal(summary="建议尽快就诊。", likelihood="possible"))
        payload["red_flags"][0]["discriminator"] = "有没有伴随大汗或向左肩放射？"
        loop = _Loop(json.dumps(payload, ensure_ascii=False))
        turn = _pipeline(loop).advance("我胸口有点闷")

        assert "有没有伴随大汗或向左肩放射？" in turn.message

    def test_a_ruled_out_signal_gets_no_question(self):
        """写清排除依据的 `unlikely` 算排除：不该凭空多问一句。

        `ruled_out_by` 为空时按未排除处理（缺省值不许把安全问题抹掉）——那条由下一个用例钉住。
        """
        loop = _Loop(
            _payload_with_signal(
                summary="目前更像肌肉拉伤，建议门诊就诊。",
                likelihood="unlikely",
                ruled_out_by="按压可诱发、无大汗、无放射痛",
            )
        )
        turn = _pipeline(loop).advance("我胸口有点闷")

        assert safety_outcome(turn)["followup"] == ""
        assert "急性冠脉综合征" not in turn.message

    def test_an_unexplained_unlikely_is_treated_as_unresolved(self):
        """`unlikely` 但没写排除依据 ⇒ **仍算未排除**，照样要追问。"""
        loop = _Loop(_payload_with_signal(summary="建议门诊就诊。", likelihood="unlikely"))
        turn = _pipeline(loop).advance("我胸口有点闷")

        assert safety_outcome(turn)["followup"], "没有依据的「已排除」不算排除"
        assert turn.action == "ask"

    def test_emergency_does_not_ask_first(self):
        """判了急症就先给动作，不许反过来先追问（安全性优先）。"""
        loop = _Loop(
            _payload_with_signal(summary="请立即拨打 120。", likelihood="very_likely", severity="critical")
        )
        turn = _pipeline(loop).advance("我胸痛")

        assert safety_outcome(turn)["followup"] == ""
        assert "120" in turn.message

    def test_a_critical_signal_with_urgent_urgency_is_escalated_by_the_gate(self):
        """模型判出 `critical` 方向却把 `urgency` 写成 urgent ⇒ 安全门按信号升级。

        这一条不能只看 `urgency`：契约层会记一条自相矛盾，但那是**留档**，
        患者手上必须有急救动作，前端也才能弹出横幅。
        """
        loop = _Loop(
            _payload_with_signal(
                summary="这个情况建议尽快处理，请重视。", likelihood="possible", severity="critical"
            )
        )
        turn = _pipeline(loop).advance("我胸口闷得厉害")

        assert "120" in turn.message, "判出即刻危险的信号时必须给出急救动作"
        assert turn.action == "emergency", "按信号升级后动作必须是 emergency（前端据此弹横幅）"
        assert turns_invariant(turn, "emergency_judgment_requires_directive")


class TestDeterministicSafetyGateLayer:
    """**第三层：确定性安全门**。三条不变量里，急救指引那条是唯一"补内容"的。"""

    def test_emergency_without_a_directive_is_amended(self):
        """判了急症却只写"需要尽快处理" ⇒ 服务端把可执行的急救动作补上。"""
        loop = _Loop(
            json.dumps(
                {
                    "urgency": "emergency",
                    "red_flags": [
                        {
                            "condition": "急性冠脉综合征",
                            "severity": "critical",
                            "why": "胸痛伴大汗",
                            "angle": "组合共现",
                            "likelihood": "likely",
                            "evidence_ids": [],
                        }
                    ],
                    "departments": [{"name": "心内科", "confidence": 0.9}],
                    "differentials": [],
                    "exams": [],
                    "medications": [],
                    "patient_summary": "这个情况需要尽快处理，请重视。",
                    "followup_questions": [],
                    "information_gaps": [],
                    "confidence": 0.9,
                    "citations": [],
                },
                ensure_ascii=False,
            )
        )
        turn = _pipeline(loop).advance("我胸痛")

        assert "120" in turn.message, "判了急症却没给动作时，服务端必须补固定指引"
        assert turns_invariant(turn, "emergency_judgment_requires_directive")

    def test_a_compliant_emergency_answer_is_left_untouched(self):
        loop = _Loop(_payload(summary="请立即拨打 120 或前往最近医院急诊科。", urgency="emergency"))
        turn = _pipeline(loop).advance("我胸痛")

        assert turn.message == "请立即拨打 120 或前往最近医院急诊科。"
        assert safety_outcome(turn)["fired"] == []

    def test_detected_violations_show_up_as_invariants(self):
        """`verify.py` 拦下的那两类要同时登记成不变量（同一件事的两个视角）。"""
        loop = _Loop(
            _payload(summary="已经为您挂上呼吸内科的号了。"),
            _payload(summary="已经为您挂上呼吸内科的号了。"),
        )
        turn = _pipeline(loop).advance("帮我挂个号")

        assert turns_invariant(turn, "no_fabricated_booking_action")

    def test_safety_payload_shape_is_stable(self):
        turn = _pipeline(_Loop(_payload(summary="建议到呼吸内科就诊。"))).advance("我咳嗽")
        assert set(safety_outcome(turn)) == {"fired", "ids", "followup"}

    def test_audit_is_written_when_an_index_root_is_configured(self, tmp_path):
        """治理审计：触发的不变量要落一行 JSONL，并且**不带患者原话**。"""
        from kubedocs.clinical.safety import read_audit

        loop = _Loop(
            _payload(summary="已经为您挂上呼吸内科的号了。"),
            _payload(summary="已经为您挂上呼吸内科的号了。"),
        )
        pipeline = ClinicalPipeline(
            backend=_Backend(),
            reasoner=SimpleNamespace(),
            tool_loop=loop,
            index_root=tmp_path,
        )
        turn = pipeline.advance("帮我挂个号", patient_id="p-audit")

        rows = read_audit(tmp_path)
        assert rows, "触发了不变量却没写审计"
        assert rows[0]["invariant"] == "no_fabricated_booking_action"
        assert rows[0]["trace_id"] == turn.encounter_id, "审计要能对上就诊记录"
        assert "挂上" not in json.dumps(rows, ensure_ascii=False), "审计里不许出现患者原话"

    def test_no_audit_file_without_a_configured_root(self, tmp_path):
        """没配落点（单测/嵌入式）时不许往磁盘上写东西。"""
        loop = _Loop(
            _payload(summary="已经为您挂上呼吸内科的号了。"),
            _payload(summary="已经为您挂上呼吸内科的号了。"),
        )
        _pipeline(loop).advance("帮我挂个号")
        assert list(tmp_path.iterdir()) == []


def safety_outcome(turn: Any) -> dict[str, Any]:
    """取安全门留档（缺字段就直接失败：那说明契约没接上）。"""
    assert turn.safety is not None, "ClinicalTurn 必须带 safety 留档"
    return dict(turn.safety)


def turns_invariant(turn: Any, invariant: str) -> bool:
    return invariant in safety_outcome(turn)["ids"]
