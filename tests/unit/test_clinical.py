"""产品层（`kubedocs.clinical`）测试。

这一层是**契约与流程**，所以测试的重点不是"算得对不对"，而是：

1. 契约能不能挡住不合规的输出（没有依据的检查/用药建议、编造的证据 id、加了剂量）；
2. 流程的**安全性质**能不能被证伪（未确认绝不取号、证据不足不许给结论、红线必须进上下文）；
3. 提示词的硬性条款有没有被真的写进去（提示词是产品契约的一部分）。

全部离线：检索器、挂号后端、LLM 客户端一律用假对象。
"""

from __future__ import annotations

import re

import pytest

from kubedocs.clinical import contracts, rag
from kubedocs.clinical import evidence as ev
from kubedocs.clinical import intake as intake_mod
from kubedocs.clinical import triage as triage_mod
from kubedocs.clinical.pipeline import ClinicalPipeline
from kubedocs.clinical.prompts import prompt_version, system_prompt
from kubedocs.clinical.reasoning import ClinicalReasoner
from kubedocs.clinical.skills import BookingSkill, SkillRegistry, SkillRequest


# --------------------------------------------------------------------- 假对象
class _Chunk:
    """一个检索命中。字段与 `retrieve.hybrid.RetrievedChunk` 对齐。"""

    def __init__(self, title: str, body: str, score: float = 1.0) -> None:
        self.chunk_id = f"c-{abs(hash(title)) % 10_000}"
        self.doc_id = "d1"
        self.ordinal = 0
        self.title = title
        self.breadcrumb = "示例"
        self.source_url = "https://example.invalid/x"
        self.body = body
        self.score = score
        self.ranks: dict[str, int] = {}


class _Result:
    def __init__(self, results: list[_Chunk]) -> None:
        self.query = "q"
        self.results = results


class _FakeRetriever:
    def __init__(self, results: list[_Chunk] | None = None, boom: bool = False) -> None:
        self._results = results or []
        self._boom = boom

    def search(
        self, query: str, *, top_k: int | None = None, channels: tuple[str, ...] = ("bm25", "dense")
    ) -> _Result:
        del query, top_k, channels
        if self._boom:
            raise RuntimeError("后端炸了")
        return _Result(self._results)


class _FakeGuard:
    def __init__(self, word: str | None = None) -> None:
        self._word = word

    def check(self, text: str) -> object | None:
        del text
        if self._word is None:
            return None
        return type(
            "Hit",
            (),
            {"word": self._word, "category": "chest", "experiencer": "self", "temporality": "present"},
        )()


class _FakeAvailability:
    def __init__(self, remaining: int) -> None:
        self._remaining = remaining

    def total_remaining(self) -> int:
        return self._remaining


class _FakeBooking:
    def __init__(self, remaining: int = 3, departments: list[str] | None = None) -> None:
        self.booked: list[dict[str, object]] = []
        self._remaining = remaining
        self._departments = departments or ["心血管内科", "消化内科"]

    def departments(self) -> list[dict]:
        return [{"name": d} for d in self._departments]

    def resolve_department(self, name: str) -> str:
        if name not in self._departments:
            raise ValueError(f"没有这个科：{name}")
        return name

    def availability(self, department: str, date: str | None = None) -> _FakeAvailability:
        del department, date
        return _FakeAvailability(self._remaining)

    def first_available_slot(self, department: str, date: str | None = None) -> str | None:
        del department, date
        return "09:00-09:30" if self._remaining else None

    def book(
        self,
        department: str,
        patient_id: str,
        *,
        date: str | None = None,
        time_slot: str | None = None,
        patient_name: str = "",
    ) -> dict:
        self.booked.append({"department": department, "patient_id": patient_id, "time_slot": time_slot})
        return {
            "department": department,
            "date": date or "2026-10-15",
            "time_slot": time_slot or "09:00-09:30",
            "ticket_no": "A0001",
            "patient_name": patient_name,
        }


class _FakeLLM:
    """按脚本返回。`reply` 可以是字符串（原样返回）或异常。"""

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.seen: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.seen.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _assessment_json(**over: object) -> str:
    payload: dict[str, object] = {
        "urgency": "routine",
        "departments": [{"name": "心血管内科", "confidence": 0.8, "reason": "胸痛相关"}],
        "differentials": [
            {"condition": "心绞痛", "likelihood": "medium", "rationale": "x", "evidence_ids": ["K1"]}
        ],
        "exams": [{"name": "心电图", "why": "排查心脏问题", "evidence_ids": ["K1"]}],
        "medications": [{"name": "硝酸酯类", "drug_class": "抗心绞痛", "why": "y", "evidence_ids": ["K1"]}],
        "patient_summary": "建议尽快到心血管内科就诊，可能需要做心电图。",
        "followup_questions": [],
        "citations": [
            {"index": 1, "title": "T", "source_url": "https://example.invalid/x", "chunk_id": "K1"}
        ],
    }
    payload.update(over)
    import json

    return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------- 契约
class TestContracts:
    def test_medication_advice_has_no_dosage_field(self) -> None:
        """**硬约束用类型守住**：想给剂量就必须先改契约类型，那是一次显式决策。"""
        assert "dose" not in contracts.MedicationAdvice.model_fields
        assert not any("dose" in f for f in contracts.MedicationAdvice.model_fields)

    def test_extra_fields_are_rejected(self) -> None:
        """模型多吐字段要被发现，而不是静默忽略。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            contracts.DepartmentAdvice(name="内科", unexpected="x")  # type: ignore[call-arg]

    def test_exam_without_evidence_is_a_problem(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="routine",
            departments=[{"name": "内科"}],
            exams=[{"name": "血常规", "evidence_ids": []}],
            patient_summary="去医院看看",
        )
        problems = contracts.validate_against(a, {"K1"})
        assert any("没有给出依据" in p for p in problems)

    def test_unknown_evidence_id_is_a_problem(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="routine",
            departments=[{"name": "内科"}],
            exams=[{"name": "血常规", "evidence_ids": ["K9"]}],
            patient_summary="去医院看看",
        )
        problems = contracts.validate_against(a, {"K1"})
        assert any("不存在的证据 id" in p for p in problems)

    def test_emergency_requires_actionable_guidance(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="emergency",
            departments=[{"name": "急诊科"}],
            patient_summary="先观察一下",
        )
        problems = contracts.validate_against(a, set())
        assert any("没有可执行的紧急指引" in p for p in problems)

    def test_valid_assessment_has_no_problems(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="routine",
            departments=[{"name": "内科"}],
            exams=[{"name": "血常规", "evidence_ids": ["K1"]}],
            patient_summary="建议到内科就诊",
        )
        assert contracts.validate_against(a, {"K1"}) == []


# --------------------------------------------------------------------- RAG 后端
class TestRetrievalBackend:
    def test_local_backend_adapts_the_retriever(self) -> None:
        backend = rag.LocalIndexBackend(_FakeRetriever([_Chunk("标题", "正文", 0.5)]))
        hits = backend.search("q", top_k=3)
        assert hits[0].title == "标题" and hits[0].body == "正文"
        assert backend.health().ok is True

    def test_http_backend_fails_loudly_instead_of_returning_empty(self) -> None:
        """**关键取舍**：没实现就大声抛，不许安静地返回空列表。"""
        with pytest.raises(NotImplementedError):
            rag.HttpRagBackend("http://rag.invalid").search("q")

    def test_assert_is_backend_rejects_incomplete_object(self) -> None:
        with pytest.raises(TypeError):
            rag.assert_is_backend(type("X", (), {"name": "x", "search": lambda self, q: []})())


# --------------------------------------------------------------------- 分诊
class TestTriage:
    def test_direct_mention_outranks_card_hits(self) -> None:
        engine = triage_mod.RetrievalTriage(_FakeBooking(departments=["心血管内科", "消化内科"]))
        chunks = [_Chunk("消化内科常见病", "胃痛、反酸", 1.0)]
        ranked = engine.rank("我想挂心血管内科", chunks)
        assert ranked and ranked[0].name == "心血管内科"

    def test_alias_is_matched(self) -> None:
        engine = triage_mod.RetrievalTriage(_FakeBooking(departments=["心血管内科"]))
        ranked = engine.rank("我心内科复查", [])
        assert ranked and ranked[0].name == "心血管内科"

    def test_no_signal_returns_empty(self) -> None:
        engine = triage_mod.RetrievalTriage(_FakeBooking())
        assert engine.rank("你好", []) == []

    def test_no_departments_configured_returns_empty(self) -> None:
        assert triage_mod.RetrievalTriage(_FakeBooking(departments=[])).rank("胸痛", []) == []


class TestTriageFromCardLabels:
    """分诊 v1：按召回卡片的**科室标签**投票（v0 只看"科室名有没有出现在文本里"）。

    这里钉住三件在真实数据上踩过的事：

    1. 卡片正文里那句「建议就诊科室：X、Y。」要能解析出来；
    2. **占位符「未标注」必须被滤掉**——实测 `data/clinic/cards.json` 里正文含该子句的
       27,303 张卡中有 **12,863 张（47.1%）只写着「未标注」**，不滤掉它会把
       「未标注」投成一个"科室"，而实测它真的出现在了人工金标的 top-3 里
       （看起来像答案、实际什么都没有）；
    3. 多科室卡片按科室数**均分**票权，否则列了 3 个科室的卡片等于 3 票。
    """

    def test_clause_is_parsed(self) -> None:
        assert triage_mod.departments_in_card("咳嗽。建议就诊科室：呼吸内科、儿科。") == ["呼吸内科", "儿科"]

    def test_missing_clause_is_not_an_error(self) -> None:
        """没有科室的卡片是正常的，不该抛异常。"""
        assert triage_mod.departments_in_card("只是一句普通正文") == []

    def test_placeholder_is_filtered(self) -> None:
        """**核心回归**：占位符不是科室。"""
        assert triage_mod.departments_in_card("某病。建议就诊科室：未标注。") == []
        assert triage_mod.departments_in_card("某病。建议就诊科室：其他科室。") == []

    def test_department_map_fills_placeholder_cards(self) -> None:
        """**核心回归**：卡片正文写着「未标注」时，补全映射表能给出科室。

        语料里 47.1% 的卡片是占位符（源图谱缺关系），映射表由
        `scripts/clinic_derive_departments.py` 从原始三元组推导。若这条测试挂了，
        说明分诊又退化成"只信正文"，那批卡片会集体弃权。
        """
        engine = triage_mod.RetrievalTriage(known_departments=None, department_map={"耳朵里疼": ["耳鼻喉科"]})
        chunk = rag.RetrievedChunk(
            chunk_id="c1", title="耳朵里疼", body="耳朵里疼。可能相关的疾病：暂无。建议就诊科室：未标注。"
        )
        assert engine._departments_of("耳朵里疼", chunk.body) == ["耳鼻喉科"]
        assert engine.map_hits == 1
        # 映射表里没有的 title 必须退回正文解析（不能因为查表失败就丢标签）
        assert engine._departments_of("咳嗽", "咳嗽。建议就诊科室：呼吸内科。") == ["呼吸内科"]

    def test_confidence_tier_controls_which_labels_are_trusted(self, tmp_path: object) -> None:
        """`min_confidence="high"` 不该放进模糊归并的结果（那一档有错例，需抽检）。"""
        import json as _json
        from pathlib import Path as _P

        p = _P(str(tmp_path)) / "m.json"
        p.write_text(
            _json.dumps(
                {
                    "meta": {},
                    "map": {
                        "权威": {"departments": ["心内科"], "source": "alias:心脏病", "confidence": "high"},
                        "模糊": {
                            "departments": ["产科"],
                            "source": "fuzzy:血流:0.50",
                            "confidence": "medium",
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        strict = triage_mod.load_department_map(p, min_confidence="high")
        loose = triage_mod.load_department_map(p, min_confidence="medium")
        assert strict == {"权威": ["心内科"]}
        assert loose == {"权威": ["心内科"], "模糊": ["产科"]}
        # 文件不存在时**不能抛异常**：分诊要退化成正文解析，而不是整体挂掉
        assert triage_mod.load_department_map(p.parent / "nope.json") == {}

    def test_vote_comes_from_card_labels(self) -> None:
        engine = triage_mod.RetrievalTriage(known_departments=None)
        chunks = [
            _Chunk("咳嗽", "咳嗽。建议就诊科室：呼吸内科。", 1.0),
            _Chunk("发热", "发热。建议就诊科室：呼吸内科、感染科。", 1.0),
        ]
        ranked = [d.name for d in engine.rank("咳嗽发烧", chunks)]
        assert ranked and ranked[0] == "呼吸内科"

    def test_placeholder_only_pool_yields_nothing(self) -> None:
        """整池都是「未标注」⇒ 返回空，而不是返回一个假的科室名。"""
        engine = triage_mod.RetrievalTriage(known_departments=None)
        chunks = [_Chunk("某病", "某病。建议就诊科室：未标注。", 1.0)]
        assert engine.rank("某病", chunks) == []

    def test_multi_department_card_splits_its_vote(self) -> None:
        """一张列 3 个科室的卡不该等于 3 票：单科室的两张卡应当胜过它。"""
        engine = triage_mod.RetrievalTriage(known_departments=None)
        chunks = [
            _Chunk("甲", "甲。建议就诊科室：皮肤科。", 1.0),
            _Chunk("乙", "乙。建议就诊科室：皮肤科。", 1.0),
            _Chunk("丙", "丙。建议就诊科室：皮肤科、骨外科、妇科。", 1.0),
        ]
        ranked = engine.rank("甲 乙 丙", chunks)
        assert ranked[0].name == "皮肤科"


# --------------------------------------------------------------------- 证据装配
class TestEvidence:
    def test_red_line_is_rendered_first_and_marked_must(self) -> None:
        bundle = ev.build_evidence(
            "我胸痛",
            backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])),
            guard=_FakeGuard("胸痛"),
        )
        text = bundle.render()
        assert text.startswith("【红线信号｜MUST 处理】")
        assert bundle.red_line_positive is True

    def test_no_knowledge_explicitly_forbids_advice(self) -> None:
        """检索为空时，上下文里必须**明说**不许给检查/用药建议。"""
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([])))
        assert "不得给出检查建议或用药建议" in bundle.render()

    def test_backend_failure_degrades_instead_of_raising(self) -> None:
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever(boom=True)))
        assert bundle.items == []
        assert bundle.degraded and "失败" in bundle.degraded[0]
        assert "检索链路问题" in bundle.render()

    def test_evidence_ids_are_stable_and_sequential(self) -> None:
        chunks = [_Chunk("A", "a"), _Chunk("B", "b")]
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever(chunks)))
        assert [i.id for i in bundle.items] == ["K1", "K2"]


# --------------------------------------------------------------------- 采问
class TestIntake:
    def test_critical_slot_is_asked_first(self) -> None:
        st = intake_mod.IntakeState()
        nxt = st.next_question()
        assert nxt is not None and nxt.critical

    def test_merge_fills_slots(self) -> None:
        st = intake_mod.IntakeState()
        filled = st.merge("胸口疼，已经三天了", intake_mod.RuleSlotExtractor())
        assert "chief" in filled and "duration" in filled

    def test_unfillable_slot_stops_being_asked(self) -> None:
        """**防无限追问**：问满 MAX_ASK 次还抽不出来，就不再问、并记为"问不出来"。"""
        st = intake_mod.IntakeState()
        st.filled["chief"] = "胸口疼"
        for _ in range(intake_mod.MAX_ASK):
            slot = st.next_question()
            assert slot is not None
            st.mark_asked(slot.key)
        unfillable = set(st.unfillable())
        assert unfillable, "问满次数后应当有槽位进入 unfillable"
        assert all(s.key not in unfillable for s in [st.next_question()] if s is not None)

    def test_can_conclude_requires_critical_slot_only(self) -> None:
        st = intake_mod.IntakeState(filled={"chief": "胸口疼"})
        assert st.can_conclude() is True

    def test_render_separates_askable_from_unfillable(self) -> None:
        st = intake_mod.IntakeState(filled={"chief": "疼"}, asked={"meds": intake_mod.MAX_ASK})
        text = st.render()
        assert "仍缺（可以继续问）" in text
        assert "问不出来（已问满次数，不要再问）" in text


# --------------------------------------------------------------------- 提示词
class TestSystemPrompt:
    def test_version_is_a_stable_12_hex(self) -> None:
        assert re.fullmatch(r"[0-9a-f]{12}", prompt_version())

    @pytest.mark.parametrize(
        "clause",
        [
            "绝不给用药剂量",  # R2
            "只能来自【知识依据】",  # R3
            "检索不到就不给",  # R3
            "不给确定诊断",  # R4
            "不要编造依据编号",  # R6
            "看到【红线信号】就必须给紧急处理指引",  # R1
            "120",  # 急救动作
            "followup_questions",  # 输出契约
            "证据_ids",  # 依据引用
        ],
    )
    def test_hard_clauses_are_present(self, clause: str) -> None:
        """提示词是产品契约的一部分：硬性条款必须真的写在里面，且被测试锁住。"""
        text = system_prompt()
        assert clause.replace("证据_ids", "evidence_ids") in text

    def test_prompt_demands_json_only(self) -> None:
        assert "只输出 JSON" in system_prompt()


# --------------------------------------------------------------------- Skill 框架
class TestSkillRegistry:
    def test_duplicate_registration_is_rejected(self) -> None:
        reg = SkillRegistry()
        reg.register(BookingSkill(_FakeBooking()))
        with pytest.raises(ValueError):
            reg.register(BookingSkill(_FakeBooking()))

    def test_two_side_effect_steps_are_rejected(self) -> None:
        """副作用最多一处：不可逆点分散会让"失败回退"说不清。"""
        from kubedocs.clinical.skills.base import SkillStep

        class _Bad:
            name = "bad"

            steps = (
                SkillStep("read", "只读"),
                SkillStep("commit_a", "副作用一", side_effect=True),
                SkillStep("commit_b", "副作用二", side_effect=True),
            )

            def run(self, request: SkillRequest) -> object:
                del request
                raise AssertionError

        with pytest.raises(ValueError, match="多个副作用步骤"):
            SkillRegistry().register(_Bad())  # type: ignore[arg-type]

    def test_side_effect_as_first_step_is_rejected(self) -> None:
        """第一步就动手的 skill 重跑不幂等——这类形状要被挡住。"""
        from kubedocs.clinical.skills.base import SkillStep

        class _Bad:
            name = "bad2"

            steps = (SkillStep("commit", "直接副作用", side_effect=True),)

            def run(self, request: SkillRequest) -> object:
                del request
                raise AssertionError

        with pytest.raises(ValueError, match="第一步就是副作用"):
            SkillRegistry().register(_Bad())  # type: ignore[arg-type]

    def test_read_only_step_after_the_side_effect_is_allowed(self) -> None:
        """挂号是 commit → receipt（副作用之后还有只读步），这条形状必须被允许。"""
        reg = SkillRegistry()
        reg.register(BookingSkill(_FakeBooking()))
        names = [s.name for s in reg.get("book_appointment").steps]
        assert names.index("commit") < names.index("receipt")

    def test_describe_exposes_steps(self) -> None:
        reg = SkillRegistry()
        reg.register(BookingSkill(_FakeBooking()))
        described = reg.describe()
        assert described[0]["name"] == "book_appointment"
        assert [s["name"] for s in described[0]["steps"]][-1] == "receipt"


# --------------------------------------------------------------------- 挂号 skill
class TestBookingSkill:
    def test_unconfirmed_run_does_not_book(self) -> None:
        """**安全性质**：未确认时绝不产生副作用——这里断言后端压根没被调用。"""
        backend = _FakeBooking()
        outcome = BookingSkill(backend).run(
            SkillRequest(patient_id="p1", params={"department": "心血管内科"})
        )
        assert outcome.needs_confirmation is True
        assert outcome.stopped_at == "confirm"
        assert backend.booked == [], "未确认却取了号"

    def test_confirmed_run_books_and_returns_receipt(self) -> None:
        backend = _FakeBooking()
        outcome = BookingSkill(backend).run(
            SkillRequest(patient_id="p1", confirmed=True, params={"department": "心血管内科"})
        )
        assert outcome.ok is True
        assert outcome.data["ticket_no"] == "A0001"
        assert backend.booked[0]["patient_id"] == "p1"
        assert outcome.steps_done[-1] == "receipt"

    def test_unknown_department_stops_at_resolve(self) -> None:
        outcome = BookingSkill(_FakeBooking()).run(
            SkillRequest(patient_id="p1", params={"department": "外星科"})
        )
        assert outcome.ok is False and outcome.stopped_at == "resolve_department"

    def test_no_slot_stops_before_confirm(self) -> None:
        outcome = BookingSkill(_FakeBooking(remaining=0)).run(
            SkillRequest(patient_id="p1", params={"department": "心血管内科"})
        )
        assert outcome.ok is False and outcome.stopped_at == "list_candidates"

    def test_missing_department_parameter_is_reported(self) -> None:
        outcome = BookingSkill(_FakeBooking()).run(SkillRequest(patient_id="p1"))
        assert outcome.stopped_at == "resolve_department"


# --------------------------------------------------------------------- 主判
class TestReasoner:
    def test_valid_json_becomes_assessment(self) -> None:
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])))
        result = ClinicalReasoner(_FakeLLM(_assessment_json())).reason("q", bundle, intake_mod.IntakeState())
        assert result.assessment is not None
        assert result.assessment.departments[0].name == "心血管内科"
        assert result.problems == []

    def test_fenced_json_is_tolerated(self) -> None:
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])))
        raw = "```json\n" + _assessment_json() + "\n```"
        result = ClinicalReasoner(_FakeLLM(raw)).reason("q", bundle, intake_mod.IntakeState())
        assert result.assessment is not None

    def test_llm_failure_degrades_without_inventing(self) -> None:
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])))
        result = ClinicalReasoner(_FakeLLM(RuntimeError("网络断了"))).reason(
            "q", bundle, intake_mod.IntakeState()
        )
        assert result.assessment is not None
        assert result.assessment.exams == [] and result.assessment.medications == []
        assert "降级" in result.degraded or "失败" in result.degraded

    def test_garbage_output_degrades(self) -> None:
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([])))
        result = ClinicalReasoner(_FakeLLM("我不知道该怎么回答")).reason(
            "q", bundle, intake_mod.IntakeState()
        )
        assert result.assessment is not None
        assert result.problems and "契约" in result.problems[0]

    def test_red_line_conflict_is_recorded_not_silently_ignored(self) -> None:
        """按决定 2b 模型有权判定，但**冲突必须留痕**。"""
        bundle = ev.build_evidence(
            "q",
            backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])),
            guard=_FakeGuard("胸痛"),
        )
        result = ClinicalReasoner(_FakeLLM(_assessment_json(urgency="routine"))).reason(
            "q", bundle, intake_mod.IntakeState()
        )
        assert any("红线阳性" in p for p in result.problems)

    def test_red_line_fallback_without_llm_is_emergency(self) -> None:
        """**决策者缺席时，证据本身成为结论**：模型挂了 + 红线阳性 ⇒ 必须给急救指引。"""
        bundle = ev.build_evidence(
            "q", backend=rag.LocalIndexBackend(_FakeRetriever([])), guard=_FakeGuard("胸痛")
        )
        result = ClinicalReasoner(_FakeLLM(RuntimeError("挂了"))).reason(
            "q", bundle, intake_mod.IntakeState()
        )
        assert result.assessment is not None
        assert result.assessment.urgency == "emergency"
        assert "120" in result.assessment.patient_summary


class TestLiteLLMAdapter:
    """把仓库的 `LLMResult` 适配成 `str` 的那一层。

    **这个坑是实测踩到的**：`ClinicalReasoner` 直接对返回值调 `.strip()`，而仓库真实的
    `LiteLLMClient.complete()` 返回 `LLMResult` 对象 → `AttributeError` → 整条主判
    **安静地走降级**（表面上"有回答"，实际一次模型判断都没发生）。
    """

    def test_reads_text_from_a_result_like_object(self) -> None:
        from kubedocs.clinical.reasoning import LiteLLMAdapter

        class _Result:
            text = "正文"

        class _Client:
            def complete(self, system: str, user: str) -> _Result:
                del system, user
                return _Result()

        assert LiteLLMAdapter(_Client()).complete("s", "u") == "正文"

    def test_raises_loudly_when_result_has_no_text(self) -> None:
        """类型不对要**大声抛**，不许变成"模型输出不合契约"这种误导性的降级理由。"""
        from kubedocs.clinical.reasoning import LiteLLMAdapter

        class _Client:
            def complete(self, system: str, user: str) -> object:
                del system, user
                return object()

        with pytest.raises(TypeError, match="没有可用的 .text"):
            LiteLLMAdapter(_Client()).complete("s", "u")

    def test_adapter_output_reaches_the_contract(self) -> None:
        """端到端接一遍：适配器 + reasoner + 契约解析。"""
        from kubedocs.clinical.reasoning import LiteLLMAdapter

        class _Result:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Client:
            def complete(self, system: str, user: str) -> _Result:
                del system, user
                return _Result(_assessment_json())

        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])))
        result = ClinicalReasoner(LiteLLMAdapter(_Client())).reason("q", bundle, intake_mod.IntakeState())
        assert result.assessment is not None
        assert result.problems == []


# --------------------------------------------------------------------- 编排
class TestPipeline:
    def _pipeline(self, reply: str | Exception = "", *, word: str | None = None) -> ClinicalPipeline:
        from kubedocs.clinical.skills import default_registry

        return ClinicalPipeline(
            backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("心电图检查", "心电图用于排查心脏问题")])),
            reasoner=ClinicalReasoner(_FakeLLM(reply or _assessment_json())),
            skills=default_registry(booking_backend=_FakeBooking()),
            guard=_FakeGuard(word),
            triage=triage_mod.RetrievalTriage(_FakeBooking(departments=["心血管内科", "消化内科"])),
        )

    def test_asks_when_critical_slot_is_missing(self) -> None:
        """证据不足不得给结论。"""
        turn = self._pipeline().advance("我不舒服")
        assert turn.action == "ask"
        assert turn.assessment is None
        assert turn.message.endswith("？")

    def test_concludes_when_slot_filled(self) -> None:
        turn = self._pipeline().advance("胸口疼")
        assert turn.action == "answer"
        assert turn.assessment is not None
        assert turn.prompt_version != ""

    def test_emergency_action_when_model_says_so(self) -> None:
        turn = self._pipeline(
            _assessment_json(urgency="emergency", patient_summary="请立即拨打 120 或去急诊。")
        ).advance("胸口疼")
        assert turn.action == "emergency"

    def test_red_line_positive_is_exposed_even_if_model_disagrees(self) -> None:
        turn = self._pipeline(word="胸痛").advance("胸口疼")
        assert turn.red_line_positive is True
        assert any("红线阳性" in p for p in turn.problems)

    def test_booking_requires_confirmation_then_books(self) -> None:
        pipe = self._pipeline()
        params = {"department": "心血管内科"}
        first = pipe.advance("胸口疼", want_booking=True, booking_params=params)
        assert first.action == "confirm_booking"
        assert first.skill is not None and first.skill.needs_confirmation

        # 会话状态由调用方持有并回传（`ClinicalTurn.intake`）——不传就等于新开一轮，
        # 连主诉都要重问。这一条既是用法示范，也是那个设计的回归测试。
        second = pipe.advance(
            "好",
            state=first.intake,
            want_booking=True,
            booking_params=params,
            confirmed=True,
        )
        assert second.action == "booked"
        assert second.skill is not None and second.skill.data["ticket_no"] == "A0001"

    def test_forgetting_to_thread_state_loses_what_was_collected(self) -> None:
        """不传 `state` 就等于新开一轮——把"状态属于会话层"这件事钉死。"""
        pipe = self._pipeline()
        first = pipe.advance("胸口疼，已经三天了")
        assert first.action == "answer"
        assert "duration" in first.intake.filled

        again = pipe.advance("胸口疼")  # 没传 state
        assert again.intake is not first.intake
        assert "duration" not in again.intake.filled, "状态没传却看起来还记得上一轮"
