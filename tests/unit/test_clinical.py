"""产品层（`kubedocs.clinical`）测试：守护契约与流程，而不是"算得对不对"。

契约拦得住不合规输出（无依据的检查/用药建议、编造的证据 id、剂量）；流程的安全性质
成立（急重症优先、红线进上下文、工具循环失败也要降级而不是把裸文本给患者）；
提示词的硬性条款真写在里面。全部离线：检索器、LLM 客户端一律用假对象。
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from kubedocs.clinical import contracts, rag
from kubedocs.clinical import evidence as ev
from kubedocs.clinical import intake as intake_mod
from kubedocs.clinical import triage as triage_mod
from kubedocs.clinical.pipeline import ClinicalPipeline
from kubedocs.clinical.prompts import prompt_version, system_prompt
from kubedocs.clinical.reasoning import ClinicalReasoner


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
    """按召回卡片的**科室标签**投票分诊，钉住三条不变式。

    正文的「建议就诊科室：X、Y。」要能解析、缺这句属正常数据；占位符「未标注」必须滤掉
    （含该子句的卡片里 47.1% 只写「未标注」）；多科室卡片按科室数**均分**票权。
    """

    def test_clause_is_parsed(self) -> None:
        assert triage_mod.departments_in_card("咳嗽。建议就诊科室：呼吸内科、儿科。") == ["呼吸内科", "儿科"]

    def test_missing_clause_is_not_an_error(self) -> None:
        """正文缺科室子句是正常数据，不该抛异常。"""
        assert triage_mod.departments_in_card("只是一句普通正文") == []

    def test_placeholder_is_filtered(self) -> None:
        """占位符不是科室：必须被滤掉。"""
        assert triage_mod.departments_in_card("某病。建议就诊科室：未标注。") == []
        assert triage_mod.departments_in_card("某病。建议就诊科室：其他科室。") == []

    def test_department_map_fills_placeholder_cards(self) -> None:
        """正文写着「未标注」时，补全映射表必须给出科室。

        语料里 47.1% 的卡片是占位符；查表失败要退回正文解析，不能丢标签。
        """
        engine = triage_mod.RetrievalTriage(known_departments=None, department_map={"耳朵里疼": ["耳鼻喉科"]})
        chunk = rag.RetrievedChunk(
            chunk_id="c1", title="耳朵里疼", body="耳朵里疼。可能相关的疾病：暂无。建议就诊科室：未标注。"
        )
        assert engine._departments_of("耳朵里疼", chunk.body) == ["耳鼻喉科"]
        assert engine.map_hits == 1
        # 映射表里没有的 title 必须退回正文解析
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
        # 文件不存在时不能抛异常：分诊退化成正文解析
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
    def test_keyword_hit_is_not_rendered_into_the_prompt(self) -> None:
        """关键词命中不进模型上下文：提示词里一个字都不能提关键词。

        用关键词约束医学判断会漏掉词表覆盖不到的表述，也会把不该报的转成急诊；急重症改由
        模型按病症 + 严重程度判断（`system.md` R1），命中事实只留在 `red_line_positive` 里。
        """
        bundle = ev.build_evidence(
            "我胸痛",
            backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])),
            guard=_FakeGuard("胸痛"),
        )
        text = bundle.render()
        assert "胸痛" not in text  # 命中词不得出现在上下文里
        assert "MUST" not in text
        assert "红线" not in text
        assert bundle.red_line_positive is True
        assert bundle.red_line is not None and bundle.red_line.word == "胸痛"

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


class TestContractEnforcement:
    """契约违规必须真的拦掉内容，不能只记一笔账。

    `enforce()` 丢弃违规条目并记入 `problems`：只返回问题字符串会让 R3 退化成事后记账。
    """

    def test_exam_without_evidence_is_dropped(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="routine",
            departments=[contracts.DepartmentAdvice(name="呼吸内科", confidence=0.5)],
            exams=[
                contracts.ExamAdvice(name="有依据的检查", evidence_ids=["K1"]),
                contracts.ExamAdvice(name="凭空来的检查"),  # 无依据 ⇒ 该被丢
            ],
            patient_summary="建议就诊。",
        )
        out = contracts.enforce(a, {"K1"})
        assert [e.name for e in out.assessment.exams] == ["有依据的检查"]
        assert any("凭空来的检查" in d for d in out.dropped)
        assert any("已按 R3 丢弃" in p for p in out.problems)

    def test_medication_with_fake_evidence_id_is_dropped(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="routine",
            departments=[contracts.DepartmentAdvice(name="呼吸内科", confidence=0.5)],
            medications=[contracts.MedicationAdvice(name="编的假药", evidence_ids=["K99"])],
            patient_summary="建议就诊。",
        )
        out = contracts.enforce(a, {"K1"})
        assert out.assessment.medications == []
        assert any("K99" in d for d in out.dropped)

    def test_emergency_content_is_never_dropped(self) -> None:
        """急重症判断与科室建议不因 R3 被削弱：漏报急诊比多报危险得多。

        引用不存在的 id 只丢检查/用药，`red_flags`、`departments`、`patient_summary` 原样留下。
        """
        a = contracts.ClinicalAssessment(
            urgency="emergency",
            red_flags=[
                contracts.RedFlag(
                    condition="急性冠脉综合征",
                    severity="critical",
                    why="胸痛伴大汗",
                    evidence_ids=["不存在的id"],
                )
            ],
            departments=[contracts.DepartmentAdvice(name="急诊科", confidence=0.9)],
            exams=[contracts.ExamAdvice(name="心电图", evidence_ids=["不存在的id"])],
            patient_summary="请立刻拨打 120。",
        )
        out = contracts.enforce(a, {"K1"})
        assert out.assessment.red_flags
        assert out.assessment.departments
        assert out.assessment.patient_summary == "请立刻拨打 120。"
        assert out.assessment.exams == []  # 有依据才留
        assert any("急重症判断" in p for p in out.problems)  # 只记不丢

    def test_clean_assessment_passes_untouched(self) -> None:
        a = contracts.ClinicalAssessment(
            urgency="routine",
            departments=[contracts.DepartmentAdvice(name="呼吸内科", confidence=0.5)],
            exams=[contracts.ExamAdvice(name="血常规", evidence_ids=["K1"])],
            medications=[contracts.MedicationAdvice(name="止咳糖浆", evidence_ids=["K2"])],
            patient_summary="建议就诊。",
        )
        out = contracts.enforce(a, {"K1", "K2"})
        assert out.dropped == []
        assert out.problems == []
        assert out.assessment.exams and out.assessment.medications


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
            # R1：急重症由模型按病症 + 严重程度判断，不查关键词表
            "急重症由你判断",
            "病症",
            "严重程度",
            "red_flags",
            "拿不准时往重的一侧判",  # 保守偏置必须写在提示词里
            "120",
            "followup_questions",
            "证据_ids",
        ],
    )
    def test_hard_clauses_are_present(self, clause: str) -> None:
        """提示词是产品契约的一部分：硬性条款必须真的写在里面，且被测试锁住。"""
        text = system_prompt()
        assert clause.replace("证据_ids", "evidence_ids") in text

    def test_prompt_does_not_tell_the_model_to_obey_a_keyword_list(self) -> None:
        """提示词里不得出现"照红线信号执行"这类关键词约束：急重症判断权在模型。"""
        text = system_prompt()
        assert "【红线信号" not in text
        assert "MUST 处理" not in text
        assert "不是重新评估它" not in text

    def test_prompt_demands_json_only(self) -> None:
        assert "只输出 JSON" in system_prompt()

    def test_prompt_is_declared_as_package_data(self) -> None:
        """`system.md` 必须声明为 package-data，否则不进 wheel。

        setuptools 默认只带 `.py`，editable 安装发现不了，装出来的包读提示词会直接失败。
        """
        from kubedocs.config import repo_root

        pyproject = (repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        assert "[tool.setuptools.package-data]" in pyproject
        assert '"kubedocs.clinical.prompts" = ["*.md"]' in pyproject

    def test_prompt_path_lives_next_to_the_module(self) -> None:
        """提示词必须与 `prompts/__init__.py` 同目录 —— 换位置就会静默读到别人。"""
        from kubedocs.clinical.prompts import SYSTEM_PROMPT_PATH

        assert SYSTEM_PROMPT_PATH.name == "system.md"
        assert SYSTEM_PROMPT_PATH.parent.name == "prompts"
        assert SYSTEM_PROMPT_PATH.is_file()


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

    def test_keyword_hit_without_red_flag_is_recorded_not_silently_ignored(self) -> None:
        """关键词命中而模型没判出急重症时必须留痕，但不推翻模型：判断权归模型，程序只记进 `problems`。"""
        bundle = ev.build_evidence(
            "q",
            backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("T", "B")])),
            guard=_FakeGuard("胸痛"),
        )
        result = ClinicalReasoner(_FakeLLM(_assessment_json(urgency="routine"))).reason(
            "q", bundle, intake_mod.IntakeState()
        )
        assert any("关键词表命中" in p for p in result.problems)
        # 留痕归留痕，模型的结论**不被改写**
        assert result.assessment is not None and result.assessment.urgency == "routine"

    def test_red_flags_force_emergency_by_contract(self) -> None:
        """判出急重症却给常规档 ⇒ 契约判违规：自相矛盾的输出会被程序拦下并记进 `problems`。"""
        bundle = ev.build_evidence("q", backend=rag.LocalIndexBackend(_FakeRetriever([])))
        raw = json.dumps(
            {
                "urgency": "routine",
                "red_flags": [{"condition": "急性冠脉综合征", "severity": "critical", "why": "胸痛伴大汗"}],
                "departments": [{"name": "心内科", "confidence": 0.6}],
                "patient_summary": "建议尽快就诊。",
            },
            ensure_ascii=False,
        )
        result = ClinicalReasoner(_FakeLLM(raw)).reason("q", bundle, intake_mod.IntakeState())
        assert any("却把 urgency 定为 routine" in p for p in result.problems)

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

    真实 `LiteLLMClient.complete()` 返回对象而非字符串，`ClinicalReasoner` 却对返回值调
    `.strip()`：类型不匹配必须抛错，否则主判会安静地走降级，表面上"有回答"而实际没判断。
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
class _FakeToolLoop:
    """最小工具循环替身：记录被调用次数，返回固定契约 JSON。"""

    def __init__(self, final_text: str) -> None:
        self.final_text = final_text
        self.calls = 0
        self.questions: list[str] = []
        self.histories: list[object] = []

    def run(
        self,
        *,
        system: str,
        question: str,
        intake: object,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> object:
        # 管线会把 `patient_id` 经 `context=` 交给工具循环（挂号记账要靠它）。
        # 这个替身不执行真工具，只把签名对齐——**不等齐就会 TypeError**，而管线会
        # 捕获它并静默降级，测试于是从"验证编排"变成"验证降级路径"。
        del system, intake, context
        self.calls += 1
        self.questions.append(question)
        self.histories.append(history)

        from kubedocs.clinical.evidence import EvidenceBundle

        class _R:
            final_text = self.final_text
            calls: list[object] = []
            evidence = EvidenceBundle(items=[])

        return _R()


class TestPipeline:
    """编排层：只有一条路径（工具循环），这里钉住它推导出的 `action`。"""

    def _pipeline(
        self,
        *,
        word: str | None = None,
        tool_loop: object | None = None,
    ) -> ClinicalPipeline:
        return ClinicalPipeline(
            backend=rag.LocalIndexBackend(_FakeRetriever([_Chunk("心电图检查", "心电图用于排查心脏问题")])),
            # reasoner 只提供降级产物（工具循环失败时的那句有限结论），正常轮次不经过它
            reasoner=ClinicalReasoner(_FakeLLM(_assessment_json())),
            guard=_FakeGuard(word),
            triage=triage_mod.RetrievalTriage(_FakeBooking(departments=["心血管内科", "消化内科"])),
            tool_loop=tool_loop or _FakeToolLoop(_assessment_json()),
        )

    def test_intake_never_preempts_the_model(self) -> None:
        """编排层不设追问闸门：追不追问归模型，急症不能被拦成一句追问。

        「突发言语不清，右侧肢体无力两小时」这类卒中描述在闸门下会被判成 `action=ask`；
        现在"够不够下结论"必须由模型自己决定（提示词 R5）。
        """
        loop = _FakeToolLoop(
            json.dumps(
                {
                    "urgency": "emergency",
                    "red_flags": [
                        {"condition": "急性脑卒中", "severity": "critical", "why": "突发神经功能缺失"}
                    ],
                    "departments": [{"name": "急诊科", "confidence": 0.9}],
                    "patient_summary": "这是急症，请立刻拨打 120。",
                },
                ensure_ascii=False,
            )
        )
        turn = self._pipeline(tool_loop=loop).advance("突发言语不清，右侧肢体无力两小时")
        assert loop.calls == 1, "模型必须真的被调用过（而不是被编排层拦住）"
        assert turn.action == "emergency"
        assert turn.assessment is not None and turn.assessment.red_flags

    def test_followup_questions_become_an_ask_but_emergency_wins(self) -> None:
        """模型追问 ⇒ `action=ask`；但**急重症优先**，不得给"先追问"的动作。"""
        ask_loop = _FakeToolLoop(
            json.dumps(
                {
                    "urgency": "routine",
                    "departments": [{"name": "呼吸内科", "confidence": 0.5}],
                    "patient_summary": "这种情况持续多久了？",
                    "followup_questions": ["持续多久了"],
                },
                ensure_ascii=False,
            )
        )
        assert self._pipeline(tool_loop=ask_loop).advance("我咳嗽").action == "ask"

        emg_loop = _FakeToolLoop(
            json.dumps(
                {
                    "urgency": "emergency",
                    "red_flags": [{"condition": "脑卒中", "severity": "critical", "why": "突发偏瘫"}],
                    "departments": [{"name": "急诊科", "confidence": 0.9}],
                    "patient_summary": "请立刻拨打 120。",
                    "followup_questions": ["什么时候开始的"],
                },
                ensure_ascii=False,
            )
        )
        assert self._pipeline(tool_loop=emg_loop).advance("半边身子不能动").action == "emergency"

    def test_keyword_hit_is_exposed_but_does_not_decide(self) -> None:
        """关键词命中要如实暴露（供排查），但**不改写模型的结论**。"""
        turn = self._pipeline(word="胸痛").advance("胸口疼")
        assert turn.red_line_positive is True
        # 桩判的是 routine，结论就得是 routine，不被关键词改写
        assert turn.assessment is not None
        assert turn.action != "emergency"

    def test_forgetting_to_thread_state_loses_what_was_collected(self) -> None:
        """不传 `state` 就等于新开一轮——把"状态属于会话层"这件事钉死。"""
        pipe = self._pipeline()
        first = pipe.advance("胸口疼，已经三天了")
        assert "duration" in first.intake.filled

        again = pipe.advance("胸口疼")  # 没传 state
        assert again.intake is not first.intake
        assert "duration" not in again.intake.filled, "状态没传却看起来还记得上一轮"
