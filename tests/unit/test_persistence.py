"""就诊记录持久化的测试：连**真 Postgres**，钉住"真的落库、真的读回"这几件事。

1. **往返**：写进去的结论（含 `red_flags`、检查/用药、引用）读回来字段一致，
   对话由消息表重建、顺序正确；
2. **耐久**：关掉连接池、重新连同一个库（同一个 schema），数据还在——这是"持久化"
   这个词的最小证明；
3. **幂等**：`upsert_patient` 同 id 两次只有一行，且非空字段被更新；`ensure_schema()`
   反复调用不报错、不丢数据；
4. **失败策略**：写入即抛异常的假 store 不能让 `advance()` 失败，只留一条 warning——
   宁可少一条记录，也不能因为落库失败让患者拿不到答复。

**Postgres 不可达时整文件 skip**，并写明探测到的原因——写法与
`tests/unit/test_pgvector_store.py` 一致。每个用例用独立的 `test_clinical_*` schema，
结束即 `DROP SCHEMA CASCADE`：既不污染真实库的 `clinical` schema，也不互相看见对方的数据。

探测只连一次、**不查任何扩展**：就诊记录只需要 Postgres 本体，pgvector / pg_search 都
不是它的前置条件（`pgvector_store.probe` 会要求 vector 扩展，用在这里会把一个能跑的
临床记录库误判成不可用）。

全部离线：检索后端、模型、工具循环都是假对象。
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg import sql

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.persistence import (
    EncounterRecord,
    NoopPersistence,
    PatientRecord,
    PersistenceError,
    PgPersistence,
)
from kubedocs.clinical.pipeline import ClinicalPipeline
from kubedocs.clinical.rag import LocalIndexBackend
from kubedocs.clinical.reasoning import ClinicalReasoner
from kubedocs.config import Settings

_DSN = Settings().database_url


def _probe(dsn: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    """这个 Postgres 能不能连（**不要求任何扩展**，理由见模块文档）。"""
    try:
        with psycopg.connect(dsn, connect_timeout=int(max(1, timeout))) as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - 探测就是要吞掉一切异常并转成原因字符串
        return False, f"{type(e).__name__}: {e}"
    return True, ""


_PG_OK, _PG_WHY = _probe(_DSN)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _PG_OK,
        reason=f"Postgres 不可达，本文件未验证：{_PG_WHY}（DSN={_DSN}）",
    ),
]


def _drop_schema(schema: str) -> None:
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def schema() -> Iterator[str]:
    """每个用例一个独立 schema：名字带随机后缀，避免并发/残留互相影响。"""
    name = f"test_clinical_{uuid.uuid4().hex[:10]}"
    yield name
    _drop_schema(name)


@pytest.fixture
def store(schema: str) -> Iterator[PgPersistence]:
    s = PgPersistence(_DSN, schema=schema)
    try:
        yield s
    finally:
        s.close()


# --------------------------------------------------------------------- 假对象
class _Chunk:
    """一条检索命中（字段与 `retrieve.hybrid.RetrievedChunk` 对齐）。"""

    chunk_id = "c1"
    doc_id = "d1"
    ordinal = 0
    title = "咳嗽"
    breadcrumb = "呼吸"
    source_url = "https://example.invalid/x"
    body = "咳嗽常见于呼吸道感染"
    score = 1.0
    ranks: dict[str, int] = {}


class _Result:
    def __init__(self) -> None:
        self.query = "q"
        self.results = [_Chunk()]


class _Retriever:
    def search(self, query: str, *, top_k: int | None = None, channels: Any = None) -> _Result:
        del query, top_k, channels
        return _Result()


class _LLM:
    """只需能给出契约 JSON（正常轮次不走它，它只服务降级路径）。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        del system, user
        return self.reply


class _ToolLoop:
    """最小工具循环替身：返回固定契约 JSON（模型那一侧不是本模块要测的东西）。"""

    def __init__(self, final_text: str) -> None:
        self.final_text = final_text
        self.calls = 0

    def run(self, *, system: str, question: str, intake: Any, history: Any = None) -> Any:
        del system, intake, history
        from kubedocs.clinical.evidence import EvidenceBundle

        self.calls += 1
        loop = self

        class _R:
            final_text = loop.final_text
            calls: list[Any] = []
            evidence = EvidenceBundle(items=[])

        return _R()


class _BrokenStore:
    """写入即抛：验证"落库失败不能打断问诊"这条产品取舍。

    抛的是普通 `RuntimeError`（而不是某个驱动的异常）：这条取舍与具体数据库无关，
    换引擎不该让这条测试失去意义——真实世界里的对应物是"库不可达 / 磁盘满 / 权限不对"。
    """

    def __init__(self) -> None:
        self.calls = 0

    def _boom(self) -> None:
        self.calls += 1
        raise RuntimeError("模拟记录库不可用：connection refused")

    def upsert_patient(self, record: PatientRecord) -> None:
        del record
        self._boom()

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None:
        del encounter_id, role, text, institution
        self._boom()

    def save_encounter(self, record: EncounterRecord) -> None:
        del record
        self._boom()

    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None:
        del patient_id, institution
        self._boom()
        return None

    def list_encounters(
        self, patient_id: str, *, institution: str = "", limit: int = 20
    ) -> list[EncounterRecord]:
        del patient_id, institution, limit
        self._boom()
        return []


# --------------------------------------------------------------------- 夹具
def _assessment(**over: Any) -> ClinicalAssessment:
    payload: dict[str, Any] = {
        "urgency": "emergency",
        "red_flags": [
            {
                "condition": "急性冠脉综合征",
                "severity": "critical",
                "why": "胸痛伴大汗",
                "evidence_ids": ["K1"],
            }
        ],
        "departments": [{"name": "心血管内科", "confidence": 0.8, "reason": "胸痛相关"}],
        "differentials": [
            {"condition": "心绞痛", "likelihood": "medium", "rationale": "x", "evidence_ids": ["K1"]}
        ],
        "exams": [{"name": "心电图", "why": "排查心脏问题", "evidence_ids": ["K1"]}],
        "medications": [{"name": "硝酸酯类", "drug_class": "抗心绞痛", "why": "y", "evidence_ids": ["K1"]}],
        "patient_summary": "胸痛伴大汗属于急症，请立刻拨打 120。",
        "followup_questions": [],
        "citations": [
            {"index": 1, "title": "T", "source_url": "https://example.invalid/x", "chunk_id": "K1"}
        ],
    }
    payload.update(over)
    return ClinicalAssessment.model_validate(payload)


_ROUTINE_JSON = json.dumps(
    {
        "urgency": "routine",
        "departments": [{"name": "呼吸内科", "confidence": 0.7, "reason": "咳嗽"}],
        "patient_summary": "建议到呼吸内科就诊。",
        "followup_questions": [],
    },
    ensure_ascii=False,
)


def _pipeline(store: Any, final_text: str = _ROUTINE_JSON) -> ClinicalPipeline:
    return ClinicalPipeline(
        backend=LocalIndexBackend(_Retriever()),
        reasoner=ClinicalReasoner(_LLM(final_text)),
        tool_loop=_ToolLoop(final_text),
        store=store,
    )


# --------------------------------------------------------------------- 往返
class TestRoundTrip:
    def test_encounter_and_assessment_survive_the_round_trip(self, store: PgPersistence) -> None:
        """写进去的结论读回来**字段一致**（含 `red_flags`），对话按写入顺序重建。"""
        assessment = _assessment()
        store.upsert_patient(PatientRecord(patient_id="p1", display_name="张三", attributes={"age": "40"}))
        store.save_encounter(
            EncounterRecord(
                encounter_id="e1",
                patient_id="p1",
                chief_complaint="胸口疼，出大汗",
                assessment=assessment,
            )
        )
        store.append_message(encounter_id="e1", role="user", text="胸口疼，出大汗")
        store.append_message(encounter_id="e1", role="assistant", text="请立刻拨打 120。")

        records = store.list_encounters("p1")
        assert len(records) == 1
        rec = records[0]
        assert rec.encounter_id == "e1"
        assert rec.patient_id == "p1"
        assert rec.chief_complaint == "胸口疼，出大汗"
        assert rec.created_at  # 时间落下来了（读路径要显示"什么时候看的"）

        # 契约对象经 JSON 往返后字段一致——急重症判断是最不能丢的那部分
        assert rec.assessment is not None
        assert rec.assessment.urgency == "emergency"
        assert [f.condition for f in rec.assessment.red_flags] == ["急性冠脉综合征"]
        assert rec.assessment.red_flags[0].severity == "critical"
        assert rec.assessment.red_flags[0].why == "胸痛伴大汗"
        assert rec.assessment.red_flags[0].evidence_ids == ["K1"]
        assert rec.assessment.departments[0].name == "心血管内科"
        assert rec.assessment.exams[0].name == "心电图"
        assert rec.assessment.medications[0].drug_class == "抗心绞痛"
        assert rec.assessment.patient_summary == assessment.patient_summary
        assert rec.assessment.citations[0].source_url == "https://example.invalid/x"

        # transcript 由消息表重建（用户说了什么、助手回了什么不能错位）
        assert [(m["role"], m["content"]) for m in rec.transcript] == [
            ("user", "胸口疼，出大汗"),
            ("assistant", "请立刻拨打 120。"),
        ]

    def test_transcript_in_save_encounter_is_not_dead_weight(self, store: PgPersistence) -> None:
        """**只用 `save_encounter` 的调用方也不丢对话**：这次就诊还没有消息时，用它播种。

        （管线走的是"先 `append_message` 再 `save_encounter`"，那时不会重复播种。）
        """
        store.save_encounter(
            EncounterRecord(
                encounter_id="e1",
                patient_id="p1",
                chief_complaint="咳嗽",
                assessment=_assessment(urgency="routine", red_flags=[]),
                transcript=[
                    {"role": "user", "content": "咳嗽"},
                    {"role": "assistant", "content": "多久了？"},
                ],
            )
        )
        assert store.counts()["messages"] == 2
        rec = store.list_encounters("p1")[0]
        assert [m["content"] for m in rec.transcript] == ["咳嗽", "多久了？"]

    def test_missing_patient_is_none_not_an_error(self, store: PgPersistence) -> None:
        assert store.get_patient("nobody") is None
        assert store.list_encounters("nobody") == []

    def test_broken_assessment_json_does_not_break_the_read(self, store: PgPersistence, schema: str) -> None:
        """库里可能有旧版本契约写下的行：坏 JSON 只让**那一条**结论缺失，不许整个读路径失败。"""
        store.save_encounter(EncounterRecord(encounter_id="e1", patient_id="p1", chief_complaint="咳嗽"))
        with psycopg.connect(_DSN, connect_timeout=5) as raw:
            raw.execute(
                sql.SQL("UPDATE {} SET assessment_json = %s WHERE encounter_id = %s").format(
                    sql.Identifier(schema, "encounters")
                ),
                ("{{", "e1"),
            )
        records = store.list_encounters("p1")
        assert len(records) == 1
        assert records[0].assessment is None
        assert records[0].chief_complaint == "咳嗽"


# --------------------------------------------------------------------- 耐久
class TestDurability:
    def test_data_survives_closing_and_reopening_the_store(self, schema: str) -> None:
        """**这是"持久化"这个词的最小证明**：连接池关掉、重新连同一个库，数据还在。"""
        first = PgPersistence(_DSN, schema=schema)
        first.upsert_patient(PatientRecord(patient_id="p1", display_name="张三"))
        first.save_encounter(
            EncounterRecord(
                encounter_id="e1", patient_id="p1", chief_complaint="咳嗽", assessment=_assessment()
            )
        )
        first.append_message(encounter_id="e1", role="user", text="咳嗽")
        first.close()

        again = PgPersistence(_DSN, schema=schema)
        patient = again.get_patient("p1")
        assert patient is not None and patient.display_name == "张三"
        records = again.list_encounters("p1")
        assert len(records) == 1
        assert records[0].encounter_id == "e1"
        assert records[0].assessment is not None and records[0].assessment.red_flags
        assert [m["content"] for m in records[0].transcript] == ["咳嗽"]
        assert again.counts() == {"patients": 1, "encounters": 1, "messages": 1}
        again.close()

    def test_ensure_schema_is_idempotent_and_keeps_data(self, store: PgPersistence, schema: str) -> None:
        """`ensure_schema()` 可反复调用：第二次构造同一个 schema 不该报错、也不该清数据。"""
        store.upsert_patient(PatientRecord(patient_id="p1", display_name="张三"))
        store.ensure_schema()  # 同一个对象上再来一次

        other = PgPersistence(_DSN, schema=schema)  # 换一个对象（新连接池）再建一次
        try:
            patient = other.get_patient("p1")
            assert patient is not None and patient.display_name == "张三"
            assert other.counts()["patients"] == 1
        finally:
            other.close()

    def test_invalid_schema_name_is_rejected(self) -> None:
        """schema 名会进 SQL 标识符：非法取值必须在**建任何东西之前**被拒。"""
        with pytest.raises(PersistenceError, match="schema"):
            PgPersistence(_DSN, schema='clinical"; DROP SCHEMA public CASCADE; --')


# --------------------------------------------------------------------- 患者 upsert
class TestUpsertPatient:
    def test_same_id_twice_is_one_row_with_updated_fields(self, store: PgPersistence) -> None:
        store.upsert_patient(PatientRecord(patient_id="p1", display_name="张三", attributes={"age": "40"}))
        store.upsert_patient(PatientRecord(patient_id="p1", display_name="张三丰", attributes={"age": "41"}))
        assert store.counts()["patients"] == 1
        patient = store.get_patient("p1")
        assert patient is not None
        assert patient.display_name == "张三丰"
        assert patient.attributes == {"age": "41"}

    def test_empty_fields_do_not_wipe_what_was_recorded(self, store: PgPersistence) -> None:
        """管线每一轮都只带 `patient_id` 来挂归属：**不能**把别处录的姓名/属性抹成空。"""
        store.upsert_patient(PatientRecord(patient_id="p1", display_name="张三", attributes={"age": "40"}))
        store.upsert_patient(PatientRecord(patient_id="p1"))
        patient = store.get_patient("p1")
        assert patient is not None
        assert patient.display_name == "张三"
        assert patient.attributes == {"age": "40"}

    def test_attributes_round_trip_non_ascii(self, store: PgPersistence) -> None:
        store.upsert_patient(PatientRecord(patient_id="p1", attributes={"过敏史": "青霉素"}))
        patient = store.get_patient("p1")
        assert patient is not None and patient.attributes == {"过敏史": "青霉素"}


# --------------------------------------------------------------------- 查询
class TestListEncounters:
    def test_newest_first_and_limit(self, store: PgPersistence) -> None:
        for i, stamp in enumerate(
            ["2026-01-01T00:00:01+00:00", "2026-01-01T00:00:02+00:00", "2026-01-01T00:00:03+00:00"]
        ):
            store.save_encounter(
                EncounterRecord(
                    encounter_id=f"e{i}",
                    patient_id="p1",
                    chief_complaint=f"第{i}次",
                    created_at=stamp,
                )
            )
        assert [r.encounter_id for r in store.list_encounters("p1")] == ["e2", "e1", "e0"]
        assert [r.encounter_id for r in store.list_encounters("p1", limit=2)] == ["e2", "e1"]

    def test_records_are_scoped_to_the_patient(self, store: PgPersistence) -> None:
        """`patient_id` 是记录归属：别人的记录查不到（它**不是认证**，见 SECURITY.md）。"""
        store.save_encounter(EncounterRecord(encounter_id="e1", patient_id="p1", chief_complaint="咳嗽"))
        store.save_encounter(EncounterRecord(encounter_id="e2", patient_id="p2", chief_complaint="头痛"))
        assert [r.encounter_id for r in store.list_encounters("p1")] == ["e1"]
        assert [r.encounter_id for r in store.list_encounters("p2")] == ["e2"]

    def test_institution_is_a_reserved_column_not_tenancy(self, store: PgPersistence) -> None:
        """多机构不在范围内：写入都是空机构；拿一个真实机构名去查会**查不到**——
        这正是"保留列、不做隔离"的语义，将来实现隔离时以本测试为起点改。
        """
        store.save_encounter(EncounterRecord(encounter_id="e1", patient_id="p1", institution=""))
        assert len(store.list_encounters("p1")) == 1  # 空串 = 不过滤
        assert store.list_encounters("p1", institution="某三甲医院") == []

    def test_same_timestamp_keeps_insertion_order(self, store: PgPersistence) -> None:
        """同一秒内的连续几轮也要有稳定次序（`created_at` 相同时按内部 `seq` 倒序兜底）。

        Postgres 没有 SQLite 的 `rowid`，所以表里有一个内部 `bigserial seq` 承担同一职责；
        `ON CONFLICT DO UPDATE` 不会改它，与 `rowid` 的行为一致。
        """
        for i in range(3):
            store.save_encounter(
                EncounterRecord(
                    encounter_id=f"e{i}",
                    patient_id="p1",
                    chief_complaint=f"第{i}次",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            )
        assert [r.encounter_id for r in store.list_encounters("p1")] == ["e2", "e1", "e0"]

    def test_counts_reflect_writes(self, store: PgPersistence) -> None:
        store.upsert_patient(PatientRecord(patient_id="p1"))
        store.save_encounter(EncounterRecord(encounter_id="e1", patient_id="p1"))
        store.append_message(encounter_id="e1", role="user", text="q")
        store.append_message(encounter_id="e1", role="assistant", text="a")
        assert store.counts() == {"patients": 1, "encounters": 1, "messages": 2}


class TestNoopPersistence:
    """默认实现是 `PgPersistence`；`NoopPersistence` 是"显式不落库"，行为必须诚实。"""

    def test_nothing_is_stored_and_nothing_is_pretended(self) -> None:
        s = NoopPersistence()
        s.upsert_patient(PatientRecord(patient_id="p1", display_name="张三"))
        s.save_encounter(EncounterRecord(encounter_id="e1", patient_id="p1", chief_complaint="咳嗽"))
        s.append_message(encounter_id="e1", role="user", text="咳嗽")
        assert s.get_patient("p1") is None
        assert s.list_encounters("p1") == []
        assert s.counts() == {"patients": 0, "encounters": 0, "messages": 0}
        s.close()  # 与 PgPersistence 同形，上层可以不区分地关掉它


# --------------------------------------------------------------------- 管线落库
class TestPipelineWrites:
    def test_turn_is_written_and_readable_by_patient(self, store: PgPersistence) -> None:
        """**`patient_id` 要有真实读者**：它决定记录挂在谁名下，读回路径按它查。"""
        turn = _pipeline(store).advance("我咳嗽三天了，没有发烧", patient_id="p7")

        assert turn.encounter_id, "每轮都要有就诊记录主键（HTTP 层要回给前端）"
        records = store.list_encounters("p7")
        assert [r.encounter_id for r in records] == [turn.encounter_id]
        assert records[0].chief_complaint == "我咳嗽三天了，没有发烧"
        assert records[0].assessment is not None
        assert records[0].assessment.urgency == "routine"
        assert [m["role"] for m in records[0].transcript] == ["user", "assistant"]
        assert records[0].transcript[0]["content"] == "我咳嗽三天了，没有发烧"
        assert records[0].transcript[1]["content"] == turn.message
        assert store.counts() == {"patients": 1, "encounters": 1, "messages": 2}

        # 匿名与具名是两个归属，互不串门
        assert store.list_encounters("anon") == []

    def test_history_is_passed_as_the_transcript_snapshot(self, store: PgPersistence) -> None:
        """`_persist` 拿到的 transcript 是"上文 + 本轮"（管线的 agent 传的是同一份 history）。"""
        history = [{"role": "user", "content": "我咳嗽"}, {"role": "assistant", "content": "多久了？"}]
        _pipeline(store).advance("三天了", state=None, history=history, patient_id="p1")
        # 消息表只收本轮增量（上文属于它自己那几轮）——不会把整段上文按轮重复落库
        assert store.counts()["messages"] == 2
        assert [m["content"] for m in store.list_encounters("p1")[0].transcript] == [
            "三天了",
            "建议到呼吸内科就诊。",
        ]

    def test_store_failure_does_not_break_the_turn(self, caplog: Any) -> None:
        """**核心取舍**：库不可达 / 磁盘满 / 权限不对时，患者仍然要拿到这一轮的答复。"""
        broken = _BrokenStore()
        with caplog.at_level(logging.WARNING, logger="kubedocs.clinical.pipeline"):
            turn = _pipeline(broken).advance("我咳嗽", patient_id="p1")

        assert broken.calls > 0, "假 store 必须真的被调用过（否则这条测试是空跑）"
        assert turn.action == "answer"
        assert turn.assessment is not None and turn.assessment.patient_summary
        assert turn.message == "建议到呼吸内科就诊。"
        assert turn.encounter_id, "落库失败也要生成主键（前端仍要对账这一轮）"
        assert any("落库失败" in r.message for r in caplog.records)

    def test_default_store_is_noop(self) -> None:
        """不注入 store 时**不落库**（测试与"只是跑一下"的场景不该偷偷写数据）。"""
        pipe = ClinicalPipeline(
            backend=LocalIndexBackend(_Retriever()),
            reasoner=ClinicalReasoner(_LLM(_ROUTINE_JSON)),
            tool_loop=_ToolLoop(_ROUTINE_JSON),
        )
        turn = pipe.advance("我咳嗽")
        assert turn.encounter_id  # 主键仍然生成（它标识的是这一轮，不是"已落库"）
        assert pipe._store.counts() == {"patients": 0, "encounters": 0, "messages": 0}  # noqa: SLF001
