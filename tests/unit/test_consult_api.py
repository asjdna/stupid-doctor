"""`/consult` 交互层的契约测试：钉住**用户可见的协议**，不测医学逻辑。

1. 多轮采问：`action=ask` 那轮要能接着答，`session_id` 复用同一份 `IntakeState`；
2. 急诊红线由服务端强制：`red_line_positive` 为真时，响应里 `emergency` 必须为真并带
   `emergency_note`——前端只渲染、不判断，两处判断必然会不一致；
3. 降级要说出来：模型没走通时必须回传 `degraded`，不能把兜底文案伪装成模型结论；
4. 索引不可用是 503 + 可执行指引，不是 500 堆栈。全部用假管线，不联网、不花钱。
5. **就诊记录真的落库**：`POST /consult` 之后 `GET /consult/encounters` 能把它读回来
   （用独立的 `test_consult_*` schema，不碰真实库的 `clinical` schema），
   `/consult/status` 能看到行数在涨。这几条要连真 Postgres：不可达时**只跳过它们**
   （探测写法与 `tests/unit/test_pgvector_store.py` 一致），其余用例照跑。
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg import sql

from kubedocs.api import consult as consult_module
from kubedocs.api.consult import (
    DISCLAIMER,
    EMERGENCY_NOTE,
    ConsultRuntime,
    build_consult_router,
    to_payload,
)
from kubedocs.clinical.contracts import (
    ClinicalAssessment,
    DepartmentAdvice,
    DifferentialItem,
    ExamAdvice,
    MedicationAdvice,
    RedFlag,
)
from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem, RedLineSignal
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.persistence import NoopPersistence, PgPersistence
from kubedocs.clinical.pipeline import ClinicalPipeline, ClinicalTurn
from kubedocs.clinical.rag import LocalIndexBackend
from kubedocs.clinical.reasoning import ClinicalReasoner
from kubedocs.config import Settings

_DSN = Settings().database_url


def _probe(dsn: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    """这个 Postgres 能不能连（就诊记录不要求 pgvector 之类的扩展）。"""
    try:
        with psycopg.connect(dsn, connect_timeout=int(max(1, timeout))) as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - 探测要把任何失败转成原因字符串
        return False, f"{type(e).__name__}: {e}"
    return True, ""


_PG_OK, _PG_WHY = _probe(_DSN)

_requires_pg = pytest.mark.skipif(
    not _PG_OK,
    reason=f"Postgres 不可达，本条未验证：{_PG_WHY}（DSN={_DSN}）",
)


@pytest.fixture
def pg_schema() -> Iterator[str]:
    """独立 schema（随机后缀）：**不污染真实库的 `clinical` schema**，结束即删。"""
    name = f"test_consult_{uuid.uuid4().hex[:10]}"
    yield name
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))


def _assessment(urgency: str = "routine", *, red_flags: bool = False) -> ClinicalAssessment:
    flags = [RedFlag(condition="急性冠脉综合征", severity="critical", why="胸痛伴大汗")] if red_flags else []
    return ClinicalAssessment(
        urgency=urgency,  # type: ignore[arg-type]
        red_flags=flags,
        patient_summary="最近咳嗽，建议呼吸内科就诊。",
        departments=[DepartmentAdvice(name="呼吸内科", confidence=0.8, reason="咳嗽")],
        differentials=[DifferentialItem(condition="急性支气管炎", likelihood="medium")],
        exams=[ExamAdvice(name="血常规", why="看有没有感染")],
        medications=[MedicationAdvice(name="止咳糖浆", drug_class="镇咳药", why="对症")],
    )


class _FakePipeline:
    """两轮：先追问，再作答。用来验证会话状态确实被复用。"""

    def __init__(self, *, red_line: bool = False, degraded: str = "") -> None:
        self.red_line = red_line
        self.degraded = degraded
        self.seen: list[str] = []
        self.histories: list[Any] = []

    def advance(self, question: str, *, state: Any = None, history: Any = None, **_: Any) -> ClinicalTurn:
        intake = state if state is not None else IntakeState()
        self.seen.append(question)
        self.histories.append(history)
        # 判定依据必须是该会话自己的状态（用跨会话计数器会让"会话隔离"那条测试假阴性）
        if "duration" not in intake.filled:
            intake.filled["duration"] = "两天"
            return ClinicalTurn(action="ask", message="这种情况持续多久了？", intake=intake)
        return ClinicalTurn(
            action="answer",
            message="根据您的描述，建议到呼吸内科就诊。",
            assessment=_assessment(),
            evidence=EvidenceBundle(
                items=[EvidenceItem(id="c1", kind="knowledge", text="咳嗽常见于呼吸道感染")],
                red_line=RedLineSignal(word="胸痛") if self.red_line else None,
            ),
            degraded=self.degraded,
            red_line_positive=self.red_line,
            intake=intake,
        )


class _Runtime:
    def __init__(self, pipeline: _FakePipeline | None) -> None:
        self.settings = None
        self._pipeline = pipeline

    def retriever(self) -> Any:
        return object()


@pytest.fixture
def client_factory(tmp_path: Any) -> Any:
    def make(pipeline: _FakePipeline | Any, store: Any = None) -> TestClient:
        page = tmp_path / "consult.html"
        page.write_text("<html>consult page</html>", encoding="utf-8")
        app = FastAPI()
        app.include_router(
            build_consult_router(
                runtime_provider=lambda _req: _Runtime(pipeline),
                page_path=page,
                pipeline=pipeline,
                store=store,
            )
        )
        return TestClient(app)

    return make


def test_page_is_served(client_factory: Any) -> None:
    r = client_factory(_FakePipeline()).get("/consult")
    assert r.status_code == 200
    assert "consult page" in r.text


def test_missing_page_is_loud_not_blank(tmp_path: Any) -> None:
    """页面文件缺失要报错，不能静默返回空白页。"""
    app = FastAPI()
    app.include_router(
        build_consult_router(
            runtime_provider=lambda _req: _Runtime(_FakePipeline()),
            page_path=tmp_path / "nope.html",
        )
    )
    r = TestClient(app).get("/consult")
    assert r.status_code == 500
    assert "缺失" in r.text


def test_followup_round_reuses_session_state(client_factory: Any) -> None:
    """多轮采问：第二轮必须拿到第一轮累积的 `IntakeState`。"""
    pipe = _FakePipeline()
    c = client_factory(pipe)
    first = c.post("/consult", json={"question": "我咳嗽", "session_id": "s1"}).json()
    assert first["action"] == "ask"
    assert first["message"] == "这种情况持续多久了？"
    assert first["intake"]["filled"] == {"duration": "两天"}
    assert first["turns"] == 1

    second = c.post("/consult", json={"question": "两天了", "session_id": "s1"}).json()
    assert second["action"] == "answer"
    assert second["turns"] == 2
    # 结构化字段仍然回传（后端留档 / 契约校验），但**前端不渲染它们**
    assert second["departments"][0]["name"] == "呼吸内科"
    assert second["differentials"][0]["likelihood"] == "medium"
    assert second["exams"][0]["name"] == "血常规"
    assert second["medications"][0]["drug_class"] == "镇咳药"
    assert second["evidence"][0]["id"] == "c1"
    assert second["disclaimer"] == DISCLAIMER
    # 患者只看得到 message（== patient_summary），所以它必须就是那段答复
    assert second["message"] == "根据您的描述，建议到呼吸内科就诊。"


def test_sessions_are_isolated(client_factory: Any) -> None:
    pipe = _FakePipeline()
    c = client_factory(pipe)
    c.post("/consult", json={"question": "我咳嗽", "session_id": "a"})
    # 另一个会话应当是**第一轮**（继续追问），不能继承 a 的轮次
    other = c.post("/consult", json={"question": "我头痛", "session_id": "b"}).json()
    assert other["action"] == "ask"
    assert other["turns"] == 1


def test_conversation_history_is_threaded_into_the_next_turn(client_factory: Any) -> None:
    """**产品前提**：追问要答得一致，就必须带上文。
    追问"那该挂哪个科"时模型看不到上一轮说过什么就会重跑判断，同一个病例可能问出另一个科室。
    """
    pipe = _FakePipeline()
    c = client_factory(pipe)
    c.post("/consult", json={"question": "我咳嗽", "session_id": "h1"})  # 第 1 轮：追问，无 assessment
    c.post("/consult", json={"question": "两天了", "session_id": "h1"})
    assert pipe.histories[0] == []

    c.post("/consult", json={"question": "那该挂哪个科", "session_id": "h1"})
    second_history = pipe.histories[2]
    assert [m["role"] for m in second_history] == ["user", "assistant", "user", "assistant"]
    # 追问轮（无 assessment）在上文里就是那句追问原文
    assert second_history[1]["content"] == "这种情况持续多久了？"
    # 有结论的那一轮，助手那侧必须是契约 JSON：存散文会诱导模型也写散文、不再输出 JSON ⇒ 整轮降级
    assert second_history[3]["content"].strip().startswith("{")
    assert "呼吸内科" in second_history[3]["content"]
    # 散文信息仍在 JSON 里（生产路径里 message == assessment.patient_summary）
    assert "patient_summary" in second_history[3]["content"]


def test_history_is_bounded(client_factory: Any) -> None:
    """上文有上界：不设界等于每次调用都把整段对话重新计费。"""
    from kubedocs.api.consult import MAX_HISTORY_TURNS

    pipe = _FakePipeline()
    c = client_factory(pipe)
    for i in range(MAX_HISTORY_TURNS + 3):
        c.post("/consult", json={"question": f"第{i}问", "session_id": "h2"})
    last = pipe.histories[-1]
    assert len(last) == MAX_HISTORY_TURNS * 2
    # 第 N 轮拿到的上文是「第 1..N-1 轮」，裁到最近 MAX_HISTORY_TURNS 轮
    assert last[-2]["content"] == f"第{MAX_HISTORY_TURNS + 1}问"


def test_reset_clears_the_session(client_factory: Any) -> None:
    c = client_factory(_FakePipeline())
    c.post("/consult", json={"question": "我咳嗽", "session_id": "s1"})
    assert c.post("/consult/reset", json={"session_id": "s1"}).json()["ok"] is True
    again = c.post("/consult", json={"question": "我咳嗽", "session_id": "s1"}).json()
    assert again["turns"] == 1


def test_degraded_is_surfaced(client_factory: Any) -> None:
    """模型没走通时必须说出来，否则用户会把兜底文案当成模型结论。"""
    pipe = _FakePipeline(degraded="模型调用失败：网络断了")
    c = client_factory(pipe)
    c.post("/consult", json={"question": "我咳嗽", "session_id": "s1"})
    p = c.post("/consult", json={"question": "两天", "session_id": "s1"}).json()
    assert "网络断了" in p["degraded"]


def test_unavailable_index_is_503_with_guidance(client_factory: Any) -> None:
    """索引没建好是可预期的运维状态 ⇒ 503 + 指引，不是 500 堆栈。"""

    class _NoIndex(_Runtime):
        def retriever(self) -> Any:
            return None

    app = FastAPI()
    app.include_router(
        build_consult_router(
            runtime_provider=lambda _req: _NoIndex(None),
            page_path=__import__("pathlib").Path(__file__),
        )
    )
    r = TestClient(app).post("/consult", json={"question": "我咳嗽", "session_id": "x"})
    assert r.status_code == 503
    assert "索引" in r.json()["detail"]


class _Turn:
    """最小 turn，用于直接测 `to_payload` 的横幅/一致性逻辑。"""

    def __init__(self, *, red_line: bool, urgency: str, degraded: str = "", red_flags: bool = False) -> None:
        self.action = "answer"
        self.message = "m"
        self.encounter_id = "e1"
        self.assessment = _assessment(urgency, red_flags=red_flags)
        self.evidence = EvidenceBundle(items=[], red_line=RedLineSignal(word="胸痛") if red_line else None)
        self.problems = []
        self.degraded = degraded
        self.prompt_version = "v1"
        self.red_line_positive = red_line
        self.tool_calls: list[dict[str, Any]] = []
        self.intake = IntakeState()


@pytest.mark.parametrize("urgency", ["routine", "urgent"])
def test_keyword_hit_alone_does_not_raise_the_banner(urgency: str) -> None:
    """横幅只跟**模型判出的急重症**，不跟关键词（判断标准见 system.md R1）。

    这是架构决定，改动请连同本条一起改。
    """
    p = to_payload(_Turn(red_line=True, urgency=urgency), "s")  # type: ignore[arg-type]
    assert p["emergency"] is False
    assert p["emergency_note"] == ""
    assert p["keyword_hit"] == "胸痛"  # 命中事实仍如实回传（供排查）
    assert p["red_flags"] == []


@pytest.mark.parametrize("urgency", ["routine", "urgent"])
def test_keyword_conflict_is_spoken_in_the_dialogue(urgency: str) -> None:
    """**核心回归**：关键词命中但模型没判急重症 ⇒ 不弹横幅，但风险提示必须出现在对话里。
    判断权归模型后"模型漏判"成了静默失败路径；这条提示把它留成可被复核的一句话。
    """
    p = to_payload(_Turn(red_line=True, urgency=urgency), "s")  # type: ignore[arg-type]
    assert p["keyword_hit_conflict"] is True
    assert "胸痛" in p["message"]
    assert "120" in p["message"]  # 可执行的急救指引
    assert p["message"].startswith("m")  # 原回答仍在，提示是**追加**而不是替换


def test_model_emergency_raises_the_banner() -> None:
    p = to_payload(_Turn(red_line=False, urgency="emergency"), "s")  # type: ignore[arg-type]
    assert p["emergency"] is True
    assert p["emergency_note"] == EMERGENCY_NOTE
    assert p["keyword_hit_conflict"] is False  # 模型判了急重症就不算不一致
    assert "120" in p["emergency_note"]


def test_model_red_flags_are_exposed_for_audit() -> None:
    """**红线的新形态**：模型判出的急重症要带"病症 + 严重程度 + 依据"，可审计。"""
    p = to_payload(_Turn(red_line=False, urgency="emergency", red_flags=True), "s")  # type: ignore[arg-type]
    assert p["emergency"] is True
    assert len(p["red_flags"]) == 1
    flag = p["red_flags"][0]
    assert flag["condition"] == "急性冠脉综合征"
    assert flag["severity"] == "critical"
    assert flag["why"]  # 依据必填——没有依据的判断无法复核


def test_degraded_emergency_does_not_raise_the_banner() -> None:
    """**兜底产物不算模型的判断**：`reasoning._fallback` 写的 `urgency=emergency` 不该弹横幅，120 指引仍在正文里。"""
    turn = _Turn(red_line=True, urgency="emergency", degraded="模型调用失败：网络断了")
    p = to_payload(turn, "s")  # type: ignore[arg-type]
    assert p["emergency"] is False
    assert p["emergency_note"] == ""
    assert p["keyword_hit_conflict"] is True  # 仍然要说出风险
    assert p["degraded"] != ""


def test_no_emergency_when_neither_signals() -> None:
    p = to_payload(_Turn(red_line=False, urgency="routine"), "s")  # type: ignore[arg-type]
    assert p["emergency"] is False
    assert p["emergency_note"] == ""
    assert p["keyword_hit_conflict"] is False
    assert p["disclaimer"] == DISCLAIMER


def test_payload_carries_the_encounter_id() -> None:
    """这一轮的就诊记录主键必须回给前端：否则前端无法对账"这一轮到底存下来没有"。"""
    p = to_payload(_Turn(red_line=False, urgency="routine"), "s")  # type: ignore[arg-type]
    assert p["encounter_id"] == "e1"


def test_status_endpoint_does_not_build_the_pipeline() -> None:
    """状态端点要便宜：不能因为探活就把嵌入器加载起来。"""
    rt = ConsultRuntime(_Runtime(_FakePipeline()))
    app = FastAPI()
    app.include_router(
        build_consult_router(
            runtime_provider=lambda _req: _Runtime(_FakePipeline()),
            page_path=__import__("pathlib").Path(__file__),
        )
    )
    body = TestClient(app).get("/consult/status").json()
    assert body["pipeline_built"] is False
    assert body["sessions"] == 0
    assert "in-process" in body["session_store"]
    assert body["disclaimer"] == DISCLAIMER  # 前端开场白从这里取，页面里不写第二份
    # 没有 settings（假 runtime）⇒ 不落盘：如实报 Noop，而不是假装有库
    assert body["persistence_backend"] == "NoopPersistence"
    assert body["persistence_rows"] == {"patients": 0, "encounters": 0, "messages": 0, "bookings": 0}
    assert rt.pipeline.__self__ is rt  # 静默：避免"未使用变量"告警同时确认绑定正确


# ===========================================================================
# 就诊记录：POST /consult 真的写进去了，GET /consult/encounters 能读回来
# ===========================================================================
class _StubChunk:
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


class _StubRetriever:
    def search(self, query: str, *, top_k: int | None = None, channels: Any = None) -> Any:
        del query, top_k, channels
        return type("_R", (), {"query": "q", "results": [_StubChunk()]})()


class _StubLoop:
    """最小工具循环替身：回一个合法契约 JSON（医学逻辑不在这里测）。"""

    def run(
        self, *, system: str, question: str, intake: Any, history: Any = None, context: Any = None
    ) -> Any:
        # `context` 是管线注入的环境参数（`patient_id`）。签名必须与真实循环一致，
        # 否则待测路径会 TypeError 并静默降级，"验证答复"就变成了"验证降级"。
        del system, question, intake, history, context
        text = json.dumps(
            {
                "urgency": "routine",
                "departments": [{"name": "呼吸内科", "confidence": 0.7}],
                "patient_summary": "建议到呼吸内科就诊。",
            },
            ensure_ascii=False,
        )
        return type("_R", (), {"final_text": text, "calls": [], "evidence": EvidenceBundle(items=[])})()


class _StubLLM:
    """只服务降级路径（正常轮次不经过主判）。"""

    def complete(self, system: str, user: str) -> str:
        del system, user
        return "{}"


def _real_pipeline(store: Any) -> ClinicalPipeline:
    """真的 `ClinicalPipeline`（检索/模型/工具循环是替身）：**落库这一段必须走真代码**。"""
    return ClinicalPipeline(
        backend=LocalIndexBackend(_StubRetriever()),
        reasoner=ClinicalReasoner(_StubLLM()),
        tool_loop=_StubLoop(),
        store=store,
    )


@_requires_pg
def test_consult_writes_a_record_that_encounters_reads_back(client_factory: Any, pg_schema: str) -> None:
    """**"持久化有作用"的证据**：POST 一轮 → GET /consult/encounters 能查到同一条。

    库是**独立的 `test_consult_*` schema**（结束即 `DROP SCHEMA CASCADE`）：不碰真实库的
    `clinical` schema，测试之间也互不可见。
    """
    store = PgPersistence(_DSN, schema=pg_schema)
    try:
        c = client_factory(_real_pipeline(store), store=store)

        posted = c.post(
            "/consult", json={"question": "我咳嗽三天了", "session_id": "s1", "patient_id": "p-1"}
        ).json()
        assert posted["action"] == "answer"
        assert posted["encounter_id"], "响应必须回传就诊记录主键，前端才能对账"

        got = c.get("/consult/encounters", params={"patient_id": "p-1"}).json()
        assert got["count"] == 1
        assert got["persistence_backend"] == "PgPersistence"
        enc = got["encounters"][0]
        assert enc["encounter_id"] == posted["encounter_id"]
        assert enc["chief_complaint"] == "我咳嗽三天了"
        assert enc["created_at"]
        assert enc["assessment"]["patient_summary"] == "建议到呼吸内科就诊。"
        assert [m["role"] for m in enc["transcript"]] == ["user", "assistant"]
        assert enc["transcript"][1]["content"] == posted["message"]

        # 归属：换个人查就查不到（`patient_id` 是记录归属，不是认证）
        assert c.get("/consult/encounters", params={"patient_id": "p-2"}).json()["count"] == 0

        # 运维可见性：状态端点能看到行数真的在涨（会话状态仍是内存态，如实说明）
        status = c.get("/consult/status").json()
        assert status["persistence_backend"] == "PgPersistence"
        assert status["persistence_rows"] == {"patients": 1, "encounters": 1, "messages": 2, "bookings": 0}
        # 落点描述是 `schema@脱敏 DSN`：说清落在哪，口令不回显
        assert status["persistence_path"].startswith(f"{pg_schema}@")
        assert "***" in status["persistence_path"]
        assert "in-process" in status["session_store"]
    finally:
        store.close()


def test_encounters_endpoint_is_empty_without_persistence(client_factory: Any) -> None:
    """没有注入存储（假 runtime 没有 settings）时读回空列表并如实报后端，而不是 500。"""
    got = client_factory(_FakePipeline()).get("/consult/encounters").json()
    assert got["count"] == 0
    assert got["encounters"] == []
    assert got["persistence_backend"] == "NoopPersistence"


def test_encounters_limit_is_validated(client_factory: Any) -> None:
    """`limit` 有上下界：`0` 或超界是 422，不是"悄悄按 0 条返回"。"""
    c = client_factory(_FakePipeline())
    assert c.get("/consult/encounters", params={"limit": 0}).status_code == 422
    assert c.get("/consult/encounters", params={"limit": 1000}).status_code == 422


@_requires_pg
def test_runtime_builds_the_store_from_settings_without_self_deadlock(
    pg_schema: str, monkeypatch: Any
) -> None:
    """`pipeline()` 持锁时会调 `_build()` 去取 store（**同一把锁**）。

    锁因此必须是可重入的：普通 `Lock` 会让第一次 `/consult` 直接自锁死（挂住，不报错）。
    这条测试用带超时的线程把"挂住"变成"5 秒后断言失败"。

    存储按 settings 里的 `database_url` 真造（证明"有 settings 就该建真库，而不是静默退回
    Noop"）；`PgPersistence` 被换成**注入独立 schema** 的版本，以免在真实库里建 `clinical`。
    """

    class _WithSettings:
        settings = SimpleNamespace(database_url=_DSN)

        def retriever(self) -> Any:
            return None

    monkeypatch.setattr(consult_module, "PgPersistence", lambda dsn: PgPersistence(dsn, schema=pg_schema))
    rt = ConsultRuntime(_WithSettings())
    got: dict[str, Any] = {}

    def take() -> None:
        with rt._lock:  # noqa: SLF001 - 这就是 `pipeline()` 持锁调 `_build()` 的形状
            got["store"] = rt.store()

    worker = threading.Thread(target=take, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "持锁取 store 自锁死了：`ConsultRuntime._lock` 必须是 RLock"
    assert isinstance(got["store"], PgPersistence), "有 settings 就该建真库，而不是退回 Noop"
    got["store"].close()


def test_unreachable_database_degrades_to_noop_instead_of_failing(caplog: Any) -> None:
    """**核心取舍**：Postgres 不可达时 `/consult` 必须照常可用，只是这轮不留档。

    用一个**解析不了的主机名**（`.invalid` 是保留后缀）：与"防火墙丢包"同样是"库不可达"，
    但它立刻失败——换成 127.0.0.1 的 1 号端口 会白等满 10 秒连接超时，把单测拖慢。
    """

    class _BadDsn:
        settings = SimpleNamespace(database_url="postgresql://kubedocs:kubedocs@no-such-host.invalid/db")

        def retriever(self) -> Any:
            return None

    with caplog.at_level(logging.WARNING, logger="kubedocs.api.consult"):
        store = ConsultRuntime(_BadDsn()).store()

    assert isinstance(store, NoopPersistence), "连不上库就该降级，而不是让 /consult 起不来"
    assert store.counts() == {"patients": 0, "encounters": 0, "messages": 0, "bookings": 0}
    assert any("不可用" in r.message for r in caplog.records), "降级必须留痕（warning）"


@pytest.fixture(scope="module")
def page_html() -> str:
    """`consult.html` 的原始文本（模块级 fixture：类作用域当实例方法已被 pytest 弃用）。"""
    from kubedocs.config import repo_root

    return (repo_root() / "data" / "web" / "consult.html").read_text(encoding="utf-8")


class TestPageIsDialogueOnly:
    """**产品契约：主页只留对话区**（结构化面板移到后台，由模型在正文里讲）。
    有人要把面板加回主页时，这条测试应当先红，而不是产品形态被默默改回去。
    """

    @pytest.mark.parametrize("panel_id", ["depts", "diffs", "exams", "meds", "summary", "ev", "raw"])
    def test_structured_panels_are_gone(self, page_html: str, panel_id: str) -> None:
        assert f'id="{panel_id}"' not in page_html, f"主页不应再有 {panel_id} 面板"

    @pytest.mark.parametrize("title", ["分诊建议", "可能病症", "建议检查", "用药提示", "小结与依据"])
    def test_panel_headings_are_gone(self, page_html: str, title: str) -> None:
        assert title not in page_html, f"主页不应再有「{title}」区块"

    def test_dialogue_essentials_remain(self, page_html: str) -> None:
        """只留对话 ≠ 功能缺失：输入、发送、重新开始、急诊横幅都要在。"""
        for need in ('id="stream"', 'id="q"', 'id="send"', 'id="reset"', 'id="emg"'):
            assert need in page_html, f"对话区缺少 {need}"

    def test_banner_is_hidden_until_the_server_says_so(self, page_html: str) -> None:
        """横幅默认不显示，只有脚本按服务端的 `emergency` 加上 `.show` 才出现。"""
        assert "#emg { display:none; }" in page_html
        assert "#emg.show" in page_html
        assert "p.emergency" in page_html
        # 客户端**不得**自己拿红线或 urgent 去判断急诊（那会和服务端两处判断不一致）
        assert "red_line_positive" not in page_html
        assert '"urgent"' not in page_html

    def test_client_does_not_render_structured_fields(self, page_html: str) -> None:
        """科室/病症/检查/用药这些字段由模型写进正文，前端不该再去读它们。"""
        for field in (
            'p["departments"]',
            "p.departments",
            "p.differentials",
            "p.exams",
            "p.medications",
        ):
            assert field not in page_html, f"前端不应再渲染 {field}"
