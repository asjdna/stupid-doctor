"""`/consult` 交互层的契约测试。

这里钉住的是**用户可见的协议**，不是医学逻辑（那一层已有自己的测试）：

1. 多轮采问：`action=ask` 的那一轮要能接着答，且 `session_id` 复用同一份 `IntakeState`；
2. **急诊红线由服务端强制**：即使模型把 urgency 判成 `routine`，
   只要 `red_line_positive` 为真，响应里 `emergency` 就必须为真并带 `emergency_note`——
   前端只渲染、不判断，两处判断必然会不一致；
3. **降级要说出来**：模型没走通时必须回传 `degraded`，不能把兜底文案伪装成模型结论；
4. 索引不可用时是 **503 + 可执行指引**，不是 500 堆栈。

全部用假管线，不联网、不花钱、不需要语料。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
)
from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem, RedLineSignal
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.pipeline import ClinicalTurn


def _assessment(urgency: str = "routine") -> ClinicalAssessment:
    return ClinicalAssessment(
        urgency=urgency,  # type: ignore[arg-type]
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

    def advance(self, question: str, *, state: Any = None, **_: Any) -> ClinicalTurn:
        intake = state if state is not None else IntakeState()
        self.seen.append(question)
        # 判定依据必须是**该会话自己的状态**，不能用跨会话的计数器——
        # 否则"会话隔离"这条测试会因桩写错而假阴性（实测踩过一次）。
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
    def make(pipeline: _FakePipeline | None) -> TestClient:
        page = tmp_path / "consult.html"
        page.write_text("<html>consult page</html>", encoding="utf-8")
        app = FastAPI()
        app.include_router(
            build_consult_router(
                runtime_provider=lambda _req: _Runtime(pipeline),
                page_path=page,
                pipeline=pipeline,
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
    assert second["departments"][0]["name"] == "呼吸内科"
    assert second["differentials"][0]["likelihood"] == "medium"
    assert second["exams"][0]["name"] == "血常规"
    assert second["medications"][0]["drug_class"] == "镇咳药"
    assert second["evidence"][0]["id"] == "c1"
    assert second["disclaimer"] == DISCLAIMER


def test_sessions_are_isolated(client_factory: Any) -> None:
    pipe = _FakePipeline()
    c = client_factory(pipe)
    c.post("/consult", json={"question": "我咳嗽", "session_id": "a"})
    # 另一个会话应当是**第一轮**（继续追问），不能继承 a 的轮次
    other = c.post("/consult", json={"question": "我头痛", "session_id": "b"}).json()
    assert other["action"] == "ask"
    assert other["turns"] == 1


def test_reset_clears_the_session(client_factory: Any) -> None:
    c = client_factory(_FakePipeline())
    c.post("/consult", json={"question": "我咳嗽", "session_id": "s1"})
    assert c.post("/consult/reset", json={"session_id": "s1"}).json()["ok"] is True
    again = c.post("/consult", json={"question": "我咳嗽", "session_id": "s1"}).json()
    assert again["turns"] == 1  # 从第一轮重新开始


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
    """最小 turn，用于直接测 `to_payload` 的红线强制逻辑。"""

    def __init__(self, *, red_line: bool, urgency: str, degraded: str = "") -> None:
        self.action = "answer"
        self.message = "m"
        self.assessment = _assessment(urgency)
        self.evidence = EvidenceBundle(items=[], red_line=RedLineSignal(word="胸痛") if red_line else None)
        self.problems = []
        self.degraded = degraded
        self.prompt_version = "v1"
        self.red_line_positive = red_line
        self.skill = None
        self.intake = IntakeState()


@pytest.mark.parametrize("urgency", ["routine", "urgent"])
def test_red_line_forces_emergency_regardless_of_model(urgency: str) -> None:
    """**核心回归**：红线阳性时，模型把 urgency 判成什么都必须报急诊。

    红线的价值就在于"确定性、LLM 不可绕过"；若 `emergency` 跟着模型的 urgency 走，
    模型一旦漏判，前端就不会显示急诊横幅——这正是最不能发生的一种失败。
    """
    p = to_payload(_Turn(red_line=True, urgency=urgency), "s")  # type: ignore[arg-type]
    assert p["emergency"] is True
    assert p["red_line_positive"] is True
    assert p["emergency_note"] == EMERGENCY_NOTE
    assert "120" in p["emergency_note"]


def test_model_emergency_also_raises_the_flag() -> None:
    """模型自己判成急诊时也要报（红线没命中不代表不急）。"""
    p = to_payload(_Turn(red_line=False, urgency="emergency"), "s")  # type: ignore[arg-type]
    assert p["emergency"] is True
    assert p["emergency_note"] == EMERGENCY_NOTE


def test_no_emergency_when_neither_signals() -> None:
    p = to_payload(_Turn(red_line=False, urgency="routine"), "s")  # type: ignore[arg-type]
    assert p["emergency"] is False
    assert p["emergency_note"] == ""
    assert p["disclaimer"] == DISCLAIMER


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
    assert rt.pipeline.__self__ is rt  # 静默：避免"未使用变量"告警同时确认绑定正确
