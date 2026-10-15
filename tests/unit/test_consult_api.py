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
    RedFlag,
)
from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem, RedLineSignal
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.pipeline import ClinicalTurn


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

    "科室/用药作为回答之后的可选扩展项、等用户来问"——患者追问"那该挂哪个科"时，
    模型若看不到自己上一轮说过什么，就会重跑判断，**同一个病例可能问出另一个科室**。
    """
    pipe = _FakePipeline()
    c = client_factory(pipe)
    c.post("/consult", json={"question": "我咳嗽", "session_id": "h1"})  # 第 1 轮：追问，无 assessment
    c.post("/consult", json={"question": "两天了", "session_id": "h1"})  # 第 2 轮：给结论
    # 第 1 轮没有上文
    assert pipe.histories[0] == []

    c.post("/consult", json={"question": "那该挂哪个科", "session_id": "h1"})  # 第 3 轮：追问细节
    second_history = pipe.histories[2]
    assert [m["role"] for m in second_history] == ["user", "assistant", "user", "assistant"]
    # 追问轮（无 assessment）在上文里就是那句追问原文——这是对的
    assert second_history[1]["content"] == "这种情况持续多久了？"
    # **有结论的那一轮，助手那一侧必须是契约 JSON，不是散文**：
    # 存散文时模型会跟着写散文、不再输出 JSON ⇒ 整轮降级
    # （实测：用户追问"那该挂哪个科？"时 action=error、degraded=模型输出不可解析）。
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
    """最小 turn，用于直接测 `to_payload` 的横幅/一致性逻辑。"""

    def __init__(self, *, red_line: bool, urgency: str, degraded: str = "", red_flags: bool = False) -> None:
        self.action = "answer"
        self.message = "m"
        self.assessment = _assessment(urgency, red_flags=red_flags)
        self.evidence = EvidenceBundle(items=[], red_line=RedLineSignal(word="胸痛") if red_line else None)
        self.problems = []
        self.degraded = degraded
        self.prompt_version = "v1"
        self.red_line_positive = red_line
        self.skill = None
        self.tool_calls: list[dict[str, Any]] = []
        self.intake = IntakeState()


@pytest.mark.parametrize("urgency", ["routine", "urgent"])
def test_keyword_hit_alone_does_not_raise_the_banner(urgency: str) -> None:
    """**架构决定（改动请连同本条一起改）**：横幅只跟**模型判出的急重症**，不跟关键词。

    底层约束是大模型：它按"病症 + 严重程度"判断（`system.md` R1），
    关键词表已退出判断链。
    """
    p = to_payload(_Turn(red_line=True, urgency=urgency), "s")  # type: ignore[arg-type]
    assert p["emergency"] is False
    assert p["emergency_note"] == ""
    assert p["keyword_hit"] == "胸痛"  # 命中事实仍如实回传（供排查）
    assert p["red_flags"] == []


@pytest.mark.parametrize("urgency", ["routine", "urgent"])
def test_keyword_conflict_is_spoken_in_the_dialogue(urgency: str) -> None:
    """**核心回归**：关键词命中但模型没判急重症 ⇒ 不弹横幅，但**风险提示必须出现在对话里**。

    判断权归模型之后，"模型漏判"就成了一条静默失败路径。这条提示把它留成
    **可被复核的一句话**：横幅不弹（尊重模型的结论），但系统不假装自己没担心过。
    """
    p = to_payload(_Turn(red_line=True, urgency=urgency), "s")  # type: ignore[arg-type]
    assert p["keyword_hit_conflict"] is True
    assert "胸痛" in p["message"]  # 命中的词
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
    """**兜底产物不算模型的判断**：主判没走通时结论来自 `reasoning._fallback`，
    即使它写了 `urgency=emergency`，也不该让横幅冒充"模型判出急重症"。
    兜底给患者的 120 指引仍然在对话正文里（信息不丢）。
    """
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
    assert rt.pipeline.__self__ is rt  # 静默：避免"未使用变量"告警同时确认绑定正确


@pytest.fixture(scope="module")
def page_html() -> str:
    """`consult.html` 的原始文本（模块级 fixture：类作用域当实例方法已被 pytest 弃用）。"""
    from kubedocs.config import repo_root

    return (repo_root() / "data" / "web" / "consult.html").read_text(encoding="utf-8")


class TestPageIsDialogueOnly:
    """**产品契约：主页只留对话区**（结构化面板一律移到后台，由模型在正文里讲）。

    这条测试是给未来的人看的：如果哪天有人想把科室表格加回主页，
    他会先看到这条测试失败，而不是默默把产品形态改回去。
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
