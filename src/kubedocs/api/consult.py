"""临床问诊的**交互式 HTTP 层**（`/consult`）：把 `ClinicalPipeline` 接到前端。

## 为什么单独一个模块

`ClinicalPipeline` 的每一段依赖都是注入的（这是它可测的原因），但也因此
**一直没有一个地方真正装配它**——`clinical/__init__.py` 只导出、不构造，
所以"接上映射表能 +14 点"这类收益线上拿不到（CHANGELOG 里记为 Known gap）。
本模块就是那个装配点，并且只做装配与协议转换，不含医学逻辑：

    检索后端 ← 复用 app 的 clinic 运行时（同一份 store / embedder，不重复加载模型）
    重点：`RetrievalTriage(department_map=load_department_map())` ← 这条是实测 +14 点的关键
    主判   ← `ClinicalReasoner` + `LiteLLMAdapter` + `BudgetedClient`（**与 CLI/eval 共用同一本日预算账**）
    红线   ← `RiskGuard`（确定性，LLM 不可绕过）
    挂号   ← `default_registry(booking_backend=BookingService(MockAppointmentProvider()))`
    会话   ← 进程内 `IntakeState`（多轮采问的状态属于会话层，不属于流程）

## 会话状态**在进程内**，这是明确的限制

- 单进程内存字典，**重启即丢**；多副本部署下同一 `session_id` 会落到不同实例。
- 有条数上限（`MAX_SESSIONS`），超出按最久未用淘汰——避免被无限建会话撑爆内存。
- 真要持久化得换外部存储；`clinical/persistence.py` 目前还是占位实现，**不假装它有**。

## 急诊横幅**只在模型自己判出急重症时**出现

- `emergency` = 模型这一轮的 `urgency == "emergency"`（它由模型判出的 `red_flags` 推出），
  且**该判断真的出自模型**（`degraded` 非空 ⇒ 结论来自兜底，不算模型判断，不弹横幅）。
- **关键词红线不参与**：`domains.clinic.redline` 的命中结果既不进提示词、也不影响
  `urgency`（见 `clinical/evidence.py` 的 `render`）。它只在主判不可用时给一句对话提示。
- **不一致要留痕，不静默**：关键词表命中而模型没判出急重症时，回传
  `keyword_hit_conflict=true`，并在对话正文**追加**一句风险提示
  （`RED_LINE_CONFLICT_NOTE`）。**只进对话、不弹横幅**——
  判断权归模型，但"系统担心过这件事"不能被抹掉。

"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.pipeline import ClinicalPipeline, ClinicalTurn

logger = logging.getLogger(__name__)

CONSULT_HTML = ("data", "web", "consult.html")
"""交互式问诊页相对仓库根的位置。"""

MAX_SESSIONS = 200
"""进程内会话上限（超出按最久未用淘汰）。取值只为"别把内存撑爆"，不是容量规划。"""

EMERGENCY_NOTE = (
    "检测到需要立即处理的危险信号（如胸痛、呼吸困难、意识不清、大出血等）。"
    "**请立即拨打 120 或直接前往最近医院急诊科**，不要等待在线建议。"
)
"""模型判出 `emergency` 时横幅的固定文案。放在服务端是刻意的：前端只渲染，不另写一份。"""

RED_LINE_CONFLICT_NOTE = (
    "另外要提醒您：您提到的「{word}」属于需要警惕的信号。"
    "即使系统判断目前不属于最紧急的情况，也请**尽快就医**；"
    "如果它加重、或出现持续胸痛、呼吸困难、意识不清，请**立即拨打 120**。"
)
"""关键词表命中、但模型未判急重症时的**对话内**风险提示（不弹横幅）。

为什么还要留着它：判断权已经交给模型，但"关键词表担心过这件事"若被完全抹掉，
一旦模型漏判就没有任何痕迹。这条提示把它留成**可被复核的一句话**：
横幅不弹（尊重模型的结论），但系统不会假装自己没担心过。
"""

DISCLAIMER = "本结果由系统根据您描述的症状与知识卡片生成，仅供导诊参考，不能替代医生诊断，也不构成处方。"
"""随每轮返回；前端把它作为**对话里的开场系统消息**，而不是页面上的独立面板。"""


# ===========================================================================
# 请求 / 响应契约
# ===========================================================================
class ConsultRequest(BaseModel):
    """一轮问诊请求。`session_id` 由前端生成并复用，多轮采问靠它串起来。"""

    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=128)
    want_booking: bool = False
    confirmed: bool = False


class ResetRequest(BaseModel):
    session_id: str = Field(default="", max_length=128)


# ===========================================================================
# 运行期：管线 + 会话
# ===========================================================================
@dataclass
class _Session:
    intake: IntakeState = field(default_factory=IntakeState)
    turns: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class ConsultRuntime:
    """装配 `ClinicalPipeline` 并持有会话状态。**每个 app 一份**（闭包缓存）。

    `runtime` 参数是一个鸭子类型对象，需要有：
      - `.settings`（clinic 域的 Settings，用于 LLM/预算配置）
      - `.retriever()`（可能返回 `None`——索引没建好时）
    故意不 import `api.app`：那会形成循环导入，也会让本模块无法单测。
    """

    def __init__(self, runtime: Any, *, pipeline: Any = None) -> None:
        self._runtime = runtime
        # 测试接缝：允许直接塞一个现成管线，从而不必加载嵌入器/语料/密钥。
        # 生产路径不传，走下面的惰性装配。
        self._pipeline: Any = pipeline
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._sessions_lock = threading.Lock()

    # ---------------------------------------------------------------- 装配
    def pipeline(self) -> Any:
        """惰性装配一次并缓存（嵌入器与预算文件都贵，不能每请求重来）。

        返回 `Any` 而不是 `ClinicalPipeline`：测试接缝允许塞入任何满足
        `advance()` 协议的对象，标注成具体类型会逼测试桩去继承真实类。
        """
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._build()
            return self._pipeline

    def _build(self) -> ClinicalPipeline:
        from kubedocs.clinical.orchestration import ClinicalToolLoop
        from kubedocs.clinical.rag import LocalIndexBackend
        from kubedocs.clinical.reasoning import BudgetedClient, ClinicalReasoner, LiteLLMAdapter
        from kubedocs.clinical.skills import default_registry
        from kubedocs.clinical.triage import RetrievalTriage, load_department_map
        from kubedocs.domains.clinic.redline import RiskGuard
        from kubedocs.generate import DailyBudget, LiteLLMClient

        settings = self._runtime.settings
        retriever = self._runtime.retriever()
        if retriever is None:
            raise RuntimeError(
                "clinic 索引不可用（store 未打开或索引未建）⇒ 无法装配问诊管线。"
                "先跑 `kubedocs ingest` 建索引。"
            )
        budget = DailyBudget(
            settings.llm_spend_path,
            settings.llm_daily_budget_usd,
            unpriced_usd=settings.llm_unpriced_call_usd,
        )
        # **把 `search_docs` 注册进 clinic 域**——这一步是工具循环能不能工作的前提。
        #
        # 为什么启动时（`app._register_tools`）故意不注册它：那会强制在启动时构造检索器、
        # 加载嵌入模型（首次几十秒），与"资源惰性构造"冲突。
        # 而这里**retriever 已经建好了**（上面那行），此时注册不增加任何成本。
        # 实测教训：漏了这一步，`tool_schemas(domain="clinic")` 在过滤掉副作用工具后是**空的**，
        # 模型一个工具都没得调，于是退回用自己的通用医学知识作答——
        # 表面上"有回答"，实际上 R3「检索不到就不给」被架空了。
        from kubedocs.agent.bootstrap import register_domain_tools

        register_domain_tools("clinic", service=self._booking_backend(), retriever=retriever)
        reasoner = ClinicalReasoner(LiteLLMAdapter(BudgetedClient(LiteLLMClient(settings), budget)))
        # **工具循环**：模型自己决定调哪些模块取证（架构决定：底层约束为大模型）。
        # 用同一个 BudgetedClient ⇒ 循环里每一次调用都过同一本日预算账。
        budgeted = BudgetedClient(LiteLLMClient(settings), budget)
        tool_loop = ClinicalToolLoop(budgeted, domain="clinic")
        # **这一行是实测 +14 点（top-3 67.6%→81.7%）的来源**：
        # 语料里 47.1% 的卡片科室是「未标注」，靠这张补全表把票补回来。
        # 只信 high 档（图谱直接关系 + SameAs 别名）；含模糊归并的 medium 档实测更差。
        dep_map = load_department_map(min_confidence="high")
        if not dep_map:
            logger.warning(
                "科室补全映射表为空（data/clinic/department_map.json 缺失）⇒ 分诊退化为只解析卡片正文，"
                "口语召回会明显下降。跑 scripts/clinic_derive_departments.py 生成。"
            )
        return ClinicalPipeline(
            backend=LocalIndexBackend(retriever),
            reasoner=reasoner,
            skills=default_registry(booking_backend=self._booking_backend()),
            guard=RiskGuard(),
            triage=RetrievalTriage(department_map=dep_map),
            tool_loop=tool_loop,
            top_k=10,
        )

    def _booking_backend(self) -> Any:
        from kubedocs.domains.clinic.booking import BookingService
        from kubedocs.domains.clinic.mock import MockAppointmentProvider

        return BookingService(MockAppointmentProvider())

    # ---------------------------------------------------------------- 会话
    def _session(self, session_id: str) -> _Session:
        key = session_id or "anon"
        with self._sessions_lock:
            sess = self._sessions.get(key)
            if sess is None:
                sess = _Session()
                self._sessions[key] = sess
            sess.last_used = time.time()
            self._sessions.move_to_end(key)
            while len(self._sessions) > MAX_SESSIONS:
                evicted, _ = self._sessions.popitem(last=False)
                logger.info("会话数超上限，淘汰最久未用：%s", evicted)
            return sess

    def reset(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id or "anon", None)

    def session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    def session_turns(self, session_id: str) -> int:
        with self._sessions_lock:
            sess = self._sessions.get(session_id or "anon")
            return sess.turns if sess else 0

    # ---------------------------------------------------------------- 推进
    def advance(
        self,
        session_id: str,
        question: str,
        *,
        want_booking: bool = False,
        confirmed: bool = False,
    ) -> ClinicalTurn:
        sess = self._session(session_id)
        pipeline = self.pipeline()
        turn = pipeline.advance(
            question,
            state=sess.intake,
            want_booking=want_booking,
            confirmed=confirmed,
            patient_id="anon",
        )
        sess.turns += 1
        turn_typed: ClinicalTurn = turn
        return turn_typed


# ===========================================================================
# 协议转换
# ===========================================================================
def _assessment_payload(a: ClinicalAssessment | None) -> dict[str, Any]:
    if a is None:
        return {}
    return {
        "urgency": a.urgency,
        "patient_summary": a.patient_summary,
        "departments": [
            {"name": d.name, "confidence": d.confidence, "reason": d.reason} for d in a.departments
        ],
        "differentials": [
            {"condition": d.condition, "likelihood": d.likelihood, "rationale": d.rationale}
            for d in a.differentials
        ],
        "exams": [{"name": e.name, "why": e.why} for e in a.exams],
        "medications": [{"name": m.name, "drug_class": m.drug_class, "why": m.why} for m in a.medications],
        "followup_questions": list(a.followup_questions),
        "citations": [{"title": c.title, "source_url": c.source_url} for c in a.citations],
    }


def to_payload(turn: ClinicalTurn, session_id: str, *, turns: int = 0) -> dict[str, Any]:
    """把一轮产物转成前端契约。

    ## 三条规则（都由产品决定，改动请连同测试一起改）

    1. **横幅只跟模型的判断**：`emergency` = 模型这一轮给出 `urgency="emergency"`
       且**该结论真的出自模型**（`degraded` 为空）。`urgency` 由模型判出的
       `red_flags` 推出，关键词表不参与。
    2. **关键词命中只作为不一致留痕**：命中而模型未判急重症时，往对话正文追加一句
       风险提示，并回传 `keyword_hit_conflict=true`（不弹横幅）。
    3. **结构化结论不下发到界面**：`departments` / `differentials` / `exams` /
       `medications` / `patient_summary` / `red_flags` 仍然出现在 JSON 里
       （后端留档、契约校验、运维排查都要用），但**前端只渲染 `message`**——
       也就是 `patient_summary`。提示词已要求模型把科室、可能方向、检查、用药、
       以及急重症都写进那段话里（见 `system.md` 三之二）。
    """
    assessment = turn.assessment
    degraded = bool(turn.degraded)
    model_emergency = bool(assessment is not None and assessment.urgency == "emergency" and not degraded)
    keyword_hit_conflict = bool(turn.red_line_positive) and not model_emergency
    message = turn.message
    if keyword_hit_conflict:
        word = turn.evidence.red_line.word if turn.evidence and turn.evidence.red_line else "该症状"
        note = RED_LINE_CONFLICT_NOTE.format(word=word)
        message = f"{message}\n\n{note}" if message else note
    payload: dict[str, Any] = {
        "session_id": session_id,
        "action": turn.action,
        "message": message,
        "emergency": model_emergency,
        "emergency_note": EMERGENCY_NOTE if model_emergency else "",
        # 模型自己判出的急重症（可审计：病症 + 严重程度 + 依据）
        "red_flags": [
            {
                "condition": f.condition,
                "severity": f.severity,
                "why": f.why,
                "evidence_ids": list(f.evidence_ids),
            }
            for f in (assessment.red_flags if assessment else [])
        ],
        # 关键词表的命中（**仅供排查**，不参与判断）：用来发现"模型是否漏了一个真急症"
        "keyword_hit": (turn.evidence.red_line.word if turn.evidence and turn.evidence.red_line else ""),
        "keyword_hit_conflict": keyword_hit_conflict,
        # 模型这一轮**自己决定**调用了哪些模块（架构决定的可见落点：工具由模型驱动）
        "tool_calls": list(turn.tool_calls),
        "degraded": turn.degraded,
        "disclaimer": DISCLAIMER,
        "problems": list(turn.problems),
        "prompt_version": turn.prompt_version,
        "turns": turns,
        "intake": {
            "filled": dict(turn.intake.filled),
            "asked": dict(turn.intake.asked),
            "unfillable": list(turn.intake.unfillable()),
        },
        "skill": (
            {
                "ok": turn.skill.ok,
                "stopped_at": turn.skill.stopped_at,
                "needs_confirmation": turn.skill.needs_confirmation,
                "message": turn.skill.message,
                "steps_done": list(turn.skill.steps_done),
                "data": dict(turn.skill.data),
            }
            if turn.skill is not None
            else None
        ),
    }
    payload.update(_assessment_payload(assessment))
    bundle = turn.evidence
    payload["evidence"] = (
        [
            {"id": i.id, "kind": i.kind, "text": i.text[:400], "score": i.score}
            for i in (bundle.items if bundle else [])
        ]
        if bundle
        else []
    )
    payload["red_line_word"] = (
        bundle.red_line.word if bundle is not None and bundle.red_line is not None else ""
    )
    return payload


# ===========================================================================
# 路由
# ===========================================================================
def build_consult_router(
    *,
    runtime_provider: Callable[[Request], Any],
    page_path: Path,
    pipeline: Any = None,
) -> APIRouter:
    """构造 `/consult` 路由。

    `runtime_provider` 由 `create_app` 注入（返回 clinic 域的运行时对象），
    这样本模块既不 import `app`（避免循环导入），又能复用同一份 store/embedder。
    `pipeline` 是测试接缝：传了就直接用，不装配真管线（见 `ConsultRuntime.__init__`）。
    """
    router = APIRouter(tags=["consult"])
    runtimes: dict[int, ConsultRuntime] = {}
    runtimes_lock = threading.Lock()

    def _rt(request: Request) -> ConsultRuntime:
        key = id(request.app)
        with runtimes_lock:
            rt = runtimes.get(key)
            if rt is None:
                rt = ConsultRuntime(runtime_provider(request), pipeline=pipeline)
                runtimes[key] = rt
            return rt

    @router.get("/consult", response_class=HTMLResponse)
    def consult_page() -> HTMLResponse:
        if not page_path.is_file():
            return HTMLResponse(f"<h1>交互式问诊页缺失</h1><p>未找到 {page_path}</p>", status_code=500)
        return HTMLResponse(page_path.read_text(encoding="utf-8"))

    @router.post("/consult")
    def consult(request: Request, body: ConsultRequest) -> dict[str, Any]:
        rt = _rt(request)
        try:
            turn = rt.advance(
                body.session_id,
                body.question,
                want_booking=body.want_booking,
                confirmed=body.confirmed,
            )
        except RuntimeError as e:
            # 索引没建好是**可预期的运维状态**，给出可执行的指引而不是 500 堆栈
            raise HTTPException(status_code=503, detail=str(e)) from e
        return to_payload(turn, body.session_id, turns=rt.session_turns(body.session_id))

    @router.post("/consult/reset")
    def consult_reset(request: Request, body: ResetRequest) -> dict[str, Any]:
        _rt(request).reset(body.session_id)
        return {"ok": True, "session_id": body.session_id}

    @router.get("/consult/status")
    def consult_status(request: Request) -> dict[str, Any]:
        """运维可见性：会话数、管线是否已装配、补全映射表条数。

        **不触发管线装配**（那会加载嵌入器），所以这个端点便宜、可以随便探。
        """
        from kubedocs.clinical.triage import load_department_map

        rt = _rt(request)
        dep_map = load_department_map(min_confidence="high")
        return {
            "sessions": rt.session_count(),
            "max_sessions": MAX_SESSIONS,
            "pipeline_built": rt._pipeline is not None,  # noqa: SLF001
            "department_map_entries": len(dep_map),
            "department_map_warning": (
                "" if dep_map else "空：data/clinic/department_map.json 缺失 ⇒ 分诊退化为只解析卡片正文"
            ),
            "session_store": "in-process（重启即丢，多副本不共享）",
            "emergency_flag": (
                "只由模型判出的 red_flags/urgency 决定（且该结论必须出自模型）；"
                "关键词表已退出判断链，命中只在对话里留一句不一致提示，不打横幅"
            ),
            "redline_source": "大模型按「病症 + 严重程度」判断（system.md R1）；关键词表不参与",
            "ui_contract": "前端只渲染 message 与急诊横幅；结构化字段仅供后端留档",
            # 前端开场白从这里取，避免在 HTML 里写第二份免责声明
            "disclaimer": DISCLAIMER,
        }

    return router
