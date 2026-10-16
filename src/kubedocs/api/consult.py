"""临床问诊的**交互式 HTTP 层**（`/consult`）：把 `ClinicalPipeline` 接到前端。

本模块只做装配与协议转换，不含医学逻辑。管线各段依赖在此注入：

    检索后端 ← 复用 app 的 clinic 运行时（同一份 store / embedder，不重复加载模型）
    重点     ← `RetrievalTriage(department_map=load_department_map())` 科室补全表
    **工具编排** ← `ClinicalToolLoop`：**模型自己决定调哪些工具取证**（检索 / 引用核对…）
    主判     ← `ClinicalReasoner` + `LiteLLMAdapter` + `BudgetedClient`（**与 CLI/eval 共用同一本日预算账**）
    急诊     ← **模型按「病症 + 严重程度」判出的 `red_flags`**（`system.md` R1）
    挂号     ← `default_registry(booking_backend=BookingService(MockAppointmentProvider()))`
    会话     ← 进程内 `IntakeState` + 对话上文（多轮采问的状态属于会话层，不属于流程）
    就诊记录 ← `PgPersistence(settings.database_url)`（患者 / 就诊 / 消息，落 Postgres 的 `clinical` schema）

## ⚠️ 急诊红线在本路径上**不是** `RiskGuard`（与图路径相反）

本仓库有**两条临床路径，急诊设计相反**，改代码前必须先看清走的是哪条：

| 路径 | 入口 | 急重症怎么判 | `RiskGuard` 的作用 |
|---|---|---|---|
| **本路径**（临床管线） | `/consult` | **模型**按病症 + 严重程度判，输出 `red_flags` | **已退出判断链**：不进提示词、不影响 `urgency`、不触发横幅；只在主判不可用时给一句保守提示，并在"命中而模型未判"时留一条不一致痕迹 |
| 图路径 | `/chat` | `RiskGuard` 关键词命中即强制转急诊（`agent/kernel.py` 的强制边） | **仍是确定性强制边，LLM 不可绕过** |

两条路径**故意并存**，但取舍完全相反：图路径宁可我多发一次急诊（词表覆盖不到的表述
会漏），本路径把判断权交给模型（模型漏判时没有确定性兜底）。
**做产品决策时不要把它们混为一谈。**

## 会话状态与急诊横幅

- 会话状态**在进程内**：单进程内存字典，**重启即丢**，多副本部署下同一 `session_id`
  会落到不同实例；有条数上限（`MAX_SESSIONS`），超出按最久未用淘汰。
- **就诊记录（患者 / 就诊 / 消息）落在 Postgres（`clinical` schema）里**，重启与多副本共享同一份；
  `GET /consult/encounters` 就是读它（`POST /consult` 的 `encounter_id` 是那里的主键）。
  **Postgres 不可达时存储降级为 `NoopPersistence`（一条 warning），`/consult` 照常可用**；
  落库失败**同样不影响本轮答复**（管线记 warning 后继续，见 `clinical/pipeline.py`）。
- `patient_id` 决定记录挂在谁名下，**不是认证**：调用方可以声明任意值。单机内网可信
  部署下没问题，多用户场景必须由网关按调用方身份限制可读的 `patient_id`（见 SECURITY.md）。
- `emergency` = 模型这一轮的 `urgency == "emergency"`（由模型判出的 `red_flags` 推出），
  且 `degraded` 为空——结论来自兜底时不算模型判断，不弹横幅。
- **关键词红线不参与判断**：`domains.clinic.redline` 的命中既不进提示词、也不影响
  `urgency`（见 `clinical/evidence.py` 的 `render`），只在主判不可用时给一句对话提示。
  关键词表命中而模型没判出急重症时，回传 `keyword_hit_conflict=true` 并在对话正文
  **追加**一句风险提示（`RED_LINE_CONFLICT_NOTE`）：**只进对话、不弹横幅**——
  判断权归模型，但"系统担心过这件事"不能被抹掉。
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.persistence import (
    DEFAULT_ENCOUNTER_LIMIT,
    EncounterRecord,
    NoopPersistence,
    PgPersistence,
)
from kubedocs.clinical.pipeline import ClinicalPipeline, ClinicalTurn
from kubedocs.observability import current_trace_id, span, span_attrs

logger = logging.getLogger(__name__)

CONSULT_HTML = ("data", "web", "consult.html")
"""交互式问诊页相对仓库根的位置。"""

MAX_SESSIONS = 200
"""进程内会话上限（超出按最久未用淘汰）。取值只为"别把内存撑爆"，不是容量规划。"""

MAX_HISTORY_TURNS = 6
"""保留的最近问答轮数（一轮 = 用户一句 + 助手一句）。

产品要求"科室/用药作为回答之后的可选扩展项、等用户来问"，追问要答得一致就得有上文。
取 6 轮：够覆盖"主诉→追问→结论→再问科室→再问用药"，又不至于让上下文无限增长
（每条消息都要计费，历史越长每次调用越贵）。
"""

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

判断权已经交给模型，但"关键词表担心过这件事"若被完全抹掉，一旦模型漏判就没有任何
痕迹。这条提示把它留成**可被复核的一句话**：横幅不弹（尊重模型的结论），
但系统不会假装自己没担心过。
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
    patient_id: str = Field(default="anon", max_length=128)
    """这次就诊**记在谁名下**（默认 `anon` = 匿名）。

    它**不是认证**：谁能读到记录取决于调用方声明了什么。单机 / 内网可信部署下由网关
    做鉴权；多用户场景必须在那里限制可读的 `patient_id`（见 SECURITY.md）。
    """


class ResetRequest(BaseModel):
    session_id: str = Field(default="", max_length=128)


# ===========================================================================
# 运行期：管线 + 会话
# ===========================================================================
@dataclass
class _Session:
    intake: IntakeState = field(default_factory=IntakeState)
    turns: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    """本会话的对话上文（user/assistant 成对）。

    没有它，"科室/用药留待用户追问"就做不成：模型看不到自己上一轮说过什么，追问时会
    重跑一遍判断，**同一个病例可能问出另一个科室**。`MAX_HISTORY_TURNS` 只是上界——
    不设上界等于每次调用都把整段对话重新计费。
    """

    def remember(self, question: str, turn: Any) -> None:
        """把这一轮记进上文。

        **助手那一侧存的是契约 JSON，不是给患者看的那段散文。** 存散文时模型会以为
        自己上一轮输出的是散文，这一轮接着写散文、**不再输出契约 JSON**，于是解析失败、
        整轮降级。存 JSON 两个好处都占：`patient_summary` 本来就在 JSON 里（散文信息
        不丢），而且每一轮都在提醒模型"你的输出格式是这个"。
        """
        payload = ""
        assessment = getattr(turn, "assessment", None)
        if assessment is not None and hasattr(assessment, "model_dump_json"):
            payload = assessment.model_dump_json()
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": payload or str(turn.message)})
        keep = MAX_HISTORY_TURNS * 2
        if len(self.history) > keep:
            del self.history[: len(self.history) - keep]


class ConsultRuntime:
    """装配 `ClinicalPipeline` 并持有会话状态。**每个 app 一份**（闭包缓存）。

    `runtime` 参数是一个鸭子类型对象，需要有：
      - `.settings`（clinic 域的 Settings，用于 LLM/预算配置）
      - `.retriever()`（可能返回 `None`——索引没建好时）
    故意不 import `api.app`：那会形成循环导入，也会让本模块无法单测。
    """

    def __init__(self, runtime: Any, *, pipeline: Any = None, store: Any = None) -> None:
        self._runtime = runtime
        # 测试接缝：允许直接塞一个现成管线，从而不必加载嵌入器/语料/密钥。
        # 生产路径不传，走下面的惰性装配。
        self._pipeline: Any = pipeline
        # 存储同样可以注入（测试接缝）：不注入就按 settings 惰性构造真库。
        self._store: Any = store
        # RLock 而不是 Lock：`pipeline()` 持锁时会调 `_build()`，后者要取 store（同一把锁），
        # 用普通 Lock 会在第一次请求上自锁死。
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._sessions_lock = threading.Lock()

    # ---------------------------------------------------------------- 存储
    def store(self) -> Any:
        """就诊记录库（惰性构造一次并缓存）。

        **不触发管线装配**（那会加载嵌入器）：连一次 Postgres + `CREATE ... IF NOT EXISTS`
        是毫秒级的，所以 `/consult/status` 与 `GET /consult/encounters` 都能便宜地拿到它。
        """
        with self._lock:
            if self._store is None:
                self._store = self._build_store()
            return self._store

    def _build_store(self) -> Any:
        """按 settings 构造存储。**数据库不可达也不能让服务起不来**：退回 `NoopPersistence()`
        并留 warning——索引/模型装不上是致命的，记录库装不上只是"这轮不留档"。

        Postgres 没起、口令不对、网络不通都会在这里被挡住：宁可少一条记录，也不能让患者
        拿不到答复（同一条取舍在 `clinical/pipeline.py` 的 `_persist` 里还有一层兜底）。
        """
        settings = getattr(self._runtime, "settings", None)
        dsn = getattr(settings, "database_url", "") if settings is not None else ""
        if not dsn:
            # 没有 settings = 测试接缝里的假 runtime：不落库，也不去碰真实数据库。
            return NoopPersistence()
        try:
            return PgPersistence(dsn)
        except Exception as e:  # noqa: BLE001 - 存储不可用不该拖垮整个问诊服务
            logger.warning(
                "就诊记录库（Postgres）不可用（%s）：本次起不落库，问诊不受影响：%s",
                type(e).__name__,
                e,
            )
            return NoopPersistence()

    # ---------------------------------------------------------------- 装配
    def pipeline(self) -> Any:
        """惰性装配一次并缓存（嵌入器与预算文件都贵，不能每请求重来）。

        返回 `Any` 而不是 `ClinicalPipeline`：测试接缝允许塞入任何满足 `advance()`
        协议的对象，标注成具体类型会逼测试桩去继承真实类。
        """
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._build()
            return self._pipeline

    def _build(self) -> ClinicalPipeline:
        from kubedocs.clinical.orchestration import ClinicalToolLoop
        from kubedocs.clinical.rag import LocalIndexBackend
        from kubedocs.clinical.reasoning import BudgetedClient, ClinicalReasoner, LiteLLMAdapter
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
        # 启动时（`app._register_tools`）故意不注册它：那会强制在启动时构造检索器、
        # 加载嵌入模型（首次几十秒），与"资源惰性构造"冲突；而这里 retriever 已经建好了
        # （上面那行），此时注册不增加任何成本。
        # 漏了这一步，`tool_schemas(domain="clinic")` 在过滤掉副作用工具后是**空的**，
        # 模型一个工具都没得调，于是退回用自己的通用医学知识作答——表面上"有回答"，
        # 实际上 R3「检索不到就不给」被架空了。
        from kubedocs.agent.bootstrap import register_domain_tools

        register_domain_tools("clinic", service=self._booking_backend(), retriever=retriever)
        reasoner = ClinicalReasoner(LiteLLMAdapter(BudgetedClient(LiteLLMClient(settings), budget)))
        # **工具循环**：模型自己决定调哪些模块取证（架构决定：底层约束为大模型）。
        # 用同一个 BudgetedClient ⇒ 循环里每一次调用都过同一本日预算账。
        budgeted = BudgetedClient(LiteLLMClient(settings), budget)
        tool_loop = ClinicalToolLoop(budgeted, domain="clinic")
        # 科室补全映射表：语料里近半卡片的科室是「未标注」，靠这张表把票补回来。
        # 只信 high 档（图谱直接关系 + SameAs 别名）；含模糊归并的 medium 档准确率更差。
        dep_map = load_department_map(min_confidence="high")
        if not dep_map:
            logger.warning(
                "科室补全映射表为空（data/clinic/department_map.json 缺失）⇒ 分诊退化为只解析卡片正文，"
                "口语召回会明显下降。跑 scripts/clinic_derive_departments.py 生成。"
            )
        return ClinicalPipeline(
            backend=LocalIndexBackend(retriever),
            reasoner=reasoner,
            guard=RiskGuard(),
            triage=RetrievalTriage(department_map=dep_map),
            tool_loop=tool_loop,
            store=self.store(),
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
        patient_id: str = "anon",
    ) -> ClinicalTurn:
        sess = self._session(session_id)
        pipeline = self.pipeline()
        turn = pipeline.advance(
            question,
            state=sess.intake,
            history=list(sess.history),
            patient_id=patient_id,
        )
        sess.turns += 1
        # 记住这一轮：下一轮追问（"那该挂哪个科"）要在上文里看到本轮结论
        sess.remember(question, turn)
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
    """把一轮产物转成前端契约。三条规则都由产品决定，改动请连同测试一起改：

    1. **横幅只跟模型的判断**：`emergency` = 模型这一轮给出 `urgency="emergency"`
       且 `degraded` 为空。`urgency` 由模型判出的 `red_flags` 推出，关键词表不参与。
    2. **关键词命中只作为不一致留痕**：命中而模型未判急重症时，往对话正文追加一句
       风险提示，并回传 `keyword_hit_conflict=true`（不弹横幅）。
    3. **结构化结论不下发到界面**：`departments` / `differentials` / `exams` /
       `medications` / `patient_summary` / `red_flags` 仍然出现在 JSON 里
       （后端留档、契约校验、运维排查都要用），但**前端只渲染 `message`**——
       也就是 `patient_summary`；提示词已要求模型把科室、可能方向、检查、用药以及
       急重症都写进那段话里。
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
        # 这一轮的就诊记录主键：前端拿它去 GET /consult/encounters 对账（空串 = 没落库）
        "encounter_id": turn.encounter_id,
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


def _encounter_payload(record: EncounterRecord) -> dict[str, Any]:
    """一条就诊记录的对外形状（读路径）。`assessment` 复用 `_assessment_payload`，
    所以写进去的结论与读回来的结论是同一组字段，不会两处漂移。
    """
    return {
        "encounter_id": record.encounter_id,
        "patient_id": record.patient_id,
        "institution": record.institution,  # 保留列：当前恒为空串（多机构不在范围内）
        "chief_complaint": record.chief_complaint,
        "created_at": record.created_at,
        "assessment": _assessment_payload(record.assessment),
        "transcript": [dict(m) for m in record.transcript],
    }


_EMPTY_ROWS: dict[str, int] = {"patients": 0, "encounters": 0, "messages": 0}


def _store_row_counts(store: Any) -> dict[str, int]:
    """就诊记录三张表的行数（`/consult/status` 用）。

    `counts()` 不在 `ClinicalStore` 协议里（它是运维读数，不是产品能力），所以存储没实现
    就报零；读取失败也只报零 + warning——**状态端点不该因为存储能力不全而 500**。
    """
    fn = getattr(store, "counts", None)
    if not callable(fn):
        return dict(_EMPTY_ROWS)
    try:
        return dict(fn())
    except Exception as e:  # noqa: BLE001
        logger.warning("读取就诊记录行数失败：%s: %s", type(e).__name__, e)
        return dict(_EMPTY_ROWS)


# ===========================================================================
# 路由
# ===========================================================================
def build_consult_router(
    *,
    runtime_provider: Callable[[Request], Any],
    page_path: Path,
    pipeline: Any = None,
    store: Any = None,
) -> APIRouter:
    """构造 `/consult` 路由。

    `runtime_provider` 由 `create_app` 注入（返回 clinic 域的运行时对象），这样本模块
    既不 import `app`（避免循环导入），又能复用同一份 store/embedder。
    `pipeline` 与 `store` 是测试接缝：传了就直接用，不装配真管线、也不去碰真实数据目录
    （见 `ConsultRuntime.__init__`）。
    """
    router = APIRouter(tags=["consult"])
    runtimes: dict[int, ConsultRuntime] = {}
    runtimes_lock = threading.Lock()

    def _rt(request: Request) -> ConsultRuntime:
        key = id(request.app)
        with runtimes_lock:
            rt = runtimes.get(key)
            if rt is None:
                rt = ConsultRuntime(runtime_provider(request), pipeline=pipeline, store=store)
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
            # 一次问诊 = 一条 trace：端点开根 span，模型调用与工具执行是它的子 span
            with span(
                "http.consult",
                session_id=body.session_id or "anon",
                question_chars=len(body.question),
            ) as current:
                turn = rt.advance(
                    body.session_id,
                    body.question,
                    body.patient_id,
                )
                span_attrs(
                    current,
                    action=turn.action,
                    urgency=str(getattr(turn.assessment, "urgency", "")),
                )
                # trace id 必须在 span 结束**之前**取：结束后当前 span 已失效，
                # 取出来是空串（这正是它一开始为空的bug）。
                trace_id = current_trace_id()
        except RuntimeError as e:
            # 索引没建好是**可预期的运维状态**，给出可执行的指引而不是 500 堆栈
            raise HTTPException(status_code=503, detail=str(e)) from e
        payload = to_payload(turn, body.session_id, turns=rt.session_turns(body.session_id))
        # 回写 trace id：日志/前端/评测报告都能用它把这次问诊与追踪对上
        payload["trace_id"] = trace_id
        return payload

    @router.post("/consult/reset")
    def consult_reset(request: Request, body: ResetRequest) -> dict[str, Any]:
        _rt(request).reset(body.session_id)
        return {"ok": True, "session_id": body.session_id}

    @router.get("/consult/encounters")
    def consult_encounters(
        request: Request,
        patient_id: str = Query(default="anon", max_length=128),
        limit: int = Query(default=DEFAULT_ENCOUNTER_LIMIT, ge=1, le=100),
    ) -> dict[str, Any]:
        """**读回**某患者的最近若干次就诊（新 → 旧）。

        这是"持久化真的在写"的对外证据：`POST /consult` 落的每一轮都能在这里查到，
        `encounter_id` 与那次响应里回传的完全一致（重启进程后依然查得到）。

        **归属不是认证**：`patient_id` 由调用方声明，本仓库不做认证与授权。
        单机 / 内网可信部署下由网关兜住；多用户场景必须在那里限制可读的 `patient_id`。
        """
        store = _rt(request).store()
        try:
            records = store.list_encounters(patient_id, limit=limit)
        except Exception as e:  # noqa: BLE001 - 库坏了是可预期的运维故障 ⇒ 503 + 原因
            logger.warning("读取就诊记录失败：%s: %s", type(e).__name__, e)
            raise HTTPException(503, detail=f"就诊记录不可读：{type(e).__name__}: {e}") from e
        return {
            "patient_id": patient_id,
            "limit": limit,
            "count": len(records),
            "persistence_backend": type(store).__name__,
            "encounters": [_encounter_payload(r) for r in records],
        }

    @router.get("/consult/status")
    def consult_status(request: Request) -> dict[str, Any]:
        """运维可见性：会话数、管线是否已装配、补全映射表条数、就诊记录落库情况。

        **不触发管线装配**（那会加载嵌入器），所以这个端点便宜、可以随便探。
        存储会惰性构造（连 Postgres + 建 schema/表，毫秒级），这样 `persistence_rows`
        是真实读数而不是"还没装配所以显示 0"。
        """
        from kubedocs.clinical.triage import load_department_map

        rt = _rt(request)
        store = rt.store()
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
            # 就诊记录是**真的落库**的：这几行的读数就是"它真的在写"的证明。
            "persistence_backend": type(store).__name__,
            # 落点描述：`PgPersistence` 给的是 `schema@脱敏 DSN`（口令不回显）；
            # Noop 没有落点，如实给空串。
            "persistence_path": str(getattr(store, "location", "")),
            "persistence_rows": _store_row_counts(store),
            "persistence_note": (
                "就诊记录（患者/就诊/消息）已落 Postgres（`clinical` schema）；"
                "备份 / PITR / 权限走数据库既有机制。"
                "会话状态与号源仍是进程内，重启即丢。"
            ),
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
