"""临床问诊的**交互式 HTTP 层**（`/consult`）：把 `ClinicalPipeline` 接到前端。

本模块只做装配与协议转换，不含医学逻辑。管线各段依赖在此注入：

    检索后端 ← 复用 app 的 clinic 运行时（同一份 store / embedder，不重复加载模型）
    重点     ← `RetrievalTriage(department_map=load_department_map())` 科室补全表
    工具编排 ← `ClinicalToolLoop`：**模型自己决定调哪些工具取证**（检索 / 科室 / 号源 / 挂号提议）
    主判     ← `ClinicalReasoner` + `LiteLLMAdapter` + `BudgetedClient`（**与 CLI/eval 共用同一本日预算账**）
    急诊     ← **模型按「病症 + 严重程度」判出的 `red_flags`**（`system.md` R1）
    号源     ← `PgAppointmentProvider`（`booking` schema）；连不上时降级为进程内 Mock
    挂号     ← `PendingBookings`（待确认单）+ 本模块三个写端点（**模型没有写权限**）
    会话     ← 进程内 `IntakeState` + 对话上文（多轮采问的状态属于会话层，不属于流程）
    就诊记录 ← `PgPersistence(settings.database_url)`（患者/就诊/消息/挂号关联，落 `clinical` schema）

## 挂号：模型提议，患者点确认，服务端执行

写操作**只由服务端执行**，模型这一侧只能读：

| 谁 | 能做什么 | 走哪条路 |
|---|---|---|
| 模型 | 列科室 / 查号源 / 看放号窗口 / 查本人已挂的号 / **提议**一张待确认单 | 工具循环（`propose_booking` 只读） |
| 患者 | **点确认**才真的取号；点退号才真的退号 | `POST /consult/booking/confirm`、`POST /consult/booking/cancel` |
| 前端 | 渲染服务端给的卡片，**不能**自己拼科室/日期/费用 | 响应里的 `pending_action` / `appointments` |

确认请求只带 `proposal_id`——科室、日期、时段、挂号费都取自服务端那张单子。
待确认单**一次性消费**（双击重放只会成功一次）、15 分钟过期、进程内存储（见 `pending.py`）。

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
    BookingLink,
    EncounterRecord,
    NoopPersistence,
    PgPersistence,
)
from kubedocs.clinical.pipeline import ClinicalPipeline, ClinicalTurn
from kubedocs.domains.clinic.pending import PendingBookings
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


class ConfirmBookingRequest(BaseModel):
    """确认一张待确认的挂号单（**用户点了按钮**才发这个请求）。

    只带 `proposal_id`，**不带科室/日期/时段/费用**：那些参数留在服务端那张单子上。
    若让前端回传它们，界面就成了价格的真相源——改一个数字就能少付 5 块钱。
    """

    proposal_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(default="", max_length=128)
    patient_id: str = Field(default="anon", max_length=128)
    patient_name: str = Field(default="", max_length=32)


class CancelBookingRequest(BaseModel):
    """退号。凭据号 + 就诊人——归属在号源层核对（只有本人能退自己的号）。"""

    ticket_no: str = Field(min_length=1, max_length=32)
    session_id: str = Field(default="", max_length=128)
    patient_id: str = Field(default="anon", max_length=128)


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

    last_encounter_id: str = ""
    """最近一轮落库的就诊记录主键。挂号关联要挂到**具体哪次就诊**上，而按钮是异步点的，
    所以这个值必须由会话记住——不能指望确认请求里带着它。"""

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


def build_booking_service(settings: Any) -> Any:
    """号源服务的构造点（实现见 `domains.clinic.assembly`）。**保留这个转发**是为了让
    `/consult` 的调用点不必知道号源装配在哪个模块；真正的逻辑只有一份。"""
    from kubedocs.domains.clinic.assembly import build_booking_service as _build

    return _build(settings)


class ConsultRuntime:
    """装配 `ClinicalPipeline` 并持有会话状态。**每个 app 一份**（闭包缓存）。

    `runtime` 参数是一个鸭子类型对象，需要有：
      - `.settings`（clinic 域的 Settings，用于 LLM/预算配置）
      - `.retriever()`（可能返回 `None`——索引没建好时）
    故意不 import `api.app`：那会形成循环导入，也会让本模块无法单测。

    `booking_provider` 与 `pending` 是**跨路由共享**的接缝：`/consult` 与 `/chat`
    必须用同一个号源服务与同一份待确认单，否则"在问诊页提议、在对话页确认"会两边都对不上。
    它们由 `create_app` 建好挂在 `app.state` 上再传进来；不传就各自惰性构造（单测用）。
    """

    def __init__(
        self,
        runtime: Any,
        *,
        pipeline: Any = None,
        store: Any = None,
        pending: PendingBookings | None = None,
        booking_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._runtime = runtime
        # 测试接缝：允许直接塞一个现成管线，从而不必加载嵌入器/语料/密钥。
        # 生产路径不传，走下面的惰性装配。
        self._pipeline: Any = pipeline
        # 存储同样可以注入（测试接缝）：不注入就按 settings 惰性构造真库。
        self._store: Any = store
        # 待确认挂号单：**进程内**、一次性消费（理由见 `pending.py`）。它不需要惰性构造，
        # 建一个空字典的成本可以忽略。
        self._pending = pending if pending is not None else PendingBookings()
        self._booking_provider = booking_provider
        self._booking: Any = None
        # RLock 而不是 Lock：`pipeline()` 持锁时会调 `_build()`，后者要取 store（同一把锁），
        # 用普通 Lock 会在第一次请求上自锁死。
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._sessions_lock = threading.Lock()

    # ---------------------------------------------------------------- 号源
    def pending(self) -> PendingBookings:
        """待确认挂号单的存储（工具层与确认端点共用同一份）。"""
        return self._pending

    def booking(self) -> Any:
        """号源服务（`BookingService`），惰性构造一次并缓存。

        `booking_provider` 由 `create_app` 注入时**直接沿用**（不做二次包装）：
        `/chat` 与 `/consult` 拿到的必须是同一个对象，否则号源库（连接池）会被建两遍。
        """
        with self._lock:
            if self._booking is None:
                if self._booking_provider is not None:
                    self._booking = self._booking_provider()
                else:
                    from kubedocs.domains.clinic.assembly import build_booking_service

                    self._booking = build_booking_service(getattr(self._runtime, "settings", None))
            return self._booking

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

        register_domain_tools(
            "clinic",
            service=self.booking(),
            retriever=retriever,
            # 没有 `pending` 就不会注册 `propose_booking`，模型也就没法发起挂号——
            # 挂号的入口是"提议一张待确认单"，不是"直接取号"。
            pending=self._pending,
        )
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
            # 参考标准面板由**域层**装配：症状词表、科室表、号源事实都是 clinic 域的知识，
            # 管线那一层不该知道（它只认 `FactSet` 这个形状）。
            reference_builder=self.build_reference,
            # **有没有大模型**这条状态由 HTTP 层判定（密钥在 settings/env 里）：
            # 管线拿到它就知道该不该产出判断——没有判断者时它只回一句话。
            has_llm=self.has_llm,
        )

    # ---------------------------------------------------------------- 参考标准
    def has_llm(self) -> bool:
        """这个部署到底有没有可用的模型密钥。**一次判定，两处使用**（管线与端点）。"""
        from kubedocs.generate import has_api_key

        settings = getattr(self._runtime, "settings", None)
        if settings is None:  # 测试接缝里的假 runtime：当"有"，否则所有用例都会退化成无模型
            return True
        return bool(has_api_key(settings.llm_model))

    def link_map(self) -> Any:
        """症状→科室关联表（进程内缓存一次）。缺语料时返回空表，面板会如实写出来。"""
        from kubedocs.domains.clinic.pack import link_vocabulary

        keys = link_vocabulary()
        if not keys:
            return {}
        from kubedocs.domains.clinic.domain import ClinicPack

        return ClinicPack().triage_links()

    def booking_facts(self, patient_id: str) -> tuple[list[Any], list[str]]:
        """把**服务端已经确认的挂号/退号**读成面板事实。返回 (事实, 装配问题)。

        这是防"没挂上却说挂上"最直接的一条：模型手里有一份"本轮到底发生过什么"的清单。
        读不到时**不猜**——留一条 note 让模型知道这份事实这次不可用（而不是让它以为"没有挂号"）。
        """
        from kubedocs.clinical.reference import BookingFact

        notes: list[str] = []
        try:
            rows = self.booking().my_appointments(patient_id, 5)
        except Exception as e:  # noqa: BLE001 - 号源不可读不该让整轮问诊失败
            notes.append(f"挂号记录本次读不到（{type(e).__name__}）：请勿声称任何挂号结果")
            return [], notes
        out: list[BookingFact] = []
        for row in rows:
            status = str(row.get("status") or "")
            out.append(
                BookingFact(
                    kind="cancelled" if status == "cancelled" else "confirmed",
                    department=str(row.get("department") or ""),
                    date=str(row.get("date") or ""),
                    time_slot=str(row.get("time") or ""),
                    fee=int(row.get("fee") or 0),
                    ticket_no=str(row.get("ticket_no") or ""),
                    queue_no=int(row.get("queue_no") or 0),
                    status=status,
                )
            )
        return out, notes

    def build_reference(self, question: str, evidence: Any, patient_id: str) -> Any:
        """装配这一轮的参考标准面板（`FactSet`）。**注入给管线**，见 `_build`。"""
        from kubedocs.domains.clinic.data.departments import DEPARTMENTS
        from kubedocs.domains.clinic.redline import RiskGuard
        from kubedocs.domains.clinic.reference import clinic_reference

        links = self.link_map()
        notes: list[str] = []
        if not links:
            notes.append("症状→科室关联表为空（卡片语料缺失）")
        booking, booking_notes = self.booking_facts(patient_id)
        notes.extend(booking_notes)
        return clinic_reference(
            question,
            links=links,
            guard=RiskGuard(),
            evidence=evidence,
            booking=booking,
            known_departments=DEPARTMENTS,
            notes=notes,
        )

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
        sess.last_encounter_id = str(getattr(turn, "encounter_id", "") or "")
        # 记住这一轮：下一轮追问（"那该挂哪个科"）要在上文里看到本轮结论
        sess.remember(question, turn)
        turn_typed: ClinicalTurn = turn
        return turn_typed

    # ---------------------------------------------------------------- 挂号
    def confirm_booking(
        self, *, session_id: str, proposal_id: str, patient_id: str, patient_name: str = ""
    ) -> dict[str, Any]:
        """执行一张待确认挂号单。**这是本仓库唯一的挂号写入口**。

        顺序刻意如此：

        1. `take()` 先取出单子——**一次性**，所以双击/重放只会有一张单子生效；
        2. 再用单子上的参数取号（号源层用一句条件更新占号，并发不超卖）；
        3. 最后写挂号关联（就诊 → 号，单向）。关联写失败**不回滚挂号**：号已经挂出去了，
           退掉它比少一条关联更糟。失败留 warning，`/consult/status` 能看到关联行数。

        **取单与占号必须在同一个 `try` 里**：两者都会抛 `BookingError`（单子过期 / 不是本人 /
        号被占满 / 停诊），而这四件事对界面是同一类结果——**409 + 一句人话**。只兜住占号
        那一段的话，"单子过期"会以未捕获异常的形态变成 500，而它恰恰是最常见的一种
        （患者想了两分钟才点确认）。
        """
        from kubedocs.domains.clinic.provider import BookingError

        try:
            proposal = self._pending.take(proposal_id, patient_id=patient_id)
            ticket = self.booking().book(
                proposal.department,
                proposal.patient_id,
                date=proposal.date,
                time_slot=proposal.time_slot,
                patient_name=patient_name or proposal.patient_name,
            )
        except BookingError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        self._link_booking(session_id, ticket)
        # 显式标注：`self.booking()` 返回 Any（号源实现是注入的），直接 return 会把
        # Any 漏进本方法声明的返回类型。
        booked: dict[str, Any] = ticket
        return booked

    def cancel_booking(self, *, session_id: str, ticket_no: str, patient_id: str) -> dict[str, Any]:
        """退号。归属由号源层核对（只有本人能退自己的号）。"""
        from kubedocs.domains.clinic.provider import BookingError

        try:
            result = self.booking().cancel(ticket_no, patient_id)
        except BookingError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        store = self.store()
        sync = getattr(store, "set_booking_status", None)
        if callable(sync):
            try:
                sync(ticket_no, str(result.get("status", "cancelled")))
            except Exception as e:  # noqa: BLE001 - 关联表只是投影，同步失败不该让退号失败
                logger.warning("退号后同步挂号关联失败（%s: %s）：号源那边已经退掉了", type(e).__name__, e)
        del session_id  # 退号不需要会话：凭据号 + 就诊人已经唯一确定了那张单子
        cancelled: dict[str, Any] = result
        return cancelled

    def appointments(self, patient_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """患者本人的挂号记录（新 → 旧）。号源层只按 `patient_id` 过滤，别人的看不见。"""
        rows = self.booking().my_appointments(patient_id, limit)
        return [dict(r) for r in rows]

    def _link_booking(self, session_id: str, ticket: dict[str, Any]) -> None:
        """把"这次就诊挂出了哪张号"写进就诊记录（**单向**关联）。

        目标就诊 = **本会话最近一轮**。挂号按钮是异步点的，请求里不可能带着
        `encounter_id`；而"刚才那轮问诊挂的号"正是唯一有意义的归属。
        会话不存在（例如患者直接调端点、没走问诊）时 `encounter_id` 留空，
        关联仍然记下来——号是真的挂出去了，不该因为找不到就诊行就丢掉这条事实。
        """
        with self._sessions_lock:
            sess = self._sessions.get(session_id or "anon")
            encounter_id = sess.last_encounter_id if sess else ""
        store = self.store()
        add = getattr(store, "add_booking", None)
        if not callable(add):
            return
        try:
            add(
                BookingLink(
                    ticket_no=str(ticket.get("ticket_no", "")),
                    encounter_id=encounter_id,
                    department=str(ticket.get("department", "")),
                    date=str(ticket.get("date", "")),
                    time_slot=str(ticket.get("time", "")),
                    fee=int(ticket.get("fee") or 0),
                    status=str(ticket.get("status", "")),
                )
            )
        except Exception as e:  # noqa: BLE001 - 号已经挂出去了，少一条关联不能反过来退号
            logger.warning("挂号关联写入失败（%s: %s）：号已挂出，关联缺失", type(e).__name__, e)


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


def _booking_cards(tool_calls: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """从这一轮的工具调用留档里拣出**待确认挂号单**与**患者已有的挂号记录**。

    卡片的内容来自**工具的结构化结果**，不是模型那段散文。模型可以说得天花乱坠，
    但界面上"呼吸内科 2026-10-17 上午 35 元"这几个字必然来自服务端写下的那张单子——
    这是"确认按钮可以被信任"的前提。
    """
    pending: dict[str, Any] | None = None
    appointments: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        proposal = result.get("proposal")
        if call.get("name") == "propose_booking" and isinstance(proposal, dict):
            pending = dict(proposal)
        rows = result.get("appointments")
        if call.get("name") == "my_appointments" and isinstance(rows, list):
            appointments = [dict(r) for r in rows if isinstance(r, dict)]
    return pending, appointments


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
    4. **挂号例外**：`pending_action` / `appointments` 是**可操作**的卡片，前端必须渲染。
       它们不是"模型的结论"，而是工具的结构化产出（待确认单 / 挂号记录），
       文案里的科室、日期、费用都来自服务端，模型改不动。
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
    pending_action, appointments = _booking_cards(getattr(turn, "tool_calls", None))
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
        # 这一轮**给它看了什么依据**、以及**它的输出哪里越界被拦了**。
        # 两者必须成对回传：只有同时看到"依据"与"违规"，才能回答
        # "这次是依据不够，还是模型没守规矩"——这是这套架构唯一可信的自证材料。
        "reference": turn.reference,
        "contract_violations": list(turn.contract_violations),
        # 挂号的两张卡片：`pending_action` 需要患者点"确认"才会真的取号；
        # `appointments` 是患者本人的挂号记录（卡片上带"退号"按钮）。
        # 两者都由工具结果生成，不由模型那段话生成（见 `_booking_cards`）。
        "pending_action": pending_action,
        "appointments": appointments,
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


def _booking_payload(link: BookingLink) -> dict[str, Any]:
    """一条挂号关联的对外形状（读路径）。`status` 是投影，权威在号源库里。"""
    return {
        "ticket_no": link.ticket_no,
        "encounter_id": link.encounter_id,
        "department": link.department,
        "date": link.date,
        "time_slot": link.time_slot,
        "fee": link.fee,
        "status": link.status,
        "created_at": link.created_at,
    }


def _encounter_payload(record: EncounterRecord, bookings: list[BookingLink] | None = None) -> dict[str, Any]:
    """一条就诊记录的对外形状（读路径）。`assessment` 复用 `_assessment_payload`，
    所以写进去的结论与读回来的结论是同一组字段，不会两处漂移。

    `bookings` 是这次就诊挂出的号（**单向**关联）：号源那边不知道就诊记录，
    而"这次问诊最后挂了哪个科"正是回看时最想知道的一件事。
    """
    return {
        "encounter_id": record.encounter_id,
        "patient_id": record.patient_id,
        "institution": record.institution,  # 保留列：当前恒为空串（多机构不在范围内）
        "chief_complaint": record.chief_complaint,
        "created_at": record.created_at,
        "assessment": _assessment_payload(record.assessment),
        "transcript": [dict(m) for m in record.transcript],
        "bookings": [_booking_payload(b) for b in (bookings or [])],
    }


_EMPTY_ROWS: dict[str, int] = {"patients": 0, "encounters": 0, "messages": 0, "bookings": 0}


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
    pending: PendingBookings | None = None,
    booking_provider: Callable[[], Any] | None = None,
) -> APIRouter:
    """构造 `/consult` 路由。

    `runtime_provider` 由 `create_app` 注入（返回 clinic 域的运行时对象），这样本模块
    既不 import `app`（避免循环导入），又能复用同一份 store/embedder。
    `pipeline` 与 `store` 是测试接缝：传了就直接用，不装配真管线、也不去碰真实数据目录
    （见 `ConsultRuntime.__init__`）。

    `pending` / `booking_provider` 同样是跨路由共享的接缝：`/chat` 与 `/consult` 必须
    对着同一个号源服务与同一份待确认单说话（见 `ConsultRuntime`）。
    """
    router = APIRouter(tags=["consult"])
    runtimes: dict[int, ConsultRuntime] = {}
    runtimes_lock = threading.Lock()

    def _rt(request: Request) -> ConsultRuntime:
        app = request.app
        # 优先复用**已经挂在 app.state 上的那一份**：`/chat` 可能先一步把它建出来了
        # （它也要同一个号源服务）。各建各的会让"问诊页提议、对话页确认"两边对不上。
        rt = getattr(app.state, "consult_runtime", None)
        if rt is not None:
            found: ConsultRuntime = rt
            return found
        with runtimes_lock:
            rt = getattr(app.state, "consult_runtime", None)
            if rt is None:
                rt = ConsultRuntime(
                    runtime_provider(request),
                    pipeline=pipeline,
                    store=store,
                    pending=pending,
                    booking_provider=booking_provider,
                )
                app.state.consult_runtime = rt
                runtimes[id(app)] = rt
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

    # ------------------------------------------------------------------ 挂号
    @router.post("/consult/booking/confirm")
    def booking_confirm(request: Request, body: ConfirmBookingRequest) -> dict[str, Any]:
        """**确认挂号**（界面上的"确认"按钮调这个）。

        这是全仓库唯一的挂号写入口，也是"写操作只由服务端执行"这条原则的落点：
        模型只能提议（`propose_booking` 是只读的），参数留在服务端那张单子上，
        前端只回传一个单号。

        业务失败（单子过期 / 号被占满 / 停诊）返回 **409 + 一句人话**；
        参数本身不合法由 pydantic 拦成 422。两者对界面的处理方式不同：
        409 要展示那句话并引导改时段，422 是前端 bug。
        """
        rt = _rt(request)
        try:
            with span("http.consult.booking_confirm", session_id=body.session_id or "anon") as current:
                ticket = rt.confirm_booking(
                    session_id=body.session_id,
                    proposal_id=body.proposal_id,
                    patient_id=body.patient_id,
                    patient_name=body.patient_name,
                )
                span_attrs(current, department=ticket.get("department", ""), ok=True)
                trace_id = current_trace_id()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return {
            "ok": True,
            "ticket": ticket,
            "trace_id": trace_id,
            "message": (
                f"已为 {ticket.get('department', '')} 取号："
                f"{ticket.get('date', '')} {ticket.get('time', '')}，"
                f"凭据号 {ticket.get('ticket_no', '')}，"
                f"队列号 {ticket.get('queue_no', '')}，挂号费 {ticket.get('fee', 0)} 元。"
            ),
        }

    @router.post("/consult/booking/cancel")
    def booking_cancel(request: Request, body: CancelBookingRequest) -> dict[str, Any]:
        """退号（挂号卡片上的"退号"按钮调这个）。只有本人能退自己的号。"""
        rt = _rt(request)
        result = rt.cancel_booking(
            session_id=body.session_id,
            ticket_no=body.ticket_no,
            patient_id=body.patient_id,
        )
        return {"ok": True, "cancelled": result, "message": f"凭据号 {body.ticket_no} 已退号。"}

    @router.get("/consult/appointments")
    def consult_appointments(
        request: Request,
        patient_id: str = Query(default="anon", max_length=128),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        """患者本人的挂号记录。**归属不是认证**，与 `GET /consult/encounters` 同一套取舍：
        单机/内网可信部署由网关兜住，多用户场景必须在网关限制可读的 `patient_id`。"""
        rows = _rt(request).appointments(patient_id, limit)
        return {"patient_id": patient_id, "count": len(rows), "appointments": rows}

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
            # 挂号关联一次取回（不是每条就诊查一次）：这是"这次问诊挂出了哪张号"的
            # 唯一可见落点，`booking.appointments` 那边不知道就诊记录的存在。
            many = getattr(store, "list_bookings_many", None)
            links = many([r.encounter_id for r in records]) if callable(many) else {}
        except Exception as e:  # noqa: BLE001 - 库坏了是可预期的运维故障 ⇒ 503 + 原因
            logger.warning("读取就诊记录失败：%s: %s", type(e).__name__, e)
            raise HTTPException(503, detail=f"就诊记录不可读：{type(e).__name__}: {e}") from e
        return {
            "patient_id": patient_id,
            "limit": limit,
            "count": len(records),
            "persistence_backend": type(store).__name__,
            "encounters": [_encounter_payload(r, links.get(r.encounter_id, [])) for r in records],
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
                "就诊记录（患者/就诊/消息/**挂号关联**）已落 Postgres（`clinical` schema）；"
                "备份 / PITR / 权限走数据库既有机制。"
                "会话状态与待确认挂号单仍是进程内，重启即丢。"
            ),
            # 号源：默认 Postgres（跨重启、跨副本），连不上时降级为进程内 Mock。
            "booking_backend": type(getattr(rt.booking(), "provider", None)).__name__,
            "pending_bookings": rt.pending().count(),
            "pending_note": (
                "待确认挂号单只在内存里且**一次性消费**（15 分钟过期）；"
                "确认与退号都走 `/consult/booking/*`，模型拿不到任何写工具"
            ),
            "emergency_flag": (
                "只由模型判出的 red_flags/urgency 决定（且该结论必须出自模型）；"
                "关键词表已退出判断链，命中只在对话里留一句不一致提示，不打横幅"
            ),
            "redline_source": "大模型按「病症 + 严重程度」判断（system.md R1）；关键词表不参与",
            "ui_contract": (
                "前端渲染 message、急诊横幅，以及 `pending_action` / `appointments` 两类挂号卡片"
                "（卡片内容来自服务端，前端不自己拼）；其余结构化字段仅供后端留档"
            ),
            # 前端开场白从这里取，避免在 HTML 里写第二份免责声明
            "disclaimer": DISCLAIMER,
        }

    return router
