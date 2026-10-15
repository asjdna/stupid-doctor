"""FastAPI 服务层：把 CLI 已经跑通的能力包成一组 HTTP 端点。

本层只做协议转换：JSON → 参数 → 内核 → JSON。检索与生成一律复用 CLI 的同一套实现
（`kubedocs.retrieve` / `kubedocs.generate` / `kubedocs.agent`），不在这里重写，
否则"CLI 能跑通"这条可复现性基线就失效了。

## 空索引不是故障

空索引是部署过程中的正常中间态（先起服务、再灌语料），也可能是这个域还没建索引，
所以它不是 500：`/healthz` 给 `ok=true, n_chunks=0`；`/search` `/ask` 给
**200 + 空结果 + 一句说明**（`note` / `degraded`）；索引与配置**不兼容**
（向量维度 / 嵌入模型 / 归一化器不符 → `StoreError`）走同一条路——记下
`index_error`、端点返回 200 + 说明。必须失败的只有"回答里出现没有依据的内容"：
没有语料就明说没有语料。

## 资源只打开一次

`Settings` → `SqliteStore` 是进程级资源（一个连接 + 一份常驻内存的向量矩阵），
在 `create_app` 里按域建好并缓存到 `app.state.runtimes`（约定 `app.state.store` =
当前域的 store），**每次请求不再打开**；embedder 与 retriever 惰性构造——bge 首次
加载要几十秒，`/healthz` 与 `/tools` 不该付这个代价；lifespan 结束时统一 `close()`。

## SSE 事件格式（`/chat`）

`event:` 名字 + **单行 JSON** `data:`（统一由 `_sse()` 构造）：

    event: thread   data: {"thread_id": "..."}       <- 首个事件，前端据此记住会话
    event: tool     data: {"tool": "book_appointment", ...}   <- 0..n 个，真执行了才发
    event: message  data: {"text": "...", ...}       <- 最终文本（或等待用户回答的追问）
    event: done     data: {"thread_id": "..."}       <- 流结束

`data` 一律 `json.dumps(..., ensure_ascii=False)` 并压成一行：SSE 的 data 遇到换行会被
拆成多条 `data:` 行、需要接收端自己拼回来。这里流的是**事件**，不是 LLM 的 token——
clinic 内核的推理是确定性的规则 + 词表（`ClinicAgent` 是同步 `graph.invoke`），本文件
用线程池把它挪出事件循环，使 `thread` 事件能先于这一轮推理送达客户端；token 级流式
要等生成层给出流式接口。

`thread_id` 由客户端自带，服务端管住两件事（实现见 `api/threads.py`）：**属主校验**
（首次出现即绑定 `patient_id`，换一个 `patient_id` 用同一个 `thread_id` → **409**；
校验在**端点里**做，响应头还没发出所以能给明确的 4xx，工具循环路径与确定性图路径
同受这道检查）与**有界**（LRU：`MAX_THREADS` 个会话 × 每会话 `MAX_MESSAGES_PER_THREAD`
条消息，淘汰最久未用者）。

加一个端点：在 `create_app` 里声明 `@app.get` / `@app.post`，把请求与响应形状放进
`api/schemas.py`，重活交给内核（`kubedocs.retrieve` / `kubedocs.generate` /
`kubedocs.agent`）；有状态或有副作用的资源挂到 `app.state`，并在 lifespan 里释放。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from kubedocs.agent.tools import ToolRegistry
from kubedocs.api.consult import CONSULT_HTML, build_consult_router
from kubedocs.api.schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    CitationOut,
    HealthResponse,
    Hit,
    SearchRequest,
    SearchResponse,
    ToolsResponse,
)
from kubedocs.api.threads import ThreadOwnershipError, ThreadStore
from kubedocs.config import Settings, get_settings, repo_root
from kubedocs.domains import load_all
from kubedocs.domains.base import DomainRegistry
from kubedocs.embed import Embedder, build_embedder
from kubedocs.generate import Answer, Answerer, build_answerer, has_api_key
from kubedocs.observability import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    configure_logging,
    metrics_payload,
    publish_config_gauges,
    setup_tracing,
    span,
)
from kubedocs.retrieve.hybrid import HybridRetriever
from kubedocs.store import Store, StoreError, from_settings
from kubedocs.store.base import StoreStats

if TYPE_CHECKING:  # 只为标注：不 import langgraph / agent 内核的实现细节
    from kubedocs.agent.kernel import ClinicAgent
    from kubedocs.agent.state import ClinicState

logger = logging.getLogger(__name__)

INDEX_HTML = ("data", "web", "index.html")
"""演示前端相对仓库根的位置。缺失是正常情况（见 `index` 端点）。"""

CONSOLE_HTML = ("data", "web", "console.html")
"""控制台页相对仓库根的位置。

与演示页分成两个文件：演示页是"最小可用示例"，控制台是"把运维信号摆出来"
（索引体检、`/metrics` 摘要、每路检索名次、降级原因）。两者共用同一套设计令牌，
视觉上是一家的，但职责不同、可以各自演进。
"""

STATS_TTL_S = 2.0
"""`_DomainRuntime.stats()` 的缓存有效期（秒），取值理由见该方法 docstring。"""

PATIENT_HTML = ("data", "web", "patient.html")
"""患者端页面。**只**保留问诊 agent，面向真实使用者（见 `patient` 端点的说明）。"""

# `CONSULT_HTML` 由 `kubedocs.api.consult` 定义并 import（单一来源，避免两处不一致）。
# 与 `patient.html` 的区别：`patient` 走**图 + 工具循环**，`consult` 走 `ClinicalPipeline`
# （契约化输出 + 确定性红线 + 检索式分诊）。两者并存是刻意的：两条路径的取舍不同。


# ===========================================================================
# 运行期资源：一个域一份，只构造一次
# ===========================================================================
class _DomainRuntime:
    """一个域的运行期资源：store / embedder / retriever / answerer。

    按域分开的理由：索引目录是 `<index_dir>/<domain>/`，两个域的向量维度、嵌入模型、
    归一化器、语料都不同。共用一个 store 会把两个域的数据混进同一份矩阵。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_error = ""
        self._retriever: HybridRetriever | None = None
        self._answerer: Answerer | None = None
        # 惰性构造持锁：同步端点跑在线程池里，两个并发请求不该各自构造一遍
        # 嵌入器（bge 那一次就够贵了），也不该各自开一个回答器（会重复读预算文件）
        self._lock = threading.Lock()
        # 统计缓存：见 `stats()` 的 docstring（探针与检索热路径共用这条）
        self._stats_cache: StoreStats | None = None
        self._stats_at = 0.0
        self._stats_lock = threading.Lock()
        try:
            self.store: Store | None = from_settings(settings)
        except (StoreError, sqlite3.Error, OSError) as e:
            # 索引不可用 ≠ 服务不可用：记下原因，端点返回 200 + 说明
            self.store = None
            self.index_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ 只读
    def stats(self, *, max_age_s: float = STATS_TTL_S) -> StoreStats | None:
        """取索引统计（**带短 TTL 缓存**）；索引不可用或库读坏了都返回 None。

        必须缓存：`StoreStats.n_with_code` 在 SQLite 侧是
        `SELECT COUNT(*) FROM chunks WHERE body LIKE '%```%'`，即大文本列的全表扫描
        （约 20 ms）。而 `/healthz` 每个请求都走 `n_chunks()` → `stats()`（`/search`
        也会，只为拼一句提示），并且整段都在 store 的 `threading.Lock` 里，于是并发
        完全失去收益。TTL 取 2 秒（`STATS_TTL_S`）：compose 的探针间隔是 10 s，
        探针看到的仍是"最多 2 秒前"的真实状态。要即时值就传 `max_age_s=0`
        （CLI `stats` 走的是 store 直连，不经过这里）。
        """
        if self.store is None:
            return None
        now = time.monotonic()
        with self._stats_lock:
            if self._stats_cache is not None and (now - self._stats_at) < max_age_s:
                return self._stats_cache
            try:
                fresh = self.store.stats()
            except (StoreError, sqlite3.Error, OSError) as e:
                self.index_error = f"{type(e).__name__}: {e}"
                logger.warning("读索引统计失败：%s", self.index_error)
                return None
            self._stats_cache = fresh
            self._stats_at = now
            return fresh

    def n_chunks(self) -> int:
        st = self.stats()
        return st.n_chunks if st is not None else 0

    # ------------------------------------------------------------------ 惰性构造
    def retriever(self) -> HybridRetriever | None:
        """惰性构造检索器（**第一次真正检索时**才建嵌入器）。"""
        if self.store is None:
            return None
        with self._lock:
            if self._retriever is None:
                embedder = build_embedder(self.settings)
                self._retriever = _build_retriever(self.settings, self.store, embedder)
            return self._retriever

    def answerer(self) -> Answerer | None:
        """惰性构造回答器（复用同一个预算文件与 LLM 配置）。

        生成层**完全复用 CLI `ask` 那一套**：`generate.build_answerer` → `Answerer`
        → `LiteLLMClient`（`litellm.completion`）+ 引用越界剥除 + 日预算。在这里重写
        提示词或引用校验，只会让"CLI 能用"与"HTTP 能用"变成两件事。
        """
        retriever = self.retriever()
        if retriever is None:
            return None
        with self._lock:
            if self._answerer is None:
                self._answerer = build_answerer(retriever, self.settings)
            return self._answerer

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None


def _build_retriever(
    settings: Settings,
    store: Store,
    embedder: Embedder,
) -> HybridRetriever:
    """按域构造检索器：优先用域包自己的 `retriever()`，域包未注册时退回默认实现。

    检索器配置属于域（不同域可以换精排、换候选池大小）。保留默认实现做兜底，
    是为了让"域包没注册"不会变成 500。
    """
    try:
        pack = DomainRegistry.get(settings.domain)
    except KeyError:
        return HybridRetriever(store, embedder, settings)
    return pack.retriever(store, embedder, settings)


# ===========================================================================
# 辅助函数
# ===========================================================================
def _known_domains() -> list[str]:
    """已注册的域名（`load_all()` 幂等，重复调用安全）。"""
    load_all()
    return DomainRegistry.names()


def _require_domain(name: str) -> str:
    """校验 `domain` 参数；空串表示"用当前配置的域"。

    未知域必须 400 并把可用域列出来：静默回退到默认域会让调用方以为"查的就是 clinic"，
    实际拿到 k8s 文档的结果——这种错误比直接报错更难发现。
    """
    known = _known_domains()
    if name and name not in known:
        raise HTTPException(
            status_code=400,
            detail=f"未知的域 {name!r}；可用域：{', '.join(known)}",
        )
    return name


def _runtime_for_app(app: FastAPI, domain: str) -> _DomainRuntime:
    """按域取运行期资源（不存在则建一次并缓存）。"""
    target = domain or app.state.settings.domain
    with app.state.runtimes_lock:
        runtimes: dict[str, _DomainRuntime] = app.state.runtimes
        runtime = runtimes.get(target)
        if runtime is None:
            runtime = _DomainRuntime(app.state.settings.model_copy(update={"domain": target}))
            runtimes[target] = runtime
        return runtime


def _runtime(request: Request, domain: str = "") -> _DomainRuntime:
    return _runtime_for_app(request.app, domain)


def _index_note(runtime: _DomainRuntime, n_hits: int) -> str:
    """空索引 / 无命中时给一句人话；有命中时保持空串。"""
    if n_hits:
        return ""
    if runtime.index_error:
        return f"索引不可用，无法检索：{runtime.index_error}"
    if runtime.n_chunks() == 0:
        return (
            "索引为空（0 块）：先跑 `python -m kubedocs.cli ingest` 建索引；"
            "空索引不是服务故障，所以这里返回 200 而不是 500"
        )
    return "没有任何块命中：换一种说法再试，或确认语料里确实包含这个话题"


def _ask_response(ans: Answer, runtime: _DomainRuntime) -> AskResponse:
    """把 `generate.Answer` 映射成 HTTP 响应，并把"为什么没有回答"写清楚。

    `degraded` 按"最相关的原因在前"拼装：先讲检索有没有依据，再讲生成为什么没跑，
    最后把检索层自己的降级（例如向量路不可用、已退化成纯 BM25）带出来。这些调用方
    必须知道，但**都不改变 HTTP 状态码**：200 是"服务正常回答了它能回答的部分"，
    不是"一切完美"。
    """
    hits = list(ans.retrieval.results) if ans.retrieval is not None else []
    notes: list[str] = []
    if not hits:
        notes.append("检索结果为空，没有可依据的片段，因此不调用大模型")
    if not has_api_key(runtime.settings.llm_model):
        notes.append("未配置 LLM 密钥，仅返回检索结果")
    elif not ans.text:
        notes.append(ans.refusal_reason or "未能生成回答，仅返回检索结果")
    else:
        notes.extend(ans.degraded)  # 生成成功时才需要：引用越界被剥除之类的问题
    if ans.retrieval is not None:
        retrieval_notes = ans.retrieval.diagnostics.get("degraded")
        if isinstance(retrieval_notes, list):
            notes.extend(str(note) for note in retrieval_notes)
    return AskResponse(
        answer=ans.text,
        citations=[CitationOut.model_validate(c) for c in ans.citations],
        n_hits=len(hits),
        degraded="；".join(notes),
    )


# ===========================================================================
# clinic agent（/chat 用）
# ===========================================================================
@lru_cache(maxsize=1)
def _clinic_links() -> dict[str, tuple[str, ...]]:
    """症状 -> 候选关联表（8MB 的卡片语料只解析一次）。

    语料缺失时返回空表而不是报错：clinic 的规则轨（红线 / 症状词典 / 挂号）不依赖它。
    """
    from kubedocs.domains.clinic.pack import build_links_from_cards, clinic_cards_path

    links = build_links_from_cards(clinic_cards_path())
    return {k: tuple(v) for k, v in links.items()}


def _make_checkpointer(settings: Settings) -> object | None:
    """按配置造图的持久化 checkpointer；未配置时返回 None（内存态）。

    配了 `KUBEDOCS_CLINIC_STATE_DB` 时把图状态落 SQLite，使追问到一半的会话能跨请求、
    跨重启续跑（`build_clinic_graph` 的 `checkpointer` 参数即为此而留）。

    `check_same_thread=False` 是必需的：`/chat` 跑在 `asyncio.to_thread` 的线程池里，
    连接会跨线程使用。
    """
    path = str(getattr(settings, "clinic_state_db", "") or "").strip()
    if not path:
        return None
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    logger.info("clinic 图状态落盘：%s", target)
    return SqliteSaver(conn)


def _get_clinic_agent(app: FastAPI) -> ClinicAgent:
    """惰性构造 clinic agent 并缓存到 `app.state`。

    必须缓存：agent 持有 `MockAppointmentProvider` 与图的 checkpointer，每请求新建会
    丢掉上一轮的追问状态，`thread_id` 续跑静默失效。必须惰性：构造要编译 LangGraph 图
    （秒级），不碰 `/chat` 的请求不该付这个代价。必须持锁：`/chat` 跑在线程池里，
    两个并发的首次请求各自建一个 agent 的话，只有一个能留下。
    """
    agent: ClinicAgent | None = getattr(app.state, "clinic_agent", None)
    if agent is not None:
        return agent
    with app.state.agent_lock:
        agent = getattr(app.state, "clinic_agent", None)  # 双检：等锁期间可能有人建好了
        if agent is None:
            from kubedocs.agent.kernel import ClinicAgent as _ClinicAgent
            from kubedocs.domains.clinic.mock import MockAppointmentProvider
            from kubedocs.domains.clinic.pack import build_clinic_deps

            wiring = build_clinic_deps(MockAppointmentProvider(), links=_clinic_links())
            checkpointer = _make_checkpointer(app.state.settings)
            if checkpointer is not None:
                # 记下来给 lifespan 关连接用（SQLite 连接不关会在 Windows 上锁住文件）
                app.state.clinic_checkpointer = checkpointer
            agent = _ClinicAgent(wiring.deps, checkpointer=checkpointer)  # type: ignore[arg-type]
            app.state.clinic_agent = agent
        return agent


def _sse(event: str, payload: Mapping[str, Any]) -> ServerSentEvent:
    """构造一个 SSE 事件：事件名 + 单行 JSON data（格式理由见模块文档）。"""
    return ServerSentEvent(event=event, data=json.dumps(dict(payload), ensure_ascii=False))


def _tool_events(state: ClinicState) -> list[tuple[str, dict[str, Any]]]:
    """把内核**真的执行过**的步骤映射成 `tool` 事件。

    clinic 内核的节点是确定性闭包（`book` / 红线检查），它并不经过 `ToolRegistry` 的
    function calling，所以只能按状态的痕迹发事件：宁可一个都不发，也不为了"看起来像
    agent"发假事件。
    """
    events: list[tuple[str, dict[str, Any]]] = []
    red = state.get("red_line")
    if red:
        events.append(("check_red_line", {"word": red.get("word", ""), "status": red.get("status", "")}))
    ticket = state.get("ticket")
    if ticket:
        events.append(("book_appointment", {"ticket": ticket}))
    return events


class _ChatOutcome:
    """一轮 `/chat` 的结果：要发的 tool 事件 + 最终文本 + 是否又停在追问上。"""

    def __init__(self, state: ClinicState, *, resuming: bool, pending: str | None = None) -> None:
        self.state = state
        self.resuming = resuming
        # 这一轮结束后**仍然挂着**的追问问题（内核在等用户回答）；None 表示流程已结束
        self.pending = pending
        self.tools = _tool_events(state)
        answer = state.get("answer") or {}
        self.text = str(answer.get("text", "")) if isinstance(answer, dict) else ""
        self.citations = list(answer.get("citations", [])) if isinstance(answer, dict) else []
        self.status = str(state.get("status", ""))
        # 图路径不走工具循环，没有"算不出价"这个概念（它的 LLM 调用在别处记账）
        self.unpriced_calls = 0

    @classmethod
    def from_clinic_turn(cls, turn: Any) -> _ChatOutcome:
        """从**工具循环**的一轮结果构造。

        工具循环是无状态的（不像图有 checkpointer），多轮上下文由本文件用
        `app.state.clinic_threads` 按 thread_id 维护——这是与图路径的**真实差异**：
        图路径重启进程后能续聊，工具循环路径不能。这份 history 是**真的会进模型**的
        （`_converse_with_tools` 交给 `ClinicReactRuntime.turn`，再由它传给
        `run_tool_loop(history=...)`）。
        """
        self = cls.__new__(cls)
        self.state = {}  # 工具循环无图状态，留空字典占位
        self.resuming = False
        self.pending = None
        # 事件形状与 SSE 的 `{"tool": name, **payload}` 对齐
        self.tools = [
            (str(tc.get("tool", "")), {k: v for k, v in tc.items() if k != "tool"}) for tc in turn.tool_calls
        ]
        self.text = turn.text
        self.citations = list(getattr(turn, "citations", []) or [])
        self.status = "emergency" if turn.emergency else "done"
        # "算不出价"的调用次数透传到 SSE：让未知成本在**响应里**也看得见
        self.unpriced_calls = int(getattr(turn, "unpriced_calls", 0) or 0)
        return self


def _get_clinic_react(app: FastAPI) -> Any:
    """装配 LLM 驱动的 clinic 运行期（幂等缓存）。

    与 `_get_clinic_agent` 并存：**有 LLM 密钥时优先走这条**（控制流由模型决定），
    没有密钥时回退到确定性图（红线与挂号仍可用，只是不会主动追问）。
    """
    rt = getattr(app.state, "clinic_react", None)
    if rt is not None:
        return rt
    from kubedocs.agent.clinic_react import ClinicReactRuntime
    from kubedocs.domains.clinic.booking import BookingService
    from kubedocs.domains.clinic.mock import MockAppointmentProvider

    settings = app.state.settings
    rt = ClinicReactRuntime(settings, BookingService(MockAppointmentProvider()))
    app.state.clinic_react = rt
    return rt


def _converse_with_tools(app: FastAPI, request: ChatRequest, thread_id: str) -> _ChatOutcome:
    """走工具循环：**LLM 通过 function calling 自己决定调哪个工具、调几次**。

    历史存在 `app.state.clinic_threads`（属主 + LRU 上界，见 `api/threads.py`）：
    属主校验在**端点**里已经做过一次，这里的 `history/append` 是第二道——两次都必须过，
    因为这一层才是真正读写别人历史的地方。
    """
    rt = _get_clinic_react(app)
    store: ThreadStore = app.state.clinic_threads
    past = store.history(thread_id, request.patient_id)
    # `patient_id` 必须一路传到工具执行体：号源记的是"谁在调用"。
    # 只传到这一层为止的话，`book_appointment` 读不到它，号会记在 "anon" 名下，
    # 票据也无法按属主查回。
    turn = rt.turn(request.message, history=past, patient_id=request.patient_id)
    # 历史上界由 ThreadStore 按 max_messages 执行（默认最近 8 条 = 4 轮）
    store.append(
        thread_id,
        request.patient_id,
        [{"role": "user", "content": request.message}, {"role": "assistant", "content": turn.text}],
    )
    return _ChatOutcome.from_clinic_turn(turn)


def _run_turn(app: FastAPI, request: ChatRequest, thread_id: str) -> _ChatOutcome:
    """同步跑一轮问诊（在线程池里调用，别挡住事件循环）。

    **两条路径的取舍**：有 LLM 密钥 → 走 `run_tool_loop`，控制流交给模型，它会自己
    决定"先查号源再回答"还是"先追问"，工具调用是模型选的而不是代码排的；无密钥 →
    回退到确定性图（`ClinicAgent`），红线与挂号照常可用，只是不会主动追问。
    **宁可降级也不报错。**

    红线在两条路径里都在**进循环之前**用确定性规则判定（见 clinic_react），所以这条
    安全不变量不因走哪条路径而改变。
    """
    react = None
    try:
        react = _get_clinic_react(app)
    except Exception:  # noqa: BLE001 - 装配失败就退图，不能因此让 /chat 挂掉
        logger.exception("装配工具循环失败，回退到确定性图")
    with span("chat.turn", thread_id=thread_id, llm=bool(react is not None and react.has_llm())):
        if react is not None and react.has_llm():
            return _converse_with_tools(app, request, thread_id)

        agent = _get_clinic_agent(app)
        resuming = agent.pending_question(thread_id=thread_id) is not None
        if resuming:
            state = agent.resume(request.message, thread_id=thread_id)
        else:
            state = agent.start(request.message, thread_id=thread_id, patient_id=request.patient_id)
        # 跑完再看一次：这一轮可能又停在一个新追问上
        return _ChatOutcome(state, resuming=resuming, pending=agent.pending_question(thread_id=thread_id))


async def _chat_events(app: FastAPI, request: ChatRequest, thread_id: str) -> AsyncIterator[ServerSentEvent]:
    """`/chat` 的事件序列（见模块文档里的格式说明）。"""
    # 首个事件就把会话 id 交出去：调用方不必先知道 id 才能续聊
    yield _sse("thread", {"thread_id": thread_id, "domain": "clinic"})
    try:
        # 同步 agent 挪到线程池：thread 事件已经在网络上了，事件循环也不会被阻塞
        outcome = await asyncio.to_thread(_run_turn, app, request, thread_id)
    except Exception as e:  # noqa: BLE001 - 响应头已发出，改不了状态码，只能把失败当事件发
        logger.exception("chat 轮次失败：thread_id=%s", thread_id)
        yield _sse("error", {"thread_id": thread_id, "error": f"{type(e).__name__}: {e}"})
        yield _sse("done", {"thread_id": thread_id, "ok": False})
        return

    for name, payload in outcome.tools:
        yield _sse("tool", {"thread_id": thread_id, "tool": name, **payload})

    pending = outcome.pending
    text = pending or outcome.text
    if not text:
        # 兜底：图里既没有答案也没有追问时，宁可给一句人话，也不要让前端拿到空串
        text = "这一轮没有产生可回复的内容；请换一种说法，或说明具体症状。"
    yield _sse(
        "message",
        {
            "thread_id": thread_id,
            "text": text,
            "citations": outcome.citations,
            "awaiting_user": bool(pending),
            "status": "asking" if pending else outcome.status,
            "resumed": outcome.resuming,
            # 0 是常态；>0 表示有调用的价算不出来（已按保守上限记账，**不是免费**）
            "unpriced_llm_calls": outcome.unpriced_calls,
        },
    )
    yield _sse("done", {"thread_id": thread_id, "ok": True})


# ===========================================================================
# 首页
# ===========================================================================
def _fallback_html(reason: str) -> str:
    """没有前端产物时的说明页。

    返回 **200** 而不是 404：404 会被浏览器与探针当成"服务没起来"，而实际情况是
    "服务起来了，只是没构建前端"。这两件事必须能区分。
    """
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>kubedocs</title></head><body style='font-family:system-ui;max-width:44rem'>"
        "<h1>kubedocs 服务已启动</h1>"
        f"<p>未找到前端产物（{reason}），但服务本身可用。"
        "本页是说明页，不是错误页——缺少前端不等于服务故障。</p>"
        "<ul>"
        "<li><code>GET /healthz</code> 存活与索引规模</li>"
        "<li><code>GET /domains</code> 可用域</li>"
        "<li><code>GET /tools</code> 已注册工具</li>"
        "<li><code>POST /search</code> 混合检索</li>"
        "<li><code>POST /ask</code> 检索 + 生成（无 LLM 密钥时降级返回检索结果）</li>"
        "<li><code>POST /chat</code> clinic 问诊 / 挂号（SSE 流）</li>"
        "</ul>"
        "</body></html>"
    )


# ===========================================================================
# 工具注册
# ===========================================================================
def _register_tools() -> None:
    """把域工具装配进 `ToolRegistry`，让 `/tools` 列的与运行期用的是同一批。

    只装配 clinic 的号源三件套 + `check_citations`：

    - clinic 需要注入 `BookingService`，而 API 的对话域固定走进程内号源替身
      （`MockAppointmentProvider`，见 `/chat`），所以这里注入同一个替身；
    - `search_docs` **不在这里注册**：注册它必须在启动时传入一个真检索器，而那会强制
      加载嵌入模型（bge 首次几十秒），与"资源惰性构造"冲突——检索能力由 `/search`
      提供，功能没有缺失。clinic 域内的 `search_docs` 由 `consult.py` 在检索器就绪后注册。
      另外 `ToolRegistry` 按**工具名**索引，注册表不支持同名多域，两个域都注册会直接撞名。

    `register_domain_tools` 内部先按域清理再注册，反复调用（`uvicorn --reload`）安全。
    """
    from kubedocs.agent.bootstrap import register_domain_tools
    from kubedocs.domains.clinic.booking import BookingService
    from kubedocs.domains.clinic.mock import MockAppointmentProvider

    register_domain_tools("clinic", service=BookingService(MockAppointmentProvider()))


# ===========================================================================
# 应用工厂
# ===========================================================================
def create_app(settings: Settings | None = None) -> FastAPI:
    """构造 FastAPI 应用。

    `settings` 传空时用 `get_settings()`（进程内单例，读 `.env` 与环境变量）。
    资源在这里建好并挂到 `app.state`：**同一个 Settings 只打开一次 store**。
    """
    # 把 .env 导出到 os.environ：LiteLLM 与 has_api_key() 读的都是 os.environ，
    # 而 pydantic-settings 只把 .env 填进 Settings 对象。不导出的话，即使 .env 里有
    # 密钥，/chat 也会永远走"无 LLM"的降级分支。
    from kubedocs.config import load_env_file

    load_env_file()
    cfg = settings or get_settings()
    load_all()  # 域包注册：/domains 与 domain 校验都依赖它

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """lifespan 只做一件事：关掉所有 store。

        为什么显式关：SQLite 连接不关会在 Windows 上锁住索引目录（删不掉、换不了），
        `--reload` 反复重启还会累积连接。
        资源**不在这里创建**：`TestClient(app)`（不进入 lifespan）也必须能直接用。
        """
        try:
            yield
        finally:
            for runtime in app.state.runtimes.values():
                runtime.close()
            # 图状态的 SQLite 连接同样要关（理由同上：Windows 上会锁住文件）。
            checkpointer = getattr(app.state, "clinic_checkpointer", None)
            if checkpointer is not None:
                conn = getattr(checkpointer, "conn", None)
                if conn is not None:
                    conn.close()

    app = FastAPI(title="kubedocs", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg
    # 两个"设了要有反应"的开关在装配点生效（见 observability.py 的说明）：
    # 日志级别真正设到 root logger，OTel 真正建 provider。
    configure_logging(cfg)
    setup_tracing(cfg)
    # 预算上限与"有没有调用过 LLM"无关，必须在装配点就发出去；否则控制台会把一个
    # 真实存在的上限读成 gauge 默认值 0，而 0 的约定是"不限"（假信息）。
    publish_config_gauges(cfg)
    app.state.index_html = repo_root().joinpath(*INDEX_HTML)
    app.state.console_html = repo_root().joinpath(*CONSOLE_HTML)
    app.state.patient_html = repo_root().joinpath(*PATIENT_HTML)
    app.state.consult_html = repo_root().joinpath(*CONSULT_HTML)
    app.state.runtimes_lock = threading.Lock()
    app.state.agent_lock = threading.Lock()
    app.state.clinic_agent = None
    app.state.clinic_checkpointer = None
    """图状态的持久化 checkpointer（仅当 `KUBEDOCS_CLINIC_STATE_DB` 配了路径时才非 None）。"""
    # /chat 的会话表：属主绑定 + LRU 上界。**不放在 `_get_clinic_react` 里**——
    # 那个函数只在有 LLM 密钥、且真的走到工具循环时才被调用，而属主校验必须在端点里、
    # 对两条路径都生效，所以它必须是 create_app 的常驻状态。
    app.state.clinic_threads = ThreadStore()
    current = _DomainRuntime(cfg)
    app.state.runtimes = {cfg.domain: current}
    app.state.store = current.store  # 约定：当前域的 store 挂在 app.state.store
    _register_tools()

    # 交互式问诊（`/consult`）：装配 `ClinicalPipeline`，**复用 clinic 域的同一份
    # store / embedder**（不重复加载模型）。运行时对象惰性构造，见 consult.py。
    app.include_router(
        build_consult_router(
            runtime_provider=lambda req: _runtime_for_app(req.app, "clinic"),
            page_path=app.state.consult_html,
        )
    )

    @app.middleware("http")
    async def _observe_requests(request: Request, call_next: Any) -> Any:
        """按端点统计请求数与耗时。

        端点名取**路由模板**（`request.scope["route"].path`）而不是原始 URL：用 URL 会把
        `/search?q=...` 的每个不同查询变成一个新的时间序列标签，指标基数随流量无界增长。
        `/metrics` 自身不计数，否则抓取得越勤、它自己的曲线越陡。
        """
        started = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", None) or request.url.path
        if route != "/metrics":
            HTTP_REQUESTS.labels(endpoint=route, status=str(response.status_code)).inc()
            HTTP_LATENCY.labels(endpoint=route).observe(time.perf_counter() - started)
        return response

    # ---------------------------------------------------------------- 指标
    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus 文本格式指标。`otel_enable` 与它无关：指标始终可用。"""
        payload, content_type = metrics_payload()
        return Response(content=payload, media_type=content_type)

    # ---------------------------------------------------------------- 探针与自省
    @app.get("/healthz")
    def healthz(request: Request) -> HealthResponse:
        """存活探针：**索引为空 / 索引不可用都不算失败**（理由见模块文档）。"""
        runtime = _runtime(request)
        return HealthResponse(ok=True, domain=runtime.settings.domain, n_chunks=runtime.n_chunks())

    @app.get("/domains")
    def domains() -> list[dict[str, Any]]:
        """列出可用域。域包与索引是两件事：没有索引照样能列。"""
        return [dict(item) for item in DomainRegistry.describe()]

    @app.get("/tools")
    def tools(domain: str = "") -> ToolsResponse:
        """列出已注册的工具；`?domain=clinic` 只列该域的。"""
        _require_domain(domain)
        if not domain:
            return ToolsResponse(tools=ToolRegistry.describe())
        return ToolsResponse(tools=[spec.describe() for spec in ToolRegistry.for_domain(domain)])

    # ---------------------------------------------------------------- 检索与回答
    @app.post("/search")
    def search(request: Request, body: SearchRequest) -> SearchResponse:
        """混合检索。索引为空时返回 200 + 空 hits + 一句说明。"""
        _require_domain(body.domain)
        runtime = _runtime(request, body.domain)
        retriever = runtime.retriever()
        if retriever is None:
            return SearchResponse(
                query=body.q,
                hits=[],
                diagnostics={"degraded": [f"索引不可用：{runtime.index_error}"]},
                note=_index_note(runtime, 0),
            )
        with span("http.search", domain=runtime.settings.domain, top_k=body.top_k):
            result = retriever.search(body.q, top_k=body.top_k)
        hits = [Hit.from_chunk(chunk) for chunk in result.results]
        return SearchResponse(
            query=result.query,
            hits=hits,
            diagnostics=result.diagnostics,
            note=_index_note(runtime, len(hits)),
        )

    @app.post("/ask")
    def ask(request: Request, body: AskRequest) -> AskResponse:
        """检索 + 生成（LiteLLM）。

        没有 LLM 密钥**不是 500**：检索结果与引用照常返回，`degraded` 里写明原因。
        这条路径让没有密钥的环境（离线机器）也能完整走通整条链路。
        """
        _require_domain(body.domain)
        runtime = _runtime(request, body.domain)
        answerer = runtime.answerer()
        if answerer is None:
            return AskResponse(
                answer="",
                citations=[],
                n_hits=0,
                degraded=f"索引不可用，仅返回检索结果：{runtime.index_error}",
            )
        return _ask_response(answerer.answer(body.q, top_k=body.top_k), runtime)

    # ---------------------------------------------------------------- 对话（SSE）
    @app.post("/chat")
    async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
        """clinic 问诊 / 挂号，SSE 流式返回（域固定为 clinic）。

        `thread_id` 留空时由服务端生成，并在**首个事件** `thread` 里返回；
        带同一个 `thread_id` 再来一次即续聊（内核用 `interrupt` 挂起的追问会被 `resume`）。
        `patient_id` 由调用方在请求体里注入（默认 `"anon"`），模型不能提供或改写它——
        号源记账与属主绑定都用这个值。

        **属主校验在这里做**（不是流里面）：`thread_id` 首次出现即绑定 `patient_id`，
        换一个 `patient_id` 用同一个 `thread_id` → **409**。放在这里是因为响应头还没发出，
        能给出真正的 4xx 而不是流里的一条 error 事件，而且它同时覆盖工具循环与确定性图
        两条路径。
        """
        thread_id = body.thread_id or uuid.uuid4().hex
        store: ThreadStore = request.app.state.clinic_threads
        try:
            store.claim(thread_id, body.patient_id)
        except ThreadOwnershipError as e:
            logger.warning("拒绝跨属主续聊：thread_id=%s patient_id=%s", thread_id, body.patient_id)
            raise HTTPException(status_code=409, detail=str(e)) from e
        return EventSourceResponse(_chat_events(request.app, body, thread_id))

    # ---------------------------------------------------------------- 前端页面
    def _serve_page(path: Path) -> HTMLResponse:
        """返回静态页；没有该文件就返回说明页（**200，不是 404**）。"""
        if path.is_file():
            try:
                return HTMLResponse(path.read_text(encoding="utf-8"))
            except OSError as e:
                logger.warning("读取页面 %s 失败：%s", path, e)
                return HTMLResponse(_fallback_html(f"读取 {path} 失败：{e}"))
        return HTMLResponse(_fallback_html(f"未找到 {path}"))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        """演示页：检索 / 工具清单 / 问诊的最小示例。"""
        return _serve_page(request.app.state.index_html)

    @app.get("/console", response_class=HTMLResponse)
    def console(request: Request) -> HTMLResponse:
        """控制台页：索引体检 + 指标 + 检索诊断 + 问诊。

        必须由同一个 app 提供（而不是另起静态服务器）：页面直接对同源发
        `/search`、`/chat` 等请求，而 API 没有开 CORS，换源就会全线失败。
        """
        return _serve_page(request.app.state.console_html)

    @app.get("/patient", response_class=HTMLResponse)
    def patient(request: Request) -> HTMLResponse:
        """患者端：**只**保留问诊 agent，面向真实使用者。

        三个页面的分工：`/patient` 给患者，
        只有对话、急诊提示、追问与依据；`/` 给开发者（检索 / 工具清单 / 问诊的最小示例）；
        `/console` 给运维（索引体检、`/metrics`、每路检索名次、降级原因）。
        `/` 与 `/console` 上的输入项（`API 地址`、`域`、`top_k`）对患者是额外的认知负担，
        而且**能改到不该改的东西**，所以患者端一律**同源**、不暴露任何可调参数。
        """
        return _serve_page(request.app.state.patient_html)

    return app


app = create_app()
"""模块级实例：`uvicorn kubedocs.api.app:app`。

`create_app()` 只打开索引、不加载嵌入模型，所以这一步是廉价的；换来的是"配置错误
在启动时就报，而不是在第一个请求时炸"。
"""
