"""API 层测试：路由形状、降级路径、SSE 事件格式。

## 三条硬约束

1. **绝不联网、绝不加载真实模型**：索引一律建在 `tmp_path`，
   `embedding_provider="hashing"`（conftest 的 autouse 夹具还会清掉所有供应商密钥）；
2. **不依赖磁盘语料**：需要数据时直接构造 store 灌块——这里要验证的是 HTTP 层的
   协议转换，不是切块与摄取；
3. **空索引是重点用例**：`/healthz` `/domains` `/tools` `/search` `/ask` `/chat`
   在**没有任何索引**时都必须能返回，而且不能是 500。这条"部署中间态"是最容易
   被写崩的地方（例如 `cli.open_store(require_data=True)` 那种前置检查直接抛错）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kubedocs.api.app import create_app
from kubedocs.api.threads import ThreadOwnershipError
from kubedocs.config import Settings
from kubedocs.embed import HashingEmbedder
from kubedocs.generate import Answerer, LLMResult
from kubedocs.store import from_settings
from kubedocs.store.base import ChunkInput, DocInput

DIM = 32
"""与 `Settings.embedding_dim` 一致；hashing 维度只影响向量空间大小，不影响管道正确性。"""

LONG_BODY = "kubelet eviction 硬阈值说明，这一段刻意写得很长用来验证截断。" * 40
"""明显超过 600 字符的正文，用于验证 `/search` 的截断行为。"""

STARTUP_PAGE = "<h1>真实首页</h1>"


class _StubLLM:
    """打桩 LLM：返回一句带合法引用标记的答案，不碰任何网络。"""

    def complete(self, system: str, user: str) -> LLMResult:
        return LLMResult(text="打桩答案 [1]", prompt_tokens=1, completion_tokens=1)


# --------------------------------------------------------------------------- 辅助
def _seed_index(settings: Settings, bodies: list[str], *, title: str = "Pod 驱逐") -> None:
    """直接构造 store 灌入 1~2 个块（等价于 ingest 的后半段）。

    为什么不用 CLI 的 ingest：那需要磁盘上的 markdown 语料，而本文件要验证的是
    HTTP 层——索引里有块就够了。写完就 `close()`，随后 `create_app` 会自己再开一次
    （这也顺带验证了"索引文件是持久的、不靠进程内状态"）。
    """
    store = from_settings(settings)
    try:
        docs: list[DocInput] = []
        chunks: list[ChunkInput] = []
        for i, body in enumerate(bodies):
            doc_id = f"doc{i}"
            url = f"https://kubernetes.io/docs/doc{i}/"
            digest = f"h-{i}"
            docs.append(
                DocInput(
                    doc_id=doc_id, rel_path=f"{doc_id}.md", title=title, source_url=url, content_hash=digest
                )
            )
            chunks.append(
                ChunkInput(
                    chunk_id=f"{doc_id}::0000",
                    doc_id=doc_id,
                    ordinal=0,
                    title=title,
                    breadcrumb="配置",
                    source_url=url,
                    body=body,
                    content_hash=digest,
                    n_chars=len(body),
                )
            )
        matrix = HashingEmbedder(dim=settings.embedding_dim).encode([c.body for c in chunks])
        store.upsert_docs(docs, chunks, {c.chunk_id: matrix[i] for i, c in enumerate(chunks)})
        store.finalize()
    finally:
        store.close()


def _sse_events(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 `(event, data)` 列表（`\\r\\n` 统一成 `\\n` 再切块）。"""
    events: list[tuple[str, dict]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = "", {}
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if name:
            events.append((name, data))
    return events


def _event(events: list[tuple[str, dict]], name: str) -> dict:
    """取第一个名为 `name` 的事件数据；没有就断言失败（而不是静默返回空 dict）。"""
    for got, data in events:
        if got == name:
            return data
    raise AssertionError(f"没有 {name!r} 事件；实际收到：{[n for n, _ in events]}")


# --------------------------------------------------------------------------- 夹具
@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    """索引指向 tmp_path、嵌入用 hashing：**绝不联网、绝不加载真实模型**。"""
    return Settings(
        index_dir=str(tmp_path / "index"),
        domain="k8s_docs",
        embedding_provider="hashing",
        embedding_dim=DIM,
        embedding_model="hashing-test",
        rerank_enable=False,
    )


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    """空索引的客户端。

    用 `with` 进入 lifespan：测试结束会关掉 SQLite 连接，
    否则 Windows 上 tmp_path 会因为文件被占用而删不掉（表现为 teardown 报错）。
    """
    with TestClient(create_app(api_settings)) as c:
        yield c


# ===========================================================================
# 没有索引时的探针与自省
# ===========================================================================
class TestWithoutIndex:
    def test_healthz_returns_200_even_without_index(self, client: TestClient) -> None:
        """没有索引时 /healthz 仍然 200：判活探针关心"进程活着"，不关心有没有语料。"""
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "domain": "k8s_docs", "n_chunks": 0}

    def test_domains_returns_200_even_without_index(self, client: TestClient) -> None:
        """域包与索引是两件事：索引为空照样要能列出可用域。"""
        resp = client.get("/domains")
        assert resp.status_code == 200
        names = {item["name"] for item in resp.json()}
        assert {"k8s_docs", "clinic"} <= names

    def test_tools_returns_200_even_without_index(self, client: TestClient) -> None:
        """工具注册与索引无关：空索引下 /tools 也要能自省出工具清单。"""
        resp = client.get("/tools")
        assert resp.status_code == 200
        assert "book_appointment" in {t["name"] for t in resp.json()["tools"]}

    def test_tools_domain_filter_and_unknown_domain(self, client: TestClient) -> None:
        """`?domain=` 只列该域的工具；未知域 400 且把可用域列出来。"""
        all_tools = client.get("/tools").json()["tools"]
        filtered = client.get("/tools", params={"domain": "clinic"}).json()["tools"]
        assert "book_appointment" in {t["name"] for t in filtered}, filtered
        # 按域过滤 = 只保留声明为 clinic 的工具（比对全量清单自己算出的期望值，
        # 这样断言不依赖"哪个工具被归属到哪个域"的实现细节）
        assert filtered == [t for t in all_tools if t["domain"] == "clinic"]

        bad = client.get("/tools", params={"domain": "no_such_domain"})
        assert bad.status_code == 400
        detail = bad.json()["detail"]
        assert "k8s_docs" in detail and "clinic" in detail, detail


# ===========================================================================
# /search
# ===========================================================================
class TestSearch:
    def test_empty_index_returns_200_with_empty_hits(self, client: TestClient) -> None:
        """空索引 → 200 + 空 hits + 一句说明（绝不是 500）。"""
        resp = client.post("/search", json={"q": "pod"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["hits"] == []
        assert body["note"], "空索引必须给出说明，别让调用方自己去猜"
        assert "diagnostics" in body

    def test_hits_carry_source_url_and_truncated_body(self, api_settings: Settings) -> None:
        """有数据时 hits 里有 source_url，且 body 被截断（引用能追溯到原文）。"""
        assert len(LONG_BODY) > 600, "测试数据本身要先超过截断阈值"
        _seed_index(api_settings, [LONG_BODY])
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/search", json={"q": "kubelet", "top_k": 5})

        assert resp.status_code == 200
        hits = resp.json()["hits"]
        assert hits, resp.json()
        hit = hits[0]
        assert hit["source_url"].startswith("https://")
        assert len(hit["body"]) == 600, "截断后总长（含省略号）必须正好等于上限"
        assert hit["body"].endswith("…"), "截断必须留痕迹，否则读者以为正文就这样结束了"
        assert hit["chunk_id"] and hit["title"] and hit["breadcrumb"]

    def test_unknown_domain_is_400_and_lists_available(self, client: TestClient) -> None:
        """未知域 → 400 并列出可用域（静默回退到默认域会让人拿到错的域的结果）。"""
        resp = client.post("/search", json={"q": "pod", "domain": "clinic_typo"})
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "k8s_docs" in detail and "clinic" in detail, detail

    @pytest.mark.parametrize("path", ["/search", "/ask"])
    @pytest.mark.parametrize("top_k", [0, 21])
    def test_top_k_out_of_range_is_422(self, client: TestClient, path: str, top_k: int) -> None:
        """top_k 越界（0 或 21）→ 422：范围由 `Field(ge=1, le=20)` 定，不靠下游兜底。"""
        resp = client.post(path, json={"q": "pod", "top_k": top_k})
        assert resp.status_code == 422

    def test_blank_query_is_422(self, client: TestClient) -> None:
        """只有空白的 q → 422：它注定无结果，却会白跑一次嵌入与全文检索。"""
        assert client.post("/search", json={"q": "   "}).status_code == 422

    def test_too_long_query_is_422(self, client: TestClient) -> None:
        """q 超过 300 字符 → 422（外部输入必须有长度上限）。"""
        assert client.post("/search", json={"q": "k" * 301}).status_code == 422


# ===========================================================================
# /ask
# ===========================================================================
class TestAsk:
    def test_without_llm_key_degrades_instead_of_500(self, api_settings: Settings) -> None:
        """没有 LLM 密钥 → 200 + degraded 说明 + 引用照常返回（**不是** 500）。

        conftest 的 autouse 夹具清掉了所有供应商密钥，所以这条分支是确定性的。
        """
        _seed_index(api_settings, [LONG_BODY])
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/ask", json={"q": "kubelet"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == ""
        assert body["degraded"] == "未配置 LLM 密钥，仅返回检索结果"
        assert body["n_hits"] >= 1
        assert body["citations"], "回答降级不等于引用也降级：来源必须照常返回"
        assert body["citations"][0]["source_url"].startswith("https://")

    def test_empty_index_returns_200_with_explanation(self, client: TestClient) -> None:
        """空索引下 /ask 也返回 200 + 说明：没有语料就明说没有，不许编造。"""
        resp = client.post("/ask", json={"q": "pod"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == ""
        assert body["n_hits"] == 0
        assert body["degraded"]

    def test_answer_and_citations_are_passed_through(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "有可用 LLM"时 /ask 把生成层的答案与引用原样透出（用打桩 LLM，绝不联网）。

        打桩打在**装配点** `build_answerer` 而不是 litellm 内部：只要有一处没打中，
        真调用就会发到供应商，而测试不允许存在这种可能。这一条覆盖的是
        "检索 -> 生成 -> 响应映射"这段胶水，生成层自身由 `test_generate.py` 覆盖。
        """
        _seed_index(api_settings, [LONG_BODY])
        monkeypatch.setattr("kubedocs.api.app.has_api_key", lambda _model: True)
        monkeypatch.setattr(
            "kubedocs.api.app.build_answerer",
            lambda retriever, settings: Answerer(retriever, _StubLLM(), settings),
        )
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/ask", json={"q": "kubelet"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "打桩答案 [1]"
        assert body["degraded"] == "", "生成成功且检索无降级时不该有降级说明"
        assert [c["index"] for c in body["citations"]] == [1]
        assert body["n_hits"] >= 1


# ===========================================================================
# /chat（SSE）
# ===========================================================================
class TestChat:
    def test_returns_sse_stream_with_expected_events(self, client: TestClient) -> None:
        """/chat 返回 SSE：content-type 是 text/event-stream，事件序列是 thread→message→done。"""
        resp = client.post("/chat", json={"message": "我最近有点咳嗽"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "event:" in resp.text and "data:" in resp.text

        events = _sse_events(resp.text)
        names = [name for name, _ in events]
        assert names[0] == "thread", names
        assert names[-1] == "done", names
        assert events[0][1]["thread_id"], "首个事件必须把 thread_id 交出去"
        text = _event(events, "message")["text"]
        assert text, "最终文本不能是空串"

    def test_thread_id_is_respected_and_conversation_continues(self, client: TestClient) -> None:
        """带同一个 thread_id 再来一次 = 续聊：内核的 interrupt 挂起点会被 resume。"""
        first = client.post("/chat", json={"message": "我最近咳嗽", "thread_id": "t-api-1"})
        assert _event(_sse_events(first.text), "thread")["thread_id"] == "t-api-1"

        second = client.post("/chat", json={"message": "帮我挂号", "thread_id": "t-api-1"})
        assert second.status_code == 200
        message = _event(_sse_events(second.text), "message")
        assert message["thread_id"] == "t-api-1"
        assert message["text"]

    def test_chat_works_without_index_or_corpus(self, api_settings: Settings) -> None:
        """无索引/无语料也要能返回：clinic 的规则轨（红线）不依赖索引，这是设计目标。

        「胸痛」是急诊红线词，所以这一轮必定走 emergency 分支并给出急诊建议；
        同时它以 `tool` 事件的形式可见——即"agent 真的做了这件事"，而不是只在文本里提一句。
        """
        settings = api_settings.model_copy(update={"domain": "clinic"})
        with TestClient(create_app(settings)) as client:
            resp = client.post("/chat", json={"message": "我现在胸痛", "patient_id": "p1"})

        assert resp.status_code == 200
        events = _sse_events(resp.text)
        tools = [data for name, data in events if name == "tool"]
        assert any(t["tool"] == "check_red_line" for t in tools), tools
        assert "120" in _event(events, "message")["text"]

    def test_too_long_message_is_422(self, client: TestClient) -> None:
        """message 超过 2000 字符 → 422（单轮对话文本的上限在 schema 里）。"""
        assert client.post("/chat", json={"message": "咳" * 2001}).status_code == 422

    def test_interrupted_question_is_streamed_and_resumed(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """被 `interrupt` 挂起的追问以 message 事件送出（`awaiting_user=true`），
        同一个 `thread_id` 再来一次即 `resumed=true`——这是"真挂起"而不是"内存里记个标志位"。

        症状关联表用打桩：真实卡片语料不入库（`data/clinic/cards.json` 是派生文件），
        靠它会让这条用例在干净克隆里变成"永远只走兜底回答"，失去验证价值。
        """
        monkeypatch.setattr(
            "kubedocs.api.app._clinic_links",
            lambda: {"咳嗽": ("支气管炎", "呼吸内科"), "发热": ("支气管炎", "感染科")},
        )
        with TestClient(create_app(api_settings)) as client:
            first = client.post("/chat", json={"message": "我咳嗽", "thread_id": "t-resume"})
            asking = _event(_sse_events(first.text), "message")
            assert asking["awaiting_user"] is True, asking
            assert asking["status"] == "asking"
            assert asking["text"], "追问的问题不能是空串"

            second = client.post("/chat", json={"message": "没有别的了", "thread_id": "t-resume"})
            resumed = _event(_sse_events(second.text), "message")
            assert resumed["resumed"] is True, "第二轮必须是从挂起点续跑，而不是开新会话"
            assert resumed["thread_id"] == "t-resume"

    def test_message_event_reports_zero_unpriced_calls(self, client: TestClient) -> None:
        """`unpriced_llm_calls` 是"未知成本可见"的契约：正常情况下必须是 0（不是缺失）。"""
        resp = client.post("/chat", json={"message": "我咳嗽"})
        assert _event(_sse_events(resp.text), "message")["unpriced_llm_calls"] == 0


# ===========================================================================
# /chat 的 thread_id 属主校验（缺陷回归）
#
# 缺陷形态：端点直接拿客户端传来的 `thread_id` 当 `app.state.clinic_history` 的键，
# 没有任何属主校验 ⇒ 猜到/拿到别人的 thread_id 就能续聊别人的会话（SECURITY.md S-3）。
# 现在的契约：thread_id 首见即绑定 patient_id；换属主 ⇒ **409**（不静默当新会话、
# 也不静默接到别人的历史上）；**一直用同一个 patient_id + thread_id 的行为完全不变**。
# ===========================================================================
class TestChatThreadOwnership:
    def test_same_patient_can_continue_its_own_thread(self, client: TestClient) -> None:
        """兼容性钉子：同一属主沿用同一个 thread_id 必须完全正常。"""
        for message in ("我最近咳嗽", "还有点发热"):
            resp = client.post("/chat", json={"message": message, "thread_id": "t-own-1", "patient_id": "p1"})
            assert resp.status_code == 200, resp.text
        assert _event(_sse_events(resp.text), "message")["thread_id"] == "t-own-1"

    def test_default_patient_id_is_also_a_stable_owner(self, client: TestClient) -> None:
        """不传 patient_id（默认 `anon`）的用户同样不受影响。"""
        for _ in range(3):
            resp = client.post("/chat", json={"message": "我咳嗽", "thread_id": "t-anon"})
            assert resp.status_code == 200, resp.text

    def test_server_generated_thread_id_is_always_accepted(self, client: TestClient) -> None:
        """留空 thread_id（服务端生成）不受属主检查影响：新 id 归本次请求者。"""
        resp = client.post("/chat", json={"message": "我咳嗽", "patient_id": "p9"})
        assert resp.status_code == 200
        assert _event(_sse_events(resp.text), "thread")["thread_id"]

    def test_another_patient_using_the_same_thread_id_is_rejected(self, client: TestClient) -> None:
        """**关键断言**：换一个 patient_id 用同一个 thread_id ⇒ 409 + 可读原因。"""
        first = client.post("/chat", json={"message": "我咳嗽", "thread_id": "t-own-2", "patient_id": "p1"})
        assert first.status_code == 200

        bad = client.post("/chat", json={"message": "继续说", "thread_id": "t-own-2", "patient_id": "p2"})

        assert bad.status_code == 409, bad.text
        detail = bad.json()["detail"]
        assert "t-own-2" in detail and "patient_id" in detail
        # 拒绝理由里**不能**回显属主是谁（否则等于帮着做身份探测）
        assert "p1" not in detail

    def test_rejected_turn_does_not_get_the_other_patients_history(self, client: TestClient) -> None:
        """被拒之后越权者拿不到任何东西：读别人的线程直接抛（而不是返回别人的历史）。"""
        client.post("/chat", json={"message": "我咳嗽", "thread_id": "t-own-3", "patient_id": "p1"})
        bad = client.post("/chat", json={"message": "继续说", "thread_id": "t-own-3", "patient_id": "p2"})
        assert bad.status_code == 409

        store = client.app.state.clinic_threads  # type: ignore[attr-defined]
        with pytest.raises(ThreadOwnershipError):
            store.history("t-own-3", "p2")
        assert store.owner_of("t-own-3") == "p1", "被拒的请求改写了属主"

    def test_thread_store_is_bounded(self, client: TestClient) -> None:
        """会话表有上界：超过上限的旧线程会被淘汰（缺陷回归：原先是无界 dict）。"""
        store = client.app.state.clinic_threads  # type: ignore[attr-defined]
        store.max_threads = 3
        for i in range(5):
            resp = client.post("/chat", json={"message": "我咳嗽", "thread_id": f"t-cap-{i}"})
            assert resp.status_code == 200
        assert len(store) == 3
        assert store.n_evicted == 2


# ===========================================================================
# /chat 的成本上限（缺陷回归）
#
# 缺陷形态：`DailyBudget` 只在 `Answerer.answer()` 里 check()，而 `/chat` 走
# `LiteLLMToolCaller`，它只累加 `self.cost`、从不与预算比较（SECURITY.md S-2）。
# 现在的契约：超限时**一次模型调用都不发生**，并把原因如实送到流上。
# ===========================================================================
class TestChatBudget:
    def _exhaust(self, settings: Settings) -> None:
        """把当日花费写成远超上限：模拟"今天已经花完了"。"""
        path = settings.llm_spend_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({time.strftime("%Y-%m-%d"): 99.0}), encoding="utf-8")

    def test_over_budget_stops_the_llm_and_is_reported_on_the_stream(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api_settings.llm_daily_budget_usd = 1.0
        self._exhaust(api_settings)
        calls: list[int] = []

        class _CountingCaller:
            def __init__(self, _settings: Any) -> None:
                pass

            def choose(self, *_a: Any, **_k: Any) -> dict[str, Any]:
                calls.append(1)
                return {"content": "答", "tool_calls": []}

        monkeypatch.setattr("kubedocs.agent.clinic_react.ClinicReactRuntime.has_llm", lambda self: True)
        monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _CountingCaller)
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/chat", json={"message": "我咳嗽", "thread_id": "t-budget"})

        assert resp.status_code == 200  # 响应头早就发出去了，只能把失败当事件发
        err = _event(_sse_events(resp.text), "error")
        assert "BudgetExceeded" in err["error"] and "上限" in err["error"]
        assert calls == [], "超预算后仍然调用了模型：账单没被挡住"


# ===========================================================================
# 首页
# ===========================================================================
class TestIndexPage:
    def test_missing_html_returns_200_explanation(self, api_settings: Settings, tmp_path: Path) -> None:
        """`data/web/index.html` 不存在时返回 200 说明页（404 会被当成"服务没起来"）。"""
        app = create_app(api_settings)
        app.state.index_html = tmp_path / "no-such-index.html"
        with TestClient(app) as client:
            resp = client.get("/")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "kubedocs" in resp.text

    def test_existing_html_is_served_verbatim(self, api_settings: Settings, tmp_path: Path) -> None:
        """有构建产物时原样返回它：说明页只在真的缺前端时出现。"""
        page = tmp_path / "index.html"
        page.write_text(STARTUP_PAGE, encoding="utf-8")
        app = create_app(api_settings)
        app.state.index_html = page
        with TestClient(app) as client:
            resp = client.get("/")

        assert resp.status_code == 200
        assert STARTUP_PAGE in resp.text


class TestStatsCache:
    """`_DomainRuntime.stats()` 的短 TTL 缓存。

    回归的是一个**实测出来的性能缺陷**：`stats()` 里那条
    `COUNT(*) FROM chunks WHERE body LIKE '%```%'` 是 18822 行大文本列的全表扫描
    （本机实测 17.3 ms，占 `stats()` 的 24.4 ms 里的绝大部分），而 `/healthz`
    **每个请求**都走它、`/search` 也会走（`_index_note`），且全程持 store 锁。
    实测后果：本地直跑 `/healthz` 吞吐被硬卡在 ≈19.6 req/s，并发 1→16 时
    p50 从 54.6 ms 线性涨到 802 ms（并发毫无收益）。
    """

    def _runtime_with_counting_store(self, api_settings: Settings) -> tuple[object, list[int]]:
        from kubedocs.api.app import _DomainRuntime

        runtime = _DomainRuntime(api_settings)
        calls: list[int] = []

        class _CountingStore:
            def stats(self) -> object:
                calls.append(1)
                return SimpleNamespace(n_chunks=18822, n_docs=1722)

            def close(self) -> None:
                """lifespan 关闭时会调用它 —— 桩少了这个会让测试在 teardown 炸。"""

        runtime.store = _CountingStore()  # type: ignore[assignment]
        return runtime, calls

    def test_second_call_within_ttl_does_not_hit_the_store(self, api_settings: Settings) -> None:
        runtime, calls = self._runtime_with_counting_store(api_settings)
        first = runtime.stats()
        second = runtime.stats()

        assert len(calls) == 1, "TTL 内又查了一次库（探针又变成全表扫描了）"
        assert first is second, "TTL 内应当返回同一个对象"

    def test_max_age_zero_bypasses_the_cache(self, api_settings: Settings) -> None:
        """逃生口：要即时值时必须能拿到（缓存不是不可绕过的）。"""
        runtime, calls = self._runtime_with_counting_store(api_settings)
        runtime.stats()
        runtime.stats(max_age_s=0.0)

        assert len(calls) == 2

    def test_healthz_uses_the_cache(self, api_settings: Settings) -> None:
        """端到端：连打三次 `/healthz` 只应查一次库。"""
        app = create_app(api_settings)
        calls: list[int] = []
        runtime = app.state.runtimes["k8s_docs"] if hasattr(app.state, "runtimes") else None
        if runtime is None:  # 运行期表是惰性建的，先触发一次
            with TestClient(app) as client:
                client.get("/healthz")
            runtime = app.state.runtimes["k8s_docs"]

        class _CountingStore:
            def stats(self) -> object:
                calls.append(1)
                return SimpleNamespace(n_chunks=18822, n_docs=1722)

            def close(self) -> None:
                """同上：lifespan 会调用。"""

        runtime.store = _CountingStore()
        runtime._stats_cache = None  # noqa: SLF001 - 清掉上一次的缓存，让计数从 0 开始
        with TestClient(app) as client:
            for _ in range(3):
                assert client.get("/healthz").json()["n_chunks"] == 18822

        assert len(calls) == 1, f"三次 /healthz 查了 {len(calls)} 次库（缓存没生效）"


class TestConsolePage:
    """`/console` 控制台页。

    和演示页共用 `_serve_page`，所以这里既测"缺文件时的降级"，也测"仓库里的
    真文件真的被端到端送达"——后者是防路径写错的那一道闸：路由里写的是
    `("data", "web", "console.html")`，拼错了在浏览器里才发现就太晚了。
    """

    def test_missing_html_returns_200_explanation(self, api_settings: Settings, tmp_path: Path) -> None:
        app = create_app(api_settings)
        app.state.console_html = tmp_path / "no-such-console.html"
        with TestClient(app) as client:
            resp = client.get("/console")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "kubedocs" in resp.text

    def test_existing_html_is_served_verbatim(self, api_settings: Settings, tmp_path: Path) -> None:
        page = tmp_path / "console.html"
        page.write_text(STARTUP_PAGE, encoding="utf-8")
        app = create_app(api_settings)
        app.state.console_html = page
        with TestClient(app) as client:
            resp = client.get("/console")

        assert resp.status_code == 200
        assert STARTUP_PAGE in resp.text

    def test_repo_asset_is_reachable_and_self_contained(self, api_settings: Settings) -> None:
        """仓库里真有 data/web/console.html，且送达时没有外链依赖（离线可用）。"""
        app = create_app(api_settings)
        assert app.state.console_html.is_file(), f"控制台页缺失：{app.state.console_html}"
        with TestClient(app) as client:
            resp = client.get("/console")

        assert resp.status_code == 200
        assert "kubedocs" in resp.text
        # 页面必须自带样式与脚本：容器里没有外网，CDN 一挂就是白板。
        assert "<style>" in resp.text and "<script>" in resp.text
        assert "http://cdn" not in resp.text and "https://cdn" not in resp.text


class TestPatientPage:
    """`/patient` 患者端：**只**保留问诊 agent。

    这个页面的关键约束不是"能打开"，而是**"不该有的东西一个都没有"**——
    `/` 与 `/console` 上的域名选择、API 地址、`top_k`、指标、工具清单对患者
    是纯认知负担，而且能改到不该改的东西。所以下面有一组**否定断言**，
    它们的作用是防止以后有人"顺手"把开发向控件加回来。
    """

    def test_missing_html_returns_200_explanation(self, api_settings: Settings, tmp_path: Path) -> None:
        app = create_app(api_settings)
        app.state.patient_html = tmp_path / "no-such-patient.html"
        with TestClient(app) as client:
            resp = client.get("/patient")

        assert resp.status_code == 200
        assert "kubedocs" in resp.text

    def test_existing_html_is_served_verbatim(self, api_settings: Settings, tmp_path: Path) -> None:
        page = tmp_path / "patient.html"
        page.write_text(STARTUP_PAGE, encoding="utf-8")
        app = create_app(api_settings)
        app.state.patient_html = page
        with TestClient(app) as client:
            resp = client.get("/patient")

        assert resp.status_code == 200
        assert STARTUP_PAGE in resp.text

    def test_repo_asset_is_reachable_and_offline_safe(self, api_settings: Settings) -> None:
        app = create_app(api_settings)
        assert app.state.patient_html.is_file(), f"患者端页面缺失：{app.state.patient_html}"
        with TestClient(app) as client:
            resp = client.get("/patient")

        assert resp.status_code == 200
        assert "<style>" in resp.text and "<script>" in resp.text
        assert "http://cdn" not in resp.text and "https://cdn" not in resp.text

    def test_has_no_developer_or_ops_controls(self, api_settings: Settings) -> None:
        """**否定断言**：开发/运维向的控件一个都不许出现。"""
        with TestClient(create_app(api_settings)) as client:
            text = client.get("/patient").text

        for forbidden, why in (
            ("#domainSel", "域选择器"),
            ('id="apiBase"', "API 地址输入框"),
            ('id="topK"', "top_k 输入框"),
            ("/metrics", "指标端点"),
            ("/tools", "工具清单端点"),
            ("工具调用轨迹", "工具轨迹面板"),
            ("检索诊断", "检索诊断面板"),
        ):
            assert forbidden not in text, f"患者端出现了{why}（{forbidden}）"

    def test_keeps_the_safety_critical_parts(self, api_settings: Settings) -> None:
        """**肯定断言**：安全关键的东西必须在。

        急诊红线是这个域的安全底线——患者端把它做丢，比多几个控件严重得多。
        """
        with TestClient(create_app(api_settings)) as client:
            text = client.get("/patient").text

        assert "emergency" in text, "急诊提示的渲染路径不见了"
        assert "120" in text, "急诊指引（拨打 120）不见了"
        assert "/chat" in text, "没有调用问诊接口"
        assert "awaiting_user" in text, "追问状态没有被处理"
        assert "不能替代医生诊断" in text, "缺少免责声明"
