"""API 层测试：路由形状、无密钥收口、SSE 事件格式。

三条硬约束：
1. 绝不联网、绝不加载真实模型：索引建在 `tmp_path`、嵌入用 `hashing`，
   conftest 的 autouse 夹具还会清掉所有供应商密钥；
2. 不依赖磁盘语料：直接构造 store 灌块——这里验证的是 HTTP 层的协议转换；
3. 空索引是重点用例：`/healthz` `/domains` `/tools` `/search` `/ask` `/chat`
   在没有索引时都必须能返回而不是 500，这是部署中间态。

## 无密钥时的契约（新架构）

`/ask` 与 `/chat` 都**只回一句话**（`NO_LLM_MESSAGE`）：`/ask` 给空引用 + `n_hits=0`，
`/chat` 的 message 事件就是那句话（`status="no_llm"`）。确定性图（`ClinicAgent`）
**不在请求路径上**，它保留为"规则判断 vs 模型判断"的对照臂——下面有一条用例专门钉住
"请求期间它一次都没被调用"。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
from fastapi.testclient import TestClient

from kubedocs.agent.clinic_react import ClinicReactRuntime
from kubedocs.api.app import create_app
from kubedocs.api.threads import ThreadOwnershipError
from kubedocs.config import Settings
from kubedocs.embed import HashingEmbedder
from kubedocs.generate import Answerer, LLMResult
from kubedocs.llm_status import LLM_FAILED_MESSAGE, NO_LLM_MESSAGE
from kubedocs.store import from_settings
from kubedocs.store.base import ChunkInput, DocInput

DIM = 32
"""必须与 `Settings.embedding_dim` 一致；hashing 的维度只影响向量空间大小。"""

LONG_BODY = "kubelet eviction 硬阈值说明，这一段刻意写得很长用来验证截断。" * 40
"""明显超过 600 字符的正文，用于验证 `/search` 的截断行为。"""

STARTUP_PAGE = "<h1>真实首页</h1>"


class _StubLLM:
    """打桩 LLM：返回一句带合法引用标记的答案，不碰任何网络。"""

    def complete(self, system: str, user: str) -> LLMResult:
        return LLMResult(text="打桩答案 [1]", prompt_tokens=1, completion_tokens=1)


class _FailingLLM:
    """打桩 LLM：这次调用失败（限流/超时那一类），不碰任何网络。"""

    def complete(self, system: str, user: str) -> LLMResult:
        raise RuntimeError("供应商 429")


# --------------------------------------------------------------------------- 辅助
def _seed_index(settings: Settings, bodies: list[str], *, title: str = "Pod 驱逐") -> None:
    """直接构造 store 灌入 1~2 个块（等价于 ingest 的后半段）。

    不走 CLI 的 ingest：那需要磁盘语料，而这里验证的是 HTTP 层。写完即 `close()`，
    随后 `create_app` 自己再开一次，顺带验证了索引文件持久、不靠进程内状态。
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
    """索引指向 tmp_path、嵌入用 hashing：**绝不联网、绝不加载真实模型**。

    显式 `store="sqlite-vec"`：`--index-dir` 式的隔离只对本地后端成立（pgvector 的索引
    在 Postgres 的 schema 里，目录参数对它没有作用）。默认后端是 pgvector，不写这一项
    这些用例就会去连真实 Postgres，既是环境依赖、又会被本机索引的维度/内容影响。

    `database_url` 指向一个必然连不上的端口：**测试不该碰真实的号源库**（那会真的占掉
    库存，几天后某条用例会因为"某个时段挂满了"而红）。指到死端口后号源走进程内降级。
    """
    return Settings(
        store="sqlite-vec",
        database_url="postgresql://kubedocs:kubedocs@127.0.0.1:1/kubedocs",
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

    用 `with` 进入 lifespan：测试结束会关掉 SQLite 连接，否则 Windows 上 tmp_path
    因文件被占用而删不掉（teardown 报错）。
    """
    with TestClient(create_app(api_settings)) as c:
        yield c


# --------------------------------------------------------------------------- 没有索引时的探针与自省
class TestWithoutIndex:
    def test_healthz_returns_200_even_without_index(self, client: TestClient) -> None:
        """没有索引时 /healthz 仍然 200：判活探针关心"进程活着"，不关心有没有语料。

        `store` / `store_fallback` 是"用的是哪个向量库、有没有降级"的对外读数，
        这里只断言它们存在（具体后端取决于测试环境的配置与库可达性）。
        """
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert {k: body[k] for k in ("ok", "domain", "n_chunks")} == {
            "ok": True,
            "domain": "k8s_docs",
            "n_chunks": 0,
        }
        assert "store" in body and "store_fallback" in body

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
        # 期望值由全量清单自己算出，这样断言不依赖"哪个工具属于哪个域"的实现细节
        assert filtered == [t for t in all_tools if t["domain"] == "clinic"]

        bad = client.get("/tools", params={"domain": "no_such_domain"})
        assert bad.status_code == 400
        detail = bad.json()["detail"]
        assert "k8s_docs" in detail and "clinic" in detail, detail


# --------------------------------------------------------------------------- /search
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


# --------------------------------------------------------------------------- /ask
class TestAsk:
    def test_without_llm_key_returns_only_the_message(self, api_settings: Settings) -> None:
        """**新契约**：没有 LLM 密钥 → 200 + 只回 `NO_LLM_MESSAGE`，**不回检索兜底内容**。

        以前这里返回"检索结果 + degraded 说明"。降级层已经删掉：`citations=[]`、`n_hits=0`，
        而且**连检索都不跑**（autouse 夹具清掉了所有密钥，分支是确定的）。
        """
        _seed_index(api_settings, [LONG_BODY])
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/ask", json={"q": "kubelet"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == NO_LLM_MESSAGE
        assert body["citations"] == [], "无密钥时不许回检索兜底内容"
        assert body["n_hits"] == 0
        assert "未连接大模型" in body["degraded"]

    def test_provider_failure_says_a_different_sentence(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """配了密钥但这次调用失败 → `LLM_FAILED_MESSAGE`（**不是**"未连接大模型"）。

        两句必须分开：把"连了但失败"说成"未连接"会让运维去翻一个其实配好的密钥。
        具体原因（异常类型）留在 `degraded` 里。
        """
        _seed_index(api_settings, [LONG_BODY])
        monkeypatch.setattr("kubedocs.api.app.has_api_key", lambda _model: True)
        monkeypatch.setattr(
            "kubedocs.api.app.build_answerer",
            lambda retriever, settings: Answerer(retriever, _FailingLLM(), settings),
        )
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/ask", json={"q": "kubelet"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == LLM_FAILED_MESSAGE
        assert body["answer"] != NO_LLM_MESSAGE
        assert body["citations"] == []
        assert body["n_hits"] == 0
        assert "RuntimeError" in body["degraded"]

    def test_empty_index_returns_200_with_explanation(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**有模型**而索引为空时：200 + 说明"检索结果为空"，不编造（空索引不是故障）。

        需要把密钥判成有：无密钥时 `/ask` 在**检索之前**就收口成那一句话
        （见上一条用例），根本走不到"空索引"这条分支。
        """
        monkeypatch.setattr("kubedocs.api.app.has_api_key", lambda _model: True)
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/ask", json={"q": "pod"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == ""
        assert body["n_hits"] == 0
        assert body["citations"] == []
        assert "检索结果为空" in body["degraded"]

    def test_answer_and_citations_are_passed_through(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """有可用 LLM 时，`/ask` 把生成层的答案与引用原样透出（打桩 LLM，绝不联网）。

        打桩打在装配点 `build_answerer`：只要漏一处，真调用就会发到供应商。
        这一条覆盖"检索 → 生成 → 响应映射"的胶水，生成层自身由 `test_generate.py` 覆盖。
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


# --------------------------------------------------------------------------- /chat（SSE）
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

    def test_without_llm_key_the_message_is_one_sentence(self, api_settings: Settings) -> None:
        """**新契约**：没有密钥的 `/chat` 只回 `NO_LLM_MESSAGE`，不发 tool 事件、不回落规则轨。

        以前这里会回退到确定性图（`ClinicAgent`）跑红线与挂号。降级层已经删掉：
        没有判断者就没有结论，`status="no_llm"` 让调用方能与正常轮次分开。
        """
        settings = api_settings.model_copy(update={"domain": "clinic"})
        with TestClient(create_app(settings)) as client:
            resp = client.post("/chat", json={"message": "我现在胸痛", "patient_id": "p1"})

        assert resp.status_code == 200
        events = _sse_events(resp.text)
        message = _event(events, "message")
        assert message["text"] == NO_LLM_MESSAGE
        assert message["status"] == "no_llm"
        assert message["citations"] == []
        assert message["pending_action"] is None
        # 无密钥时那一轮没有依据可言，也没有参考标准面板
        assert message["reference"] is None
        assert [data for name, data in events if name == "tool"] == [], "无密钥时不该有 tool 事件"

    def test_the_deterministic_graph_is_not_on_the_request_path(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**回归（新契约）**：请求期间 `_get_clinic_agent` 一次都不许被调用。

        确定性图保留为对照臂（评测里比较"规则判断 vs 模型判断"），但 `/chat` 只有模型
        驱动的一条路；把它接回请求路径会让同一个问题又由两个"脑"分别作答。
        """
        called: list[int] = []

        def _boom(_app: Any) -> Any:
            called.append(1)
            raise AssertionError("确定性图被接回请求路径了")

        monkeypatch.setattr("kubedocs.api.app._get_clinic_agent", _boom)
        settings = api_settings.model_copy(update={"domain": "clinic"})
        with TestClient(create_app(settings)) as client:
            resp = client.post("/chat", json={"message": "我现在胸痛"})

        assert resp.status_code == 200
        assert _event(_sse_events(resp.text), "message")["text"] == NO_LLM_MESSAGE
        assert called == [], "确定性图不在请求路径上"

    def test_message_event_carries_the_reference_panel(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """有密钥时 `message` 事件带 `reference`（键名与 `/consult` 一致），可逐轮对账。

        面板是**依据**的留档：模型说了什么在 `text` 里，依据是什么在 `reference` 里。
        """
        monkeypatch.setattr("kubedocs.api.app.has_api_key", lambda _model: True)
        monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: True)
        # 真实关联表来自 8MB 卡片语料：单测不读它，换一张打桩的。
        monkeypatch.setattr("kubedocs.agent.clinic_react._triage_links", lambda: {"咳嗽": ("呼吸内科",)})

        class _Capturing:
            def __init__(self, _settings: Any) -> None:
                pass

            def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
                return {"content": "建议挂呼吸内科。", "tool_calls": []}

        monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _Capturing)
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/chat", json={"message": "我咳嗽", "patient_id": "p-ref"})

        message = _event(_sse_events(resp.text), "message")
        assert message["text"] == "建议挂呼吸内科。"
        assert message["status"] == "done"
        reference = message["reference"]
        assert reference, "有密钥的那一轮必须带上面板"
        assert {"red_line", "symptoms", "booking", "known_departments"} <= set(reference)
        assert reference["booking"] == [], "本轮服务端没有挂号动作"

    def test_thread_id_is_respected_and_conversation_continues(self, client: TestClient) -> None:
        """带同一个 thread_id 再来一次 = 续聊：会话属于调用方，服务端按 id 记账。"""
        first = client.post("/chat", json={"message": "我最近咳嗽", "thread_id": "t-api-1"})
        assert _event(_sse_events(first.text), "thread")["thread_id"] == "t-api-1"

        second = client.post("/chat", json={"message": "帮我挂号", "thread_id": "t-api-1"})
        assert second.status_code == 200
        message = _event(_sse_events(second.text), "message")
        assert message["thread_id"] == "t-api-1"
        assert message["text"]

    def test_chat_works_without_index_or_corpus(self, api_settings: Settings) -> None:
        """无索引/无语料也要能返回：这一轮不检索，也不许 500。"""
        settings = api_settings.model_copy(update={"domain": "clinic"})
        with TestClient(create_app(settings)) as client:
            resp = client.post("/chat", json={"message": "我咳嗽", "patient_id": "p1"})

        assert resp.status_code == 200
        assert _event(_sse_events(resp.text), "message")["text"] == NO_LLM_MESSAGE

    def test_too_long_message_is_422(self, client: TestClient) -> None:
        """message 超过 2000 字符 → 422（单轮对话文本的上限在 schema 里）。"""
        assert client.post("/chat", json={"message": "咳" * 2001}).status_code == 422

    def test_message_event_reports_zero_unpriced_calls(self, client: TestClient) -> None:
        """`unpriced_llm_calls` 是"未知成本可见"的契约：正常情况下必须是 0（不是缺失）。"""
        resp = client.post("/chat", json={"message": "我咳嗽"})
        assert _event(_sse_events(resp.text), "message")["unpriced_llm_calls"] == 0


class TestChatThreadOwnership:
    """`/chat` 的 thread_id 属主校验：首见即绑定 patient_id，换属主 ⇒ 409 + 可读原因。

    既不静默当成新会话，也不静默接到别人的历史上；既有行为与被拒后的属主都不变。
    """

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
        """会话表有上界：超过上限的旧线程按 LRU 被淘汰，表不会无界增长。"""
        store = client.app.state.clinic_threads  # type: ignore[attr-defined]
        store.max_threads = 3
        for i in range(5):
            resp = client.post("/chat", json={"message": "我咳嗽", "thread_id": f"t-cap-{i}"})
            assert resp.status_code == 200
        assert len(store) == 3
        assert store.n_evicted == 2


class TestChatBudget:
    """`/chat` 同样受当日成本上限约束：超限时一次模型调用都不发生，原因如实送到流上。

    这条路径走 `LiteLLMToolCaller`（只累加 `self.cost`、不与预算比较），由本层把关。
    """

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

        # 有密钥才走得到工具循环（无密钥时 `/chat` 在检索之前就收口成一句话）；
        # 打桩关联表避免单元测试去读 8MB 卡片语料。
        monkeypatch.setattr("kubedocs.api.app.has_api_key", lambda _model: True)
        monkeypatch.setattr("kubedocs.agent.clinic_react.ClinicReactRuntime.has_llm", lambda self: True)
        monkeypatch.setattr("kubedocs.agent.clinic_react._triage_links", lambda: {})
        monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _CountingCaller)
        with TestClient(create_app(api_settings)) as client:
            resp = client.post("/chat", json={"message": "我咳嗽", "thread_id": "t-budget"})

        assert resp.status_code == 200  # 响应头早就发出去了，只能把失败当事件发
        err = _event(_sse_events(resp.text), "error")
        assert "BudgetExceeded" in err["error"] and "上限" in err["error"]
        assert calls == [], "超预算后仍然调用了模型：账单没被挡住"


# --------------------------------------------------------------------------- 首页
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

    `stats()` 要在正文大文本列上全表扫描，而 `/healthz` 每请求都走它、`/search` 也走
    （`_index_note`），且全程持锁：没有缓存时吞吐被卡住、并发毫无收益。`max_age_s=0` 可绕过。
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
                """lifespan 关闭时会调用它；桩缺了它 teardown 会炸。"""

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


class TestClinicStatePersistence:
    """图状态的持久化 checkpointer：不配路径时行为不变（内存态），配了就落盘且新实例能读同一 thread。"""

    def test_unset_means_in_memory(self, api_settings: Settings) -> None:
        """留空 = 不落盘、不建文件（纯内存态，行为不变）。"""
        from kubedocs.api.app import _make_checkpointer

        assert _make_checkpointer(api_settings) is None

    def test_configured_persists_and_survives_a_new_saver(self, tmp_path: Path) -> None:
        """配了路径就落盘，且**新实例能读到同一个 thread**——这才是续跑的关键。"""
        from langgraph.graph import END, START, StateGraph

        from kubedocs.api.app import _make_checkpointer

        db = tmp_path / "state" / "clinic.db"
        settings = Settings(
            store="sqlite-vec",
            index_dir=str(tmp_path / "index"),
            domain="k8s_docs",
            embedding_provider="hashing",
            embedding_dim=DIM,
            embedding_model="hashing-test",
            rerank_enable=False,
            clinic_state_db=str(db),
        )

        class _S(TypedDict):
            n: int

        def bump(state: _S) -> _S:
            return {"n": state.get("n", 0) + 1}

        g = StateGraph(_S)
        g.add_node("a", bump)
        g.add_edge(START, "a")
        g.add_edge("a", END)
        cfg = {"configurable": {"thread_id": "t1"}}

        first = _make_checkpointer(settings)
        assert first is not None
        assert g.compile(checkpointer=first).invoke({"n": 0}, cfg) == {"n": 1}  # type: ignore[arg-type]
        first.conn.close()  # type: ignore[attr-defined]
        assert db.is_file(), "配了路径却没有落盘文件"

        second = _make_checkpointer(settings)
        assert second is not None
        assert g.compile(checkpointer=second).get_state(cfg).values == {"n": 1}  # type: ignore[arg-type]
        second.conn.close()  # type: ignore[attr-defined]

    def test_app_does_not_build_one_by_default(self, api_settings: Settings) -> None:
        """装配层默认不建：不配就不该多出一个文件、多一条连接。

        这里**直接调用** `_get_clinic_agent` 是刻意的：它不在请求路径上（见类文档与
        `test_the_deterministic_graph_is_not_on_the_request_path`），对照实验与回归测试
        仍然要能单独构造它。
        """
        from kubedocs.api.app import _get_clinic_agent

        app = create_app(api_settings)
        with TestClient(app):
            _get_clinic_agent(app)
        assert app.state.clinic_checkpointer is None, "没配路径却建了 checkpointer"


class TestConsolePage:
    """`/console` 控制台页：与演示页共用 `_serve_page`。

    既测缺文件时的降级，也测仓库里的真文件端到端送达（防路由里的路径拼错）。
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
