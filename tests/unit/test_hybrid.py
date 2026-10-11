"""混合检索测试：两路召回与融合、降级可观测性、结果可引用。"""

from __future__ import annotations

import numpy as np
import pytest

from kubedocs.config import Settings
from kubedocs.embed import EmbedderError
from kubedocs.retrieve.hybrid import HybridRetriever
from kubedocs.store.base import ChunkInput, DocInput
from kubedocs.store.sqlite_store import SqliteStore

DIM = 32


def _v(*xs: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    return v


class FixedEmbedder:
    """永远返回同一个查询向量，让"哪一路排第一"完全可预测。"""

    name = "fixed"
    dim = DIM

    def __init__(self, vec: np.ndarray) -> None:
        self._v = vec

    def encode(self, texts, *, is_query: bool = False) -> np.ndarray:
        return np.tile(self._v, (len(texts), 1))


class BrokenEmbedder:
    name = "broken"
    dim = DIM

    def encode(self, texts, *, is_query: bool = False) -> np.ndarray:
        raise EmbedderError("模型不可用（测试用）")


def _seed(store: SqliteStore) -> None:
    """直接写入两篇文档，向量手工指定，便于精确断言。

    A：正文含 bm25 独有词，向量与查询正交（向量路应当排后面）
    B：正文不含查询词，向量与查询一致（向量路应当排第一）
    """
    docs, chunks, vecs = [], [], {}
    for doc_id, body, vec in (
        ("a", "## Section\n\nalphauniqueword describes the thing in detail here", _v(0.0, 1.0)),
        ("b", "## Section\n\nbetauniqueword explains something else entirely", _v(1.0, 0.0)),
    ):
        docs.append(
            DocInput(
                doc_id=doc_id,
                rel_path=f"{doc_id}.md",
                title=doc_id.title(),
                source_url=f"https://kubernetes.io/docs/{doc_id}/",
                content_hash=f"h-{doc_id}",
            )
        )
        cid = f"{doc_id}::0000"
        chunks.append(
            ChunkInput(
                chunk_id=cid,
                doc_id=doc_id,
                ordinal=0,
                title=doc_id.title(),
                breadcrumb="Section",
                source_url=f"https://kubernetes.io/docs/{doc_id}/",
                body=body,
                content_hash=f"h-{doc_id}-0",
                n_chars=len(body),
            )
        )
        vecs[cid] = vec
    store.upsert_docs(docs, chunks, vecs)
    store.finalize()


@pytest.fixture
def retriever(store: SqliteStore, settings: Settings) -> HybridRetriever:
    _seed(store)
    return HybridRetriever(store, FixedEmbedder(_v(1.0, 0.0)), settings)


class TestChannels:
    def test_bm25_only_uses_lexical_match(self, retriever: HybridRetriever):
        res = retriever.search("alphauniqueword", channels=("bm25",))
        assert res.chunk_ids == ["a::0000"]

    def test_dense_only_uses_vector_similarity(self, retriever: HybridRetriever):
        res = retriever.search("alphauniqueword", channels=("dense",))
        # 查询向量是 e0，只有 b 的向量与之一致
        assert res.chunk_ids[0] == "b::0000"
        assert "a::0000" in res.chunk_ids  # 仍会作为次优返回（余弦更低）

    def test_hybrid_contains_both_channels(self, retriever: HybridRetriever):
        res = retriever.search("alphauniqueword")
        assert set(res.chunk_ids) == {"a::0000", "b::0000"}

    def test_fusion_ranks_two_channel_hit_higher(self, retriever: HybridRetriever):
        """A 在全文路第 1、向量路第 2；B 只在向量路第 1。
        两路都被命中的 A 应当胜出——这正是融合要的效果。"""
        res = retriever.search("alphauniqueword")
        assert res.results[0].chunk_id == "a::0000"
        assert res.results[0].ranks == {"bm25": 1, "dense": 2}
        assert res.results[1].ranks == {"dense": 1}

    def test_diagnostics_report_channel_stats(self, retriever: HybridRetriever):
        diag = retriever.search("alphauniqueword").diagnostics
        assert diag["channels"] == ["bm25", "dense"]
        assert diag["overlap"] == 1
        detail = diag["channels_detail"]
        assert detail["bm25"]["n"] == 1 and detail["dense"]["n"] == 2  # type: ignore[index]


class TestDegradation:
    def test_vector_channel_failure_still_returns_lexical_results(
        self, store: SqliteStore, settings: Settings
    ):
        _seed(store)
        r = HybridRetriever(store, BrokenEmbedder(), settings)
        res = r.search("alphauniqueword")
        assert res.chunk_ids == ["a::0000"]  # 纯全文仍然给得出答案
        degraded = res.diagnostics["degraded"]
        assert degraded and "向量路不可用" in degraded[0]  # type: ignore[index]
        assert res.diagnostics["channels_detail"]["dense"]["n"] == 0  # type: ignore[index]

    def test_missing_term_and_broken_dense_returns_empty_not_crash(
        self, store: SqliteStore, settings: Settings
    ):
        _seed(store)
        r = HybridRetriever(store, BrokenEmbedder(), settings)
        assert r.search("no-such-token-anywhere").chunk_ids == []


class TestResultShape:
    def test_empty_query_is_handled(self, retriever: HybridRetriever):
        res = retriever.search("   ")
        assert res.results == []
        assert res.diagnostics["note"] == "空查询"

    def test_top_k_truncates(self, retriever: HybridRetriever):
        assert len(retriever.search("alphauniqueword", top_k=1).results) == 1

    def test_every_result_is_citable(self, retriever: HybridRetriever):
        for r in retriever.search("alphauniqueword").results:
            assert r.source_url.startswith("https://kubernetes.io/docs/")
            assert r.citation == f"{r.title} › {r.breadcrumb}"
            assert r.body.strip()

    def test_explain_lines_are_readable(self, retriever: HybridRetriever):
        res = retriever.search("alphauniqueword")
        text = res.explain_all()
        assert "RRF=" in text and "bm25=1" in text

    def test_chunk_ids_property_matches_results(self, retriever: HybridRetriever):
        res = retriever.search("alphauniqueword")
        assert res.chunk_ids == [r.chunk_id for r in res.results]
