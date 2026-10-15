"""SQLite 存储的单元测试：FTS5 外内容表删除顺序、rowid 回收后的向量对齐、MATCH 表达式与元信息守卫。

守的契约：MATCH 只留词元，删除或重建后两个通道都不再召回被删的块，维度或嵌入模型不符必须拒绝。"""

from __future__ import annotations

import numpy as np
import pytest

from kubedocs.store.base import ChunkInput, DocInput
from kubedocs.store.sqlite_store import (
    MissingVectorError,
    SqliteStore,
    StoreError,
    build_match_expr,
)


def _vec(*xs: float, dim: int = 32) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    return v


def _doc(doc_id: str, n: int, text_prefix: str = "body") -> tuple[DocInput, list[ChunkInput]]:
    rel = f"{doc_id}.md"
    d = DocInput(
        doc_id=doc_id, rel_path=rel, title=doc_id, source_url=f"u/{doc_id}/", content_hash=f"h-{doc_id}"
    )
    chunks = [
        ChunkInput(
            chunk_id=f"{doc_id}::{i:04d}",
            doc_id=doc_id,
            ordinal=i,
            title=doc_id,
            breadcrumb="sec",
            source_url=f"u/{doc_id}/",
            body=f"{text_prefix} {i} {doc_id}",
            content_hash=f"h-{doc_id}-{i}",
            n_chars=20,
        )
        for i in range(n)
    ]
    return d, chunks


def _upsert(store: SqliteStore, doc_id: str, n: int, axis: int, text_prefix: str = "body") -> None:
    d, chunks = _doc(doc_id, n, text_prefix)
    store.upsert_docs([d], chunks, {c.chunk_id: _vec(axis) for c in chunks})
    store.finalize()


class TestBuildMatchExpr:
    def test_strips_fts_operators(self):
        # `-` 会被 FTS5 解释成 NOT，必须只留词元
        assert "NOT" not in build_match_expr("pod -limit")
        assert "limit" in build_match_expr("pod -limit")

    def test_removes_stopwords(self):
        expr = build_match_expr("how do I set a pod")
        assert "how" not in expr.lower()
        assert "pod" in expr

    def test_quotes_each_token(self):
        assert build_match_expr("crashloopbackoff") == '"crashloopbackoff"'

    def test_dedup_is_case_insensitive(self):
        expr = build_match_expr("pod Pod POD")
        assert expr.count("OR") == 0
        assert expr.lower() == '"pod"'

    def test_only_punctuation_returns_empty(self):
        assert build_match_expr('"***"') == ""
        assert build_match_expr("!!!") == ""

    def test_quote_injection_is_neutralised(self):
        expr = build_match_expr('pod" OR "x')
        assert expr.count('"') % 2 == 0  # 引号必须成对，否则 FTS5 语法错
        assert "pod" in expr


def test_upsert_and_keyword_search(store: SqliteStore):
    _upsert(store, "a", 2, axis=0, text_prefix="kubelet eviction")
    hits = store.keyword_search("kubelet", 10)
    # 两个块都含该词，都该被召回（同一文档的多块命中是正常的）
    assert [cid for cid, _ in hits] == ["a::0000", "a::0001"]
    assert [rank for _, rank in hits] == [1, 2]
    assert store.keyword_search("nonexistentterm", 10) == []


def test_dense_search_ranks_by_cosine(store: SqliteStore):
    _upsert(store, "a", 1, axis=0)
    _upsert(store, "b", 1, axis=1)
    hits = store.dense_search(_vec(1.0), 5)
    assert hits[0][0] == "b::0000"  # b 的向量是 e1，与查询完全一致
    assert hits[1][0] == "a::0000"


def test_fts_verify_ok_after_upsert(store: SqliteStore):
    _upsert(store, "a", 3, axis=0)
    assert store.verify_fts() == "ok"


class TestExternalContentDelete:
    """FTS5 外内容表：删除顺序反了会让索引永久不一致，且只在后续查询才暴露。"""

    def test_delete_doc_keeps_index_consistent(self, store: SqliteStore):
        _upsert(store, "a", 2, axis=0, text_prefix="alpha unique")
        _upsert(store, "b", 2, axis=1, text_prefix="beta unique")
        store.delete_docs({"a.md"})
        store.finalize()
        assert store.verify_fts() == "ok"
        assert store.keyword_search("alpha", 10) == []
        assert [c for c, _ in store.keyword_search("beta", 10)] == ["b::0000", "b::0001"]

    def test_delete_all_then_reinsert(self, store: SqliteStore):
        _upsert(store, "a", 2, axis=0)
        store.delete_docs({"a.md"})
        _upsert(store, "c", 2, axis=2, text_prefix="gamma unique")
        store.finalize()
        assert store.verify_fts() == "ok"
        assert [c for c, _ in store.keyword_search("gamma", 10)] == ["c::0000", "c::0001"]

    def test_rebuild_from_scratch_clears_everything(self, store: SqliteStore):
        _upsert(store, "a", 2, axis=0, text_prefix="alpha")
        store.rebuild_from_scratch()
        assert store.stats().n_chunks == 0
        assert store.keyword_search("alpha", 10) == []
        assert store.verify_fts() == "ok"


class TestRowidRecycling:
    """SQLite 会回收最大 rowid。删掉末位文档再插入，新块会拿到更小的 rowid——
    此时"按 rowid 顺序"与"矩阵行号顺序"不再一致，向量必须跟着重排。"""

    def test_vector_row_stays_aligned_after_rowid_reuse(self, store: SqliteStore):
        _upsert(store, "a", 2, axis=0)
        _upsert(store, "b", 2, axis=1)
        store.delete_docs({"b.md"})  # b 的 rowid 最大，会被回收后交给新块复用
        store.finalize()
        _upsert(store, "c", 2, axis=2)
        store.finalize()

        # 若对齐错了，这里会拿到别的块的向量（余弦为 0），名次就乱了
        assert store.dense_search(_vec(1.0), 1)[0][0] == "c::0000"
        assert store.dense_search(_vec(0.0, 1.0), 1)[0][0] == "a::0000"

    def test_vector_matrix_shape_matches_chunks(self, store: SqliteStore):
        _upsert(store, "a", 3, axis=0)
        _upsert(store, "b", 2, axis=1)
        store.delete_docs({"a.md"})
        store.finalize()
        mat = np.load(store.vector_path)
        assert mat.shape == (store.stats().n_chunks, 32)

    def test_reopen_keeps_index_usable(self, tmp_path):
        path = tmp_path / "re.db"
        s = SqliteStore(db_path=path, vector_path=tmp_path / "re.npy", dim=32, embedding_model="hashing-test")
        d, chunks = _doc("a", 2)
        s.upsert_docs([d], chunks, {c.chunk_id: _vec(0.0, 1.0) for c in chunks})
        s.finalize()
        s.close()

        s2 = SqliteStore(
            db_path=path, vector_path=tmp_path / "re.npy", dim=32, embedding_model="hashing-test"
        )
        try:
            assert s2.stats().n_chunks == 2
            assert s2.dense_search(_vec(0.0, 1.0), 1)[0][0] == "a::0000"
            assert s2.verify_fts() == "ok"
        finally:
            s2.close()


class TestMetaGuards:
    def test_dim_mismatch_refused(self, tmp_path):
        SqliteStore(
            db_path=tmp_path / "m.db", vector_path=tmp_path / "m.npy", dim=32, embedding_model="m"
        ).close()
        with pytest.raises(StoreError, match="维度"):
            SqliteStore(
                db_path=tmp_path / "m.db", vector_path=tmp_path / "m.npy", dim=64, embedding_model="m"
            )

    def test_model_mismatch_refused(self, tmp_path):
        SqliteStore(
            db_path=tmp_path / "n.db", vector_path=tmp_path / "n.npy", dim=32, embedding_model="model-a"
        ).close()
        with pytest.raises(StoreError, match="嵌入模型"):
            SqliteStore(
                db_path=tmp_path / "n.db", vector_path=tmp_path / "n.npy", dim=32, embedding_model="model-b"
            )

    def test_missing_vector_is_refused_not_zero_filled(self, store: SqliteStore):
        d, chunks = _doc("a", 1)
        with pytest.raises(MissingVectorError):
            store.upsert_docs([d], chunks, {})


def test_existing_hash_to_vector_reuses_across_docs(store: SqliteStore):
    d1, c1 = _doc("a", 1)
    store.upsert_docs([d1], c1, {c1[0].chunk_id: _vec(1.0)})
    store.finalize()
    got = store.existing_hash_to_vector({c1[0].content_hash, "no-such-hash"})
    assert set(got) == {c1[0].content_hash}
    assert got[c1[0].content_hash][0] == pytest.approx(1.0)


def test_doc_texts_concatenates_by_doc(store: SqliteStore):
    _upsert(store, "a", 2, axis=0, text_prefix="AlphaWord")
    _upsert(store, "b", 1, axis=1, text_prefix="BetaWord")
    texts = store.doc_texts(["a", "b"])
    assert "alphaword" in texts["a"] and "betaword" not in texts["a"]
    assert "betaword" in texts["b"]


def test_stats_counts_code_chunks(store: SqliteStore):
    d = DocInput(doc_id="a", rel_path="a.md", title="a", source_url="u", content_hash="h")
    chunks = [
        ChunkInput(
            chunk_id="a::0000",
            doc_id="a",
            ordinal=0,
            title="a",
            breadcrumb="s",
            source_url="u",
            body="```\nkubectl get pods\n```",
            content_hash="h0",
            n_chars=20,
        ),
        ChunkInput(
            chunk_id="a::0001",
            doc_id="a",
            ordinal=1,
            title="a",
            breadcrumb="s",
            source_url="u",
            body="plain text",
            content_hash="h1",
            n_chars=10,
        ),
    ]
    store.upsert_docs([d], chunks, {"a::0000": _vec(1.0), "a::0001": _vec(0.0, 1.0)})
    store.finalize()
    st = store.stats()
    assert (st.n_docs, st.n_chunks, st.n_with_code) == (1, 2, 1)
    assert st.mean_chars == pytest.approx(15.0)
