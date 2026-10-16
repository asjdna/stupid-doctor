"""SQLite 存储的单元测试：FTS5 外内容表删除顺序、rowid 回收后的向量对齐、MATCH 表达式与元信息守卫、
父子两层（`parents` 表 / `chunks.parent_id`）的写入-读回-覆盖-删除，以及老库的原地升级。

守的契约：MATCH 只留词元，删除或重建后两个通道都不再召回被删的块，维度或嵌入模型不符必须拒绝；
命中块带上它所属**父块**的完整正文（取不到就留空、由 `context_text` 退回子块正文）。"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from kubedocs.store.base import ChunkInput, DocInput, ParentInput
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


def _parent(doc_id: str, i: int, body: str) -> ParentInput:
    return ParentInput(
        parent_id=f"{doc_id}::p{i:04d}",
        doc_id=doc_id,
        ordinal=i,
        title=doc_id,
        breadcrumb="sec",
        source_url=f"u/{doc_id}/",
        body=body,
        n_chars=len(body),
        content_hash=f"hp-{doc_id}-{i}",
        location=f"loc:{i}",
        n_children=1,
    )


def _parents_doc(
    doc_id: str, parent_bodies: list[str]
) -> tuple[DocInput, list[ChunkInput], list[ParentInput]]:
    """造一篇"一个父块配一个子块"的文档：父块正文互不相同，便于断言取回时**不串味**。"""
    d = DocInput(
        doc_id=doc_id,
        rel_path=f"{doc_id}.md",
        title=doc_id,
        source_url=f"u/{doc_id}/",
        content_hash=f"h-{doc_id}",
    )
    parents = [_parent(doc_id, i, body) for i, body in enumerate(parent_bodies)]
    chunks = [
        ChunkInput(
            chunk_id=f"{doc_id}::{i:04d}",
            doc_id=doc_id,
            ordinal=i,
            title=doc_id,
            breadcrumb="sec",
            source_url=f"u/{doc_id}/",
            body=f"child {i} {doc_id}",
            content_hash=f"h-{doc_id}-{i}",
            n_chars=20,
            parent_id=parents[i].parent_id,
        )
        for i in range(len(parents))
    ]
    return d, chunks, parents


def _upsert_parents(
    store: SqliteStore, doc_id: str, parent_bodies: list[str], *, axis: int = 0, write_parents: bool = True
) -> None:
    d, chunks, parents = _parents_doc(doc_id, parent_bodies)
    store.upsert_docs(
        [d], chunks, {c.chunk_id: _vec(axis) for c in chunks}, parents=parents if write_parents else None
    )
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


class TestParentChildReadback:
    """父子两层读回：命中子块要带回它所属**父块**的完整正文。"""

    def test_hit_carries_parent_id_and_parent_text(self, store: SqliteStore):
        """防的回归：子块检索命中后，喂给模型的是子块正文（300 字的碎片）而不是父块的完整语境——
        父子切割的全部收益就没了，且这种退化**不会报错**，只能靠断言 `context_text` 抓。"""
        _upsert_parents(store, "a", ["父块甲：完整小节正文"])
        hit = store.get_chunks(["a::0000"])["a::0000"]
        assert hit.parent_id == "a::p0000"
        assert hit.parent_text == "父块甲：完整小节正文"
        assert hit.context_text == "父块甲：完整小节正文"
        assert hit.body == "child 0 a"  # 子块正文仍在（引用/高亮要用它）

    def test_batch_read_fills_each_parent_without_crosstalk(self, store: SqliteStore):
        """防的回归：一次取多条命中时父块**串味**（把 A 的语境喂给 B 的块）。

        这类错误通常来自"先查第一条的父块、循环里忘了换"或把父块按行号错位映射；
        单条命中的用例抓不到它，所以必须一次取多条、每条都断言自己的父块。"""
        _upsert_parents(store, "a", ["甲父块正文", "乙父块正文", "丙父块正文"])
        hits = store.get_chunks(["a::0002", "a::0000", "a::0001"])
        assert {cid: h.parent_text for cid, h in hits.items()} == {
            "a::0000": "甲父块正文",
            "a::0001": "乙父块正文",
            "a::0002": "丙父块正文",
        }

    def test_missing_parents_fall_back_to_child_body(self, store: SqliteStore):
        """老索引兜底：`parents=None` 写入的子块（parent_id 指不到任何父块）必须能用，
        `context_text` 退回子块正文——**绝不能是空串**，那等于让模型对着空白回答。"""
        _upsert_parents(store, "a", ["父块正文根本没写进去"], write_parents=False)
        hit = store.get_chunks(["a::0000"])["a::0000"]
        assert store.stats().n_parents == 0
        assert hit.parent_id == "a::p0000"  # 子块自己记着父块 id，只是父块不在这份索引里
        assert hit.parent_text == ""
        assert hit.context_text == hit.body == "child 0 a"

    def test_chunk_without_parent_id_falls_back_to_child_body(self, store: SqliteStore):
        """父子切割之前建的块：`parent_id` 是空串，同样退回子块正文。"""
        _upsert(store, "a", 1, axis=0)
        hit = store.get_chunks(["a::0000"])["a::0000"]
        assert hit.parent_id == ""
        assert hit.parent_text == ""
        assert hit.context_text == "body 0 a"


class TestParentChildWrite:
    """父子两层的写入语义：覆盖按文档先删后写、删除与重建都要带上父块。"""

    def test_overwrite_replaces_parents(self, store: SqliteStore):
        """防的回归：同一 doc 第二次 upsert 后旧父块残留——子块会取到**过期的语境**
        （喂错上下文比喂空上下文更坏），而且残留行会越积越多。"""
        _upsert_parents(store, "a", ["旧父块甲", "旧父块乙"], axis=0)
        _upsert_parents(store, "a", ["新父块只有一个"], axis=0)

        assert store.stats().n_parents == 1
        hit = store.get_chunks(["a::0000"])["a::0000"]
        assert hit.parent_text == "新父块只有一个"
        assert hit.context_text == "新父块只有一个"

    def test_overwrite_without_parents_leaves_no_stale_context(self, store: SqliteStore):
        """`parents=None` 的重写表示"这次不提供父块"：旧父块必须一起清掉，
        否则新子块的 parent_id 会撞上旧父块，静默喂出**上一版**的语境。"""
        _upsert_parents(store, "a", ["上一版的父块"], axis=0)
        _upsert_parents(store, "a", ["这一版不传父块"], axis=0, write_parents=False)

        assert store.stats().n_parents == 0
        hit = store.get_chunks(["a::0000"])["a::0000"]
        assert hit.parent_text == ""
        assert hit.context_text == hit.body

    def test_delete_docs_removes_parents(self, store: SqliteStore):
        """防的回归：删文档只删子块，父块留在表里无限增长（`n_parents` 也会虚高）。"""
        _upsert_parents(store, "a", ["甲父块", "乙父块"], axis=0)
        _upsert_parents(store, "b", ["丙父块"], axis=1)

        assert store.delete_docs({"a.md"}) == 2  # 返回的仍是**子块**数，语义不变
        assert store.stats().n_parents == 1
        assert store.get_chunks(["b::0000"])["b::0000"].parent_text == "丙父块"

        # 同一批 parent_id 再写一次不许撞主键：这也是"删干净了"的另一种证法
        _upsert_parents(store, "a", ["甲父块"], axis=0)
        assert store.get_chunks(["a::0000"])["a::0000"].parent_text == "甲父块"

    def test_rebuild_from_scratch_clears_parents(self, store: SqliteStore):
        """防的回归：重建只清块与向量，父块表跨轮次累积，下一轮体检看到虚高的 `n_parents`。"""
        _upsert_parents(store, "a", ["甲父块", "乙父块"])
        store.rebuild_from_scratch()
        assert (store.stats().n_chunks, store.stats().n_parents) == (0, 0)

    def test_stats_reports_n_parents(self, store: SqliteStore):
        """`n_parents` 必须是真实读数：父子切割没生效时它是 0，不该等于 `n_chunks` 或恒定 0。"""
        _upsert_parents(store, "a", ["甲父块", "乙父块", "丙父块"], axis=0)
        _upsert_parents(store, "b", ["丁父块"], axis=1)
        st = store.stats()
        assert (st.n_chunks, st.n_parents) == (4, 4)

    def test_doc_texts_still_concatenates_only_chunks(self, store: SqliteStore):
        """`doc_texts` 刻意只拼子块：标注集自检问的是"内容有没有进**索引**"，
        把不进索引的父块拼进来会让"子块漏进索引"永远查不出来。"""
        _upsert_parents(store, "a", ["父块独有词 parentonlyword"])
        texts = store.doc_texts(["a"])
        assert "child 0 a" in texts["a"]
        assert "parentonlyword" not in texts["a"]


# --------------------------------------------------------------------- 老库升级
_LEGACY_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    doc_id       TEXT PRIMARY KEY,
    rel_path     TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    n_chunks     INTEGER NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE chunks (
    chunk_id       TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    title          TEXT NOT NULL,
    breadcrumb     TEXT NOT NULL,
    source_url     TEXT NOT NULL,
    body           TEXT NOT NULL,
    body_normalized TEXT NOT NULL DEFAULT '',
    content_hash   TEXT NOT NULL,
    n_chars        INTEGER NOT NULL,
    vector_row     INTEGER NOT NULL
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    body_normalized, title, breadcrumb,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""


def _make_legacy_db(db_path, vector_path) -> None:
    """造一个**父子切割之前**的库：没有 `parents` 表、`chunks` 也没有 `parent_id` 列。

    手写 schema 而不是用旧版代码：只有真正缺这两样东西，下面的用例才在测升级路径
    （函数开头会先断言这一点，防止以后有人顺手改了这里让用例变成空转）。
    """
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(_LEGACY_SCHEMA)
        con.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            [
                ("schema_version", "2"),
                ("embedding_model", "hashing-test"),
                ("embedding_dim", "32"),
                ("normalizer", "identity"),
                ("domain", ""),
                ("indexed_at", "2024-01-01T00:00:00"),
            ],
        )
        con.execute(
            "INSERT INTO documents(doc_id, rel_path, title, source_url, content_hash, n_chunks, updated_at)"
            " VALUES('a', 'a.md', 'a', 'u/a/', 'h-a', 1, '2024-01-01T00:00:00')"
        )
        # rowid 显式写死：外内容 FTS 表的索引项要指向同一行
        con.execute(
            "INSERT INTO chunks(rowid, chunk_id, doc_id, ordinal, title, breadcrumb, source_url,"
            " body, body_normalized, content_hash, n_chars, vector_row)"
            " VALUES(1, 'a::0000', 'a', 0, 'a', 'sec', 'u/a/', 'legacybody legacyword',"
            " 'legacybody legacyword', 'h-a-0', 21, 0)"
        )
        con.execute(
            "INSERT INTO chunks_fts(rowid, body_normalized, title, breadcrumb)"
            " VALUES(1, 'legacybody legacyword', 'a', 'sec')"
        )
        con.commit()
    finally:
        con.close()
    np.save(vector_path, np.zeros((1, 32), dtype=np.float32))


class TestLegacyIndexUpgrade:
    """平滑升级：换一次切块方式就要求所有人删库重建是不可接受的。老库必须**原地能读**，
    只是没有父块语境（`context_text` 退回子块正文）——那是完全可用的退化状态。"""

    def test_legacy_db_without_parent_layer_is_upgraded_in_place(self, tmp_path):
        db, npy = tmp_path / "legacy.db", tmp_path / "legacy.npy"
        _make_legacy_db(db, npy)
        # 先确认造出来的**真的**是老布局，否则这条用例会悄悄退化成"打开一个本来就有父块表的库"
        raw = sqlite3.connect(str(db))
        try:
            cols = {r[1] for r in raw.execute("PRAGMA table_info(chunks)")}
            tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            raw.close()
        assert "parent_id" not in cols and "parents" not in tables

        store = SqliteStore(db_path=db, vector_path=npy, dim=32, embedding_model="hashing-test")
        try:
            # 1) 不抛异常、老数据照旧读得到，只是没有父块语境
            hit = store.get_chunks(["a::0000"])["a::0000"]
            assert hit.body == "legacybody legacyword"
            assert hit.parent_id == ""
            assert hit.parent_text == ""
            assert hit.context_text == "legacybody legacyword"
            assert store.stats().n_parents == 0
            # 2) 全文路没被 `ALTER TABLE` 弄坏（外内容 FTS 表最容易在这里静默失配）
            assert store.keyword_search("legacyword", 5) == [("a::0000", 1)]
            assert store.verify_fts() == "ok"
            # 3) 补列与建表确已发生
            cols_now = {r["name"] for r in store._con.execute("PRAGMA table_info(chunks)")}  # noqa: SLF001
            assert "parent_id" in cols_now
            # 4) 升级后的库能接着写新格式（父子两层共存，不用重建）
            _upsert_parents(store, "b", ["升级后写入的父块"], axis=1)
            assert store.get_chunks(["b::0000"])["b::0000"].parent_text == "升级后写入的父块"
            assert store.stats().n_parents == 1
        finally:
            store.close()

    def test_reopening_upgraded_db_is_idempotent(self, tmp_path):
        """防的回归：升级逻辑不可重复执行（第二次打开时 `ALTER TABLE` 重复跑会抛
        `duplicate column name`）——"打开一次能用、打开两次就崩"是最糟的形态。"""
        db, npy = tmp_path / "legacy2.db", tmp_path / "legacy2.npy"
        _make_legacy_db(db, npy)
        for _ in range(3):
            s = SqliteStore(db_path=db, vector_path=npy, dim=32, embedding_model="hashing-test")
            try:
                assert s.get_chunks(["a::0000"])["a::0000"].parent_text == ""
            finally:
                s.close()
