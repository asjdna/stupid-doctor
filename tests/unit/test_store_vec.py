"""sqlite-vec 后端（默认后端）的专项测试：父子两层的持久化在这里必须和 `sqlite` 后端同构。

父块表与 `chunks.parent_id` 都落在基类的**普通表**里，子类只多管一张 `vec0` 虚表，所以本文件
守的是"换后端不换契约"：命中块带回父块正文、覆盖/删除/重建都不留父块残渣、老库原地能升。
任一实现单独掉了父子两层的某一步（比如重建时忘了清 `parents`），这里都会变红。"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from kubedocs.store.base import ChunkInput, DocInput, ParentInput
from kubedocs.store.sqlite_vec_store import SqliteVecStore

DIM = 32


def _vec(*xs: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    return v


@pytest.fixture
def vec_store(tmp_path) -> SqliteVecStore:
    s = SqliteVecStore(
        db_path=tmp_path / "v.db", vector_path=tmp_path / "v.npy", dim=DIM, embedding_model="hashing-test"
    )
    yield s
    s.close()


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
    """一个父块配一个子块；父块正文互不相同，便于断言取回时**不串味**。"""
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
    store: SqliteVecStore,
    doc_id: str,
    parent_bodies: list[str],
    *,
    axis: int = 0,
    write_parents: bool = True,
) -> None:
    d, chunks, parents = _parents_doc(doc_id, parent_bodies)
    store.upsert_docs(
        [d], chunks, {c.chunk_id: _vec(axis) for c in chunks}, parents=parents if write_parents else None
    )
    store.finalize()


def test_hit_carries_parent_id_and_parent_text(vec_store: SqliteVecStore):
    """默认后端也要给出父块语境：父子切割没有只在一个后端生效的道理。"""
    _upsert_parents(vec_store, "a", ["父块甲：完整小节正文"])
    hit = vec_store.get_chunks(["a::0000"])["a::0000"]
    assert hit.parent_id == "a::p0000"
    assert hit.parent_text == "父块甲：完整小节正文"
    assert hit.context_text == "父块甲：完整小节正文"
    assert hit.body == "child 0 a"
    # 向量在写入时就与块行同事务落库，父子两层不该影响这条
    assert vec_store.verify_vec() == "ok"


def test_batch_read_fills_each_parent_without_crosstalk(vec_store: SqliteVecStore):
    """防的回归：一次取多条命中时父块**串味**。单条命中的用例抓不到这类错位。"""
    _upsert_parents(vec_store, "a", ["甲父块正文", "乙父块正文", "丙父块正文"])
    hits = vec_store.get_chunks(["a::0002", "a::0000", "a::0001"])
    assert {cid: h.parent_text for cid, h in hits.items()} == {
        "a::0000": "甲父块正文",
        "a::0001": "乙父块正文",
        "a::0002": "丙父块正文",
    }


def test_missing_parents_fall_back_to_child_body(vec_store: SqliteVecStore):
    """老索引兜底：`parents=None` 写入的块取不到父块时，`context_text` 必须退回子块正文。"""
    _upsert_parents(vec_store, "a", ["父块没写进库"], write_parents=False)
    hit = vec_store.get_chunks(["a::0000"])["a::0000"]
    assert (vec_store.stats().n_parents, hit.parent_text) == (0, "")
    assert hit.context_text == hit.body == "child 0 a"


def test_overwrite_replaces_parents(vec_store: SqliteVecStore):
    """防的回归：覆盖同一 doc 时旧父块残留，子块取到过期语境。"""
    _upsert_parents(vec_store, "a", ["旧父块甲", "旧父块乙"], axis=0)
    _upsert_parents(vec_store, "a", ["新父块只有一个"], axis=0)
    assert vec_store.stats().n_parents == 1
    assert vec_store.get_chunks(["a::0000"])["a::0000"].context_text == "新父块只有一个"
    assert vec_store.verify_vec() == "ok"  # 覆盖没有在向量表里留下孤儿行


def test_delete_docs_removes_parents(vec_store: SqliteVecStore):
    """防的回归：子类覆写的 `_drop_doc_chunks`（先删向量行）漏掉父块，父块表只增不减。"""
    _upsert_parents(vec_store, "a", ["甲父块", "乙父块"], axis=0)
    _upsert_parents(vec_store, "b", ["丙父块"], axis=1)
    assert vec_store.delete_docs({"a.md"}) == 2  # 返回值仍是**子块**数，语义不变
    assert vec_store.stats().n_parents == 1
    assert vec_store.verify_vec() == "ok"
    assert vec_store.get_chunks(["b::0000"])["b::0000"].parent_text == "丙父块"
    # 同一批 parent_id 再写不许撞主键，反证父块真的删干净了
    _upsert_parents(vec_store, "a", ["甲父块"], axis=0)
    assert vec_store.get_chunks(["a::0000"])["a::0000"].parent_text == "甲父块"


def test_rebuild_from_scratch_clears_parents(vec_store: SqliteVecStore):
    """防的回归：子类覆写的重建只清块/向量/文档，`parents` 表跨轮次累积。"""
    _upsert_parents(vec_store, "a", ["甲父块", "乙父块"])
    vec_store.rebuild_from_scratch()
    assert (vec_store.stats().n_chunks, vec_store.stats().n_parents) == (0, 0)
    assert vec_store.verify_vec() == "ok"


def test_stats_reports_n_parents(vec_store: SqliteVecStore):
    """`n_parents` 必须是真实读数（父子切割没生效时是 0，而不是恒等于 `n_chunks`）。"""
    _upsert_parents(vec_store, "a", ["甲父块", "乙父块", "丙父块"], axis=0)
    _upsert_parents(vec_store, "b", ["丁父块"], axis=1)
    st = vec_store.stats()
    assert (st.n_chunks, st.n_parents) == (4, 4)


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
    """造一个父子切割之前的库（`sqlite` 后端写的旧索引）：没有 `parents` 表、没有 `parent_id`。"""
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(_LEGACY_SCHEMA)
        con.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            [
                ("schema_version", "2"),
                ("embedding_model", "hashing-test"),
                ("embedding_dim", str(DIM)),
                ("normalizer", "identity"),
                ("domain", ""),
                ("indexed_at", "2024-01-01T00:00:00"),
            ],
        )
        con.execute(
            "INSERT INTO documents(doc_id, rel_path, title, source_url, content_hash, n_chunks, updated_at)"
            " VALUES('a', 'a.md', 'a', 'u/a/', 'h-a', 1, '2024-01-01T00:00:00')"
        )
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
    np.save(vector_path, np.zeros((1, DIM), dtype=np.float32))


def test_legacy_db_upgraded_in_place_then_vec_migratable(tmp_path):
    """换默认后端后打开旧索引的第一现场：库是**父子切割之前**的布局。

    必须做到：原地补列补表、不抛异常；全文路与 `get_chunks` 立刻可用且 `parent_text` 为空
    （退化到子块正文，是合法的可用状态）；随后 `adopt_legacy_matrix()` 把向量搬进 `vec0` 后
    向量路也恢复，全程**不需要删库重建**。
    """
    db, npy = tmp_path / "legacy.db", tmp_path / "legacy.npy"
    _make_legacy_db(db, npy)
    # 确认造出来的**真的**是老布局，防止这条用例悄悄退化成"打开一个本来就有父块表的库"
    raw = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in raw.execute("PRAGMA table_info(chunks)")}
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        raw.close()
    assert "parent_id" not in cols and "parents" not in tables

    store = SqliteVecStore(db_path=db, vector_path=npy, dim=DIM, embedding_model="hashing-test")
    try:
        hit = store.get_chunks(["a::0000"])["a::0000"]
        assert (hit.body, hit.parent_id, hit.parent_text) == ("legacybody legacyword", "", "")
        assert hit.context_text == "legacybody legacyword"  # 不是空串
        assert store.keyword_search("legacyword", 5) == [("a::0000", 1)]
        assert store.stats().n_parents == 0
        # 补的表/列都在
        assert "parent_id" in {r["name"] for r in store._con.execute("PRAGMA table_info(chunks)")}  # noqa: SLF001
        # 向量路：迁移旧矩阵后恢复，且父子两层仍可继续写入
        assert store.adopt_legacy_matrix() == 1
        assert store.verify_vec() == "ok"
        _upsert_parents(store, "b", ["升级后写入的父块"], axis=1)
        assert store.get_chunks(["b::0000"])["b::0000"].parent_text == "升级后写入的父块"
        assert store.stats().n_parents == 1
    finally:
        store.close()
