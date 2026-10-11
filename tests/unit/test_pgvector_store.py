"""pgvector 生产实现的专项测试：**Postgres 不可达时整文件 skip，并写明原因**。

## 纪律

1. 连不上就 skip（`reason` 里带上探测到的具体错误），**不失败、也不假装跑过**；
   本文件里每一条断言都真的执行过一次才可能变绿；
2. schema 隔离：每个测试用一个 `test_pgvector_*` 的独立 schema，结束就 `DROP SCHEMA CASCADE`——
   测试不许污染真实索引（`k8s_docs` schema），也不许互相看见对方的数据；
3. 与 SQLite 实现对照：同一批语料在两个后端上跑，比较**可比较的部分**
   （向量路排序、命中集合、统计口径），把不可比较的部分（`ts_rank` 与 BM25 的排序）
   写清楚而不是含糊过去。

## 这一路的"未验证"是什么

起 Docker 之前这些用例整文件 skip，报告里写的是"未验证"而不是"通过"。起库之后
它们**真的跑起来了**，并暴露出两处断言本身的错误（`chunk_id` 逐位相等、
在**非池连接**上查 `hnsw.ef_search`），修法与原因见对应用例的 docstring——
这两处都是"从没执行过"才会一直存在的错，正是把它们跑起来的意义所在。
`test_store_backend.py` 里另有一组不依赖数据库的探测用例
（`probe()` 的错误路径），它们在任何环境下都会跑。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest

from kubedocs.config import Settings
from kubedocs.store.base import MissingVectorError, Store, StoreError, Vector
from kubedocs.store.pgvector_store import PgVectorStore, probe
from kubedocs.store.sqlite_vec_store import SqliteVecStore
from tests.unit.store_seed import DIM, make_seed, vec

_DSN = Settings().database_url
_PG_OK, _PG_WHY = probe(_DSN, timeout=3.0)


def _pg_search_available() -> bool:
    """这个实例能不能真上 BM25（**判据与 store 一致**：真的 `CREATE EXTENSION` 一次）。

    只看 `pg_available_extensions` 是不够的：pg_search 是 preload 型扩展，
    扩展文件在、`shared_preload_libraries` 里没有它时 `CREATE EXTENSION` 会失败。
    """
    if not _PG_OK:
        return False
    try:
        with psycopg.connect(_DSN, connect_timeout=5) as conn:
            row = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'").fetchone()
            if row is None:
                return False
            with conn.transaction():  # 嵌套 = SAVEPOINT，失败不毒化外层
                conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
        return True
    except Exception:  # noqa: BLE001 - 探测就是要把失败转成"不可用"
        return False


_PG_SEARCH_OK = _pg_search_available()


def _fts_is_healthy(store: PgVectorStore) -> bool:
    """`verify_fts()` 的三态契约：BM25 在位 = `ok`；降级 = `DEGRADED:` 开头（不是 `ok`）。

    ⚠️ 不能把两条都写成"返回 ok"：降级时**必须**返回非 ok，否则
    `cli stats --verify` 会对一个跑 `ts_rank` 的库说"一切正常"（见 ADR-0004 第十节）。
    """
    got = store.verify_fts()
    if store.fts_engine == "pg_search":
        return got == "ok"
    return got.startswith("DEGRADED")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _PG_OK,
        reason=f"Postgres/pgvector 不可达，本文件未验证：{_PG_WHY}（DSN={_DSN}）",
    ),
]


@pytest.fixture
def schema() -> str:
    """每个用例一个独立 schema：名字带随机后缀，避免并发/残留互相影响。"""
    return f"test_pgvector_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def pg(schema: str) -> Iterator[PgVectorStore]:
    store = PgVectorStore(
        _DSN, schema=schema, dim=DIM, embedding_model="hashing-test", domain=schema, pool_min=1, pool_max=4
    )
    try:
        yield store
    finally:
        store.close()
        with psycopg.connect(_DSN, connect_timeout=5) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def _seed_pg(store: PgVectorStore) -> None:
    seed = make_seed()
    store.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    store.finalize()


def _sims(ids: list[str], vectors: dict[str, Vector], q: np.ndarray) -> list[float]:
    """把后端返回的 chunk_id 序列换算成**余弦相似度**序列（用种子向量自己算）。

    为什么必须这样比而不是比 chunk_id 序列：两个后端都是**精确**近邻——
    `vec0` 是穷举扫描，而 pgvector 在小索引上的 HNSW 与关掉索引的精确检索逐位一致
    （见 `test_hnsw_matches_exact_search_on_small_index`）——所以差别只可能来自
    **并列时谁排前面**，而那不属于契约：`vec0` 按行序、HNSW 按图遍历序，两者都不会
    也不需要对齐。本种子语料在第 4 位上**必然**并列（`e_a` 对 `beta::0000` /
    `beta::0001` / `gamma::0000` 的相似度都是 0.0，而只剩 2 个位置），
    因此"chunk_id 逐位相等"这个断言在数学上就不可能稳定成立——它是本文件
    第一次真正跑起来时暴露出来的两处断言错误之一。

    可比、且必须相等的是**相似度序列**：它同时证明了两件事——名次按同一条度量
    单调降序，且岔开的位置上分数确实相等（并列），而不是"某个后端少召回了"。
    """
    qn = np.asarray(q, dtype=np.float64)
    qn = qn / np.linalg.norm(qn)
    out: list[float] = []
    for cid in ids:
        v = np.asarray(vectors[cid], dtype=np.float64)
        out.append(round(float(np.dot(v / np.linalg.norm(v), qn)), 6))
    return out


# --------------------------------------------------------------------------- schema
def test_schema_ddl_creates_tables_indexes_and_extension(pg: PgVectorStore) -> None:
    """建表/建索引是自动的（`finalize` 与 `upsert` 不必先跑初始化命令）。"""
    with psycopg.connect(_DSN) as conn:
        tables = {
            r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = %s", (pg.schema,))
        }
        indexes = {
            r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE schemaname = %s", (pg.schema,))
        }
        has_ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert {"documents", "chunks", "meta"} <= tables
    assert {"idx_chunks_tsv", "idx_chunks_embedding", "idx_chunks_doc", "idx_chunks_hash"} <= indexes
    assert has_ext is not None, "pgvector 扩展必须已启用"
    # BM25 索引**只在 pg_search 可用时**才建（见 ADR-0004 第十节）：
    # 扩展装不上时不建索引是刻意的——那时关键字路走带 DEGRADED 标记的 ts_rank，
    # 而不是"建一个用不上的索引"或"假装一切正常"。
    if _PG_SEARCH_OK:
        assert "idx_chunks_bm25" in indexes
    else:
        assert "idx_chunks_bm25" not in indexes
    assert _fts_is_healthy(pg), f"verify_fts() = {pg.verify_fts()!r}（引擎 {pg.fts_engine}）"
    assert pg.verify_hnsw() == "ok"


def test_fts_engine_is_detected_and_recorded(pg: PgVectorStore) -> None:
    """引擎判定必须留下痕迹：`meta.fts_engine` + `verify_fts()` 两处都要对上。

    这是"不许静默假装"那条纪律的**可执行版本**：库里没有 pg_search 时，
    降级必须写进元信息、并被自检报成 `DEGRADED`（不是 `ok`）。
    """
    label = pg.meta_dump().get("fts_engine", "")
    if _PG_SEARCH_OK:
        assert pg.fts_engine == "pg_search"
        assert label.startswith("pg_search-bm25"), label
        assert pg.verify_fts() == "ok"
    else:
        assert pg.fts_engine == "ts_rank"
        assert label.startswith("ts_rank-DEGRADED"), label
        assert pg.verify_fts().startswith("DEGRADED")


def test_bm25_index_definition(pg: PgVectorStore) -> None:
    """BM25 索引的**结构**必须真的落进索引定义，而不是只写在常量里。

    钉三件事：访问方法 `paradedb`、分词器（`chinese_compatible` + `stemmer=english`）、
    以及表达式字段的 `alias=`（不带它建不出索引，见 ADR-0004 第十节）。
    """
    if not _PG_SEARCH_OK:
        pytest.skip("这个 Postgres 没有可用的 pg_search：BM25 索引这一路未验证")
    with psycopg.connect(_DSN) as conn:
        row = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = 'idx_chunks_bm25'",
            (pg.schema,),
        ).fetchone()
    assert row is not None, "缺少 BM25 索引 idx_chunks_bm25"
    ddl = row[0]
    assert "USING paradedb" in ddl
    assert "pdb.chinese_compatible" in ddl and "stemmer=english" in ddl
    assert "alias=fts_text" in ddl
    # 三个字段都必须在同一个表达式里（单字段拼接，理由见 ADR-0004 第十节）
    for col in ("title", "breadcrumb", "body_normalized"):
        assert col in ddl, f"BM25 索引里没有 {col}"


def test_explicit_fts_engine_modes(schema: str) -> None:
    """显式指定引擎时的两种语义：要 BM25 而装不上 → 报错；显式要 ts_rank → 允许但带标记。"""
    kwargs = {
        "schema": schema,
        "dim": DIM,
        "embedding_model": "hashing-test",
        "domain": schema,
        "pool_min": 1,
        "pool_max": 2,
    }
    if _PG_SEARCH_OK:
        legacy = PgVectorStore(_DSN, fts_engine="ts_rank", **kwargs)
        try:
            assert legacy.fts_engine == "ts_rank"
            assert legacy.verify_fts().startswith("DEGRADED")
        finally:
            legacy.close()
    else:
        with pytest.raises(StoreError):
            PgVectorStore(_DSN, fts_engine="pg_search", **kwargs)


def test_tsv_is_a_generated_column(pg: PgVectorStore) -> None:
    """`tsv` 是生成列：全文索引不可能与正文脱钩（相对 FTS5 外内容表的结构性优势）。"""
    with psycopg.connect(_DSN) as conn:
        row = conn.execute(
            "SELECT is_generated, generation_expression FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'chunks' AND column_name = 'tsv'",
            (pg.schema,),
        ).fetchone()
    assert row is not None and row[0] == "ALWAYS"
    assert "to_tsvector" in row[1]


def test_hnsw_parameters_are_the_configured_ones(pg: PgVectorStore) -> None:
    """HNSW 的 `m` / `ef_construction` 必须真的落进索引定义，而不是只写在配置里。

    参数写错却不报错是这类改造最隐蔽的失败方式：索引建出来了、也更快了，
    只是召回悄悄掉了一截。
    """
    with psycopg.connect(_DSN) as conn:
        ddl = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = 'idx_chunks_embedding'",
            (pg.schema,),
        ).fetchone()[0]
    assert "hnsw" in ddl and "vector_cosine_ops" in ddl
    assert f"m='{pg.hnsw_m}'" in ddl or f"m={pg.hnsw_m}" in ddl
    assert (
        f"ef_construction='{pg.hnsw_ef_construction}'" in ddl
        or f"ef_construction={pg.hnsw_ef_construction}" in ddl
    )
    # ef_search 是**查询期**参数，由池的 `configure` 逐连接下发。
    # ⚠️ 因此必须在**池里取出来的连接**上查：`psycopg.connect(_DSN)` 拿到的是
    # 一条池外的裸连接，它从没走过 `configure`，`SHOW` 只会给出该 GUC 的默认值 40，
    # 断言 `== 100` 永远不成立。这正是本文件第一次真正跑起来时报出的第二处断言错误
    # ——它验证的必须是"商店（store）实际发给查询的连接"，而不是随便一条连接。
    with pg._pool.connection() as pooled:
        ef_search = pooled.execute("SHOW hnsw.ef_search").fetchone()[0]
    assert ef_search == str(pg.hnsw_ef_search)


# --------------------------------------------------------------------------- 与 SQLite 对照
def test_dense_search_matches_sqlite_vec_on_same_input(pg: PgVectorStore, tmp_path: Path) -> None:
    """核心对照：同一批语料下，pgvector 的向量路必须与 sqlite-vec 给出**同一个名次语义**。

    比的是**相似度序列**而不是 chunk_id 序列（理由见 `_sims`）：两个后端都是精确近邻，
    并列名次的先后不属于契约。这样断言既不允许"某个后端少召回"，
    也不强求 HNSW 与 vec0 在并列处对齐。
    """
    _seed_pg(pg)
    sqlite_store = SqliteVecStore(
        db_path=tmp_path / "s.db", vector_path=tmp_path / "s.npy", dim=DIM, embedding_model="hashing-test"
    )
    try:
        seed = make_seed()
        sqlite_store.upsert_docs(seed.docs, seed.chunks, seed.vectors)
        sqlite_store.finalize()
        for q in (vec(a=1.0), vec(b=1.0), vec(c=1.0)):
            got = [c for c, _ in pg.dense_search(q, 4)]
            want = [c for c, _ in sqlite_store.dense_search(q, 4)]
            got_sims, want_sims = _sims(got, seed.vectors, q), _sims(want, seed.vectors, q)
            for sims in (got_sims, want_sims):
                assert sims == sorted(sims, reverse=True), "名次必须按相似度降序"
            assert got_sims == want_sims, "两个后端的相似度序列不一致：向量路语义变了"
    finally:
        sqlite_store.close()


def test_hnsw_matches_exact_search_on_small_index(pg: PgVectorStore) -> None:
    """小索引上 HNSW 必须与**关掉索引**的精确检索给出完全相同的 top-k。

    这是"召回归因"的基准：大索引上如果两者不一致，差异就来自 HNSW 的近似性，
    而不是数据或语义变了。
    """
    _seed_pg(pg)
    for q in (vec(a=1.0), vec(a=0.8, b=0.6), vec(c=1.0)):
        assert pg.dense_search(q, 3) == pg.dense_search_exact(q, 3)


def test_keyword_hit_set_matches_sqlite(pg: PgVectorStore, tmp_path: Path) -> None:
    """全文路只保证**命中集合**一致，排序允许不同。

    SQLite 用 FTS5 的 `bm25()`，pgvector 用 pg_search（tantivy）的 BM25——**两份独立实现**，
    分词器也不同（`porter unicode61` vs `chinese_compatible('stemmer=english')`），
    所以"相同命中集合下名次可能不同"是真实差异，不是 bug（见 ADR-0004 第十节）。
    降级到 `ts_rank` 时这条断言同样成立（命中集合由同一套词元规则决定）。
    """
    _seed_pg(pg)
    sqlite_store = SqliteVecStore(
        db_path=tmp_path / "k.db", vector_path=tmp_path / "k.npy", dim=DIM, embedding_model="hashing-test"
    )
    try:
        seed = make_seed()
        sqlite_store.upsert_docs(seed.docs, seed.chunks, seed.vectors)
        sqlite_store.finalize()
        for q in ("kubelet", "crashloopbackoff", "resourcequota", "eviction threshold"):
            pg_hits = {c for c, _ in pg.keyword_search(q, 5)}
            sq_hits = {c for c, _ in sqlite_store.keyword_search(q, 5)}
            assert pg_hits == sq_hits, f"查询 {q!r} 的命中集合不一致"
            if pg_hits:
                assert [r for _, r in pg.keyword_search(q, 5)] == list(range(1, len(pg_hits) + 1))
    finally:
        sqlite_store.close()


def test_stats_and_doc_texts_match_sqlite(pg: PgVectorStore, tmp_path: Path) -> None:
    """统计口径与文档正文拼接必须与 SQLite 侧一致（标注集自检依赖后者）。"""
    _seed_pg(pg)
    sqlite_store = SqliteVecStore(
        db_path=tmp_path / "t.db", vector_path=tmp_path / "t.npy", dim=DIM, embedding_model="hashing-test"
    )
    try:
        seed = make_seed()
        sqlite_store.upsert_docs(seed.docs, seed.chunks, seed.vectors)
        sqlite_store.finalize()
        a, b = pg.stats(), sqlite_store.stats()
        assert (a.n_docs, a.n_chunks, a.total_chars, a.n_with_code) == (
            b.n_docs,
            b.n_chunks,
            b.total_chars,
            b.n_with_code,
        )
        assert pg.doc_texts(["alpha", "beta", "missing"]) == sqlite_store.doc_texts(
            ["alpha", "beta", "missing"]
        )
        assert pg.existing_doc_ids() == sqlite_store.existing_doc_ids()
        assert pg.existing_doc_hashes() == sqlite_store.existing_doc_hashes()
        assert {c: sorted(r.__dict__) for c, r in pg.get_chunks(["alpha::0000"]).items()}
        assert pg.get_chunks(["alpha::0000"]) == sqlite_store.get_chunks(["alpha::0000"])
    finally:
        sqlite_store.close()


def test_meta_guards_reject_model_and_dim_mismatch(pg: PgVectorStore) -> None:
    """元信息闸门：维度/模型/归一化器不一致必须在**打开时**拒绝，而不是查询时出怪结果。"""
    from kubedocs.store.base import StoreError

    with pytest.raises(StoreError, match="维度"):
        PgVectorStore(
            _DSN, schema=pg.schema, dim=DIM * 2, embedding_model="hashing-test", domain=pg.schema, pool_max=2
        )
    with pytest.raises(StoreError, match="嵌入模型"):
        PgVectorStore(
            _DSN, schema=pg.schema, dim=DIM, embedding_model="other-model", domain=pg.schema, pool_max=2
        )


def test_rebuild_delete_and_missing_vector(pg: PgVectorStore) -> None:
    """重建/删除/缺向量拒绝——三条写入语义在 pgvector 上必须与 SQLite 一致。"""
    _seed_pg(pg)
    assert pg.stats().n_chunks == 5
    assert pg.delete_docs({"alpha.md"}) == 2
    pg.finalize()
    assert pg.stats().n_chunks == 3
    assert pg.keyword_search("kubelet", 5) == []
    assert "alpha::0000" not in [c for c, _ in pg.dense_search(vec(a=1.0), 5)]

    with pytest.raises(MissingVectorError):
        seed = make_seed()
        pg.upsert_docs(seed.docs[:1], seed.chunks[:1], {})

    pg.rebuild_from_scratch()
    assert pg.stats().n_chunks == 0
    assert pg.dense_search(vec(a=1.0), 5) == []
    _seed_pg(pg)
    assert pg.stats().n_chunks == 5
    assert _fts_is_healthy(pg), f"verify_fts() = {pg.verify_fts()!r}（引擎 {pg.fts_engine}）"


def test_existing_hash_to_vector_roundtrip(pg: PgVectorStore) -> None:
    """向量往返必须逐位相等：`register_vector` 的适配器不许悄悄改精度。"""
    _seed_pg(pg)
    got = pg.existing_hash_to_vector({"h-alpha-0", "nope"})
    assert set(got) == {"h-alpha-0"}
    assert got["h-alpha-0"].dtype == np.float32
    assert np.array_equal(got["h-alpha-0"], vec(a=1.0))


def test_schema_name_is_validated(pg: PgVectorStore) -> None:
    """非法 schema 名（注入面）必须被拒绝——表名无法参数化，只能靠白名单。"""
    from kubedocs.store.base import StoreError

    with pytest.raises(StoreError, match="非法的 schema"):
        PgVectorStore(_DSN, schema="public; DROP TABLE x", dim=DIM)


def test_store_protocol_is_satisfied(pg: PgVectorStore) -> None:
    """运行期也满足 `Store` 协议（下游有 `isinstance` 风格的用法）。"""
    assert isinstance(pg, Store)
