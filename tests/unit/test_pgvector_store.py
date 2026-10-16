"""pgvector 后端的专项测试：**Postgres/pgvector 不可达时整文件 skip，并写明探测到的原因**。

前置条件：一个装了 pgvector 扩展的 Postgres，DSN 取自 `Settings().database_url`。连不上就
skip；每个用例用独立的 `test_pgvector_*` schema，结束即 `DROP SCHEMA CASCADE`，既不污染
真实索引，也不互相看见对方的数据。

守的契约：建表/建索引自动完成；全文引擎判定写进元信息，并由自检如实报成 `ok` 或 `DEGRADED`；
HNSW 参数落进索引定义、`ef_search` 下发到查询连接；向量路排序、统计口径与正文拼接与 SQLite
一致（并列次序除外）；非法 schema 名、维度/模型不符、缺向量都必须被拒绝。

父子两层另有一组契约：父块**不进任何索引**（`keyword_search` / `dense_search` 只可能回子块）、
`get_chunks` 按 `parent_id` **批量**带回父块正文、覆盖与删除都不留旧父块、`parents=None` 时
读路径退回子块正文；`ensure_schema()` 反复调用幂等，且老库（没有 `parent_id` / `parents` 表）
能原地升级而不是被拒绝。父块语料用本文件自己的 `_make_parent_seed`——共享的 `store_seed`
必须保持"没有父块"的形态，另一批后端一致性测试正靠它。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg import sql

from kubedocs.config import Settings
from kubedocs.store.base import (
    ChunkInput,
    DocInput,
    MissingVectorError,
    ParentInput,
    Store,
    StoreError,
    Vector,
)
from kubedocs.store.pgvector_store import PgVectorStore, probe
from kubedocs.store.sqlite_vec_store import SqliteVecStore
from tests.unit.store_seed import DIM, make_seed, vec

_DSN = Settings().database_url
_PG_OK, _PG_WHY = probe(_DSN, timeout=3.0)


def _pg_search_available() -> bool:
    """这个实例能不能真上 BM25：判据与 store 一致，真的试一次 `CREATE EXTENSION`。

    只查 `pg_available_extensions` 不够：pg_search 是 preload 型扩展，扩展文件在、但
    `shared_preload_libraries` 里没有它时 `CREATE EXTENSION` 仍会失败。
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
    """`verify_fts()` 的三态契约：BM25 在位算 `ok`，降级到 `ts_rank` 必须是 `DEGRADED:` 开头。

    降级时**必须**返回非 ok，否则自检会对一个跑 `ts_rank` 的库说"一切正常"。
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
def schema() -> Iterator[str]:
    """每个用例一个独立 schema：名字带随机后缀，避免并发/残留互相影响。

    `DROP` 放在**这个 fixture** 里而不是 `pg` 里：有些用例（显式引擎模式、幂等升级）
    自己直接构造 `PgVectorStore`、不用 `pg`，清理写在 `pg` 里就会把它们建的 schema
    永久留在 `pg_namespace` 里——下一眼看上去像真实数据。
    """
    name = f"test_pgvector_{uuid.uuid4().hex[:10]}"
    try:
        yield name
    finally:
        with psycopg.connect(_DSN, connect_timeout=5) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')


@pytest.fixture
def pg(schema: str) -> Iterator[PgVectorStore]:
    store = PgVectorStore(
        _DSN, schema=schema, dim=DIM, embedding_model="hashing-test", domain=schema, pool_min=1, pool_max=4
    )
    try:
        yield store
    finally:
        # 先关池再让 `schema` fixture 去 DROP：池里还挂着连接时删 schema 会把
        # 那条连接上的后续操作变成"relation does not exist"。
        store.close()


def _seed_pg(store: PgVectorStore) -> None:
    seed = make_seed()
    store.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    store.finalize()


def _sims(ids: list[str], vectors: dict[str, Vector], q: np.ndarray) -> list[float]:
    """把后端返回的 chunk_id 序列换算成**余弦相似度**序列（用种子向量自己算）。

    两个后端都是精确近邻（`vec0` 穷举扫描，pgvector 在小索引上的 HNSW 与精确检索一致），
    差别只可能来自**并列时谁排前面**——`vec0` 按行序、HNSW 按图遍历序，不属于契约。
    本种子语料在第 4 位上必然并列（`e_a` 对 `beta::0000` / `beta::0001` / `gamma::0000`
    都是 0.0 分而只剩 2 个位置），"chunk_id 逐位相等"因此数学上就不可能稳定成立。

    相似度序列才是可比且必须相等的：名次按同一条度量单调降序，且岔开的位置上分数确实
    相等（并列），而不是某个后端少召回了。
    """
    qn = np.asarray(q, dtype=np.float64)
    qn = qn / np.linalg.norm(qn)
    out: list[float] = []
    for cid in ids:
        v = np.asarray(vectors[cid], dtype=np.float64)
        out.append(round(float(np.dot(v / np.linalg.norm(v), qn)), 6))
    return out


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
    # BM25 索引只在 pg_search 可用时才建：装不上时关键字路走带 DEGRADED 标记的 ts_rank，
    # 而不是建一个用不上的索引。
    if _PG_SEARCH_OK:
        assert "idx_chunks_bm25" in indexes
    else:
        assert "idx_chunks_bm25" not in indexes
    assert _fts_is_healthy(pg), f"verify_fts() = {pg.verify_fts()!r}（引擎 {pg.fts_engine}）"
    assert pg.verify_hnsw() == "ok"


def test_fts_engine_is_detected_and_recorded(pg: PgVectorStore) -> None:
    """引擎判定必须留下痕迹：`meta.fts_engine` 与 `verify_fts()` 两处都要对上。

    库里没有 pg_search 时，降级必须写进元信息、并被自检报成 `DEGRADED` 而不是 `ok`。
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
    """BM25 索引的结构必须真的落进索引定义，而不是只写在常量里。

    钉三件事：访问方法 `paradedb`、分词器（`chinese_compatible` + `stemmer=english`）、
    以及表达式字段的 `alias=`——不带它建不出索引。
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
    # 三个字段都必须在同一个表达式里拼接成单字段
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

    参数写错却不报错时，索引照样建出来、也更快，只是召回悄悄掉了一截。
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
    # ef_search 是查询期参数，由连接池的 `configure` 逐连接下发，因此必须在**从池里取出的
    # 连接**上查：池外的裸连接没走过 `configure`，`SHOW` 只会给出该 GUC 的默认值。
    with pg._pool.connection() as pooled:
        ef_search = pooled.execute("SHOW hnsw.ef_search").fetchone()[0]
    assert ef_search == str(pg.hnsw_ef_search)


def test_dense_search_matches_sqlite_vec_on_same_input(pg: PgVectorStore, tmp_path: Path) -> None:
    """核心对照：同一批语料下，pgvector 的向量路必须与 sqlite-vec 给出同一个名次语义。

    比相似度序列而不是 chunk_id 序列（见 `_sims`）：两个后端都是精确近邻，并列名次的先后
    不属于契约。这条断言既不允许某个后端少召回，也不强求 HNSW 与 `vec0` 在并列处对齐。
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

    这是召回归因的基准：大索引上两者不一致时，差异来自 HNSW 的近似性，而不是数据变了。
    """
    _seed_pg(pg)
    for q in (vec(a=1.0), vec(a=0.8, b=0.6), vec(c=1.0)):
        assert pg.dense_search(q, 3) == pg.dense_search_exact(q, 3)


def test_keyword_hit_set_matches_sqlite(pg: PgVectorStore, tmp_path: Path) -> None:
    """全文路只保证**命中集合**一致，排序允许不同。

    SQLite 用 FTS5 的 `bm25()`，pgvector 用 pg_search（tantivy）的 BM25：两份独立实现，
    分词器也不同（`porter unicode61` vs `chinese_compatible('stemmer=english')`），所以
    相同命中集合下名次不同是真实差异而不是缺陷。降级到 `ts_rank` 时命中集合同样由这套
    词元规则决定，断言仍成立。
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


# ------------------------------------------------------------------ 父子两层
_PARENT_MARKER = "zxqparentonly"
"""只写进**父块**正文的标记词，子块正文里没有它。

"父块不许进索引"这条契约需要一个可判定的证据：拿这个词查关键字路，命中就必须为空。
父块正文里另外放一份与子块完全不同的句子，用来区分"取回来的是父块"和"取回来的是子块"。
"""


def _parent(doc_id: str, ordinal: int, body: str, n_children: int = 1) -> ParentInput:
    return ParentInput(
        parent_id=f"{doc_id}::p{ordinal:04d}",
        doc_id=doc_id,
        ordinal=ordinal,
        title=doc_id,
        breadcrumb="sec",
        source_url=f"https://kubernetes.io/docs/{doc_id}/",
        body=body,
        n_chars=len(body),
        content_hash=f"hp-{doc_id}-{ordinal}",
        location=f"L{ordinal}",
        n_children=n_children,
    )


def _make_parent_seed() -> tuple[list[DocInput], list[ChunkInput], dict[str, Vector], list[ParentInput]]:
    """两篇文档 / 三个父块 / 四个子块；父块与子块正文**刻意完全不同**。

    子块（进索引）：alpha 两段讲 kubelet 驱逐、一段讲 ResourceQuota，beta 一段讲退避；
    父块（不进索引）：`alpha::p0000` 覆盖前两段子块，`alpha::p0001` 覆盖第三段，
    `beta::p0000` 覆盖 beta 那段——所以"两个子块共享一个父块"与"不同父块互不串味"
    两种情形都在同一批语料里。
    """
    specs = [
        ("alpha", 0, "kubelet eviction hard threshold", 0, vec(a=1.0)),
        ("alpha", 1, "kubelet eviction soft threshold", 0, vec(a=0.8, b=0.6)),
        ("alpha", 2, "resourcequota namespace limitrange", 1, vec(a=0.6, c=0.8)),
        ("beta", 0, "crashloopbackoff restart policy", 0, vec(b=1.0)),
    ]
    docs: dict[str, DocInput] = {}
    chunks: list[ChunkInput] = []
    vectors: dict[str, Vector] = {}
    for doc_id, ordinal, body, parent_ordinal, vector in specs:
        cid = f"{doc_id}::{ordinal:04d}"
        pid = f"{doc_id}::p{parent_ordinal:04d}"
        docs.setdefault(
            doc_id,
            DocInput(
                doc_id=doc_id,
                rel_path=f"{doc_id}.md",
                title=doc_id,
                source_url=f"https://kubernetes.io/docs/{doc_id}/",
                content_hash=f"h-{doc_id}",
            ),
        )
        chunks.append(
            ChunkInput(
                chunk_id=cid,
                doc_id=doc_id,
                ordinal=ordinal,
                title=doc_id,
                breadcrumb="sec",
                source_url=f"https://kubernetes.io/docs/{doc_id}/",
                body=body,
                content_hash=f"h-{doc_id}-{ordinal}",
                n_chars=len(body),
                parent_id=pid,
            )
        )
        vectors[cid] = vector
    parents = [
        _parent("alpha", 0, f"{_PARENT_MARKER} 这一整节讲 kubelet 驱逐的硬阈值与软阈值", n_children=2),
        _parent("alpha", 1, f"{_PARENT_MARKER} 这一整节讲 ResourceQuota 与 LimitRange", n_children=1),
        _parent("beta", 0, f"{_PARENT_MARKER} 这一整节讲 CrashLoopBackOff 的指数退避", n_children=1),
    ]
    return list(docs.values()), chunks, vectors, parents


def _seed_parents(store: PgVectorStore) -> list[ParentInput]:
    docs, chunks, vectors, parents = _make_parent_seed()
    store.upsert_docs(docs, chunks, vectors, parents)
    store.finalize()
    return parents


def _raw_count(schema: str, table: str, where: sql.SQL | None = None, params: tuple[object, ...] = ()) -> int:
    """直接数库里的行——不经过 store 的方法，避免"用被测代码验证被测代码"。

    表名同样交给 `sql.Identifier`：测试里的动态 SQL 也走驱动提供的引用规则，
    免得"生产代码不许拼表名、测试里可以"变成一条没人遵守的双标。
    """
    query = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    if where is not None:
        query = query + sql.SQL(" ") + where
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        return int(conn.execute(query, params).fetchone()[0])


def test_parent_text_is_returned_and_wins_context_text(pg: PgVectorStore) -> None:
    """命中子块后必须带回**父块完整正文**，且 `context_text` 取父块而不是子块。

    这条断言防的是"父子切割名存实亡"：父块写了却没人读回来，模型拿到的仍是 300 字的
    子块——检索指标看不出来，生成质量却掉回切割前。
    """
    parents = _seed_parents(pg)
    hit = pg.get_chunks(["alpha::0000"])["alpha::0000"]
    want = next(p for p in parents if p.parent_id == "alpha::p0000")
    assert hit.parent_id == "alpha::p0000"
    assert hit.parent_text == want.body
    assert hit.context_text == want.body, "context_text 必须是父块正文"
    assert hit.body == "kubelet eviction hard threshold", "子块正文不能被父块覆盖"
    assert len(hit.context_text) > len(hit.body)


def test_multiple_hits_do_not_bleed_parent_text(pg: PgVectorStore) -> None:
    """一次取回多个命中时，每个命中的 `parent_text` 必须是**自己那个**父块的正文。

    这条断言防的是"父块串味"：批量取父块时把结果按位置错配（或干脆共用最后一条），
    表现为答案引用了别的章节的内容——比取不到父块更难发现。
    """
    parents = _seed_parents(pg)
    want = {p.parent_id: p.body for p in parents}
    hits = pg.get_chunks(["alpha::0000", "alpha::0002", "beta::0000"])
    assert hits["alpha::0000"].parent_id == "alpha::p0000"
    assert hits["alpha::0002"].parent_id == "alpha::p0001"
    assert hits["beta::0000"].parent_id == "beta::p0000"
    for cid, hit in hits.items():
        assert hit.parent_text == want[hit.parent_id], f"{cid} 的父块正文不是它自己的"
    assert len({h.parent_text for h in hits.values()}) == 3, "三个命中应当来自三个不同的父块"
    # 同一个父块被两个子块共享时，两条命中的父块正文必须相同
    both = pg.get_chunks(["alpha::0000", "alpha::0001"])
    assert both["alpha::0000"].parent_text == both["alpha::0001"].parent_text == want["alpha::p0000"]


def test_missing_parent_falls_back_to_child_body(pg: PgVectorStore) -> None:
    """`parents=None`（旧调用方）或父块行缺失时：`parent_text == ""` 且语境退回子块。

    这条断言防的是"没有父块 = 喂给模型的上下文是空的"这种回归——那会让旧索引上的
    每一个回答都凭空丢掉正文，而 `context_text` 的存在正是为了不让人直接用
    `parent_text`。
    """
    _seed_parents(pg)
    # 只重写 alpha，但**不提供父块**：子块的 parent_id 仍写着 alpha::p0000，库里却没有那行。
    # （覆盖清理由 `docs` 驱动，所以这里必须照 ingest 的做法把该篇的 DocInput 一起传——
    # 传空的 docs 会变成"往已有块上再插一遍"，那是调用方的用法错误，不是被覆盖语义。）
    docs, chunks, vectors, _ = _make_parent_seed()
    alpha_docs = [d for d in docs if d.doc_id == "alpha"]
    alpha = [c for c in chunks if c.doc_id == "alpha"]
    pg.upsert_docs(alpha_docs, alpha, {c.chunk_id: vectors[c.chunk_id] for c in alpha}, None)
    pg.finalize()

    hit = pg.get_chunks(["alpha::0000"])["alpha::0000"]
    assert hit.parent_id == "alpha::p0000", "子块仍记得自己的父块 id（只是那行不存在）"
    assert hit.parent_text == ""
    assert hit.context_text == hit.body == "kubelet eviction hard threshold"

    # 完全没有父块的老式语料（parent_id 为空串）走同一条退路
    seed = make_seed()
    pg.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    pg.finalize()
    legacy = pg.get_chunks(["gamma::0000"])["gamma::0000"]
    assert legacy.parent_id == "" and legacy.parent_text == ""
    assert legacy.context_text == legacy.body


def test_second_upsert_leaves_no_stale_parents(pg: PgVectorStore) -> None:
    """同一篇二次 upsert 后，旧父块必须整批消失，同 id 的父块正文要被新正文替换。

    这条断言防的是"覆盖语义只覆盖子块"：旧父块留在库里时，子块的 parent_id 会取回
    **上一版**的语境，答案里出现已经不存在的段落，且 `n_parents` 一路虚高。
    """
    _seed_parents(pg)
    assert pg.stats().n_parents == 3

    docs, chunks, vectors, _ = _make_parent_seed()
    alpha_docs = [d for d in docs if d.doc_id == "alpha"]
    alpha = [c for c in chunks if c.doc_id == "alpha"]  # parent_id 仍指向 p0000 与 p0001
    new_p0 = _parent("alpha", 0, f"{_PARENT_MARKER} 改写后的 alpha 小节", n_children=2)
    pg.upsert_docs(alpha_docs, alpha, {c.chunk_id: vectors[c.chunk_id] for c in alpha}, [new_p0])
    pg.finalize()

    assert _raw_count(pg.schema, "parents", sql.SQL("WHERE doc_id = %s"), ("alpha",)) == 1, "旧父块必须被清掉"
    assert pg.stats().n_parents == 2, "只剩 alpha 的新 p0000 与 beta 的 p0000"
    rewritten = pg.get_chunks(["alpha::0000"])["alpha::0000"]
    assert rewritten.parent_text == new_p0.body, "同 id 的父块必须被新正文替换"
    orphan = pg.get_chunks(["alpha::0002"])["alpha::0002"]
    assert orphan.parent_id == "alpha::p0001" and orphan.parent_text == ""
    assert orphan.context_text == orphan.body


def test_delete_docs_removes_parents_too(pg: PgVectorStore) -> None:
    """删一篇文档时它的父块一起删；别的文档的父块不受影响，返回语义仍是块数。

    这条断言防的是"父块成了删不掉的孤儿"：`chunks` 靠外键级联，而父块表如果没有显式
    删除就会永远留着——重建/删除同步之后 `n_parents` 只增不减。
    """
    _seed_parents(pg)
    removed = pg.delete_docs({"alpha.md"})
    pg.finalize()
    assert removed == 3, "返回的仍是删掉的**块数**（alpha 有 3 个子块）"
    assert _raw_count(pg.schema, "parents", sql.SQL("WHERE doc_id = %s"), ("alpha",)) == 0
    assert pg.stats().n_parents == 1 and pg.stats().n_chunks == 1
    survivor = pg.get_chunks(["beta::0000"])["beta::0000"]
    assert survivor.parent_text.startswith(_PARENT_MARKER), "别的文档的父块不能被误删"


def test_stats_counts_parents_separately(pg: PgVectorStore) -> None:
    """`stats().n_parents` 必须是真实读数，且与 `n_chunks` 是**两个**数。

    这条断言防的是"体检把父块数糊成 0 或等于块数"：父子切割之后"几段语境"和"几个检索
    单位"应当分别可见，糊在一起就看不出父块有没有真的写进去。
    """
    assert pg.stats().n_parents == 0 and pg.stats().n_chunks == 0
    _seed_parents(pg)
    st = pg.stats()
    assert (st.n_docs, st.n_chunks, st.n_parents) == (2, 4, 3)
    assert _raw_count(pg.schema, "parents") == 3


def test_searches_only_return_child_chunks(pg: PgVectorStore) -> None:
    """检索结果的 id 只可能是子块：父块不进 BM25、也没有向量。

    这条断言防的是"父块混进检索结果"：父块没有向量，一旦混进向量路就是拿 NULL/0 参与
    排序；而父块进了关键字路会以极长的正文压住真正的子块命中。
    """
    _seed_parents(pg)
    child_ids = {c.chunk_id for c in _make_parent_seed()[1]}
    parent_ids = {p.parent_id for p in _make_parent_seed()[3]}

    dense = [cid for cid, _ in pg.dense_search(vec(a=1.0), 20)]
    assert set(dense) == child_ids, "向量路只能回子块，且必须回全"
    assert not (set(dense) & parent_ids)

    assert pg.keyword_search(_PARENT_MARKER, 5) == [], "父块正文不许进 BM25 索引"
    assert {cid for cid, _ in pg.keyword_search("kubelet", 5)} == {"alpha::0000", "alpha::0001"}
    # 父块表压根没有可检索字段：把"不出现在结果里"钉在 schema 上，而不是靠查询自觉
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'parents'",
                (pg.schema,),
            )
        }
    assert not ({"embedding", "tsv", "body_normalized"} & cols), f"父块表不该有检索字段：{cols}"


def test_ensure_schema_is_idempotent_and_upgrades_legacy(schema: str) -> None:
    """`ensure_schema()` 反复调用不许抛；老库（没有 `parent_id` / `parents` 表）必须能原地升级。

    这条断言防的是"升级要丢库重建"：老库在打开时被元信息闸门或 ALTER 写法直接拒绝，
    或 `parents` 表的 `CREATE/ALTER` 只在全新库上幂等、第二次调用就报"已存在"。
    """
    kwargs = {
        "schema": schema,
        "dim": DIM,
        "embedding_model": "hashing-test",
        "domain": schema,
        "pool_min": 1,
        "pool_max": 2,
    }
    first = PgVectorStore(_DSN, **kwargs)
    try:
        _seed_parents(first)
    finally:
        first.close()

    # 第二次打开同一 schema：CREATE TABLE / ALTER / CREATE INDEX 全部再跑一遍，不许抛
    second = PgVectorStore(_DSN, **kwargs)
    try:
        assert second.stats().n_parents == 3, "幂等重开不许丢数据，也不许重复插入"
    finally:
        second.close()

    # 把库倒回**父子切割之前**的样子：没有 parents 表、chunks 里没有 parent_id 列
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        conn.execute(f'ALTER TABLE "{schema}".chunks DROP COLUMN parent_id')
        conn.execute(f'DROP TABLE "{schema}".parents')

    upgraded = PgVectorStore(_DSN, **kwargs)
    try:
        with psycopg.connect(_DSN, connect_timeout=5) as conn:
            col = conn.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'chunks' AND column_name = 'parent_id'",
                (schema,),
            ).fetchone()
            table = conn.execute(
                "SELECT 1 FROM pg_tables WHERE schemaname = %s AND tablename = 'parents'", (schema,)
            ).fetchone()
        assert col is not None, "老库必须被就地补上 chunks.parent_id 列"
        assert table is not None, "老库必须被就地建出 parents 表"
        assert _raw_count(schema, "parents") == 0, "升级不许凭空造父块"
        hit = upgraded.get_chunks(["alpha::0000"])["alpha::0000"]
        assert hit.parent_id == "" and hit.parent_text == "", "老行的 parent_id 落到空串"
        assert hit.context_text == hit.body, "老库升级后语境必须退回子块正文"
        assert upgraded.stats().n_parents == 0 and upgraded.stats().n_chunks == 4, "升级不许丢老数据"
    finally:
        upgraded.close()
