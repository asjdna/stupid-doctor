"""多后端一致性测试：同一批输入在 `sqlite` / `sqlite-vec` / `pgvector` / `milvus` 上的行为对照。

存储层的四个实现必须满足同一个契约：`dense_search` / `keyword_search` 返回 `list[(chunk_id, rank)]`
且 rank 从 1 起连续；同一批输入下向量路给出同样的排序；全文路在共用 FTS5 的三个后端之间逐字节
相同；覆盖、删除与"缺向量拒绝写入"三条写入语义在四个后端上一致。

允许的差异：`pgvector` 的全文路走 pg_search（ParadeDB）的 BM25，与 FTS5 的 `bm25()` 是两份独立
实现、分词器也不同，所以只断言**命中集合相同**，排序差异是已知的；`pgvector` / `milvus` 的向量路
是 HNSW 近似索引，在本文件 5 块的小索引上与精确检索一致，大索引上的召回率是另一个问题。

接入方式：`pgvector` 不可达时整个参数不出现（否则"本地没 Docker"会让报告里出现一堆带 skip 的
绿点，掩盖这一路没跑过）；`milvus` 依赖本地文件、始终留在参数化里，真不可用时由夹具 skip 并
打印原因——显式可见的"没跑过"胜过悄悄从参数里消失。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from kubedocs.config import Settings
from kubedocs.store.base import ChunkInput, DocInput, Store, StoreError
from kubedocs.store.milvus_store import deps_available
from kubedocs.store.sqlite_store import MissingVectorError, SqliteStore
from kubedocs.store.sqlite_vec_store import SqliteVecStore
from tests.unit.store_seed import DIM, make_seed, vec

_vec = vec
"""本文件里 `_vec` 只是共享语料模块 `vec` 的短别名（断言里出现太多次）。"""


# --------------------------------------------------------------------------- 夹具
def _pg_available() -> tuple[bool, str]:
    """探测 Postgres：**不可达就不把 pgvector 纳入参数**，并记下原因。"""
    try:
        from kubedocs.store import pgvector_store
    except ImportError as e:  # pragma: no cover - 取决于环境
        return False, f"导入 pgvector_store 失败：{e}"
    return pgvector_store.probe(Settings().database_url, timeout=2.0)


def _milvus_available() -> tuple[bool, str]:
    """探测 Milvus Lite 依赖是否装好（本地数据目录，无需服务端）。

    刻意只查依赖、不打开数据目录：milvus-lite 一旦打开数据目录，LOCK 会一直占到进程退出，
    模块级探测不该为此在每个测试进程里留一份垃圾；真正打不开的情形由夹具 skip 并打印原因。
    """
    return deps_available()


_PG_OK, _PG_WHY = _pg_available()
_MILVUS_OK, _MILVUS_WHY = _milvus_available()


@pytest.fixture(params=["sqlite", "sqlite-vec", "milvus"] + (["pgvector"] if _PG_OK else []))
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def stores(backend: str, tmp_path: Any) -> Iterator[Store]:
    """按后端构造 store：三个本地后端各用一个独立文件，pgvector 用一个独立 schema。"""
    settings = Settings(
        index_dir=str(tmp_path / "index"),
        domain="contract",
        embedding_provider="hashing",
        embedding_dim=DIM,
        embedding_model="hashing-test",
        store=backend,  # type: ignore[arg-type]
    )
    if backend == "pgvector":
        from kubedocs.store.pgvector_store import PgVectorStore

        schema = f"test_contract_{tmp_path.name[-8:]}"
        store = PgVectorStore(
            settings.database_url,
            schema=schema,
            dim=DIM,
            embedding_model="hashing-test",
            domain=schema,
            pool_min=1,
            pool_max=2,
        )
        store.rebuild_from_scratch()
        try:
            yield store
        finally:
            _drop_schema(store, schema)
            store.close()
        return

    if backend == "milvus":
        if not _MILVUS_OK:
            pytest.skip(f"Milvus Lite 不可用，milvus 这一路未验证：{_MILVUS_WHY}")
        from kubedocs.store.milvus_store import MilvusStore

        try:
            ms = MilvusStore(
                db_path=settings.sqlite_path,
                vector_path=settings.vector_path,
                dim=DIM,
                embedding_model="hashing-test",
                milvus_path=tmp_path / "milvus" / "contract.db",
            )
        except StoreError as e:
            # 打不开（例如数据目录被另一个进程独占）属于环境原因：skip 并带上原因，
            # 不失败、也不假装跑过。真正的代码缺陷会在这里之外的地方红。
            pytest.skip(f"Milvus Lite 数据目录打不开，milvus 这一路未验证：{e}")
        try:
            yield ms
        finally:
            ms.close()
        return

    store_cls = SqliteStore if backend == "sqlite" else SqliteVecStore
    s = store_cls(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=DIM,
        embedding_model="hashing-test",
    )
    try:
        yield s
    finally:
        s.close()


def _drop_schema(store: Any, schema: str) -> None:
    import psycopg

    with psycopg.connect(store.dsn, connect_timeout=5) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture
def seeded(stores: Store) -> Store:
    seed = make_seed()
    stores.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    stores.finalize()
    return stores


# --------------------------------------------------------------------------- 形态
def test_dense_search_shape_and_rank_start_at_one(seeded: Store) -> None:
    """向量路：返回 (chunk_id, rank)，rank 从 1 起连续。三个后端必须同形。"""
    hits = seeded.dense_search(_vec(a=1.0), 5)
    assert [r for _, r in hits] == list(range(1, len(hits) + 1))
    assert all(isinstance(cid, str) for cid, _ in hits)


def test_keyword_search_shape_and_rank_start_at_one(seeded: Store) -> None:
    """全文路：返回 (chunk_id, rank)，rank 从 1 起连续。"""
    hits = seeded.keyword_search("kubelet", 5)
    assert hits, "kubelet 至少该命中 alpha 的两个块"
    assert [r for _, r in hits] == list(range(1, len(hits) + 1))


def test_empty_query_and_zero_limit_return_nothing(seeded: Store) -> None:
    """空查询 / limit<=0 必须返回空——而不是"匹配全部"或抛异常。"""
    assert seeded.keyword_search("", 5) == []
    assert seeded.keyword_search("kubelet", 0) == []
    assert seeded.dense_search(_vec(a=1.0), 0) == []


def test_dense_query_dim_mismatch_is_refused(seeded: Store) -> None:
    """查询向量维度不对必须报错：静默算出垃圾相似度是更坏的结果。"""
    with pytest.raises(StoreError):
        seeded.dense_search(np.zeros(DIM + 1, dtype=np.float32), 5)


# --------------------------------------------------------------------------- 排序一致
def test_dense_ranking_is_identical_across_backends(seeded: Store) -> None:
    """契约第 2 条：向量已归一化时，四个后端的向量路排序必须**完全相同**。

    换后端（numpy 内积 / sqlite-vec 余弦 / pgvector HNSW / Milvus HNSW）不许改变检索顺序，
    否则评测指标的变化就无法归因。查询取 top-2 而不是 top-3：0 分是并列的，并列名次没有唯一
    答案，拿它做跨实现断言只会得到一个随机变红的测试。
    """
    assert seeded.dense_search(_vec(a=1.0), 2) == [("alpha::0000", 1), ("alpha::0001", 2)]
    assert seeded.dense_search(_vec(c=1.0), 2) == [("gamma::0000", 1), ("beta::0001", 2)]


def test_keyword_ranking_matches_sqlite_reference(seeded: Store, backend: str) -> None:
    """契约第 3 条。

    两个 SQLite 后端共用同一份 FTS5 索引，排序必须逐字节相同；pgvector 用 pg_search 的 BM25
    （与 FTS5 的 `bm25()` 是两份独立实现、分词器也不同），只断言**命中集合**相同。
    """
    got = seeded.keyword_search("kubelet", 5)
    if backend == "pgvector":
        assert {c for c, _ in got} == {"alpha::0000", "alpha::0001"}
    else:
        assert got == [("alpha::0000", 1), ("alpha::0001", 2)]


def test_keyword_search_is_deterministic(seeded: Store) -> None:
    """同一查询连跑两次结果必须一致（并列分数要有确定的次序）。"""
    first = seeded.keyword_search("kubelet eviction threshold", 5)
    for _ in range(3):
        assert seeded.keyword_search("kubelet eviction threshold", 5) == first


def test_keyword_query_is_not_an_injection_surface(seeded: Store) -> None:
    """查询里的引擎元语法（FTS5 的 `-` `"`、tsquery 的 `!` `&`）不许被当语法解释。

    `kubelet -eviction` 在 FTS5 里是"含 kubelet 且不含 eviction"；
    这里必须只把它当普通词元，命中集合与 `kubelet` 相同。
    """
    plain = {c for c, _ in seeded.keyword_search("kubelet", 5)}
    # `-` 若被当语法，`kubelet -eviction` 会**排除**所有 kubelet 命中（它们都含 eviction）
    # 而返回空；这里必须仍返回那两个块，说明 `-` 只是普通分隔符。
    assert {c for c, _ in seeded.keyword_search("kubelet -eviction", 5)} == plain
    # 引号同理：`kubelet" OR "x` 只被当词元，不许把 `OR` 当运算符
    assert {c for c, _ in seeded.keyword_search('kubelet" OR "x', 5)} == plain
    # 全特殊字符 -> 没有词元 -> 空结果（"搜不到"比"匹配全部"诚实）
    assert seeded.keyword_search('"***"', 5) == []
    assert seeded.keyword_search("!!!", 5) == []


def test_keyword_hits_do_not_cross_documents(seeded: Store) -> None:
    """全文路不许把别的文档的块也召回（分词语义正确性的一条基础检查）。"""
    assert {c for c, _ in seeded.keyword_search("crashloopbackoff", 5)} == {"beta::0000"}
    assert {c for c, _ in seeded.keyword_search("resourcequota", 5)} == {"gamma::0000"}


# --------------------------------------------------------------------------- 写入语义
def test_missing_vector_is_refused_on_every_backend(stores: Store) -> None:
    """没有向量的块必须被拒绝写入，而不是写进去静默漏召回。"""
    seed = make_seed()
    with pytest.raises(MissingVectorError):
        stores.upsert_docs(seed.docs[:1], seed.chunks[:2], {})


def test_upsert_overwrites_and_keeps_vectors_aligned(stores: Store) -> None:
    """覆盖语义：整篇重写后，两个通道看到的都是新内容与新向量。

    覆盖的粒度是**文档**（先删该 doc 的全部块再写），所以必须连 `DocInput` 一起传，只传块会
    撞主键；这条断言专盯"向量与块行错位"，四个后端都必须过。
    """
    seed = make_seed()
    stores.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    stores.finalize()
    # 把 beta 整篇换成一个块：新正文 + 正交方向 e4
    new_chunk = ChunkInput(
        chunk_id="beta::0000",
        doc_id="beta",
        ordinal=0,
        title="beta",
        breadcrumb="sec",
        source_url="https://kubernetes.io/docs/beta/",
        body="uniqueterm zeta",
        content_hash="h-beta-0-v2",
        n_chars=18,
    )
    new_doc = DocInput(
        doc_id="beta",
        rel_path="beta.md",
        title="beta",
        source_url="https://kubernetes.io/docs/beta/",
        content_hash="h-beta-v2",
    )
    stores.upsert_docs([new_doc], [new_chunk], {"beta::0000": _vec(e=1.0)})
    stores.finalize()

    assert stores.dense_search(_vec(e=1.0), 1) == [("beta::0000", 1)]
    assert {c for c, _ in stores.keyword_search("uniqueterm", 5)} == {"beta::0000"}
    assert stores.keyword_search("crashloopbackoff", 5) == []  # 旧正文已不在索引里
    assert stores.keyword_search("imagepullbackoff", 5) == []  # 同篇的旧兄弟块也被清掉
    assert stores.stats().n_chunks == 4


def test_delete_docs_removes_from_both_channels(seeded: Store) -> None:
    """删除：全文与向量两个通道都不许再召回被删的块。"""
    assert seeded.delete_docs({"alpha.md"}) == 2
    seeded.finalize()
    assert seeded.keyword_search("kubelet", 5) == []
    assert "alpha::0000" not in [c for c, _ in seeded.dense_search(_vec(a=1.0), 5)]
    assert seeded.existing_doc_ids() == {"beta", "gamma"}
    assert seeded.stats().n_chunks == 3
    assert seeded.get_chunks(["alpha::0000"]) == {}


def test_delete_then_reinsert_is_consistent(seeded: Store) -> None:
    """删除后重新插入（rowid 可能被回收）——向量不许与块错位。"""
    seeded.delete_docs({"beta.md"})
    seeded.finalize()
    new = ChunkInput(
        chunk_id="delta::0000",
        doc_id="delta",
        ordinal=0,
        title="delta",
        breadcrumb="sec",
        source_url="https://kubernetes.io/docs/delta/",
        body="deltaterm",
        content_hash="h-d0",
        n_chars=9,
    )
    seeded.upsert_docs(
        [
            DocInput(
                doc_id="delta",
                rel_path="delta.md",
                title="delta",
                source_url="https://kubernetes.io/docs/delta/",
                content_hash="h-delta",
            )
        ],
        [new],
        {"delta::0000": _vec(d=1.0)},
    )
    seeded.finalize()
    assert seeded.dense_search(_vec(d=1.0), 1) == [("delta::0000", 1)]
    assert seeded.dense_search(_vec(b=1.0), 1)[0][0] == "alpha::0001"


def test_rebuild_from_scratch_clears_everything(seeded: Store) -> None:
    """重建：两个通道都清空，且之后仍能正常写入。"""
    seeded.rebuild_from_scratch()
    assert seeded.stats().n_chunks == 0
    assert seeded.keyword_search("kubelet", 5) == []
    assert seeded.dense_search(_vec(a=1.0), 5) == []
    seed = make_seed()
    seeded.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    seeded.finalize()
    assert seeded.stats().n_chunks == 5
    # 全文自检是三态契约（`ok` / `DEGRADED: ...` / `FAIL: ...`）：`DEGRADED` 是合法结果
    # （库里没有 pg_search 时关键字路降级到 ts_rank），把它读成 FAIL 会掩盖真实状态。
    fts = seeded.verify_fts()
    assert fts == "ok" or fts.startswith("DEGRADED"), fts


def test_existing_hash_to_vector_reuses_vectors(seeded: Store) -> None:
    """跨篇复用：同一 content_hash 必须能取回**同一个**向量。"""
    got = seeded.existing_hash_to_vector({"h-alpha-0", "no-such-hash"})
    assert set(got) == {"h-alpha-0"}
    assert np.array_equal(got["h-alpha-0"], _vec(a=1.0))


def test_existing_doc_hashes_and_doc_texts(seeded: Store) -> None:
    """摄取计划与标注集自检依赖的两条读取路径。"""
    hashes = seeded.existing_doc_hashes()
    assert hashes["alpha.md"] == "h-alpha"
    texts = seeded.doc_texts(["alpha", "beta", "missing"])
    assert "kubelet" in texts["alpha"] and "crashloopbackoff" in texts["beta"]
    assert texts["missing"] == ""
    assert "crashloopbackoff" not in texts["alpha"]


def test_get_chunks_returns_full_bodies(seeded: Store) -> None:
    """取回块内容：正文、引用信息都必须在（生成引用靠它）。"""
    hits = seeded.get_chunks(["gamma::0000", "nope"])
    assert set(hits) == {"gamma::0000"}
    hit = hits["gamma::0000"]
    assert hit.body == "resourcequota namespace limitrange"
    assert hit.doc_id == "gamma" and hit.ordinal == 0
    assert hit.source_url.endswith("/gamma/")


def test_stats_counts_match_written_data(seeded: Store) -> None:
    """统计口径一致：文档数/块数/字符数。"""
    st = seeded.stats()
    assert (st.n_docs, st.n_chunks) == (3, 5)
    assert st.embedding_dim == DIM
    assert st.mean_chars > 0


# --------------------------------------------------------------------- sqlite-vec 专项
class TestSqliteVecBackend:
    """只对 sqlite-vec 成立的几条：页缓存参数、空向量表守卫、旧索引迁移。"""

    def test_page_cache_is_raised(self, tmp_path: Any) -> None:
        """页缓存必须按配置调大：向量表 29MB 而 SQLite 默认页缓存 2MB，不调 cache_size 时
        每次 KNN 都在穿透页缓存读盘。它不影响任何正确性断言，只能靠这条专门测出来。
        """
        s = SqliteVecStore(
            db_path=tmp_path / "c.db",
            vector_path=tmp_path / "c.npy",
            dim=DIM,
            embedding_model="hashing-test",
            cache_kib=65536,
        )
        try:
            got = int(s._con.execute("PRAGMA cache_size").fetchone()[0])  # noqa: SLF001
            assert got == -65536, "cache_size 是负值语义（KiB），必须是 -65536"
        finally:
            s.close()

    def test_missing_vector_table_is_reported_not_silently_empty(self, tmp_path: Any) -> None:
        """块表有数据、向量表却是空的：向量路必须报错并给出修复命令。

        这正是"改默认后端后打开旧索引"的第一现场；静默返回空结果会让它表现为
        "检索质量突然崩了"，而不是"你少跑了一步迁移"。
        """
        legacy = SqliteStore(
            db_path=tmp_path / "m.db", vector_path=tmp_path / "m.npy", dim=DIM, embedding_model="hashing-test"
        )
        seed = make_seed()
        legacy.upsert_docs(seed.docs, seed.chunks, seed.vectors)
        legacy.finalize()
        legacy.close()

        vec = SqliteVecStore(
            db_path=tmp_path / "m.db", vector_path=tmp_path / "m.npy", dim=DIM, embedding_model="hashing-test"
        )
        try:
            # 全文路仍然可用（不是把整个索引判死）
            assert vec.keyword_search("kubelet", 5)
            with pytest.raises(StoreError, match="向量索引与块表不一致"):
                vec.dense_search(_vec(a=1.0), 5)
            with pytest.raises(StoreError):
                vec.existing_hash_to_vector({"h-alpha-0"})
            assert vec.verify_vec().startswith("FAIL")

            # 迁移后必须**逐位**等价于旧实现（同一批向量换个地方放）
            assert vec.adopt_legacy_matrix() == 5
            assert vec.verify_vec() == "ok"
            # 只比 top-2：第 3 名是 0 分的**并列**，并列次序不是契约的一部分
            assert vec.dense_search(_vec(a=1.0), 2) == legacy_dense(tmp_path)
        finally:
            vec.close()

    def test_adopt_is_idempotent(self, tmp_path: Any) -> None:
        """迁移可以重复跑：结果不变（脚本会重跑，不能越跑越脏）。"""
        legacy = SqliteStore(
            db_path=tmp_path / "i.db", vector_path=tmp_path / "i.npy", dim=DIM, embedding_model="hashing-test"
        )
        seed = make_seed()
        legacy.upsert_docs(seed.docs, seed.chunks, seed.vectors)
        legacy.finalize()
        legacy.close()
        vec = SqliteVecStore(
            db_path=tmp_path / "i.db", vector_path=tmp_path / "i.npy", dim=DIM, embedding_model="hashing-test"
        )
        try:
            assert vec.adopt_legacy_matrix() == 5
            assert vec.adopt_legacy_matrix() == 5
            assert vec.stats().n_chunks == 5
            assert vec.verify_vec() == "ok"
        finally:
            vec.close()


def legacy_dense(tmp_path: Any) -> list[tuple[str, int]]:
    """用旧实现算一遍向量路 top-2，作为迁移正确性的参照。"""
    legacy = SqliteStore(
        db_path=tmp_path / "m.db", vector_path=tmp_path / "m.npy", dim=DIM, embedding_model="hashing-test"
    )
    try:
        return legacy.dense_search(_vec(a=1.0), 2)
    finally:
        legacy.close()
