"""`MilvusStore` 后端的专项测试：**Milvus Lite 依赖不可用时整文件 skip，并写明原因**。

前置条件：装好 `pymilvus`，milvus-lite 用本地数据目录（不需要服务端）；每个用例一份独立的
`tmp_path` 数据目录，互不干扰，也不碰真实索引（`data/index/**` 是只读资产）。数据目录被别的
进程独占这条路径无法在本进程内模拟，只覆盖"打不开就抛 `StoreError` 并给人话提示"。

守的契约：删文档必须同时清掉 Milvus 向量（两个通道都不许再召回）；重复 upsert 与重复打开
幂等而不产生重复行；HNSW 的 `M` / `efConstruction` 记进索引元信息，改了 `M` 却不重建必须报错；
依赖缺失、路径非法、数据目录打不开、维度不符都必须抛 `StoreError`；块表与向量索引不一致
（缺行或多行）必须在自检与查询处报错。与 SQLite 侧只对照可比较的部分：向量路排序、命中集合、
统计口径与向量往返（HNSW 的近似性交给 `dense_search_exact()`）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kubedocs.config import Settings
from kubedocs.store import from_settings
from kubedocs.store.base import ChunkInput, DocInput, Store, StoreError
from kubedocs.store.milvus_store import (
    BACKEND,
    COLLECTION,
    MilvusStore,
    deps_available,
)
from kubedocs.store.sqlite_store import MissingVectorError, SqliteStore
from kubedocs.store.sqlite_vec_store import SqliteVecStore
from tests.unit.store_seed import DIM, make_seed, vec

_DEPS_OK, _DEPS_WHY = deps_available()

pytestmark = pytest.mark.skipif(
    not _DEPS_OK,
    reason=f"Milvus Lite 依赖不可用，本文件未验证：{_DEPS_WHY}",
)

EMBEDDING_MODEL = "hashing-test"


def _make(tmp_path: Path, *, name: str = "m.db", **over: Any) -> MilvusStore:
    """构造一个独立的 milvus 后端（SQLite 与 Milvus 数据目录都在 tmp_path 下）。"""
    kwargs: dict[str, Any] = {
        "db_path": tmp_path / "kubedocs.db",
        "vector_path": tmp_path / "chunk_vectors.npy",
        "dim": DIM,
        "embedding_model": EMBEDDING_MODEL,
        "milvus_path": tmp_path / name,
    }
    kwargs.update(over)
    return MilvusStore(**kwargs)


@pytest.fixture
def ms(tmp_path: Path) -> Any:
    store = _make(tmp_path)
    try:
        yield store
    finally:
        store.close()


def _seed(store: Store) -> None:
    seed = make_seed()
    store.upsert_docs(seed.docs, seed.chunks, seed.vectors)
    store.finalize()


def _milvus_ids(store: MilvusStore) -> set[str]:
    """直接问 Milvus 要一份主键清单（不经过 store 的任何缓冲/缓存）。"""
    rows = store._client.query(
        COLLECTION,
        filter="",
        limit=10_000,  # noqa: SLF001
        output_fields=["chunk_id"],
    )
    return {str(r["chunk_id"]) for r in rows}


# --------------------------------------------------------------------------- 集合与幂等
def test_collection_is_created_idempotently(ms: MilvusStore, tmp_path: Path) -> None:
    """重复打开同一份索引不许报错、也不许丢数据：`create_collection` 对已存在的集合会报错
    （不像 SQL 的 IF NOT EXISTS），幂等只能靠先判断；否则"跑一次 ingest、再跑一次 eval"
    就会在第二次打开时炸。
    """
    _seed(ms)
    assert ms.stats().n_chunks == 5
    assert ms._client.has_collection(COLLECTION)  # noqa: SLF001
    ms.close()

    again = _make(tmp_path)
    try:
        assert again.stats().n_chunks == 5
        assert again.verify_vec() == "ok"
        assert again.dense_search(vec(a=1.0), 2) == [("alpha::0000", 1), ("alpha::0001", 2)]
    finally:
        again.close()


def test_repeated_upsert_does_not_duplicate_rows(ms: MilvusStore) -> None:
    """同一批输入写两遍：行数不变、主键不变——upsert 是覆盖而不是追加。

    行数翻倍在向量库里不会报错，只会让结果慢慢变脏，所以断言必须盯行数与主键而非查询结果。
    """
    _seed(ms)
    before = ms._client.get_collection_stats(COLLECTION)["row_count"]  # noqa: SLF001
    _seed(ms)
    after = ms._client.get_collection_stats(COLLECTION)["row_count"]  # noqa: SLF001
    assert before == after == 5
    assert _milvus_ids(ms) == {"alpha::0000", "alpha::0001", "beta::0000", "beta::0001", "gamma::0000"}


# --------------------------------------------------------------------------- 删除同步
def test_delete_docs_clears_milvus_vectors(ms: MilvusStore) -> None:
    """**删文档必须同时清 Milvus 向量**——两个通道都不许再召回。

    断言分两层：SQLite 侧不再召回，且直接问 Milvus 要主键清单时也没有被删的块，而不只是看
    查询结果（块行与向量分离的实现在这里最容易漏清）。
    """
    _seed(ms)
    assert ms.delete_docs({"alpha.md"}) == 2
    ms.finalize()
    assert ms.stats().n_chunks == 3
    assert _milvus_ids(ms) == {"beta::0000", "beta::0001", "gamma::0000"}
    assert ms.keyword_search("kubelet", 5) == []
    assert "alpha::0000" not in [c for c, _ in ms.dense_search(vec(a=1.0), 5)]
    assert ms.verify_vec() == "ok"


def test_delete_then_reinsert_keeps_both_sides_aligned(ms: MilvusStore) -> None:
    """删除后重新插入：块表与向量索引仍然一一对应（没有孤儿、也不缺行）。"""
    _seed(ms)
    ms.delete_docs({"beta.md"})
    ms.finalize()
    new_chunk = ChunkInput(
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
    new_doc = DocInput(
        doc_id="delta",
        rel_path="delta.md",
        title="delta",
        source_url="https://kubernetes.io/docs/delta/",
        content_hash="h-delta",
    )
    ms.upsert_docs([new_doc], [new_chunk], {"delta::0000": vec(d=1.0)})
    ms.finalize()
    assert ms.dense_search(vec(d=1.0), 1) == [("delta::0000", 1)]
    assert _milvus_ids(ms) == {"alpha::0000", "alpha::0001", "gamma::0000", "delta::0000"}
    assert ms.verify_vec() == "ok"


def test_rebuild_from_scratch_drops_the_collection(ms: MilvusStore) -> None:
    """重建：两边都清空，且之后仍能正常写入（集合被 drop 后重新建起来）。"""
    _seed(ms)
    ms.rebuild_from_scratch()
    assert ms.stats().n_chunks == 0
    assert _milvus_ids(ms) == set()
    assert ms.dense_search(vec(a=1.0), 5) == []
    _seed(ms)
    assert ms.stats().n_chunks == 5
    assert ms.verify_vec() == "ok"
    assert ms.verify_fts() == "ok"


def test_dense_search_matches_exact_search_on_small_index(ms: MilvusStore) -> None:
    """小索引上 HNSW 必须与**不用索引**的精确检索给出同一个 top-k。

    这是召回归因的基准：大索引上两者不一致时，差异来自 HNSW 的近似性而不是数据或语义变了。
    """
    _seed(ms)
    for q in (vec(a=1.0), vec(a=0.8, b=0.6), vec(c=1.0), vec(b=1.0)):
        assert ms.dense_search(q, 3) == ms.dense_search_exact(q, 3)


def test_dense_search_matches_sqlite_backend(ms: MilvusStore, tmp_path: Path) -> None:
    """核心对照：同一批语料下，milvus 的向量路排序与 `sqlite`（numpy 精确）一致。

    只比没有并列的名次（并列次序不是契约的一部分）：a、c 方向名次唯一，b 方向只比第一名。
    """
    _seed(ms)
    reference = SqliteStore(
        db_path=tmp_path / "s.db", vector_path=tmp_path / "s.npy", dim=DIM, embedding_model=EMBEDDING_MODEL
    )
    try:
        _seed(reference)
        for q in (vec(a=1.0), vec(c=1.0)):
            assert ms.dense_search(q, 2) == reference.dense_search(q, 2)
        assert ms.dense_search(vec(b=1.0), 1) == reference.dense_search(vec(b=1.0), 1)
    finally:
        reference.close()


def test_keyword_path_is_byte_identical_to_sqlite_vec(ms: MilvusStore, tmp_path: Path) -> None:
    """全文路复用同一份 FTS5 代码，排序必须与 sqlite-vec **逐字节相同**。

    这是"两个后端之间唯一变量是向量路"的可验证形式；它不成立，换后端导致的指标变化就无法归因。
    """
    _seed(ms)
    other = SqliteVecStore(
        db_path=tmp_path / "v.db", vector_path=tmp_path / "v.npy", dim=DIM, embedding_model=EMBEDDING_MODEL
    )
    try:
        _seed(other)
        for q in ("kubelet", "eviction threshold", "crashloopbackoff", "resourcequota"):
            assert ms.keyword_search(q, 5) == other.keyword_search(q, 5)
    finally:
        other.close()


def test_existing_hash_to_vector_roundtrip_is_bit_exact(ms: MilvusStore) -> None:
    """向量往返必须**逐位相等**：摄取复用旧向量时不许被悄悄改精度/归一化。"""
    _seed(ms)
    got = ms.existing_hash_to_vector({"h-alpha-0", "no-such-hash"})
    assert set(got) == {"h-alpha-0"}
    assert got["h-alpha-0"].dtype == np.float32
    assert np.array_equal(got["h-alpha-0"], vec(a=1.0))


def test_stats_and_get_chunks_match_sqlite(ms: MilvusStore, tmp_path: Path) -> None:
    """统计口径与块取回必须与 SQLite 侧一致（评测的语料指纹依赖它）。"""
    _seed(ms)
    reference = SqliteStore(
        db_path=tmp_path / "r.db", vector_path=tmp_path / "r.npy", dim=DIM, embedding_model=EMBEDDING_MODEL
    )
    try:
        _seed(reference)
        a, b = ms.stats(), reference.stats()
        assert (a.n_docs, a.n_chunks, a.total_chars, a.n_with_code) == (
            b.n_docs,
            b.n_chunks,
            b.total_chars,
            b.n_with_code,
        )
        assert ms.get_chunks(["gamma::0000", "nope"]) == reference.get_chunks(["gamma::0000", "nope"])
        assert ms.doc_texts(["alpha", "beta"]) == reference.doc_texts(["alpha", "beta"])
        assert ms.existing_doc_ids() == reference.existing_doc_ids()
        assert ms.existing_doc_hashes() == reference.existing_doc_hashes()
    finally:
        reference.close()


# --------------------------------------------------------------------------- HNSW 参数
def test_hnsw_params_are_recorded_in_index_meta(ms: MilvusStore) -> None:
    """`M` / `efConstruction` / 度量必须落进索引元信息。

    Milvus Lite 的 `describe_index` 不回传这两个参数，所以"参数有没有生效"只能靠我们记账，
    再由"改了参数却不重建索引就报错"那条用例守住。
    """
    meta = ms.meta_dump()
    assert meta["milvus_hnsw_m"] == str(ms.hnsw_m)
    assert meta["milvus_hnsw_ef_construction"] == str(ms.hnsw_ef_construction)
    assert meta["milvus_metric"] == ms.metric == "COSINE"
    assert meta["vector_backend"] == BACKEND


def test_changing_build_params_without_rebuild_is_refused(ms: MilvusStore, tmp_path: Path) -> None:
    """改了 `M` 却不重建索引 -> 报错，而不是"配置看起来改了、索引其实没变"。"""
    _seed(ms)
    ms.close()
    with pytest.raises(StoreError, match="建索引参数与当前配置不符"):
        _make(tmp_path, hnsw_m=48)
    # ef（查询期）不进这道闸门：它本来就该随时可调
    tuned = _make(tmp_path, hnsw_ef_search=256)
    try:
        assert tuned.hnsw_ef_search == 256
        assert tuned.verify_vec() == "ok"
    finally:
        tuned.close()


def test_ef_search_reaches_the_engine(ms: MilvusStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """查询期 `ef` 必须真的传进 Milvus 的检索请求，而不是只写在配置里。

    这里能验证的只是接线；"ef 越大召回越高"的效果要在大索引上才看得出来，本文件不假造它。
    """
    _seed(ms)
    captured: dict[str, Any] = {}
    real = ms._client.search  # noqa: SLF001

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ms._client, "search", spy)  # noqa: SLF001
    ms.dense_search(vec(a=1.0), 3)
    assert captured["search_params"]["metric_type"] == "COSINE"
    assert captured["search_params"]["params"]["ef"] == ms.hnsw_ef_search
    assert captured["anns_field"] == "embedding"
    assert captured["limit"] == 3


def test_metric_is_configurable_and_recorded(tmp_path: Path) -> None:
    """度量可配置（默认 COSINE，与另两个后端一致），且改动同样记进元信息。"""
    store = _make(tmp_path, name="ip.db", metric="IP")
    try:
        assert store.metric == "IP"
        assert store.meta_dump()["milvus_metric"] == "IP"
        _seed(store)
        # 向量已归一化：IP 与 COSINE 必须同序
        assert store.dense_search(vec(a=1.0), 2) == [("alpha::0000", 1), ("alpha::0001", 2)]
    finally:
        store.close()


# --------------------------------------------------------------------------- 不可用路径
def test_missing_dependency_raises_instead_of_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**没装 pymilvus 时必须报错，绝不回退到 numpy 扫描**。

    回退之后排序会变（近似 vs 精确），调用方看到的只是一个"检索质量好像变了"的谜。
    """
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(StoreError, match="不会静默回退"):
        _make(tmp_path)


def test_factory_does_not_fall_back_when_milvus_is_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """工厂层同上：`KUBEDOCS_STORE=milvus` 起不来就抛错，不会顺手构造 sqlite。"""
    settings = Settings(
        index_dir=str(tmp_path / "index"),
        domain="kubedocs",
        embedding_provider="hashing",
        embedding_dim=DIM,
        embedding_model=EMBEDDING_MODEL,
        store="milvus",
    )
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(StoreError):
        from_settings(settings)


def test_path_without_db_suffix_is_refused_with_a_human_message(tmp_path: Path) -> None:
    """路径不以 `.db` 结尾时给出人话提示（milvus-lite 自己那句话几乎没法读）。"""
    with pytest.raises(StoreError, match="必须以 .db 结尾"):
        _make(tmp_path, name="milvus_data")


def test_unopenable_data_dir_is_reported_not_silently_degraded(tmp_path: Path) -> None:
    """数据目录打不开（这里用"路径是个普通文件"构造）必须抛 `StoreError` 并带原因。"""
    blocker = tmp_path / "blocked.db"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(StoreError, match="打开 Milvus Lite 数据目录失败"):
        _make(tmp_path, name="blocked.db")


def test_dimension_mismatch_is_refused(tmp_path: Path) -> None:
    """维度不符必须在**打开时**拒绝，而不是等查询时算出垃圾相似度。"""
    store = _make(tmp_path)
    _seed(store)
    store.close()
    with pytest.raises(StoreError, match="维度"):
        _make(tmp_path, dim=DIM * 2)


# --------------------------------------------------------------------------- 一致性闸门
def test_missing_vectors_are_detected_not_silently_ignored(ms: MilvusStore) -> None:
    """块表有数据、向量侧缺行：必须报错并给修复命令，而不是静默少召回——
    打开一个还没迁移过向量的旧索引就是这种情况。"""
    _seed(ms)
    ms._client.delete(COLLECTION, ids=["alpha::0000"])  # noqa: SLF001 - 白盒制造不一致
    with pytest.raises(StoreError, match="不一致"):
        ms.finalize()
    assert ms.verify_vec().startswith("FAIL")
    with pytest.raises(StoreError):
        ms.dense_search(vec(a=1.0), 3)
    with pytest.raises(StoreError):
        ms.existing_hash_to_vector({"h-alpha-0"})
    # 全文路仍然可用（不是把整个索引判死）
    assert ms.keyword_search("kubelet", 5)


def test_orphan_vectors_are_detected(ms: MilvusStore) -> None:
    """向量侧多出块表里没有的条目：同样是报错，而不是"检索出一个取不到正文的块"。"""
    _seed(ms)
    ms._client.upsert(
        COLLECTION,
        [
            {
                "chunk_id": "ghost::0000",  # noqa: SLF001
                "embedding": np.asarray(vec(d=1.0), dtype=np.float32),
            }
        ],
    )
    with pytest.raises(StoreError, match="不一致"):
        ms.finalize()
    report = ms.verify_vec()
    assert report.startswith("FAIL") and "ghost::0000" in report


def test_verify_vec_reports_ok_after_normal_writes(ms: MilvusStore) -> None:
    """正常路径下自检必须是 ok（否则这条自检就成了噪声，没人会看）。"""
    _seed(ms)
    assert ms.verify_vec() == "ok"
    assert ms.verify_fts() == "ok"


# --------------------------------------------------------------------------- 后端闸门与迁移
def test_backend_takeover_requires_an_explicit_flag(tmp_path: Path) -> None:
    """由 `sqlite-vec` 写的索引不许被 milvus 顺手接管，否则会静默读到陈旧向量。

    要接管必须显式打开闸门（只有迁移脚本会这么做），见 `adopt_legacy_matrix`。
    """
    legacy = SqliteVecStore(
        db_path=tmp_path / "kubedocs.db",
        vector_path=tmp_path / "chunk_vectors.npy",
        dim=DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    try:
        _seed(legacy)
    finally:
        legacy.close()
    with pytest.raises(StoreError, match="索引由另一个后端写入"):
        _make(tmp_path)
    adopted = _make(tmp_path, allow_backend_takeover=True)
    try:
        assert adopted.meta_dump()["vector_backend"] == BACKEND
    finally:
        adopted.close()


def test_adopt_legacy_matrix_is_idempotent_and_bit_exact(tmp_path: Path) -> None:
    """迁移（不重新编码）：抽样逐位相等，且可以重复跑（脚本会重跑）。"""
    legacy = SqliteStore(
        db_path=tmp_path / "kubedocs.db",
        vector_path=tmp_path / "chunk_vectors.npy",
        dim=DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    try:
        _seed(legacy)
    finally:
        legacy.close()
    mat = np.load(tmp_path / "chunk_vectors.npy")
    store = _make(tmp_path, allow_backend_takeover=True)
    try:
        assert store.adopt_legacy_matrix() == 5
        assert store.adopt_legacy_matrix() == 5  # 幂等：不产生重复行
        assert store.verify_vec() == "ok"
        got = store.existing_hash_to_vector({"h-alpha-0", "h-alpha-1"})
        assert np.array_equal(got["h-alpha-0"], np.asarray(mat[0], dtype=np.float32))
        assert store.dense_search(vec(a=1.0), 2) == [("alpha::0000", 1), ("alpha::0001", 2)]
    finally:
        store.close()


def test_adopt_legacy_matrix_without_matrix_gives_a_fix_hint(tmp_path: Path) -> None:
    """没有 `.npy` 时不许猜：报错并给出"重新编码"的可执行路径。"""
    store = _make(tmp_path)
    try:
        with pytest.raises(StoreError, match="找不到旧向量矩阵"):
            store.adopt_legacy_matrix()
    finally:
        store.close()


def test_detach_lets_another_backend_take_over(tmp_path: Path) -> None:
    """`detach()` 撤销后端标记，否则"换回 sqlite-vec"连 store 都构造不出来。

    摘除后向量侧必须明确变成空：向量路报错并给修复命令，而不是静默返回空结果。
    """
    store = _make(tmp_path)
    try:
        _seed(store)
        assert store.meta_dump()["vector_backend"] == BACKEND
        store.detach()
        assert _milvus_ids(store) == set()
        assert "vector_backend" not in store.meta_dump()
        assert store.verify_vec().startswith("FAIL")
        with pytest.raises(StoreError):
            store.dense_search(vec(a=1.0), 3)
    finally:
        store.close()
    # 摘掉之后，sqlite-vec 后端可以正常打开这份索引（块行与全文索引没被动过）
    other = SqliteVecStore(
        db_path=tmp_path / "kubedocs.db",
        vector_path=tmp_path / "chunk_vectors.npy",
        dim=DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    try:
        assert other.stats().n_chunks == 5
        assert other.keyword_search("kubelet", 5)
    finally:
        other.close()


# --------------------------------------------------------------------------- 其它协议面
def test_missing_vector_is_refused(ms: MilvusStore) -> None:
    """没有向量的块必须被拒绝写入（与另外三个后端一致）。"""
    seed = make_seed()
    with pytest.raises(MissingVectorError):
        ms.upsert_docs(seed.docs[:1], seed.chunks[:2], {})


def test_query_vector_dim_mismatch_is_refused(ms: MilvusStore) -> None:
    """查询向量维度不对必须报错，不许静默算垃圾相似度。"""
    _seed(ms)
    with pytest.raises(StoreError):
        ms.dense_search(np.zeros(DIM + 1, dtype=np.float32), 3)


def test_store_satisfies_protocol(ms: MilvusStore) -> None:
    """运行期满足 `Store` 协议；并且它**就是** SqliteStore（只换向量路）。"""
    assert isinstance(ms, Store)
    assert isinstance(ms, SqliteStore)
    assert ms.normalizer.name == "identity"
