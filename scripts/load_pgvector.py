"""把本地 SQLite 索引原样灌进 Postgres（pgvector 后端），不重新编码向量。

做的是搬家：`documents` / `chunks` / 向量按原样写进 Postgres 的 `<domain>` schema，
灌完在库里逐位校验抽样向量与统计口径。不重新切块、不重新编码（那是 `ingest` 的事）。

用法：
    python scripts/load_pgvector.py                       # data/index -> schema k8s_docs
    python scripts/load_pgvector.py --batch-docs 200      # 每批文档数（默认 120）
    python scripts/load_pgvector.py --keep                # 追加，不清空目标 schema

前置条件：本地索引已建好，Postgres 可达且 DSN 可连（配置提供），目标端有 pgvector。
默认**先清空目标 schema 再灌**（镜像语义，不是叠加）；`--keep` 关闭清空，用于向已有数据追加。
输出：逐批进度，末尾是耗时、FTS/HNSW 自检、抽样逐位比较与字符总数对比。
退出码：0 灌入并校验通过 / 1 Postgres 不可达、索引有问题或校验未通过。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.store.base import ChunkInput, DocInput, StoreError  # noqa: E402
from kubedocs.store.pgvector_store import PgVectorStore, probe  # noqa: E402

EXIT_OK = 0
EXIT_USER = 1


def _read_sidecar(settings: Settings, rows: list[sqlite3.Row], dim: int) -> dict[str, np.ndarray]:
    """从 `.npy` 侧车矩阵按 `vector_row` 取向量（首选路径）。

    它是 `finalize()` 先原子落盘的权威原始产物，读它不必在连接上载入 `vec0` 扩展。
    `vector_row` 越界即报错，不静默跳块：少搬运的向量会让向量路静默少召回，
    而那是指标掉一截却查不出原因的事故。
    """
    vec_path = settings.vector_path
    if not vec_path.exists():
        return {}
    mat = np.load(vec_path)
    if mat.ndim != 2 or mat.shape[1] != dim:
        raise StoreError(f"{vec_path} 形状 {mat.shape} 与 dim={dim} 不符；换过嵌入模型必须重建索引")
    vecs: dict[str, np.ndarray] = {}
    for r in rows:
        vr = int(r["vector_row"])
        if vr < 0 or vr >= mat.shape[0]:
            raise StoreError(f"块 {r['chunk_id']} 的 vector_row={vr} 越界 {mat.shape}")
        vecs[r["chunk_id"]] = np.asarray(mat[vr], dtype=np.float32)
    return vecs


def _read_vec0(con: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[str, np.ndarray]:
    """从 `chunks_vec` 虚表读向量（回退路径：索引里没有 `.npy` 侧车时）。

    `SqliteVecStore.rebuild_from_scratch()` 会删掉 `.npy`，其 `finalize()` 又不写它，
    所以 sqlite-vec 全新 ingest 出来的索引只有 `chunks_vec` 有向量。
    必须在当前这条连接上 `enable_load_extension` 并载入 `vec0`，载完立刻关掉；
    载入失败硬报错，不退回零向量。
    """
    try:
        import sqlite_vec
    except ImportError as e:  # pragma: no cover - 取决于环境
        raise StoreError(
            f"本地索引既没有 {'.npy'} 侧车矩阵，又缺 sqlite-vec 扩展（{e}）；"
            "请装 `sqlite-vec>=0.1.9`，或用 `KUBEDOCS_STORE=sqlite` 重建索引以生成 .npy"
        ) from e
    con.enable_load_extension(True)
    try:
        sqlite_vec.load(con)
    except (sqlite3.Error, OSError) as e:
        raise StoreError(f"sqlite-vec 的 vec0 扩展加载失败（{type(e).__name__}: {e}）") from e
    finally:
        con.enable_load_extension(False)
    return {
        r["chunk_id"]: np.frombuffer(r["embedding"], dtype="<f4").astype(np.float32)
        for r in con.execute("SELECT chunk_id, embedding FROM chunks_vec")
    }


def _read_local(
    index_dir: str, domain: str
) -> tuple[list[DocInput], dict[str, list[ChunkInput]], dict[str, np.ndarray]]:
    """从本地索引读出文档/块/向量。

    向量首选 `chunk_vectors.npy` 侧车矩阵，缺失时回退 `chunks_vec` 虚表；
    两条路的取舍见 `_read_sidecar` / `_read_vec0`。
    """
    settings = Settings(index_dir=index_dir, domain=domain)
    db = settings.sqlite_path
    if not db.exists():
        raise StoreError(f"找不到本地索引：{db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        docs = [
            DocInput(
                doc_id=r["doc_id"],
                rel_path=r["rel_path"],
                title=r["title"],
                source_url=r["source_url"],
                content_hash=r["content_hash"],
            )
            for r in con.execute("SELECT doc_id, rel_path, title, source_url, content_hash FROM documents")
        ]
        by_doc: dict[str, list[ChunkInput]] = {d.doc_id: [] for d in docs}
        rows = list(
            con.execute(
                "SELECT chunk_id, doc_id, ordinal, title, breadcrumb, source_url, body,"
                " content_hash, n_chars, vector_row FROM chunks ORDER BY rowid"
            )
        )
        for r in rows:
            by_doc[r["doc_id"]].append(
                ChunkInput(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    ordinal=r["ordinal"],
                    title=r["title"],
                    breadcrumb=r["breadcrumb"],
                    source_url=r["source_url"],
                    body=r["body"],
                    content_hash=r["content_hash"],
                    n_chars=r["n_chars"],
                )
            )

        vecs = _read_sidecar(settings, rows, settings.embedding_dim)
        source = f"侧车矩阵 {settings.vector_path}"
        if not vecs:
            have_vec_table = (
                con.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunks_vec'").fetchone() is not None
            )
            if not have_vec_table:
                raise StoreError(f"本地索引没有向量：既无 {settings.vector_path}，也无 chunks_vec 虚表")
            vecs = _read_vec0(con, rows)
            source = "chunks_vec 虚表（需载入 vec0 扩展）"
        print(f"  向量来源：{source}，{len(vecs)} 条")
    finally:
        con.close()
    return docs, by_doc, vecs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把本地索引灌进 Postgres(pgvector)")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--domain", default="k8s_docs")
    ap.add_argument("--batch-docs", type=int, default=120, help="每批文档数")
    ap.add_argument("--keep", action="store_true", help="不清空目标 schema（默认先清空）")
    ap.add_argument("--sample", type=int, default=200, help="灌完后抽样校验的块数")
    args = ap.parse_args(argv)

    settings = Settings(index_dir=args.index_dir, domain=args.domain, store="pgvector")
    ok, why = probe(settings.database_url, timeout=5.0)
    if not ok:
        print(f"Postgres 不可达，未灌入任何数据：{why}", file=sys.stderr)
        print(f"（DSN = {settings.database_url}）", file=sys.stderr)
        return EXIT_USER

    docs, by_doc, vecs = _read_local(args.index_dir, args.domain)
    n_chunks = sum(len(v) for v in by_doc.values())
    print(f"本地索引：{settings.sqlite_path}")
    print(f"  文档 {len(docs)}，块 {n_chunks}，向量 {len(vecs)}")

    store = PgVectorStore(
        settings.database_url,
        schema=settings.domain,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        domain=settings.domain,
        pool_min=1,
        pool_max=4,
    )
    try:
        if not args.keep:
            store.rebuild_from_scratch()
            print("已清空目标 schema")
        t0 = time.perf_counter()
        done = 0
        for i in range(0, len(docs), args.batch_docs):
            batch = docs[i : i + args.batch_docs]
            chunks = [c for d in batch for c in by_doc.get(d.doc_id, [])]
            store.upsert_docs(batch, chunks, {c.chunk_id: vecs[c.chunk_id] for c in chunks})
            store.finalize()
            done += len(batch)
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{len(docs)} 文档（{rate:.0f} 文档/s）", flush=True)
        elapsed = time.perf_counter() - t0

        st = store.stats()
        print(f"\n灌入完成：{elapsed:.1f}s，Postgres 侧 {st.n_docs} 文档 / {st.n_chunks} 块")
        print(f"全文索引自检：{store.verify_fts()}")
        print(f"向量索引自检：{store.verify_hnsw()}")

        # 抽样校验：向量逐位相等，且文档/块/字符数口径一致
        rng = np.random.default_rng(7)
        ids = sorted(vecs)
        picks = [ids[i] for i in rng.choice(len(ids), size=min(args.sample, len(ids)), replace=False)]
        hash_of = {c.chunk_id: c.content_hash for chunks in by_doc.values() for c in chunks}
        got = store.existing_hash_to_vector({hash_of[cid] for cid in picks})
        bad = []
        for cid in picks:
            back = got.get(hash_of[cid])
            if back is None or not np.array_equal(back, vecs[cid]):
                bad.append(cid)
        print(f"抽样 {len(picks)} 块向量逐位比较：{'一致' if not bad else f'{len(bad)} 块不一致'}")
        local_chars = sum(c.n_chars for chunks in by_doc.values() for c in chunks)
        print(f"字符总数：本地 {local_chars} / Postgres {st.total_chars}")
        if bad or st.n_chunks != n_chunks or st.total_chars != local_chars:
            print("校验未通过——不做'看起来灌完了'的结论", file=sys.stderr)
            return EXIT_USER
        print("\n校验通过。现在可以跑：")
        print("  python -m kubedocs.cli stats --verify --store pgvector")
        print("  python -m kubedocs.cli eval --gate --store pgvector")
        return EXIT_OK
    except StoreError as e:
        print(f"灌入失败：{e}", file=sys.stderr)
        return EXIT_USER
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
