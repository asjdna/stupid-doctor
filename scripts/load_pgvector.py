"""把本地 SQLite 索引**原样**灌进 Postgres（pgvector 后端），不重新编码。

## 为什么要这个脚本

全量重编码 1.9 万块在本机约 33 分钟，而向量与正文都已经在本地索引里了。
这份脚本做的是"搬家"：把 `documents` / `chunks` / 向量按原样写进 Postgres 的
`<domain>` schema，然后在库里**逐位校验**抽样向量与统计口径。

## 它不做什么

- 不重新切块、不重新编码（那是 `ingest` 的职责，且与目标端无关）；
- 不做增量对比（默认先清空目标 schema 再灌，保证"镜像"而不是"叠加"；
  `--keep` 可以关掉清空，用于向已有数据追加）。

## 用法

    python scripts/load_pgvector.py                        # data/index/k8s_docs -> schema k8s_docs
    python scripts/load_pgvector.py --batch-docs 200       # 每批文档数
    python scripts/load_pgvector.py --keep                 # 不清空目标 schema
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


def _read_local(index_dir: str, domain: str) -> tuple[
        list[DocInput], dict[str, list[ChunkInput]], dict[str, np.ndarray]]:
    """从本地索引读出文档/块/向量。

    向量来源分两种：`chunks_vec` 虚表（sqlite-vec 建的索引）优先，
    退到 `chunk_vectors.npy`（旧实现）。**按 vector_row 取行**而不是按顺序假设——
    旧实现的矩阵行号就是 `vector_row`，这是它 `finalize()` 里写下的映射。
    """
    settings = Settings(index_dir=index_dir, domain=domain)
    db = settings.sqlite_path
    if not db.exists():
        raise StoreError(f"找不到本地索引：{db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        docs = [DocInput(doc_id=r["doc_id"], rel_path=r["rel_path"], title=r["title"],
                         source_url=r["source_url"], content_hash=r["content_hash"])
                for r in con.execute(
                    "SELECT doc_id, rel_path, title, source_url, content_hash FROM documents")]
        by_doc: dict[str, list[ChunkInput]] = {d.doc_id: [] for d in docs}
        rows = list(con.execute(
            "SELECT chunk_id, doc_id, ordinal, title, breadcrumb, source_url, body,"
            " content_hash, n_chars, vector_row FROM chunks ORDER BY rowid"))
        for r in rows:
            by_doc[r["doc_id"]].append(ChunkInput(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], ordinal=r["ordinal"],
                title=r["title"], breadcrumb=r["breadcrumb"], source_url=r["source_url"],
                body=r["body"], content_hash=r["content_hash"], n_chars=r["n_chars"]))

        have_vec_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunks_vec'").fetchone() is not None
        vecs: dict[str, np.ndarray] = {}
        if have_vec_table:
            for r in con.execute("SELECT chunk_id, embedding FROM chunks_vec"):
                vecs[r["chunk_id"]] = np.frombuffer(r["embedding"], dtype="<f4").astype(np.float32)
        if not vecs:
            vec_path = settings.vector_path
            if not vec_path.exists():
                raise StoreError(f"本地索引没有向量：既无 chunks_vec，也无 {vec_path}")
            mat = np.load(vec_path)
            for r in rows:
                vr = int(r["vector_row"])
                if vr < 0 or vr >= mat.shape[0]:
                    raise StoreError(f"块 {r['chunk_id']} 的 vector_row={vr} 越界 {mat.shape}")
                vecs[r["chunk_id"]] = np.asarray(mat[vr], dtype=np.float32)
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
        settings.database_url, schema=settings.domain, dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        domain=settings.domain, pool_min=1, pool_max=4)
    try:
        if not args.keep:
            store.rebuild_from_scratch()
            print("已清空目标 schema")
        t0 = time.perf_counter()
        done = 0
        for i in range(0, len(docs), args.batch_docs):
            batch = docs[i:i + args.batch_docs]
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

        # 抽样校验：向量逐位相等 + 统计口径一致
        rng = np.random.default_rng(7)
        ids = sorted(vecs)
        picks = [ids[i] for i in rng.choice(len(ids), size=min(args.sample, len(ids)),
                                           replace=False)]
        hash_of = {c.chunk_id: c.content_hash
                   for chunks in by_doc.values() for c in chunks}
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
