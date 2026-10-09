"""把现有 SQLite 索引里的向量灌进 Milvus（本地文件），**不重新编码**。

## 为什么需要这个脚本

换后端时，向量本身一个字节都没变，变的只是"存放位置"（`.npy` 矩阵 / `chunks_vec`
虚表 → Milvus 的 HNSW 索引）。全量重编码在本机是 1.9 万块 × 约 9.4 块/秒 ≈ 33 分钟，
而这一步只需要十来秒。

它同时承担第二个职责：**把索引元信息切换到 milvus**。`MilvusStore` 与
`SqliteVecStore` 一样用 `vector_backend` 这道闸门挡住"同一份索引在两个后端之间来回写"
（块表会更新、另一个后端的向量表不会，结果是静默读到陈旧向量）。
真正搬向量的动作在 `MilvusStore.adopt_legacy_matrix()` 里（与 `sqlite-vec` 的同名方法
一个模式），本脚本只负责参数解析、进度打印与**逐位抽样校验**。

## 从哪读向量

按 `chunks.vector_row` ↔ `.npy` 矩阵行号的对应关系读（旧实现 `finalize()` 写入的那个
映射）。越界、行数不符一律报错退出，不做"部分迁移也算成功"的假动作。
**只读原索引**：Milvus 的数据写到一个独立的 `.db` 文件里。

## 用法

    python scripts/migrate_to_milvus.py                     # 默认 data/index/k8s_docs
    python scripts/migrate_to_milvus.py --domain clinic
    python scripts/migrate_to_milvus.py --check-only        # 只校验，不写入
    python scripts/migrate_to_milvus.py --milvus-db D:/x/milvus.db

迁移幂等：每次先 drop 集合再灌，重复跑不会产生重复行。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.normalize import build_normalizer  # noqa: E402
from kubedocs.store.base import StoreError  # noqa: E402
from kubedocs.store.milvus_store import MilvusStore  # noqa: E402

EXIT_OK = 0
EXIT_USER = 1


def _verify(store: MilvusStore, mat: np.ndarray, pairs: list[tuple[str, int, str]],
            sample: int) -> tuple[bool, str]:
    """抽样核对：灌进去的向量必须与矩阵**逐位相等**，不是"差不多"。

    走的是公开读取路径（`existing_hash_to_vector`），顺便证明那条路径真的能用；
    自己再写一条旁路查询只能证明"我写的另一条查询是对的"。
    """
    if not pairs:
        return True, "索引为空，无需校验"
    rng = np.random.default_rng(20261008)
    picks = [pairs[i] for i in rng.choice(len(pairs), size=min(sample, len(pairs)),
                                          replace=False)]
    wanted = {h for _, _, h in picks}
    got = store.existing_hash_to_vector(wanted)
    missing = wanted - set(got)
    if missing:
        return False, f"{len(missing)} 个 content_hash 取不到向量（例：{sorted(missing)[:3]}）"
    for cid, row, h in picks:
        if not np.array_equal(got[h], np.asarray(mat[row], dtype=np.float32)):
            return False, f"块 {cid} 的向量与矩阵不一致（逐位比较）"
    return True, f"抽样 {len(picks)} 块逐位一致"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把现有索引的向量灌进 Milvus（本地文件）")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--domain", default="k8s_docs")
    ap.add_argument("--milvus-db", default="", help="Milvus 数据文件（默认 <索引目录>/milvus.db）")
    ap.add_argument("--sample", type=int, default=200, help="灌完后抽样的块数")
    ap.add_argument("--check-only", action="store_true", help="只校验，不写入")
    ap.add_argument("--detach", action="store_true",
                    help="把索引从 milvus 上摘下来（清空集合与后端标记），之后换回别的后端需 --rebuild")
    args = ap.parse_args(argv)

    over: dict[str, object] = {"index_dir": args.index_dir, "domain": args.domain,
                               "store": "milvus"}
    if args.milvus_db:
        over["milvus_db"] = args.milvus_db
    settings = Settings(**over)  # type: ignore[arg-type]

    store = MilvusStore(
        db_path=settings.sqlite_path, vector_path=settings.vector_path,
        dim=settings.embedding_dim, embedding_model=settings.effective_embedding_model,
        normalizer=build_normalizer(settings.normalizer), domain=settings.domain,
        milvus_path=settings.milvus_path,
        hnsw_m=settings.milvus_hnsw_m,
        hnsw_ef_construction=settings.milvus_hnsw_ef_construction,
        hnsw_ef_search=settings.milvus_hnsw_ef_search,
        metric=settings.milvus_metric,
        # 只有这里会打开闸门：本脚本是"向量真的被搬过去了"的那条路径。
        # 普通代码路径（CLI / API / ingest）一律不许抢别的后端的索引。
        allow_backend_takeover=True,
    )
    try:
        if args.detach:
            before = store.verify_vec()
            store.detach()
            print(f"已从 milvus 摘除：集合已清空，后端标记已撤销（此前自检：{before}）")
            print("块表与全文索引未改动。换回 sqlite-vec / sqlite 请跑：")
            print("  python -m kubedocs.cli ingest --rebuild --store sqlite-vec")
            return EXIT_OK

        with store._lock:  # noqa: SLF001 - 迁移脚本与实现同属存储层，用内部连接读映射
            pairs = [(str(r["chunk_id"]), int(r["vector_row"]), str(r["content_hash"]))
                     for r in store._con.execute(
                         "SELECT chunk_id, vector_row, content_hash FROM chunks")]
        print(f"块表：{settings.sqlite_path}")
        print(f"向量源：{settings.vector_path}")
        print(f"向量目标：{settings.milvus_path}")
        print(f"块数：{len(pairs)}")
        if not pairs:
            print("索引为空，没有可迁移的向量（先跑 ingest）", file=sys.stderr)
            return EXIT_USER
        mat = np.load(settings.vector_path)

        if not args.check_only:
            t0 = time.perf_counter()
            n = store.adopt_legacy_matrix()
            print(f"已灌入 Milvus：{n} 行（{time.perf_counter() - t0:.1f}s）")

        ok, why = _verify(store, mat, pairs, args.sample)
        print(f"抽样校验：{'通过' if ok else '不通过'} —— {why}")
        print(f"向量索引自检：{store.verify_vec()}")
        print(f"全文索引自检：{store.verify_fts()}")
        if not ok or store.verify_vec() != "ok":
            return EXIT_USER
        print("\n迁移完成。现在可以直接跑：")
        print(f"  python -m kubedocs.cli stats --verify --store milvus --domain {settings.domain}")
        print(f"  python -m kubedocs.cli eval --gate --store milvus --domain {settings.domain}")
        return EXIT_OK
    except StoreError as e:
        print(f"迁移失败：{e}", file=sys.stderr)
        return EXIT_USER
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
