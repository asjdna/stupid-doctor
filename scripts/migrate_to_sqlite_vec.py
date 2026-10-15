"""把旧实现的 `.npy` 向量矩阵搬进 sqlite-vec 的 `chunks_vec` 虚表，不重新编码。

按 `chunks.vector_row` ↔ 矩阵行号的映射逐条搬运：向量一个字节都没变，只是换了存放位置，
没必要重跑嵌入。搬完用公开读取路径 `existing_hash_to_vector` 抽样逐位校验。

用法：
    python scripts/migrate_to_sqlite_vec.py                    # 默认 data/index/k8s_docs
    python scripts/migrate_to_sqlite_vec.py --domain clinic
    python scripts/migrate_to_sqlite_vec.py --check-only       # 只校验，不写入

前置条件：索引目录下存在 `chunk_vectors.npy`，缺失即报错、不做部分迁移；迁移幂等，每次先清空 `chunks_vec` 再重建。
退出码：0 迁移并自检通过 / 1 矩阵缺失、StoreError 或抽样与向量自检不通过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.normalize import build_normalizer  # noqa: E402
from kubedocs.store.base import StoreError  # noqa: E402
from kubedocs.store.sqlite_vec_store import SqliteVecStore  # noqa: E402

EXIT_OK = 0
EXIT_USER = 1


def _verify(
    store: SqliteVecStore, expected: dict[str, np.ndarray], hashes_of: dict[str, str], sample: int
) -> tuple[bool, str]:
    """抽样核对：搬运结果必须与矩阵逐位相等，不接受近似。

    走公开读取路径（`existing_hash_to_vector`），顺便证明那条路径真的能用。
    """
    ids = sorted(expected)
    if not ids:
        return True, "索引为空，无需校验"
    rng = np.random.default_rng(20260924)
    picks = [ids[i] for i in rng.choice(len(ids), size=min(sample, len(ids)), replace=False)]
    wanted = {hashes_of[cid] for cid in picks}
    got = store.existing_hash_to_vector(wanted)
    missing = wanted - set(got)
    if missing:
        return False, f"{len(missing)} 个 content_hash 取不到向量（例：{sorted(missing)[:3]}）"
    for cid in picks:
        if not np.array_equal(got[hashes_of[cid]], expected[cid]):
            return False, f"块 {cid} 的向量与矩阵不一致（逐位比较）"
    return True, f"抽样 {len(picks)} 块逐位一致"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把 .npy 向量矩阵迁移进 sqlite-vec")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--domain", default="k8s_docs")
    ap.add_argument("--sample", type=int, default=200, help="迁移后抽样的块数")
    ap.add_argument("--check-only", action="store_true", help="只校验现有向量表，不写入")
    args = ap.parse_args(argv)

    settings = Settings(index_dir=args.index_dir, domain=args.domain, store="sqlite-vec")
    vec_path = settings.vector_path
    if not vec_path.exists():
        print(f"找不到旧向量矩阵：{vec_path}", file=sys.stderr)
        print("没有可迁移的向量。若索引本来就是 sqlite-vec 建的，直接跑 ingest 即可。", file=sys.stderr)
        return EXIT_USER

    store = SqliteVecStore(
        db_path=settings.sqlite_path,
        vector_path=vec_path,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        normalizer=build_normalizer(settings.normalizer),
        domain=settings.domain,
    )
    try:
        mat = np.load(vec_path)
        rows = store.doc_summary(limit=10**9)
        _ = rows  # 顺带确认库可读
        with store._lock:  # noqa: SLF001 - 迁移脚本与实现同属存储层，用内部连接读映射
            pairs = [
                (r["chunk_id"], int(r["vector_row"]), r["content_hash"])
                for r in store._con.execute("SELECT chunk_id, vector_row, content_hash FROM chunks")
            ]
        expected = {cid: np.asarray(mat[vr], dtype=np.float32) for cid, vr, _ in pairs}
        hashes_of = {cid: h for cid, _, h in pairs}
        print(f"索引：{settings.sqlite_path}")
        print(f"矩阵：{vec_path}  形状 {mat.shape}")
        print(f"块数：{len(pairs)}")

        if not args.check_only:
            n = store.adopt_legacy_matrix()
            print(f"已写入 chunks_vec：{n} 行")
        ok, why = _verify(store, expected, hashes_of, args.sample)
        print(f"校验：{'通过' if ok else '不通过'} —— {why}")
        print(f"向量索引自检：{store.verify_vec()}")
        print(f"全文索引自检：{store.verify_fts()}")
        if not ok or store.verify_vec() != "ok":
            return EXIT_USER
        print("\n迁移完成。现在可以直接跑：")
        print("  python -m kubedocs.cli stats --verify")
        print("  python -m kubedocs.cli eval --gate")
        return EXIT_OK
    except StoreError as e:
        print(f"迁移失败：{e}", file=sys.stderr)
        return EXIT_USER
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
