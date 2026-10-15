"""多后端测试共用的测试语料（不是测试文件本身，pytest 不会收集它）。

三篇文档 / 五个块，配**已归一化**的向量。必须各后端共用同一份：一致性测试的核心主张是
"同一批输入在不同后端上给出同样的结果"，各测试文件各造一份，对照就失去意义。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kubedocs.store.base import ChunkInput, DocInput

DIM = 32
"""与 conftest 的 DIM 一致；向量空间大小不影响契约本身。"""

_AXES = ("a", "b", "c", "d", "e")


def vec(**axes: float) -> np.ndarray:
    """构造 unit 向量：`vec(a=1.0)` = e0 方向的单位向量。

    刻意归一化：契约"内积序 == 余弦序"只对归一化向量成立，而摄取层写入索引的向量都是归一化的。
    """
    v = np.zeros(DIM, dtype=np.float32)
    for name, value in axes.items():
        v[_AXES.index(name)] = value
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n else v


@dataclass
class Seed:
    docs: list[DocInput]
    chunks: list[ChunkInput]
    vectors: dict[str, np.ndarray]


def make_seed() -> Seed:
    """固定的测试语料：期望值是写死的（见各测试里的断言）。"""
    spec = [
        ("alpha", 0, "kubelet eviction hard threshold", vec(a=1.0)),
        ("alpha", 1, "kubelet eviction soft threshold", vec(a=0.8, b=0.6)),
        ("beta", 0, "crashloopbackoff restart policy backoff", vec(b=1.0)),
        ("beta", 1, "imagepullbackoff registry credential", vec(b=0.6, c=0.8)),
        ("gamma", 0, "resourcequota namespace limitrange", vec(c=1.0)),
    ]
    docs: dict[str, DocInput] = {}
    chunks: list[ChunkInput] = []
    vectors: dict[str, np.ndarray] = {}
    for doc_id, ordinal, body, vector in spec:
        cid = f"{doc_id}::{ordinal:04d}"
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
            )
        )
        vectors[cid] = vector
    return Seed(docs=list(docs.values()), chunks=chunks, vectors=vectors)
