"""存储层的数据契约与接口。

分两层写（接口 + 实现）的理由：检索与生成代码只依赖这里的**窄接口**，
将来把本地 SQLite 换成生产 Postgres+pgvector 时，改动被关在这一个目录里，
而不是散落到 ingest / retrieve / api 各处。

接口刻意保持"窄"，但**窄不等于缺**：摄取（`ingest`）与索引体检（CLI `stats`、
评测的语料指纹）真正需要的运维操作（重建、全文索引自检、元信息、文档清单）
同样写在这里——不写进来，pgvector 版就得靠 `cast` 或 `getattr` 绕过类型检查，
那等于把"可替换"这件事作废。判断标准是"调用方是否真的需要"，
而不是"这个操作听起来像不像运维"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


class StoreError(RuntimeError):
    """存储层可预期错误（配置/一致性问题），调用方应把它当用户错误处理。"""


class MissingVectorError(StoreError):
    """某块的向量既不在本次运行的结果里、也不在旧索引里——不能静默填 0。"""


@dataclass(frozen=True)
class DocInput:
    """一篇文档的登记信息（正文不存这里，正文在块里）。"""

    doc_id: str
    rel_path: str
    title: str
    source_url: str
    content_hash: str


@dataclass(frozen=True)
class ChunkInput:
    """一个待写入的块。向量由摄取层算好传入——存储层不做编码，避免职责混淆。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    title: str
    breadcrumb: str
    source_url: str
    body: str
    content_hash: str
    n_chars: int


@dataclass(frozen=True)
class ChunkHit:
    """检索/取回命中的块：正文 + 足以生成引用的全部来源信息。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    title: str
    breadcrumb: str
    source_url: str
    body: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """人可读的引用标签，例如 `Pod CPU 限制 › 配置 › 步骤`。"""
        return f"{self.title} › {self.breadcrumb}" if self.breadcrumb else self.title


@dataclass(frozen=True)
class StoreStats:
    n_docs: int
    n_chunks: int
    total_chars: int
    n_with_code: int
    embedding_model: str
    embedding_dim: int
    indexed_at: str
    domain: str = ""
    normalizer: str = ""

    @property
    def mean_chars(self) -> float:
        return self.total_chars / self.n_chunks if self.n_chunks else 0.0


@runtime_checkable
class Store(Protocol):
    """索引存储的最小接口。`rank` 从 1 起，越小越相关。

    三个实现：`sqlite_store.SqliteStore`（numpy 暴力内积）、
    `sqlite_vec_store.SqliteVecStore`（sqlite-vec，本地默认）、
    `pgvector_store.PgVectorStore`（Postgres + pgvector HNSW，生产）。
    三者必须给出**同构的返回形态与排序语义**，差异只允许出现在"近似检索"上，
    且必须在 ADR-0004 里写明——否则后端一换，指标就变，而没人知道为什么。
    """

    def existing_doc_hashes(self) -> dict[str, str]:
        """rel_path -> 文档级 content_hash，用于判断哪些文件没变、可整篇跳过。"""

    def existing_hash_to_vector(self, hashes: set[str]) -> dict[str, Vector]:
        """content_hash -> 已算好的向量，用于跨篇复用（同一模板块不重复编码）。"""

    def delete_docs(self, rel_paths: set[str]) -> int:
        """按 rel_path 删除整篇（含其所有块、全文索引行与向量索引项）。返回删除的块数。"""

    def upsert_docs(self, docs: list[DocInput], chunks: list[ChunkInput], vectors: dict[str, Vector]) -> None:
        """写入/覆盖若干文档及其块。同一 doc 的旧块会被先清掉（覆盖语义）。"""

    def finalize(self) -> None:
        """收尾：把本轮新增的向量落到持久结构。每次摄取跑完必须调用。"""

    def keyword_search(self, query: str, limit: int) -> list[tuple[str, int]]:
        """全文路：返回 (chunk_id, rank) 按相关性降序。"""

    def dense_search(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        """向量路：返回 (chunk_id, rank) 按余弦相似度降序。"""

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkHit]:
        """按 chunk_id 批量取回块内容（保序由调用方负责）。"""

    def existing_doc_ids(self) -> set[str]:
        """索引里已有文档的 doc_id 集合，用于校验标注集引用的文档确实存在。"""

    def doc_texts(self, doc_ids: list[str]) -> dict[str, str]:
        """把指定文档的所有块正文拼起来（供标注集自检用，不是检索路径）。"""

    def stats(self) -> StoreStats: ...

    def close(self) -> None:
        """释放连接/连接池。进程退出前调用（HTTP 层在 lifespan 关闭时调用）。"""

    # ------------------------------------------------------------------ 运维
    def rebuild_from_scratch(self) -> None:
        """清空全部数据（文档、块、全文索引、向量索引）。换模型/换分块参数后重建用。"""

    def verify_fts(self) -> str:
        """全文索引一致性自检。返回 `"ok"` 或 `"FAIL: <原因>"`（不抛异常）。"""

    def meta_dump(self) -> dict[str, str]:
        """索引元信息（模型、维度、归一化器、时间戳等），供 CLI `stats` 展示。"""

    def doc_summary(self, limit: int = 20) -> list[dict[str, object]]:
        """按 rel_path 列出文档及其块数（评测的语料指纹用它，全量时传大 limit）。"""
