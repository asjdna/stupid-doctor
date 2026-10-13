"""`k8s_docs` 域包：把已有的 Kubernetes 文档检索问答包成"一个域"。

这一层刻意做得**很薄**——它不重新实现任何检索逻辑，只是把既有的
`ingest` / `HybridRetriever` / 金标集声明成 `DomainPack` 要求的形状。
薄的代价是没有，好处是：内核看到的只有协议，看不见 Kubernetes。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kubedocs.config import Settings
from kubedocs.embed import Embedder
from kubedocs.ingest import IngestReport, ingest
from kubedocs.retrieve.hybrid import HybridRetriever
from kubedocs.store.base import Store

if TYPE_CHECKING:
    from collections.abc import Callable

GOLD = Path("data/eval/retrieval_gold.jsonl")


class K8sDocsPack:
    name = "k8s_docs"
    description = "Kubernetes 官方文档检索问答：集群概念、kubectl 用法、资源清单字段、运维操作（英文语料）"
    # 英文：FTS5 的 porter + unicode61 已经够用，不需要预分词
    normalizer_name = "identity"
    embedding_model = "BAAI/bge-small-en-v1.5"
    embedding_dim = 384

    def corpus_root(self, settings: Settings) -> Path:
        return settings.corpus_path

    def build_index(
        self,
        store: Store,
        embedder: Embedder,
        settings: Settings,
        *,
        rebuild: bool = False,
        limit_files: int | None = None,
        batch_files: int = 60,
        progress: Callable[[str], None] | None = None,
    ) -> IngestReport:
        return ingest(
            store,
            embedder,
            settings,
            rebuild=rebuild,
            limit_files=limit_files,
            batch_files=batch_files,
            progress=progress,
        )

    def retriever(self, store: Store, embedder: Embedder, settings: Settings) -> HybridRetriever:
        return HybridRetriever(store, embedder, settings)

    def tools(self) -> list[object]:
        # 工具的 schema 化在 P2 做；P0 只需要域能注册与自省。
        return []

    def eval_sets(self) -> list[Path]:
        return [GOLD]


pack = K8sDocsPack()
