"""检索层：多路召回 + 融合 + 重排。"""

from __future__ import annotations

from kubedocs.retrieve.fusion import Fused, explanation, rrf_fuse, truncate_dedup
from kubedocs.retrieve.hybrid import (
    HybridRetriever,
    RetrievedChunk,
    SearchResult,
)
from kubedocs.retrieve.rerank import CrossEncoderReranker, Reranker, apply_rerank

__all__ = [
    "CrossEncoderReranker",
    "Fused",
    "HybridRetriever",
    "Reranker",
    "RetrievedChunk",
    "SearchResult",
    "apply_rerank",
    "explanation",
    "rrf_fuse",
    "truncate_dedup",
]
