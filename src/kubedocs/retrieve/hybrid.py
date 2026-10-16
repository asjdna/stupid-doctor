"""混合检索：FTS5(BM25) + 向量近邻 → RRF 融合 →（可选）cross-encoder 精排。

两路互补：

- **全文路（BM25）** 强在精确词元：`kubelet`、`--eviction-hard`、
  `topologySpreadConstraints` 这类标识符只要出现就命中，查报错信息与参数名靠它；
- **向量路** 强在改述：问"pod 一直重启怎么办"，文档里写的是 "CrashLoopBackOff"
  与 "restart policy"，字面不等但语义相邻，纯 BM25 会漏。

两路召回后按名次做 RRF 融合（免标定、跨语料稳定，理由见 `fusion.py`）。

嵌入模型不可用时查询不会失败：退化成纯 BM25，并把原因写进
`diagnostics["degraded"]`。降级必须留下记录——静默变成纯关键词检索的系统，
在评测里只表现为"分数莫名掉了几个点"，很难定位。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kubedocs.config import Settings
from kubedocs.embed import Embedder, EmbedderError
from kubedocs.observability import RETRIEVE_LATENCY
from kubedocs.retrieve.fusion import Fused, rrf_fuse
from kubedocs.retrieve.rerank import CrossEncoderReranker, Reranker, apply_rerank
from kubedocs.store.base import ChunkHit, Store


@dataclass
class RetrievedChunk:
    """一条最终结果：块内容 + 来源 + 可解释的排序依据。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    title: str
    breadcrumb: str
    source_url: str
    body: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return f"{self.title} › {self.breadcrumb}" if self.breadcrumb else self.title

    @classmethod
    def from_hit(cls, hit: ChunkHit) -> RetrievedChunk:
        return cls(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            ordinal=hit.ordinal,
            title=hit.title,
            breadcrumb=hit.breadcrumb,
            source_url=hit.source_url,
            body=hit.body,
        )

    def explain(self) -> str:
        parts = ", ".join(f"{n}={r}" for n, r in sorted(self.ranks.items()))
        tail = f"  精排={self.rerank_score:.3f}" if self.rerank_score is not None else ""
        return f"{self.chunk_id}  RRF={self.score:.4f}  名次{{{parts}}}{tail}"


@dataclass
class SearchResult:
    query: str
    results: list[RetrievedChunk]
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [r.chunk_id for r in self.results]

    def explain_all(self) -> str:
        if not self.results:
            return "(无结果)"
        return "\n".join(f"{i:>2}. {r.explain()}" for i, r in enumerate(self.results, 1))


class HybridRetriever:
    """检索入口。构造一次可复用（嵌入模型与 SQLite 连接都会被持有）。"""

    def __init__(
        self, store: Store, embedder: Embedder, settings: Settings, reranker: Reranker | None = None
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._reranker = reranker
        self._reranker_built = reranker is not None

    def _get_reranker(self) -> Reranker:
        if not self._reranker_built:
            self._reranker = CrossEncoderReranker(self.settings.rerank_model)
            self._reranker_built = True
        assert self._reranker is not None
        return self._reranker

    # ------------------------------------------------------------------ 单路
    def keyword_channel(self, query: str, limit: int) -> tuple[list[str], dict[str, object]]:
        t0 = time.perf_counter()
        hits = self.store.keyword_search(query, limit)
        elapsed = time.perf_counter() - t0
        RETRIEVE_LATENCY.labels(channel="bm25").observe(elapsed)
        return [cid for cid, _ in hits], {
            "n": len(hits),
            "ms": round(elapsed * 1000, 1),
        }

    def dense_channel(self, query: str, limit: int) -> tuple[list[str], dict[str, object], str | None]:
        t0 = time.perf_counter()
        try:
            qv = self.embedder.encode([query], is_query=True)[0]
        except (EmbedderError, RuntimeError, OSError, ValueError) as e:
            # 嵌入不可用 -> 退化到单路，但把原因带出去
            return (
                [],
                {"n": 0, "ms": round((time.perf_counter() - t0) * 1000, 1)},
                f"{type(e).__name__}: {e}",
            )
        try:
            hits = self.store.dense_search(qv, limit)
        except Exception as e:  # noqa: BLE001 - 存储层不可用同样降级，不向上抛
            # 存储层的失败（sqlite-vec 行数不一致、维度不符等）也必须走同一条降级
            # 路径：返回空结果并在 diagnostics["degraded"] 里留下原因，交由上层用
            # 纯全文结果兜底。
            return (
                [],
                {"n": 0, "ms": round((time.perf_counter() - t0) * 1000, 1)},
                f"{type(e).__name__}: {e}",
            )
        elapsed = time.perf_counter() - t0
        RETRIEVE_LATENCY.labels(channel="dense").observe(elapsed)
        return (
            [cid for cid, _ in hits],
            {
                "n": len(hits),
                "ms": round(elapsed * 1000, 1),
            },
            None,
        )

    # ------------------------------------------------------------------ 主流程
    def search(
        self, query: str, *, top_k: int | None = None, channels: tuple[str, ...] = ("bm25", "dense")
    ) -> SearchResult:
        """跑一次检索。

        `channels` 取 `("bm25", "dense")` 的子集即可只跑一路，用于单路消融对比。
        """
        top_k = top_k or self.settings.top_k
        degraded: list[str] = []
        diag: dict[str, object] = {"query": query, "top_k": top_k, "channels": list(channels)}
        if not query.strip():
            diag["note"] = "空查询"
            return SearchResult(query=query, results=[], diagnostics=diag)

        t_all = time.perf_counter()
        kw_ids, kw_diag = (
            self.keyword_channel(query, self.settings.fts_candidate_k)
            if "bm25" in channels
            else ([], {"n": 0, "ms": 0.0, "skipped": True})
        )
        if "dense" in channels:
            dn_ids, dn_diag, dense_err = self.dense_channel(query, self.settings.dense_candidate_k)
        else:
            dn_ids, dn_diag, dense_err = [], {"n": 0, "ms": 0.0, "skipped": True}, None
        diag["channels_detail"] = {"bm25": kw_diag, "dense": dn_diag}
        if dense_err:
            degraded.append(f"向量路不可用，已降级为纯全文检索：{dense_err}")

        fused: list[Fused] = rrf_fuse([("bm25", kw_ids), ("dense", dn_ids)], k=self.settings.rrf_k)
        diag["n_fused"] = len(fused)
        diag["overlap"] = len(set(kw_ids) & set(dn_ids))

        # 精排前先取更多候选，否则精排只能在 RRF 的 top_k 里重排，收益受限
        pool = (
            fused[: max(top_k, self.settings.rerank_top_k) * 3]
            if self.settings.rerank_enable
            else fused[:top_k]
        )

        hits = self.store.get_chunks([f.key for f in pool])
        results: list[RetrievedChunk] = []
        for f in pool:
            hit = hits.get(f.key)
            if hit is None:
                continue  # 索引与结果不一致时宁可少给，也不要给空壳
            rc = RetrievedChunk.from_hit(hit)
            rc.score = f.score
            rc.ranks = f.ranks
            results.append(rc)

        if self.settings.rerank_enable and results:
            scores, err = apply_rerank(self._get_reranker(), query, [r.body for r in results])
            if err:
                degraded.append(f"精排不可用，保持 RRF 顺序：{err}")
            else:
                for rc, s in zip(results, scores, strict=True):
                    rc.rerank_score = s
                # 分数相同按 chunk_id 定序，保证可重复
                results.sort(key=lambda r: (-(r.rerank_score or 0.0), r.chunk_id))
                diag["reranked"] = True
        diag["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        diag["degraded"] = degraded

        return SearchResult(query=query, results=results[:top_k], diagnostics=diag)
