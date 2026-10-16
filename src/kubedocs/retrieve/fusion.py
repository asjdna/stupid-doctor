"""多路召回融合：RRF（Reciprocal Rank Fusion）。

score(d) = Σ_r 1 / (k + rank_r(d))

BM25 分数与余弦相似度量纲不同（前者无上界、随语料长度漂移，后者在 [0,1]），
归一化后加权需要标定权重、换语料还得重调；RRF 只用名次，免标定、跨语料稳定，
代价是丢掉分差信息，因此 `RRF_K` 需要按语料规模调（k 越小越看重头部名次）。

某一路返回空时只在**有结果的路上**求和，不把空路当作"全 0 名次"参与融合，
否则会把另一路的结果整体稀释。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fused:
    """融合结果：文档键、RRF 分数，以及它在各路里的名次（用于解释排序）。"""

    key: str
    score: float
    ranks: dict[str, int]  # 路名 -> 名次（从 1 起）


def rrf_fuse(
    ranked_lists: Sequence[tuple[str, Sequence[str]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[Fused]:
    """按名次融合多路召回结果，按分数降序返回。

    ranked_lists: [(路名, [文档键按相关性降序]), ...]
    k:            平滑常数（原论文 60；k 越小越看重头部名次）
    weights:      可选的路权重（默认等权）。只在明确知道某路更可信时才用：
                  它会部分抵消免标定的好处，用之前先有评测数据。
    """
    if k < 1:
        raise ValueError(f"k 必须 >= 1，收到 {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, keys in ranked_lists:
        w = 1.0 if not weights else float(weights.get(name, 1.0))
        for rank, key in enumerate(keys, start=1):
            if key is None:
                continue
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            ranks.setdefault(key, {})[name] = rank

    out = [Fused(key=key, score=round(score, 6), ranks=ranks.get(key, {})) for key, score in scores.items()]
    # 同分时按 key 排序：排序结果必须确定，否则下游评测会随哈希种子抖动
    out.sort(key=lambda f: (-f.score, f.key))
    return out


def explanation(fused: Fused, names: dict[str, str] | None = None) -> str:
    """把一路排序解释成人话，用于调试与 `/search?explain=1`。

    样例：`concepts/foo::0002  分数=0.0325  名次{bm25=3, dense=7}`
    """
    label = names or {}
    parts = ", ".join(f"{label.get(n, n)}={r}" for n, r in sorted(fused.ranks.items()))
    return f"{fused.key}  分数={fused.score:.4f}  名次{{{parts}}}"
