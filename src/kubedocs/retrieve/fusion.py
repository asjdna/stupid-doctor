"""多路召回融合：RRF（Reciprocal Rank Fusion）。

## 为什么用 RRF 而不是"归一化分数再加权"

BM25 分数与余弦相似度**量纲不同**（前者无上界、随语料长度漂移，后者在 [0,1]），
把两者归一化后加权需要调两个权重，且换语料就得重调。RRF 只用**名次**：

    score(d) = Σ_r 1 / (k + rank_r(d))

好处是**免标定、跨语料稳定**，这也是 Elasticsearch/OpenSearch 内置混合检索采用它的原因。
代价是丢掉"分差"信息——所以 `RRF_K` 需要按语料规模调（k 越小，头部名次差异越被放大）。

## 一个必须处理的现实问题：单路为空

某一路（例如稠密路因模型不可用）返回空时，不能把它的名次算成"全 0 参与融合"，
否则会把另一路的结果整体稀释。这里只在**有结果的路上**求和。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Fused:
    """融合结果：id、RRF 分数、以及它在各路里的名次（用于解释排序）。"""

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
    k:            平滑常数（原论文 60；k 越小越看重头部）
    weights:      可选的路权重（默认等权）。仅在"明确知道某路更可信"时才用——
                  它会把免标定的好处拿回去一部分，用之前先有评测数据。
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
    # 同分时按 key 排序，保证**确定性**（否则评测与闸门会随哈希种子抖动）
    out.sort(key=lambda f: (-f.score, f.key))
    return out


def truncate_dedup(items: Iterable[str], limit: int | None = None) -> list[str]:
    """保序去重（多路召回里同一文档重复出现是常态），可选截断。"""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
        if limit is not None and len(out) >= limit:
            break
    return out


def explanation(fused: Fused, names: dict[str, str] | None = None) -> str:
    """把一路排序解释成人话，用于调试与 `/search?explain=1`。

    样例：`concepts/foo::0002  分数=0.0325  名次{bm25=3, dense=7}`
    """
    label = names or {}
    parts = ", ".join(f"{label.get(n, n)}={r}" for n, r in sorted(fused.ranks.items()))
    return f"{fused.key}  分数={fused.score:.4f}  名次{{{parts}}}"
