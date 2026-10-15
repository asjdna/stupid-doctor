"""检索评测指标：全部以**文档**为单位判定相关性。

标注与指标都在文档级：块是**实现产物**，换一次 `chunk_max_chars` 块 id 与边界全变，
块级标注立刻作废；文档级标注（"这个问题的答案在 `concepts/.../pod-lifecycle.md` 里"）
在切块参数变化后依然有效，可以长期作为回归基线。代价是它比块级宽松一些——所以
`must_contain` 自检（见 `harness.py`）负责保证"标注的文档确实讲了这个主题"。

四个指标：`hit@k` 看 top-k 里有没有命中（最贴近"用户有没有拿到答案"）；
`recall@k` 看多目标时命中了几个（覆盖面）；`mrr@k` 看第一个正确结果排多前
（要不要往下翻）；`ndcg@k` 是位置加权的整体排序质量（更细腻，也更不直观）。
四个一起看，避免被单一指标误导：把正确答案从第 1 位挪到第 2 位，hit 不变、MRR 掉一半。
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _dedup_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def first_relevant_rank(ranked_docs: Sequence[str], relevant: set[str], k: int) -> int | None:
    """第一个相关结果的名次（从 1 起）；top-k 内没有则 None。"""
    for i, doc in enumerate(_dedup_keep_order(ranked_docs)[:k], start=1):
        if doc in relevant:
            return i
    return None


def hit_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if first_relevant_rank(ranked_docs, relevant, k) is not None else 0.0


def recall_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant 不能为空——空标注会让 recall 恒为 1，是个永远不会失败的指标")
    got = set(_dedup_keep_order(ranked_docs)[:k]) & relevant
    return len(got) / len(relevant)


def mrr_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    rank = first_relevant_rank(ranked_docs, relevant, k)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    """二值相关性的 nDCG@k。"""
    if not relevant:
        raise ValueError("relevant 不能为空")
    dcg = 0.0
    for i, doc in enumerate(_dedup_keep_order(ranked_docs)[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_one(ranked_docs: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        "hit": hit_at_k(ranked_docs, relevant, k),
        "recall": recall_at_k(ranked_docs, relevant, k),
        "mrr": mrr_at_k(ranked_docs, relevant, k),
        "ndcg": ndcg_at_k(ranked_docs, relevant, k),
    }


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """宏平均（每个查询等权）。空集合返回空 dict 而不是 0——
    "没有数据"与"得分为 0"是两件不同的事，混在一起基线比较就失去意义。"""
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: sum(r[key] for r in rows) / len(rows) for key in keys}


def percentile(values: Sequence[float], pct: float) -> float:
    """线性插值分位数（用于检索延迟 p50/p95）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
