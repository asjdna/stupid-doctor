"""评测层：标注集、指标、对照实验与闸门。"""

from __future__ import annotations

from kubedocs.evaluation.harness import (
    GoldProblem,
    GoldRow,
    check_gate,
    gold_sha256,
    load_gold,
    run_eval,
    run_mode,
    validate_gold,
)
from kubedocs.evaluation.metrics import (
    aggregate,
    evaluate_one,
    first_relevant_rank,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)

__all__ = [
    "GoldProblem",
    "GoldRow",
    "aggregate",
    "check_gate",
    "evaluate_one",
    "first_relevant_rank",
    "gold_sha256",
    "hit_at_k",
    "load_gold",
    "mrr_at_k",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
    "run_eval",
    "run_mode",
    "validate_gold",
]
