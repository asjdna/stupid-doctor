"""指标测试：手算对照 + "指标必须能失败"的守卫。"""

from __future__ import annotations

import math

import pytest

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

RANKED = ["a", "b", "c", "d"]


def test_first_relevant_rank():
    assert first_relevant_rank(RANKED, {"b"}, 4) == 2
    assert first_relevant_rank(RANKED, {"d"}, 3) is None  # 超出 k
    assert first_relevant_rank(RANKED, {"zzz"}, 4) is None


def test_hit_and_mrr():
    assert hit_at_k(RANKED, {"c"}, 4) == 1.0
    assert mrr_at_k(RANKED, {"c"}, 4) == pytest.approx(1 / 3)
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0
    assert mrr_at_k(RANKED, {"c"}, 2) == 0.0


def test_recall_counts_multiple_targets():
    # k=2 时只看到 a,b，两个目标里命中一个
    assert recall_at_k(RANKED, {"a", "c"}, 2) == pytest.approx(0.5)
    assert recall_at_k(RANKED, {"a", "c"}, 4) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation():
    # 命中在位置 2：DCG = 1/log2(3)；理想 DCG = 1/log2(2)
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(RANKED, {"b"}, 4) == pytest.approx(expected)
    # 命中在位置 1 时 nDCG 必须是 1.0（上界）
    assert ndcg_at_k(RANKED, {"a"}, 4) == pytest.approx(1.0)


def test_ndcg_penalises_lower_rank():
    high = ndcg_at_k(RANKED, {"a"}, 4)
    low = ndcg_at_k(RANKED, {"d"}, 4)
    assert high > low


def test_duplicate_docs_do_not_inflate_scores():
    """同一文档的多个块命中是常态：去重后不该被算成两次命中。
    这里传入的已经是文档级序列（同一文档出现两次）。"""
    dupes = ["a", "a", "b"]
    assert recall_at_k(dupes, {"a", "b"}, 3) == pytest.approx(1.0)
    assert recall_at_k(dupes, {"a"}, 3) == pytest.approx(1.0)  # 不是 2.0
    assert first_relevant_rank(dupes, {"b"}, 3) == 2  # 不是 3
    assert ndcg_at_k(dupes, {"b"}, 3) <= 1.0


class TestMetricsCannotSilentlyPass:
    """一组永远不会失败的指标比没有指标更危险。"""

    def test_empty_relevant_set_is_rejected(self):
        with pytest.raises(ValueError, match="不能为空"):
            recall_at_k(RANKED, set(), 4)
        with pytest.raises(ValueError, match="不能为空"):
            ndcg_at_k(RANKED, set(), 4)

    def test_no_results_gives_zero_not_error(self):
        assert hit_at_k([], {"a"}, 4) == 0.0
        assert mrr_at_k([], {"a"}, 4) == 0.0


def test_evaluate_one_returns_all_four():
    m = evaluate_one(["a", "b"], {"a"}, 2)
    assert set(m) == {"hit", "recall", "mrr", "ndcg"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_aggregate_is_macro_average():
    rows = [{"hit": 1.0, "mrr": 1.0}, {"hit": 0.0, "mrr": 0.0}]
    agg = aggregate(rows)
    assert agg["hit"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.5)


def test_aggregate_of_nothing_is_empty_not_zero():
    """没有数据 与 得分为 0 是两件事，混同会让基线比较失去意义。"""
    assert aggregate([]) == {}


class TestPercentile:
    def test_single_value(self):
        assert percentile([5.0], 0.95) == 5.0

    def test_median_and_extremes(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(xs, 0.5) == pytest.approx(3.0)
        assert percentile(xs, 0.0) == pytest.approx(1.0)
        assert percentile(xs, 1.0) == pytest.approx(5.0)

    def test_p95_interpolates(self):
        xs = [float(i) for i in range(1, 101)]
        assert percentile(xs, 0.95) == pytest.approx(95.05, abs=0.02)

    def test_empty_is_zero(self):
        assert percentile([], 0.95) == 0.0
