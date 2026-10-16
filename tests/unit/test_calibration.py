"""置信度校准的契约测试：**只报告、不上门限**，而且不许把"没数据"说成"校准完美"。"""

from __future__ import annotations

import pytest

from kubedocs.evaluation.calibration import (
    MIN_MEANINGFUL_SAMPLES,
    brier_score,
    calibration_report,
    reliability_bins,
)


class TestBrierScore:
    def test_perfect_confident_predictions_score_zero(self) -> None:
        assert brier_score([(1.0, True), (0.0, False)]) == 0.0

    def test_confidently_wrong_scores_worst(self) -> None:
        assert brier_score([(1.0, False), (0.0, True)]) == 1.0

    def test_a_hedging_model_scores_a_quarter(self) -> None:
        """永远说 0.5、正负各半 ⇒ 0.25。**报出来的 Brier 必须和这个基准一起看**。"""
        assert brier_score([(0.5, True), (0.5, False)]) == pytest.approx(0.25)

    def test_empty_input_is_none_not_zero(self) -> None:
        """没测过 ≠ 校准完美：0 分是最完美的分数，不能拿它当"没数据"。"""
        assert brier_score([]) is None


class TestReliabilityBins:
    def test_bins_are_equal_width_and_cover_the_unit_interval(self) -> None:
        bins = reliability_bins([], n_bins=5)
        assert [(b.lo, b.hi) for b in bins] == [
            (0.0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
        ]

    def test_confidence_one_lands_in_the_last_bin(self) -> None:
        """1.0 不许溢出到不存在的第 n 箱（那会让最自信的样本凭空消失）。"""
        bins = reliability_bins([(1.0, True)])
        assert bins[-1].n == 1
        assert sum(b.n for b in bins) == 1

    def test_each_bin_reports_confidence_and_accuracy(self) -> None:
        bins = reliability_bins([(0.85, True), (0.95, False)], n_bins=5)
        bin_ = bins[4]
        assert bin_.n == 2
        assert bin_.mean_confidence == pytest.approx(0.9)
        assert bin_.accuracy == pytest.approx(0.5)

    def test_empty_bins_are_kept(self) -> None:
        """空箱也要返回：**"它在哪一段从没说过话"本身就是校准信息**。"""
        bins = reliability_bins([(0.95, True)], n_bins=5)
        assert len(bins) == 5
        assert [b.n for b in bins] == [0, 0, 0, 0, 1]

    def test_invalid_bin_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            reliability_bins([], n_bins=0)


class TestCalibrationReport:
    def test_report_carries_brier_bins_and_a_sample_size_note(self) -> None:
        pairs = [(0.9, True), (0.8, True), (0.3, False)]
        report = calibration_report(pairs)
        assert report.n == 3
        assert report.brier is not None
        assert report.bins
        assert "样本量" in report.note
        assert "不足以谈校准结论" in report.note, "小样本必须明说，别让人拿 3 条当结论"

    def test_larger_samples_get_a_softer_note(self) -> None:
        pairs = [(0.9, True)] * MIN_MEANINGFUL_SAMPLES
        report = calibration_report(pairs)
        assert "可看趋势" in report.note
        assert "不构成临床校准" in report.note

    def test_empty_report_says_there_is_nothing_to_calibrate(self) -> None:
        report = calibration_report([])
        assert report.n == 0 and report.brier is None and report.bins == []
        assert "没有可校准的样本" in report.note

    def test_payload_is_json_friendly(self) -> None:
        payload = calibration_report([(0.7, True)]).to_payload()
        assert set(payload) == {"n", "brier", "bins", "note"}
        assert set(payload["bins"][0]) == {"range", "n", "mean_confidence", "accuracy"}
