"""clinic 评测测试：闸门必须**能失败**。

一个永远通过的闸门等于没有闸门。这里除了验证真实金标集的指标，还专门构造
"该失败"的场景（漏报一条红线、误报率超限），断言闸门真的会红。
"""
from __future__ import annotations

import pytest

from kubedocs.config import repo_root
from kubedocs.evaluation.clinic import (
    ClinicCase,
    ClinicReport,
    evaluate_redline,
    evaluate_triage,
    gate_failures,
    load_clinic_gold,
)

GOLD = repo_root() / "data" / "eval" / "clinic_gold.jsonl"


@pytest.fixture
def cases() -> list[ClinicCase]:
    return load_clinic_gold(GOLD)


class TestGoldSet:
    def test_gold_set_loads_and_is_big_enough(self, cases):
        assert len(cases) >= 30, "金标集太小，指标没有意义"

    def test_gold_set_covers_all_four_non_trigger_categories(self, cases):
        """家属/既往/否认/假设四类都必须有用例——这是三维门控的验收面。"""
        kinds = {c.kind for c in cases}
        for k in ("redline_other", "redline_past", "redline_negated",
                  "redline_hypothetical", "redline_trigger", "safe_normal"):
            assert k in kinds, f"金标集缺少 {k} 类用例"

    def test_gold_set_has_no_duplicate_ids(self, cases):
        ids = [c.id for c in cases]
        assert len(set(ids)) == len(ids)


class TestRedlineMetrics:
    def test_real_gold_set_passes_the_hard_gate(self, cases):
        """红线召回必须 100%：漏一条急诊都不可接受。"""
        report = evaluate_redline(cases)
        assert report.redline_fn == 0, f"漏报：{report.missed}"
        assert report.redline_recall == 1.0
        assert report.redline_false_positive_rate == 0.0, f"误报：{report.false_alarms}"
        assert report.redline_tp >= 8 and report.redline_tn >= 15

    def test_gate_fails_when_a_redline_is_missed(self):
        """**闸门必须能失败**：构造一个漏报，断言它真的会红。"""
        cases = [ClinicCase(id="x1", kind="redline_trigger",
                            text="这句里没有任何红线词", expect_red_line=True)]
        report = evaluate_redline(cases)
        fails = gate_failures(report)
        assert fails and "红线漏报" in fails[0]
        assert "容差为 0" in fails[0]

    def test_gate_fails_when_false_positives_exceed_the_limit(self):
        """把所有话都判成急诊能拿 100% 召回——误报上限就是挡这个的。"""
        report = ClinicReport(redline_tp=1, redline_fp=5, redline_tn=5)
        fails = gate_failures(report)
        assert any("误报率" in f for f in fails)

    def test_gate_passes_on_a_clean_report(self):
        assert gate_failures(ClinicReport(redline_tp=5, redline_tn=5)) == []


class TestTriageMetricIsNotFaked:
    def test_unevaluated_when_no_links(self):
        """缺语料时返回"未评测"，而不是把答不出来算成 0%——后者会误导。"""
        cases = [ClinicCase(id="t", kind="triage_department", text="咳嗽",
                            expect_department="呼吸内科")]
        total, hit, evaluated, _ = evaluate_triage(cases, candidate_fn=None)
        assert (total, hit, evaluated) == (0, 0, False)

    def test_unevaluated_when_no_triage_cases(self):
        _t, _h, evaluated, _ = evaluate_triage(
            [ClinicCase(id="x", kind="safe_normal", text="咳嗽")],
            candidate_fn={"a": 1.0})
        assert evaluated is False

    def test_accuracy_is_none_when_not_evaluated(self):
        report = ClinicReport(triage_total=5, triage_hit=0, triage_evaluated=False)
        assert report.triage_accuracy is None
        assert report.to_json()["triage"]["accuracy"] is None      # type: ignore[index]

    def test_evaluated_triage_reports_a_real_number(self):
        cases = [ClinicCase(id="t", kind="triage_department", text="咳嗽",
                            expect_department="呼吸内科")]
        total, hit, evaluated, _ = evaluate_triage(
            cases, candidate_fn={"呼吸内科": 2.0, "消化内科": 1.0})
        assert (total, hit, evaluated) == (1, 1, True)


class TestReportShape:
    def test_json_export_has_the_three_sections(self, cases):
        payload = evaluate_redline(cases).to_json()
        assert set(payload) >= {"n_cases", "redline", "booking", "triage"}
        assert {"recall", "false_positive_rate", "missed", "false_alarms"} <= set(
            payload["redline"])      # type: ignore[arg-type]
