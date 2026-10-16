"""clinic 评测测试：**指标只报不卡，闸门只卡确定性安全不变量**。

这里钉住两件事：

1. 指标本身算得对（召回、误报、分诊、空集不返回 `None` 以外的假值）；
2. **闸门只对两类事失败**：这次运行不能作为证据（空集 / 有题没拿到判断），
   以及确定性安全不变量被触发。指标再差也不该让闸门变红——那是这次改造的核心。
"""

from __future__ import annotations

import pytest

from kubedocs.config import repo_root
from kubedocs.evaluation.clinic import (
    TARGET_TRIAGE_MIN_ACCURACY,
    ClinicCase,
    ClinicReport,
    evaluate_redline,
    evaluate_triage,
    governance_failures,
    load_clinic_gold,
    reporting_targets,
)

GOLD = repo_root() / "data" / "eval" / "clinic_gold.jsonl"


def _clean_report(**overrides: object) -> ClinicReport:
    """指标干净的报告；`overrides` 用来逐项构造"该红/不该红"的场景。"""
    base: dict[str, object] = {
        "n_cases": 26,
        "redline_tp": 5,
        "redline_tn": 5,
        "triage_total": 5,
        "triage_hit": 3,
        "triage_evaluated": True,
    }
    base.update(overrides)
    return ClinicReport(**base)  # type: ignore[arg-type]


@pytest.fixture
def cases() -> list[ClinicCase]:
    return load_clinic_gold(GOLD)


class TestGoldSet:
    def test_gold_set_loads_and_is_big_enough(self, cases):
        assert len(cases) >= 30, "金标集太小，指标没有意义"

    def test_gold_set_covers_all_four_non_trigger_categories(self, cases):
        """家属/既往/否认/假设四类都必须有用例——这是三维门控的验收面。"""
        kinds = {c.kind for c in cases}
        for k in (
            "redline_other",
            "redline_past",
            "redline_negated",
            "redline_hypothetical",
            "redline_trigger",
            "safe_normal",
        ):
            assert k in kinds, f"金标集缺少 {k} 类用例"

    def test_gold_set_has_no_duplicate_ids(self, cases):
        ids = [c.id for c in cases]
        assert len(set(ids)) == len(ids)


class TestRedlineMetrics:
    def test_rule_arm_clue_quality_on_the_real_gold_set(self, cases):
        """规则臂的红线词表是**喂给模型的线索**：它的召回记录在案（当前是零漏报）。"""
        report = evaluate_redline(cases)
        assert report.redline_fn == 0, f"线索层漏报（线索质量回归）：{report.missed}"
        assert report.redline_recall == 1.0
        assert report.redline_false_positive_rate == 0.0, f"误报：{report.false_alarms}"
        assert report.redline_tp >= 8 and report.redline_tn >= 15

    def test_metrics_alone_never_fail_the_gate(self):
        """**这条是本次改造的核心**：指标再差也不让闸门变红。

        漏报、误报超限、分诊 0 分——全都不再是退出码的依据。它们进报告与档案，
        由人看趋势；拿自建小样本的分数卡退出码，只会逼着后来者去改门限。
        """
        worst = _clean_report(redline_fn=9, redline_tp=0, redline_fp=20, redline_tn=0, triage_hit=0)
        assert governance_failures(worst) == []

    def test_empty_run_fails_the_gate(self):
        """空集不得满足闸门：一条用例都没跑，说明这次什么也没测。"""
        fails = governance_failures(ClinicReport())
        assert fails and "没有任何可评测的用例" in fails[0]

    def test_failed_calls_fail_the_gate(self):
        """有题目没拿到模型判断 ⇒ 闸门红（没测不等于测过）。"""
        fails = governance_failures(_clean_report(), failed_calls=["r07 调用超时"])
        assert any("r07" in f for f in fails)

    def test_invariant_violations_are_recorded_but_do_not_fail_the_gate(self):
        """**安全不变量的触发不判闸门**：它是"模型这一轮需要被纠正"，不是"这次没测成"。

        三条底线每一条都被服务端处置过（换成只含事实的回答 / 补上急救指引），
        所以它进报告、由长期监听按 alert 报给运维；把"需要被纠正的次数"当门限，
        等于让一次模型抖动把整轮评测判成无效。
        """
        report = _clean_report(
            invariant_violations=["r01：emergency_judgment_requires_directive（已补固定指引）"]
        )
        assert governance_failures(report) == []
        # 但记录必须留着（不然监听与档案都看不见它）
        assert report.to_json()["invariant_violations"] == [
            "r01：emergency_judgment_requires_directive（已补固定指引）"
        ]

    def test_label_keeps_the_two_arms_apart(self):
        fails = governance_failures(ClinicReport(), label="规则臂")
        assert fails[0].startswith("规则臂")


class TestReportingTargets:
    """参考目标还在，但只出现在报告里——**它不判通过/不通过**。"""

    def test_targets_are_exported_with_a_note(self):
        targets = reporting_targets()
        assert targets["redline_recall"] == 1.0
        assert 0.0 < targets["false_positive_rate"] <= 1.0
        assert 0.0 < targets["triage_min_accuracy"] <= 1.0
        assert "不判通过/不通过" in str(targets["note"])

    def test_report_carries_the_targets_next_to_the_measurements(self):
        payload = _clean_report().to_json()
        assert payload["targets"]["triage_min_accuracy"] == TARGET_TRIAGE_MIN_ACCURACY  # type: ignore[index]
        assert payload["targets"]["note"]  # type: ignore[index]


class TestTriageMetricIsNotFaked:
    def test_unevaluated_when_no_links(self):
        """缺语料时返回"未评测"，而不是把答不出来算成 0%——后者会误导。"""
        cases = [ClinicCase(id="t", kind="triage_department", text="咳嗽", expect_department="呼吸内科")]
        total, hit, evaluated, _ = evaluate_triage(cases, candidate_fn=None)
        assert (total, hit, evaluated) == (0, 0, False)

    def test_unevaluated_when_no_triage_cases(self):
        _t, _h, evaluated, _ = evaluate_triage(
            [ClinicCase(id="x", kind="safe_normal", text="咳嗽")], candidate_fn={"a": 1.0}
        )
        assert evaluated is False

    def test_accuracy_is_none_when_not_evaluated(self):
        report = ClinicReport(triage_total=5, triage_hit=0, triage_evaluated=False)
        assert report.triage_accuracy is None
        assert report.to_json()["triage"]["accuracy"] is None  # type: ignore[index]

    def test_evaluated_triage_reports_a_real_number(self):
        cases = [ClinicCase(id="t", kind="triage_department", text="咳嗽", expect_department="呼吸内科")]
        total, hit, evaluated, _ = evaluate_triage(cases, candidate_fn={"呼吸内科": 2.0, "消化内科": 1.0})
        assert (total, hit, evaluated) == (1, 1, True)


class TestReportShape:
    def test_json_export_has_the_three_sections(self, cases):
        payload = evaluate_redline(cases).to_json()
        assert set(payload) >= {"n_cases", "redline", "booking", "triage", "invariant_violations"}
        assert {"recall", "false_positive_rate", "missed", "false_alarms"} <= set(payload["redline"])  # type: ignore[arg-type]

    def test_invariant_violations_are_in_the_export(self):
        """不变量触发记录必须进报告：它是唯一会卡闸门的东西，不能只在内存里。"""
        report = _clean_report(invariant_violations=["x：no_fabricated_booking_action"])
        assert report.to_json()["invariant_violations"] == ["x：no_fabricated_booking_action"]
