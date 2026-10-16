"""长期监听的契约测试：**它必须能指出变化，也必须不把波动当回归**。

两组断言：

1. `evaluation/monitor.py` 的纯逻辑——记录形状、追加/读回、与历史区间的比对、
   `alert` 与 `note` 的分级（后者默认不改退出码）；
2. `scripts/monitor_clinic.py` 的行为——`--status` 只读不跑、`--run` **复用评测脚本**、
   `--follow --rounds` 真的循环、退出码只被 `alert`（或 `--strict`）改变。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from kubedocs.config import repo_root
from kubedocs.evaluation.monitor import (
    Alert,
    TrendRecord,
    append_record,
    compare,
    format_trend,
    read_records,
    record_from_reports,
    trend_path,
)

MONITOR_SCRIPT = repo_root() / "scripts" / "monitor_clinic.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load(MONITOR_SCRIPT, "_monitor_clinic_script")


class _Report:
    """报告替身：只提供趋势记录要读的那几个属性。"""

    def __init__(
        self,
        *,
        engine: str = "tool_loop",
        evaluated: bool = True,
        n_cases: int = 10,
        recall: float | None = 1.0,
        fp: float = 0.1,
        triage: float | None = 0.8,
        failed: int = 0,
        invariants: list[str] | None = None,
        llm_calls: int = 20,
        tool_calls: int = 11,
        steps: int = 30,
        cost: float = 0.05,
    ) -> None:
        self.engine = engine
        self.evaluated = evaluated
        self.n_cases = n_cases
        self.redline_recall = recall
        self.redline_false_positive_rate = fp
        self.triage_accuracy = triage
        self.n_failed_calls = failed
        self.invariant_violations = invariants or []
        self.llm_calls = llm_calls
        self.tool_calls = tool_calls
        self.steps = steps
        self.cost_usd = cost
        self.reason = "" if evaluated else "没密钥"


class TestRecordShape:
    def test_record_is_flat_and_carries_what_monitoring_needs(self) -> None:
        record = record_from_reports(
            judge=_Report(),
            rule=_Report(engine="rule", recall=1.0, fp=0.0, triage=1.0, cost=0.0),
            prompt_version="abc123",
            gold_sha256="deadbeef",
        )
        payload = record.to_payload()
        assert payload["engine"] == "tool_loop"
        assert payload["danger_recall"] == 1.0
        assert payload["failed_calls"] == 0
        assert payload["invariants"] == 0
        assert payload["prompt_version"] == "abc123"
        assert payload["gold_sha256"] == "deadbeef"
        assert payload["rule_triage_accuracy"] == 1.0
        assert payload["cost_usd"] == 0.05
        # 扁平：没有嵌套 dict（趋势表的列就是这些字段）
        assert all(not isinstance(v, dict) for v in payload.values())
        assert record.cost_per_case == pytest.approx(0.005)

    def test_invariant_ids_are_extracted_from_the_violation_text(self) -> None:
        record = record_from_reports(
            judge=_Report(
                invariants=[
                    "r01：emergency_judgment_requires_directive（已补固定指引）",
                    "n02：no_fabricated_booking_action（编造挂号）",
                ]
            )
        )
        assert record.invariant_ids == [
            "emergency_judgment_requires_directive",
            "no_fabricated_booking_action",
        ]

    def test_missing_metrics_stay_none_not_zero(self) -> None:
        """没测到的指标写 `None`：0 与"没测"在趋势图上是两件事。"""
        record = record_from_reports(judge=_Report(recall=None, triage=None))
        assert record.danger_recall is None
        assert record.triage_accuracy is None
        assert record.to_payload()["danger_recall"] is None


class TestAppendAndRead:
    def test_records_round_trip(self, tmp_path: Path) -> None:
        path = trend_path(tmp_path)
        assert append_record(path, record_from_reports(judge=_Report())) is True
        assert append_record(path, record_from_reports(judge=_Report(failed=2))) is True
        rows = read_records(path)
        assert [r.failed_calls for r in rows] == [0, 2]

    def test_missing_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_records(trend_path(tmp_path)) == []

    def test_corrupt_line_is_skipped(self, tmp_path: Path) -> None:
        path = trend_path(tmp_path)
        append_record(path, record_from_reports(judge=_Report()))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("这不是 json\n")
        assert len(read_records(path)) == 1

    def test_unknown_fields_do_not_break_reading(self, tmp_path: Path) -> None:
        """以后加列（或旧版本删列）不该让整份历史读不出来。"""
        path = trend_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": "2026-01-01T00:00:00+00:00", "engine": "tool_loop", "未来字段": 1}) + "\n",
            encoding="utf-8",
        )
        rows = read_records(path)
        assert rows and rows[0].engine == "tool_loop"

    def test_write_failure_is_reported_not_raised(self, tmp_path: Path) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("占位文件，不是目录", encoding="utf-8")
        assert append_record(blocked / "x.jsonl", record_from_reports(judge=_Report())) is False

    def test_limit_returns_the_tail(self, tmp_path: Path) -> None:
        path = trend_path(tmp_path)
        for i in range(4):
            append_record(path, record_from_reports(judge=_Report(n_cases=i + 1)))
        assert [r.n_cases for r in read_records(path, limit=2)] == [3, 4]


def _history(*records: TrendRecord) -> list[TrendRecord]:
    return list(records)


class TestCompare:
    def test_no_records_no_alerts(self) -> None:
        assert compare([]) == []

    def test_first_record_alone_has_no_history_to_compare(self) -> None:
        """第一条记录没有可比的历史：它只报告结构性问题，不编造"回归"。"""
        only = record_from_reports(judge=_Report())
        assert [a.id for a in compare([only])] == []

    def test_invariant_firing_is_an_alert(self) -> None:
        alerts = compare(
            [
                record_from_reports(judge=_Report()),
                record_from_reports(
                    judge=_Report(invariants=["r01：emergency_judgment_requires_directive（补指引）"])
                ),
            ]
        )
        ids = {a.id for a in alerts}
        assert "invariant_fired" in ids
        assert all(a.severity == "alert" for a in alerts if a.id == "invariant_fired")

    def test_unevaluated_run_is_an_alert(self) -> None:
        alerts = compare(
            [record_from_reports(judge=_Report()), record_from_reports(judge=_Report(evaluated=False))]
        )
        assert "arm_not_evaluated" in {a.id for a in alerts}

    def test_failed_calls_are_an_alert(self) -> None:
        alerts = compare([record_from_reports(judge=_Report()), record_from_reports(judge=_Report(failed=3))])
        assert "failed_calls" in {a.id for a in alerts}

    def test_metric_drop_is_only_a_note(self) -> None:
        """**关键**：指标下滑是趋势提示，默认不改退出码——把波动当回归会让监听变噪音。"""
        history = [record_from_reports(judge=_Report(recall=1.0, triage=1.0)) for _ in range(3)]
        latest = record_from_reports(judge=_Report(recall=0.8, triage=0.6))
        alerts = compare([*history, latest])

        assert {a.id for a in alerts} == {"danger_recall_dropped", "triage_accuracy_dropped"}
        assert all(a.severity == "note" for a in alerts)

    def test_false_positive_rise_is_a_note(self) -> None:
        history = [record_from_reports(judge=_Report(fp=0.1)) for _ in range(3)]
        latest = record_from_reports(judge=_Report(fp=0.4))
        alerts = compare([*history, latest])
        assert [a.id for a in alerts] == ["danger_fp_rate_rose"]
        assert alerts[0].severity == "note"

    def test_cost_per_case_rise_is_a_note(self) -> None:
        history = [record_from_reports(judge=_Report(n_cases=10, cost=0.05)) for _ in range(2)]
        latest = record_from_reports(judge=_Report(n_cases=10, cost=0.20))
        assert [a.id for a in compare([*history, latest])] == ["cost_per_case_rose"]

    def test_within_history_range_is_not_flagged(self) -> None:
        """历史区间内的波动**不该**报警：那是模型本来就会有的抖动。"""
        history = [
            record_from_reports(judge=_Report(recall=1.0, fp=0.2)),
            record_from_reports(judge=_Report(recall=0.9, fp=0.1)),
            record_from_reports(judge=_Report(recall=1.0, fp=0.15)),
        ]
        latest = record_from_reports(judge=_Report(recall=0.95, fp=0.12))
        assert compare([*history, latest]) == []

    def test_window_limits_the_reference_range(self) -> None:
        """窗口外的极端值不该永远把区间撑大：只比最近 `window` 轮。

        构造：很久以前误报率高达 0.9，最近 5 轮都在 0.1，这一轮 0.2。
        窗口 5 ⇒ 参考区间是 0.1 ⇒ **这一轮算上升**（报警）；
        窗口 6 ⇒ 那条旧记录进来，0.2 落回历史区间 ⇒ 不报警。
        """
        ancient = record_from_reports(judge=_Report(fp=0.9))
        recent = [record_from_reports(judge=_Report(fp=0.1)) for _ in range(5)]
        latest = record_from_reports(judge=_Report(fp=0.2))

        assert [a.id for a in compare([ancient, *recent, latest], window=5)] == ["danger_fp_rate_rose"]
        assert compare([ancient, *recent, latest], window=6) == []

    def test_engine_filter_keeps_engines_from_being_mixed(self) -> None:
        """工具循环与单发臂的数量级不同：混在一起比会得到假的"没变化"。

        工具循环单题 0.05 美元、单发 0.0004；这一轮单发 0.0005。
        按引擎比 ⇒ 高于单发历史 ⇒ 报警；不按引擎比 ⇒ 被工具循环的高值盖住 ⇒ 不报警。
        """
        tool = [
            record_from_reports(judge=_Report(engine="tool_loop", n_cases=10, cost=0.5)) for _ in range(3)
        ]
        single = record_from_reports(judge=_Report(engine="reasoner", n_cases=10, cost=0.004))
        latest = record_from_reports(judge=_Report(engine="reasoner", n_cases=10, cost=0.005))

        assert [a.id for a in compare([*tool, single, latest], engine="reasoner")] == ["cost_per_case_rose"]
        assert compare([*tool, single, latest]) == []

    def test_prompt_and_gold_changes_are_noted(self) -> None:
        base = record_from_reports(judge=_Report(), prompt_version="v1", gold_sha256="g1")
        latest = record_from_reports(judge=_Report(), prompt_version="v2", gold_sha256="g2")
        ids = {a.id for a in compare([base, latest])}
        assert {"prompt_changed", "gold_changed"} <= ids

    def test_alert_render_marks_the_severity(self) -> None:
        assert Alert("x", "alert", "d").render().startswith("  ✗")
        assert Alert("y", "note", "d").render().startswith("  ⚠")


class TestTrendFormatting:
    def test_empty_history_says_how_to_start(self) -> None:
        assert "还没有趋势记录" in format_trend([])

    def test_table_has_one_row_per_record_and_percentages(self) -> None:
        rows = [record_from_reports(judge=_Report(recall=1.0, fp=0.1, triage=0.8)) for _ in range(3)]
        text = format_trend(rows, n=2)
        assert text.count("\n") == 3  # 表头 + 分隔 + 2 行
        assert "100.0%" in text and "10.0%" in text

    def test_missing_metric_prints_a_dash(self) -> None:
        text = format_trend([record_from_reports(judge=_Report(recall=None, triage=None))])
        assert "-" in text.splitlines()[-1]


class TestMonitorScript:
    def _fake_eval(self, *, code: int = 0, records: list[TrendRecord] | None = None, trend: Path) -> Any:
        """假评测：把趋势记录写进指定文件、返回给定退出码（**不联网、不花钱**）。"""
        calls: list[list[str]] = []

        def main(argv: list[str] | None = None) -> int:
            calls.append(list(argv or []))
            for record in records or [record_from_reports(judge=_Report())]:
                append_record(trend, record)
            return code

        main.calls = calls  # type: ignore[attr-defined]
        return main

    def test_status_only_reads(self, script: ModuleType, tmp_path: Path, capsys: Any) -> None:
        trend = trend_path(tmp_path)
        append_record(trend, record_from_reports(judge=_Report()))
        calls: list[int] = []
        script._load_eval_main = lambda: calls.append(1)  # type: ignore[assignment]

        code = int(script.main(["--status", "--trend-path", str(trend)]))
        out = capsys.readouterr().out
        assert code == 0
        assert calls == [], "--status 不该跑评测"
        assert "共 1 轮" in out

    def test_status_judges_the_stored_history(self, script: ModuleType, tmp_path: Path, capsys: Any) -> None:
        """`--status` 也要给结论：用最后一条当"当前"与历史比，但**不产生新数据**、永远退 0。"""
        trend = trend_path(tmp_path)
        append_record(trend, record_from_reports(judge=_Report()))
        append_record(
            trend,
            record_from_reports(judge=_Report(invariants=["x：emergency_judgment_requires_directive（补）"])),
        )

        code = int(script.main(["--status", "--trend-path", str(trend)]))
        out = capsys.readouterr().out
        assert code == 0, "--status 是只读的：即使历史里有 alert 也不改退出码"
        assert "invariant_fired" in out

    def test_run_appends_a_record_and_reports_alerts(
        self, script: ModuleType, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        trend = trend_path(tmp_path)
        append_record(trend, record_from_reports(judge=_Report()))
        fake = self._fake_eval(records=[record_from_reports(judge=_Report(failed=2))], trend=trend)
        monkeypatch.setattr(script, "_load_eval_main", lambda: fake)

        code = int(script.main(["--run", "--trend-path", str(trend), "--archive", str(tmp_path)]))
        out = capsys.readouterr().out
        assert fake.calls, "监听必须复用评测脚本的 main()"
        assert "--trend-path" in fake.calls[0], "趋势落点要透传给评测脚本"
        assert code == 1, "有题目没拿到判断 ⇒ alert ⇒ 退出码 1"
        assert "failed_calls" in out

    def test_notes_alone_do_not_change_the_exit_code(
        self, script: ModuleType, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        trend = trend_path(tmp_path)
        append_record(trend, record_from_reports(judge=_Report(recall=1.0)))
        fake = self._fake_eval(records=[record_from_reports(judge=_Report(recall=0.5))], trend=trend)
        monkeypatch.setattr(script, "_load_eval_main", lambda: fake)

        code = int(script.main(["--run", "--trend-path", str(trend), "--archive", str(tmp_path)]))
        out = capsys.readouterr().out
        assert code == 0, "指标下滑只是 note，不该改退出码"
        assert "danger_recall_dropped" in out
        assert "默认不改退出码" in out

    def test_strict_mode_turns_notes_into_a_failure(
        self, script: ModuleType, tmp_path: Path, monkeypatch: Any
    ) -> None:
        trend = trend_path(tmp_path)
        append_record(trend, record_from_reports(judge=_Report(recall=1.0)))
        fake = self._fake_eval(records=[record_from_reports(judge=_Report(recall=0.5))], trend=trend)
        monkeypatch.setattr(script, "_load_eval_main", lambda: fake)

        code = int(script.main(["--run", "--strict", "--trend-path", str(trend), "--archive", str(tmp_path)]))
        assert code == 1

    def test_invariant_firing_reaches_ops_even_though_the_eval_gate_passes(
        self, script: ModuleType, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """**分工的落点**：评测闸门对"不变量触发"是绿的（它是记录），监听把它报成 `alert`。

        硬信号因此没有被丢掉，只是搬到了运维那一侧：评测的退出码回答"这次测成了吗"，
        监听的退出码回答"系统还正常吗"。
        """
        trend = trend_path(tmp_path)
        fake = self._fake_eval(
            code=0,
            records=[
                record_from_reports(
                    judge=_Report(invariants=["x：emergency_judgment_requires_directive（补）"])
                )
            ],
            trend=trend,
        )
        monkeypatch.setattr(script, "_load_eval_main", lambda: fake)

        code = int(script.main(["--run", "--trend-path", str(trend), "--archive", str(tmp_path)]))
        out = capsys.readouterr().out
        assert code == 1, "评测绿了不等于运维看不见：监听必须照样报 alert"
        assert "invariant_fired" in out

    def test_eval_failure_also_fails_the_monitor(
        self, script: ModuleType, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """评测自己的闸门红了（例如有题目没拿到判断）⇒ 监听的退出码也是 1。"""
        trend = trend_path(tmp_path)
        fake = self._fake_eval(code=2, records=[record_from_reports(judge=_Report())], trend=trend)
        monkeypatch.setattr(script, "_load_eval_main", lambda: fake)

        code = int(script.main(["--run", "--trend-path", str(trend), "--archive", str(tmp_path)]))
        assert code == 1

    def test_follow_loops_the_requested_number_of_rounds(
        self, script: ModuleType, tmp_path: Path, monkeypatch: Any
    ) -> None:
        trend = trend_path(tmp_path)
        fake = self._fake_eval(trend=trend)
        monkeypatch.setattr(script, "_load_eval_main", lambda: fake)
        monkeypatch.setattr(script.time, "sleep", lambda _s: None)  # 单测不真的等

        code = int(
            script.main(
                [
                    "--follow",
                    "--rounds",
                    "3",
                    "--interval",
                    "0.01",
                    "--trend-path",
                    str(trend),
                    "--archive",
                    str(tmp_path),
                ]
            )
        )
        assert code == 0
        assert len(fake.calls) == 3
        assert len(read_records(trend)) == 3

    def test_invalid_interval_is_a_usage_error(self, script: ModuleType, tmp_path: Path, capsys: Any) -> None:
        code = int(script.main(["--follow", "--interval", "0", "--trend-path", str(tmp_path / "t.jsonl")]))
        assert code == 2
        assert "参数错误" in capsys.readouterr().out
