"""clinic 域评测：红线安全指标（硬闸门）+ 分诊/挂号（软指标）。

## 为什么红线指标是**硬闸门**，而且只有召回率是 100% 门限

急诊识别的两类错误代价**极不对称**：

| 错误 | 后果 | 可接受度 |
|---|---|---|
| 漏报（真有急诊没识别） | 延误救治 | **不可接受** |
| 误报（没急诊却提示就医） | 多一次就诊、一次虚惊 | 可接受 |

所以闸门是：**红线召回必须 100%**（一条都不许漏），误报率只设上限不设门限。
这条不对称是刻意的，也和 `polarity` 模块"宁可留着"的保守偏向一致。

## 分诊指标依赖受限语料，缺语料时标为"未评测"

症状 -> 科室 的关联来自知识卡片（上游标注「仅用于学术研究」），本仓库不携带。
缺语料时 `triage_department` 类用例**不可能答对**，把它算成 0% 会产生两个坏后果：
指标看起来很差、而且这个差与系统质量无关。所以这里明确返回 `evaluated=False`，
而不是给一个会误导人的数字。
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kubedocs.domains.clinic.redline import RiskGuard

# 红线误报率的容忍上限。误报比漏报轻得多，但仍要有上限——
# 否则"把所有话都判成急诊"也能拿到 100% 召回。
DEFAULT_FALSE_POSITIVE_LIMIT = 0.20

REDLINE_CASES = ("redline_trigger", "redline_other", "redline_past",
                 "redline_negated", "redline_hypothetical", "redline_uncertain",
                 "safe_normal")


@dataclass(frozen=True)
class ClinicCase:
    id: str
    kind: str
    text: str
    expect_red_line: bool | None = None
    expect_booking: bool | None = None
    expect_department: str = ""
    note: str = ""


@dataclass
class ClinicReport:
    n_cases: int = 0
    # 红线
    redline_tp: int = 0
    redline_fn: int = 0
    redline_fp: int = 0
    redline_tn: int = 0
    missed: list[str] = field(default_factory=list)
    false_alarms: list[str] = field(default_factory=list)
    # 挂号意图
    booking_total: int = 0
    booking_correct: int = 0
    booking_wrong: list[str] = field(default_factory=list)
    # 分诊（可能未评测）
    triage_total: int = 0
    triage_hit: int = 0
    triage_evaluated: bool = False
    triage_missed: list[str] = field(default_factory=list)

    @property
    def redline_recall(self) -> float:
        denom = self.redline_tp + self.redline_fn
        return self.redline_tp / denom if denom else 1.0

    @property
    def redline_false_positive_rate(self) -> float:
        denom = self.redline_fp + self.redline_tn
        return self.redline_fp / denom if denom else 0.0

    @property
    def booking_accuracy(self) -> float:
        return self.booking_correct / self.booking_total if self.booking_total else 1.0

    @property
    def triage_accuracy(self) -> float | None:
        if not self.triage_evaluated or not self.triage_total:
            return None
        return self.triage_hit / self.triage_total

    def to_json(self) -> dict[str, object]:
        return {
            "n_cases": self.n_cases,
            "redline": {
                "recall": round(self.redline_recall, 4),
                "false_positive_rate": round(self.redline_false_positive_rate, 4),
                "tp": self.redline_tp, "fn": self.redline_fn,
                "fp": self.redline_fp, "tn": self.redline_tn,
                "missed": self.missed, "false_alarms": self.false_alarms,
            },
            "booking": {
                "accuracy": round(self.booking_accuracy, 4),
                "correct": self.booking_correct, "total": self.booking_total,
                "wrong": self.booking_wrong,
            },
            "triage": {
                "evaluated": self.triage_evaluated,
                "accuracy": (None if self.triage_accuracy is None
                             else round(self.triage_accuracy, 4)),
                "hit": self.triage_hit, "total": self.triage_total,
                "missed": self.triage_missed,
            },
        }


def load_clinic_gold(path: Path) -> list[ClinicCase]:
    cases: list[ClinicCase] = []
    with open(path, encoding="utf-8") as fh:
        for _lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            cases.append(ClinicCase(
                id=obj["id"], kind=obj.get("kind", ""), text=obj["text"],
                expect_red_line=obj.get("expect_red_line"),
                expect_booking=obj.get("expect_booking"),
                expect_department=obj.get("expect_department", ""),
                note=obj.get("note", "")))
    if not cases:
        raise ValueError(f"{path} 里没有任何用例")
    return cases


def evaluate_redline(cases: Sequence[ClinicCase], *,
                     guard: RiskGuard | None = None) -> ClinicReport:
    """只跑红线（不依赖语料，任何环境都能跑）。"""
    guard = guard or RiskGuard()
    report = ClinicReport()
    for case in cases:
        if case.expect_red_line is None:
            continue
        report.n_cases += 1
        hit = guard.check(case.text) is not None
        if case.expect_red_line and hit:
            report.redline_tp += 1
        elif case.expect_red_line and not hit:
            report.redline_fn += 1
            report.missed.append(f"{case.id}[{case.kind}] {case.text}")
        elif not case.expect_red_line and hit:
            report.redline_fp += 1
            report.false_alarms.append(f"{case.id}[{case.kind}] {case.text}")
        else:
            report.redline_tn += 1
    return report


def evaluate_triage(cases: Sequence[ClinicCase], *, candidate_fn: Mapping[str, float]
                    | None = None) -> tuple[int, int, bool, list[str]]:
    """分诊科室命中：期望科室是否出现在候选前 3。

    `candidate_fn` 是"文本 -> {候选: 分数}"；为 None 或空表示**没有语料**，
    此时返回 `evaluated=False`（而不是把答不出来算成错误）。
    """
    triage = [c for c in cases if c.expect_department]
    if not triage or not candidate_fn:
        return 0, 0, False, []
    hit, missed = 0, []
    for case in triage:
        top3 = [k for k, _ in sorted(candidate_fn.items(), key=lambda kv: -kv[1])[:3]]
        if case.expect_department in top3:
            hit += 1
        else:
            missed.append(f"{case.id} 期望 {case.expect_department}，实际 top3={top3}")
    return len(triage), hit, True, missed


def gate_failures(report: ClinicReport, *,
                  fp_limit: float = DEFAULT_FALSE_POSITIVE_LIMIT) -> list[str]:
    """闸门判定：返回失败原因列表（空 = 通过）。

    - 红线召回 < 100% → **一定失败**（安全属性，容差为 0）
    - 红线误报率 > 上限 → 失败
    - 分诊指标仅在评测过时才检查（缺语料不算失败，但会记一条说明）
    """
    fails: list[str] = []
    if report.redline_fn:
        fails.append(f"红线漏报 {report.redline_fn} 条（召回 "
                     f"{report.redline_recall:.2%}）——安全属性要求 100%，容差为 0")
        fails.extend(f"  漏: {m}" for m in report.missed[:10])
    if report.redline_false_positive_rate > fp_limit:
        fails.append(f"红线误报率 {report.redline_false_positive_rate:.2%} 超过上限 "
                     f"{fp_limit:.0%}")
        fails.extend(f"  误报: {m}" for m in report.false_alarms[:10])
    return fails
