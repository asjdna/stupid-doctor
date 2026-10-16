"""clinic 域评测：**指标是治理记录，闸门只卡确定性安全不变量**。

红线召回、误报率、分诊命中率都是**要看的数**，不是通过/不通过的门限。原因很直接：
这些数字来自自建的小样本用例集，它们能说明趋势和回归，**不能**承担"这条回答安全"的责任。
把 100% 召回写成门限，只会得到两种结果——为了让闸门变绿而放宽尺子，或者一个永远红着、
没人看的闸门；而系统真正的安全来自三层防护（语义理解 / 主动问诊 / 确定性安全门，
见 `clinical/safety.py`）与治理审计。

所以这里的分工是：

- **指标**：算出来、打出来、写进档案（`to_json()` 里还带着参考目标，方便人看趋势）；
- **闸门**（`governance_failures()`）：只看两件事——这次运行**能不能作为证据**
  （有没有拿到模型判断、有没有用例），以及**有没有触发确定性安全不变量**。
  不变量是被代码强制执行的那几条底线，它们被触发就是真的出了问题，与样本量无关。

`evaluate_redline()` 仍然照旧跑：规则臂的红线词表是**喂给模型的线索**，
它的召回率是"线索质量"的读数。它不再判通过/不通过。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kubedocs.domains.clinic.redline import RiskGuard

TARGET_REDLINE_RECALL = 1.0
"""参考目标（**不判通过/不通过**）：危险信号的漏报代价不对称，目标当然是"一条都不漏"。

留着这个数的用途只有一个：报告里把它和实测值并排打出来，人一眼能看出差多远。
"""

TARGET_FALSE_POSITIVE_LIMIT = 0.20
"""参考目标：误报率上限。误报比漏报轻，但也不能无限——"把所有话都判成急诊"能拿到 100% 召回。"""

TARGET_TRIAGE_MIN_ACCURACY = 0.60
"""参考目标：分诊 top-3 命中率。

取 0.60 的理由（作为目标值仍然成立）：

1. **不要求 100%**：答错科室 = 多跑一趟、被分诊台纠正一次，代价可恢复；
2. **远高于猜测上界**：候选科室约 40 个，随机取 top-3 的命中上界是 `3/40 = 7.5%`；
3. **在 5 条分诊用例上等价于"至少答对 3 条"**，是这份样本量下能看出趋势的最小值。

它**不是闸门**：自建用例上的分诊成绩不构成临床验证，拿它卡退出码只会让人去改门限。
"""


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
    # 确定性安全不变量：这一轮真正被触发的底线（**唯一会卡闸门的东西**）
    invariant_violations: list[str] = field(default_factory=list)

    @property
    def redline_recall(self) -> float | None:
        """危险信号召回。**分母为 0 时返回 `None`，不是 1.0**。

        返回 1.0 会把"一条危险信号用例都没有"伪装成"全部识别"——空集满足指标是最危险的
        假绿形态。这里返回 `None` 只影响**报告怎么读**（写"未评测"），不再判通过/不通过。
        """
        denom = self.redline_tp + self.redline_fn
        return self.redline_tp / denom if denom else None

    @property
    def redline_false_positive_rate(self) -> float:
        denom = self.redline_fp + self.redline_tn
        return self.redline_fp / denom if denom else 0.0

    @property
    def booking_accuracy(self) -> float | None:
        """挂号意图准确率；分母为 0 时同样是 `None`（理由见 `redline_recall`）。"""
        return self.booking_correct / self.booking_total if self.booking_total else None

    @property
    def triage_accuracy(self) -> float | None:
        if not self.triage_evaluated or not self.triage_total:
            return None
        return self.triage_hit / self.triage_total

    def to_json(self) -> dict[str, object]:
        return {
            "n_cases": self.n_cases,
            # 参考目标与实测值并排给出：看趋势用，**不作为通过/不通过的门限**。
            "targets": {
                "redline_recall": TARGET_REDLINE_RECALL,
                "false_positive_rate": TARGET_FALSE_POSITIVE_LIMIT,
                "triage_min_accuracy": TARGET_TRIAGE_MIN_ACCURACY,
                "note": "参考目标，不判通过/不通过：指标来自自建小样本用例集，不构成临床验证",
            },
            "redline": {
                "recall": (None if self.redline_recall is None else round(self.redline_recall, 4)),
                "false_positive_rate": round(self.redline_false_positive_rate, 4),
                "tp": self.redline_tp,
                "fn": self.redline_fn,
                "fp": self.redline_fp,
                "tn": self.redline_tn,
                "missed": self.missed,
                "false_alarms": self.false_alarms,
            },
            "booking": {
                "accuracy": (None if self.booking_accuracy is None else round(self.booking_accuracy, 4)),
                "correct": self.booking_correct,
                "total": self.booking_total,
                "wrong": self.booking_wrong,
            },
            "triage": {
                "evaluated": self.triage_evaluated,
                "accuracy": (None if self.triage_accuracy is None else round(self.triage_accuracy, 4)),
                "hit": self.triage_hit,
                "total": self.triage_total,
                "missed": self.triage_missed,
            },
            # 唯一会卡闸门的东西：确定性安全不变量被触发的记录
            "invariant_violations": list(self.invariant_violations),
        }


def clinic_gold_sha256(cases: Sequence[ClinicCase]) -> str:
    """金标集的稳定指纹（与检索 harness 同一套做法：只哈希"尺子"本身）。

    基线与指纹绑定后，"改掉/删掉红线用例"不再与"红线真的全识别"长得一样：
    换尺子会被直接拒绝比较。
    """
    blob = json.dumps(
        [
            {
                "id": c.id,
                "kind": c.kind,
                "text": c.text,
                "expect_red_line": c.expect_red_line,
                "expect_booking": c.expect_booking,
                "expect_department": c.expect_department,
            }
            for c in cases
        ],
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_clinic_gold(path: Path) -> list[ClinicCase]:
    cases: list[ClinicCase] = []
    with open(path, encoding="utf-8") as fh:
        for _lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            cases.append(
                ClinicCase(
                    id=obj["id"],
                    kind=obj.get("kind", ""),
                    text=obj["text"],
                    expect_red_line=obj.get("expect_red_line"),
                    expect_booking=obj.get("expect_booking"),
                    expect_department=obj.get("expect_department", ""),
                    note=obj.get("note", ""),
                )
            )
    if not cases:
        raise ValueError(f"{path} 里没有任何用例")
    return cases


def evaluate_redline(cases: Sequence[ClinicCase], *, guard: RiskGuard | None = None) -> ClinicReport:
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


def evaluate_triage(
    cases: Sequence[ClinicCase], *, candidate_fn: Mapping[str, float] | None = None
) -> tuple[int, int, bool, list[str]]:
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


def governance_failures(
    report: ClinicReport,
    *,
    failed_calls: Sequence[str] = (),
    label: str = "clinic 评测",
) -> list[str]:
    """闸门判定：返回失败原因列表（空 = 通过）。**只卡两类事**。

    1. **这次运行能不能作为证据**：一条可评测的用例都没有（空集）、或者有题目没拿到模型
       判断（`failed_calls` 非空）。没跑成不等于跑得好——把"没拿到判断"当成通过，
       删掉语料或让密钥失效就成了让闸门变绿的手段。
    2. **有没有触发确定性安全不变量**（`report.invariant_violations`）。这些是被代码
       强制执行的那几条底线（见 `clinical/safety.py`），触发就是真出了问题，
       与样本量、与模型水平都无关。

    **指标一律不在这里判**：危险信号召回、误报率、分诊命中率都是"要看的数"，
    它们的值进报告与档案（`to_json()`），由人做趋势判断。把自建小样本的分数写成门限，
    只会逼着后来者去改门限——那是把安全寄托在数字上，而不是寄托在防护与治理上。

    `label` 只用在提示语里（规则臂 / LLM 臂的分工不同，读者要能分清是哪一臂）。
    """
    fails: list[str] = []
    if report.n_cases <= 0 and report.triage_total <= 0:
        fails.append(f"{label}：没有任何可评测的用例 —— 空集不得满足闸门（没测不等于没问题）")
    for item in failed_calls:
        fails.append(f"{label}：这一轮没有拿到模型判断 —— {item}")
    for item in report.invariant_violations:
        fails.append(f"{label}：触发了确定性安全不变量 —— {item}")
    return fails


def reporting_targets() -> dict[str, object]:
    """参考目标（报告用）。与 `ClinicReport.to_json()["targets"]` 是同一份数。"""
    return {
        "redline_recall": TARGET_REDLINE_RECALL,
        "false_positive_rate": TARGET_FALSE_POSITIVE_LIMIT,
        "triage_min_accuracy": TARGET_TRIAGE_MIN_ACCURACY,
        "note": "参考目标，不判通过/不通过：指标来自自建小样本用例集，不构成临床验证",
    }
