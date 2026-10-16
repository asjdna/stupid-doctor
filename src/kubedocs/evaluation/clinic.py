"""clinic 域评测：红线安全指标（硬闸门）+ 分诊/挂号（软指标）。

**红线召回必须 100%**：急诊识别的两类错误代价不对称——漏报会延误救治（不可接受），
误报只是多一次就诊（可接受）。所以闸门要求红线一条不漏，误报率只设上限不设门限。
这条不对称取向与 `polarity` 模块"宁可留着"的保守偏向一致。

**未评测也算闸门失败**：症状 -> 科室 的关联来自上游知识卡片（标注「仅用于学术研究」），
本仓库不携带；缺语料时 `triage_department` 类用例不可能答对，所以 `evaluate_triage()`
返回 `evaluated=False`，而不是一个会误导人的 0 分。但"没测"不等于"测了没问题"，
`gate_failures()` 因此把未评测与低于 `DEFAULT_TRIAGE_MIN_ACCURACY` 一并判失败——
否则删掉语料就成了让闸门变绿的手段。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kubedocs.domains.clinic.redline import RiskGuard

# 红线误报率的容忍上限。误报比漏报轻得多，但仍要有上限——
# 否则"把所有话都判成急诊"也能拿到 100% 召回。
DEFAULT_FALSE_POSITIVE_LIMIT = 0.20

DEFAULT_TRIAGE_MIN_ACCURACY = 0.60
"""分诊科室命中率（top-3）的最低可接受值，低于它闸门不通过。

分诊是 clinic 域**唯一对外给出的结论**（"该挂哪个科"），质量闸门必须覆盖它，
否则"一条分诊都没答对"与"分诊全部答对"在退出码上没有区别。

取 0.60 的理由：

1. **不要求 100%**：答错科室 = 多跑一趟、被分诊台纠正一次，代价可恢复，
   这与红线召回"容差为 0"的不对称是同一套判断。
2. **远高于猜测上界**：候选科室约 40 个（`ClinicPack().triage_links()` 的取值集合），
   随机取 top-3 的命中上界是 `3/40 = 7.5%`；0.60 是它的 8 倍，能把"措辞碰巧撞上"
   与"真的能导诊"分开。
3. **在 5 条分诊用例上等价于"至少答对 3 条"**，是这份样本量下能看出趋势的最小值。

门限是具名常量，也可用 `gate_failures(triage_min_accuracy=...)` 按次覆盖；它**不**
替代"分诊指标本身是否可信"的判断——自建用例的成绩不是临床验证，与红线 1.0 同理。
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

    @property
    def redline_recall(self) -> float | None:
        """红线召回。**分母为 0 时返回 `None`，不是 1.0**。

        返回 1.0 会把"一条红线用例都没有"伪装成"红线全部识别"——安全闸门被空集
        满足是最危险的假绿形态。
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


def gate_failures(
    report: ClinicReport,
    *,
    fp_limit: float = DEFAULT_FALSE_POSITIVE_LIMIT,
    triage_min_accuracy: float = DEFAULT_TRIAGE_MIN_ACCURACY,
) -> list[str]:
    """闸门判定：返回失败原因列表（空 = 通过）。

    - 红线**未评测**（一条 `expect_red_line` 用例都没有）→ 失败（空集不得满足安全闸门）
    - 红线召回 < 100% → **一定失败**（安全属性，容差为 0）
    - 红线误报率 > `fp_limit` → 失败
    - 分诊**未评测**（`evaluated=False`，或评测了却一条可评测用例都没有）→ 失败
    - 分诊准确率 < `triage_min_accuracy` → 失败

    "未评测"与"测了不达标"必须分开报告，读者才知道该去补语料还是改算法；但两者都算
    失败，否则删掉红线用例或分诊语料就成了让闸门变绿的手段。
    """
    fails: list[str] = []
    recall = report.redline_recall
    if recall is None:
        fails.append(
            "红线未评测（用例集里没有任何 `expect_red_line` 用例）："
            "没测不等于测了没问题——若把它当通过，删掉红线用例就成了让闸门变绿的手段"
        )
    elif report.redline_fn:
        fails.append(f"红线漏报 {report.redline_fn} 条（召回 {recall:.2%}）——安全属性要求 100%，容差为 0")
        fails.extend(f"  漏: {m}" for m in report.missed[:10])
    if report.redline_false_positive_rate > fp_limit:
        fails.append(f"红线误报率 {report.redline_false_positive_rate:.2%} 超过上限 {fp_limit:.0%}")
        fails.extend(f"  误报: {m}" for m in report.false_alarms[:10])

    accuracy = report.triage_accuracy
    if not report.triage_evaluated or accuracy is None:
        fails.append(
            "分诊未评测（triage.evaluated=False 或没有可评测用例）："
            "没测不等于测了没问题——若把它当通过，“删掉语料”就成了让闸门变绿的手段"
        )
    elif accuracy < triage_min_accuracy:
        fails.append(
            f"分诊准确率 {accuracy:.2%}（hit {report.triage_hit}/{report.triage_total}）"
            f"低于最低门限 {triage_min_accuracy:.0%}；分诊是 clinic 域对外唯一的结论，"
            "答不出/答错科室等于给出错误指引"
        )
        fails.extend(f"  未命中: {m}" for m in report.triage_missed[:10])
    return fails


def redline_gate_failures(
    report: ClinicReport, *, fp_limit: float = DEFAULT_FALSE_POSITIVE_LIMIT
) -> list[str]:
    """**只判红线**的闸门：规则臂作为"线索提供者"时的口径。

    与 `gate_failures()` 的差别只有一处——**不看分诊**。判断权已经交给大模型
    （见 `clinical/reference.py` 的分工表），规则臂给出的科室票数只剩"线索质量"的含义：
    拿它卡门限等于用错尺子，规则臂的分诊数字因此**只报不卡**。

    红线不同：它是喂给模型的**最高优先级线索**，漏一条照样是回归，所以召回仍然
    100%、零容差，"未评测"也仍然算失败（空集不得满足安全闸门）。

    为什么把红线判据抄一遍而不是改 `gate_failures()`：那条闸门是**历史口径**，
    行为被单测逐条钉住（`tests/unit/test_clinic_eval.py`），不能有丝毫移动。
    两段判据必须**同步修改**——`gate_failures()` 里那一段是本函数的镜像。
    """
    fails: list[str] = []
    recall = report.redline_recall
    if recall is None:
        fails.append(
            "红线未评测（用例集里没有任何 `expect_red_line` 用例）："
            "没测不等于测了没问题——若把它当通过，删掉红线用例就成了让闸门变绿的手段"
        )
    elif report.redline_fn:
        fails.append(f"红线漏报 {report.redline_fn} 条（召回 {recall:.2%}）——安全属性要求 100%，容差为 0")
        fails.extend(f"  漏: {m}" for m in report.missed[:10])
    if report.redline_false_positive_rate > fp_limit:
        fails.append(f"红线误报率 {report.redline_false_positive_rate:.2%} 超过上限 {fp_limit:.0%}")
        fails.extend(f"  误报: {m}" for m in report.false_alarms[:10])
    return fails
