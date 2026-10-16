"""校准：**模型自报的置信度到底可不可信**。

不确定性量化只有在能被检验时才有意义。模型说 `confidence=0.9`，如果它在这些题上的实际
判对率只有 60%，那这个数就是装饰——**下一次没人该拿它做任何决定**。本模块把
（自报置信度，是否判对）这一串配对算成两个读数：

- **Brier 分数**：`mean((confidence - correct)^2)`，0 最好、1 最差。一个"永远说 0.5"的
  模型在正负样本各半时会得到 0.25，所以**报出来的 Brier 必须和这个基准一起看**，
  否则 0.3 到底算好还是算坏没人说得清。
- **可靠性分箱**：把置信度切成若干档，每档给出 `n` / 平均置信度 / 实际判对率。
  它回答的是"说 0.8 的时候，真的对了几成"。

## 两条纪律

1. **只报告，不上门限**。校准是个统计性质，小样本上波动很大，把它写成闸门等于用噪声
   判生死。它的用途是治理：趋势、异常、以及"要不要继续信这个字段"。
2. **样本量必须一起报**。`n=5` 的 Brier 分数没有解释力，所以 `CalibrationReport.note`
   会明说要多少样本才能谈结论——不写样本量的小数是最容易被误读的那种输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_MEANINGFUL_SAMPLES = 20
"""低于这个样本量，校准数字只当"记录"看。

取 20 不是统计上的魔法数字，而是"少于它连分箱都会空箱"的下限：5 个箱 × 每箱至少几条，
20 是能看出形状的最小量级。真正的临床校准需要成百上千条带金标的病例。
"""


@dataclass
class ReliabilityBin:
    """一个置信度区间的实测表现。"""

    lo: float
    hi: float
    n: int = 0
    mean_confidence: float = 0.0
    accuracy: float = 0.0

    def to_payload(self) -> dict[str, object]:
        return {
            "range": f"{self.lo:.1f}-{self.hi:.1f}",
            "n": self.n,
            "mean_confidence": round(self.mean_confidence, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class CalibrationReport:
    """校准结果。`pairs` 为空时 `n=0`、`brier=None`（**不是 0**：没数据不等于完美校准）。"""

    n: int = 0
    brier: float | None = None
    bins: list[ReliabilityBin] = field(default_factory=list)
    note: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "n": self.n,
            "brier": self.brier,
            "bins": [b.to_payload() for b in self.bins],
            "note": self.note,
        }


def brier_score(pairs: list[tuple[float, bool]]) -> float | None:
    """`mean((confidence - correct)^2)`。空输入返回 `None`（没测过 ≠ 校准完美）。"""
    if not pairs:
        return None
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / len(pairs)


def reliability_bins(pairs: list[tuple[float, bool]], *, n_bins: int = 5) -> list[ReliabilityBin]:
    """按置信度分箱统计。等宽分箱（[0,1] 切成 `n_bins` 段），空箱也返回（形状稳定）。

    等宽而不是等频：等频箱会让"模型从不说低于 0.7"这类问题看起来不存在——
    而"它在哪一段从没说过话"本身就是校准信息。
    """
    if n_bins < 1:
        raise ValueError(f"n_bins 必须 >= 1（收到 {n_bins}）")
    width = 1.0 / n_bins
    bins = [ReliabilityBin(lo=round(i * width, 6), hi=round((i + 1) * width, 6)) for i in range(n_bins)]
    for conf, ok in pairs:
        # 置信度 1.0 落在最后一箱（而不是溢出到不存在的第 n_bins 箱）
        idx = min(int(conf / width), n_bins - 1)
        bins[idx].n += 1
        bins[idx].mean_confidence += conf
        bins[idx].accuracy += 1.0 if ok else 0.0
    for b in bins:
        if b.n:
            b.mean_confidence /= b.n
            b.accuracy /= b.n
    return bins


def calibration_report(pairs: list[tuple[float, bool]], *, n_bins: int = 5) -> CalibrationReport:
    """算一份校准报告。`pairs` 是 `(自报置信度, 是否判对)`。

    报告只用于治理与趋势：**它不参与任何门限**（见模块文档）。
    """
    n = len(pairs)
    if not n:
        return CalibrationReport(n=0, brier=None, bins=[], note="没有可校准的样本（模型没给 confidence）")
    note = (
        f"样本量 {n} < {MIN_MEANINGFUL_SAMPLES}：只作记录，不足以谈校准结论"
        if n < MIN_MEANINGFUL_SAMPLES
        else f"样本量 {n}：可看趋势，仍不构成临床校准（需要成百上千条带金标病例）"
    )
    return CalibrationReport(
        n=n, brier=brier_score(pairs), bins=reliability_bins(pairs, n_bins=n_bins), note=note
    )
