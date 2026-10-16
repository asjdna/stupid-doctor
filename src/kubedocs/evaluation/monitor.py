"""**长期监听**：把每一次评测变成一行趋势记录，并对新记录做回归比对。

## 为什么是"记录 + 比对"而不是"再来一道门限"

安全由三层防护与治理保证（见 `clinical/safety.py`、`docs/GOVERNANCE.md`），指标只说明趋势。
但"只记录"有一个代价：**没人看就等于没测**。长期监听要解决的就是这个——它不新增门限，
而是把当前这一轮与**最近若干轮**摆在一起，指出哪一项变了：

- **`alert`（要立即看）**：确定性安全不变量被触发、这一轮根本没跑成（未评测）、
  有题目没拿到判断。这些不是"数字波动"，是"系统这次没做到它承诺的事"；
- **`note`（看一眼）**：危险信号召回下降、误报率上升、分诊命中率下降、单题花费上升。
  它们是模型行为的波动，**默认不改退出码**——把波动当回归，等于让监听变成噪音源，
  而这正是当初删掉指标门限的原因。需要机器硬拦时显式加 `--strict`。

阈值不是拍出来的：`note` 一律与**最近几轮的实际区间**比较（低于历史最低 / 高于历史最高），
所以它随样本自然收紧或放宽，不需要人去调一个"合理误差"。

## 记录长什么样

每行一条 JSON（`data/eval/clinic_trend.jsonl`，追加、永不覆盖），字段刻意拍平：
时间戳、引擎、金标指纹、提示词版本、两臂的关键指标、失败调用数、不变量触发数、
模型/工具调用数与花费。**不记患者原话**——与安全审计同一条纪律。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_WINDOW = 5
"""比对窗口：默认拿最近 5 轮做参考区间。

取 5 的理由：模型每一轮都有波动，窗口太小会把抖动当回归（红得没人看），
太大则异常要等很多轮才显现。5 轮是"能看出区间、又能及时报警"的折中。
"""


@dataclass
class Alert:
    """一条监听结论。`severity` 决定它改不改退出码（见模块文档）。"""

    id: str
    severity: str  # "alert" | "note"
    detail: str

    def render(self) -> str:
        mark = "✗" if self.severity == "alert" else "⚠"
        return f"  {mark} {self.id}：{self.detail}"


@dataclass
class TrendRecord:
    """一轮评测的**扁平**记录。字段少而稳定，方便人看也方便以后加列。"""

    ts: str
    engine: str = ""
    evaluated: bool = False
    n_cases: int = 0
    danger_recall: float | None = None
    danger_fp_rate: float | None = None
    triage_accuracy: float | None = None
    failed_calls: int = 0
    invariants: int = 0
    invariant_ids: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    cost_usd: float = 0.0
    prompt_version: str = ""
    gold_sha256: str = ""
    rule_danger_recall: float | None = None
    rule_danger_fp_rate: float | None = None
    rule_triage_accuracy: float | None = None
    note: str = ""

    def to_payload(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def cost_per_case(self) -> float | None:
        """单题花费——比总额更能反映"是不是越问越贵"。"""
        return self.cost_usd / self.n_cases if self.n_cases else None


def trend_path(index_root: str | Path) -> Path:
    """趋势记录的落点。与治理档案同目录（`data/eval/`），与索引分开。"""
    return Path(index_root) / "clinic_trend.jsonl"


def record_from_reports(
    *,
    judge: Any,
    rule: Any = None,
    prompt_version: str = "",
    gold_sha256: str = "",
    note: str = "",
    now: datetime | None = None,
) -> TrendRecord:
    """把两臂的报告压成一条趋势记录。报告对象只需有同名属性（打桩替身也能用）。"""
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    recall = getattr(judge, "redline_recall", None)
    return TrendRecord(
        ts=stamp,
        engine=str(getattr(judge, "engine", "") or ""),
        evaluated=bool(getattr(judge, "evaluated", False)),
        n_cases=int(getattr(judge, "n_cases", 0) or 0),
        danger_recall=None if recall is None else round(float(recall), 4),
        danger_fp_rate=round(float(getattr(judge, "redline_false_positive_rate", 0.0) or 0.0), 4),
        triage_accuracy=_round_or_none(getattr(judge, "triage_accuracy", None)),
        failed_calls=int(getattr(judge, "n_failed_calls", 0) or 0),
        invariants=len(getattr(judge, "invariant_violations", []) or []),
        invariant_ids=_invariant_ids(getattr(judge, "invariant_violations", []) or []),
        llm_calls=int(getattr(judge, "llm_calls", 0) or 0),
        tool_calls=int(getattr(judge, "tool_calls", 0) or 0),
        steps=int(getattr(judge, "steps", 0) or 0),
        cost_usd=round(float(getattr(judge, "cost_usd", 0.0) or 0.0), 6),
        prompt_version=prompt_version,
        gold_sha256=gold_sha256,
        rule_danger_recall=None if rule is None else _round_or_none(getattr(rule, "redline_recall", None)),
        rule_danger_fp_rate=None
        if rule is None
        else round(float(getattr(rule, "redline_false_positive_rate", 0.0) or 0.0), 4),
        rule_triage_accuracy=None if rule is None else _round_or_none(getattr(rule, "triage_accuracy", None)),
        note=note,
    )


def append_record(path: str | Path, record: TrendRecord) -> bool:
    """追加一行。写失败**不让评测失败**（与审计同一条纪律），但调用方应知道没写成。"""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_payload(), ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def read_records(path: str | Path, *, limit: int = 0) -> list[TrendRecord]:
    """读回趋势记录。坏行跳过（一条脏数据不该让整份历史读不出来）。"""
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[TrendRecord] = []
    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(_record_from_payload(payload))
    return rows[-limit:] if limit else rows


def compare(records: Sequence[TrendRecord], *, window: int = DEFAULT_WINDOW, engine: str = "") -> list[Alert]:
    """拿**最新一条**与它之前的 `window` 条比对，返回监听结论。

    `engine` 给了就只在同一引擎的历史里比：工具循环臂与单发臂的数量级本来就不同
    （一个检索、一个不检索），混在一起比会得到假的"回归"。
    """
    if not records:
        return []
    latest = records[-1]
    history = [r for r in records[:-1] if not engine or r.engine == engine]
    history = history[-window:]
    alerts: list[Alert] = []

    # ---- 结构性：这些不参与"波动"讨论
    if not latest.evaluated:
        alerts.append(Alert("arm_not_evaluated", "alert", f"这一轮没跑成：{latest.note or '原因见报告'}"))
    if latest.invariants:
        alerts.append(
            Alert(
                "invariant_fired",
                "alert",
                f"触发了 {latest.invariants} 条确定性安全不变量：{latest.invariant_ids}",
            )
        )
    if latest.failed_calls:
        alerts.append(
            Alert("failed_calls", "alert", f"{latest.failed_calls}/{latest.n_cases} 条用例没拿到模型判断")
        )

    # ---- 趋势项：与历史实际区间比，不与一个拍出来的阈值比
    for label, attr, lower_is_worse in (
        ("危险信号召回", "danger_recall", True),
        ("分诊命中率", "triage_accuracy", True),
        ("误报率", "danger_fp_rate", False),
        ("单题花费", "cost_per_case", False),
    ):
        now_value = _metric(latest, attr)
        past = [v for v in (_metric(r, attr) for r in history) if v is not None]
        if now_value is None or not past:
            continue
        if lower_is_worse and now_value < min(past):
            alerts.append(
                Alert(
                    f"{attr}_dropped",
                    "note",
                    f"{label} {now_value:.4f} 低于最近 {len(past)} 轮的最低值 {min(past):.4f}",
                )
            )
        elif not lower_is_worse and now_value > max(past):
            alerts.append(
                Alert(
                    f"{attr}_rose",
                    "note",
                    f"{label} {now_value:.4f} 高于最近 {len(past)} 轮的最高值 {max(past):.4f}",
                )
            )
    if history and latest.prompt_version and latest.prompt_version != history[-1].prompt_version:
        alerts.append(
            Alert(
                "prompt_changed",
                "note",
                f"提示词版本变了（{history[-1].prompt_version} → {latest.prompt_version}）：跨版本的分数不可直接比",
            )
        )
    if history and latest.gold_sha256 and latest.gold_sha256 != history[-1].gold_sha256:
        alerts.append(Alert("gold_changed", "note", "金标集变了：这一轮与之前的记录不可比（尺子换了）"))
    return alerts


def format_trend(records: Sequence[TrendRecord], *, n: int = 8) -> str:
    """把最近 `n` 条记录打成一张小表（人看的；趋势靠它，而不是靠翻 JSON）。"""
    rows = list(records)[-n:]
    if not rows:
        return "（还没有趋势记录：跑一次 `scripts/eval_clinic.py --arm both --archive data/eval/clinic_archive`）"
    head = (
        f"{'时间':<20} {'引擎':<10} {'召回':>6} {'误报':>6} {'分诊':>6} "
        f"{'失败':>4} {'不变量':>6} {'单题$':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.ts[:19]:<20} {r.engine or '-':<10} "
            f"{_pct(r.danger_recall):>6} {_pct(r.danger_fp_rate):>6} {_pct(r.triage_accuracy):>6} "
            f"{r.failed_calls:>4} {r.invariants:>6} "
            f"{('-' if r.cost_per_case is None else f'{r.cost_per_case:.5f}'):>8}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- 内部


def _round_or_none(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _metric(record: TrendRecord, attr: str) -> float | None:
    if attr == "cost_per_case":
        return record.cost_per_case
    value = getattr(record, attr, None)
    return None if value is None else float(value)


def _invariant_ids(violations: Iterable[Any]) -> list[str]:
    """从"`用例id：不变量id（细节）`"这种字符串里取出不变量 id（保持出现顺序、去重）。"""
    ids: list[str] = []
    for item in violations:
        text = str(item)
        _, _, after = text.partition("：")
        head = after.split("（", 1)[0].strip() if after else ""
        if head and head not in ids:
            ids.append(head)
    return ids


def _record_from_payload(payload: dict[str, Any]) -> TrendRecord:
    known = set(TrendRecord.__dataclass_fields__)
    return TrendRecord(**{k: v for k, v in payload.items() if k in known})
