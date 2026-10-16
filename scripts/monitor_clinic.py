#!/usr/bin/env python
"""长期监听：**定期跑一轮评测、留一行趋势、把变化指出来**。

它不新增门限——安全由三层防护与治理保证（见 `docs/GOVERNANCE.md`），指标只说明趋势。
它解决的是"只记录、没人看"：每一轮与最近若干轮摆在一起，指出**哪一项变了**，
并区分"系统这次没做到承诺的事"（`alert`）与"模型行为的波动"（`note`）。

## 三种用法

```bash
python scripts/monitor_clinic.py --status                 # 只看历史（不花钱、不联网）
python scripts/monitor_clinic.py --run                    # 跑一轮：评测 → 留档 → 趋势 → 提醒
python scripts/monitor_clinic.py --follow --interval 3600 # 常驻：每小时一轮（容器边车/cron 都行）
```

`--run` / `--follow` **复用 `scripts/eval_clinic.py` 的 `main()`**（同一份装配、同一套闸门、
同一份报告与档案）：监听自己再写一遍评测，就会得到一份"和产品不一样"的趋势，
而那种偏差最难发现。

## 退出码

- `0`：这一轮没有 `alert`（`note` 会打印，但**不改退出码**——把波动当回归会让监听变成噪音）；
- `1`：有 `alert`（不变量被触发 / 这一轮没跑成 / 有题目没拿到判断）；
- `--strict` 时：`note` 也让退出码变 1（给"必须硬拦"的流水线用）；
- `--status` 永远 0（它不产生新数据，只看）。

参数错误返回 `2`（与评测脚本的"闸门不通过"区分开：这是用法问题，不是系统问题）。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

TREND_FILE = REPO_ROOT / "data" / "eval" / "clinic_trend.jsonl"
ARCHIVE_DIR = REPO_ROOT / "data" / "eval" / "clinic_archive"

EXIT_OK = 0
EXIT_ALERT = 1
EXIT_USAGE = 2


def _emit(text: str) -> None:
    print(text, flush=True)


def _load_eval_main() -> Callable[[list[str] | None], int]:
    """取评测脚本的 `main()`。**复用**它=复用同一份装配与闸门（见模块文档）。"""
    import eval_clinic  # scripts/ 已在 sys.path 上

    return eval_clinic.main


def run_round(
    *,
    eval_main: Callable[[list[str] | None], int],
    gold: str,
    trend: Path,
    archive: Path | str,
    arm: str = "both",
    extra: Sequence[str] = (),
) -> tuple[int, list[Any]]:
    """跑一轮：评测（会写档案与趋势）→ 读回趋势 → 比对。

    返回 `(评测退出码, 监听结论)`。评测退出码原样保留：它是"这次运行能不能作为证据"
    的判定（闸门只卡不变量与拿到没拿到判断），监听只在其之上补"与历史比怎么样"。
    """
    from kubedocs.evaluation.monitor import compare, read_records

    argv = [
        "--gold",
        gold,
        "--arm",
        arm,
        "--archive",
        str(archive),
        "--trend-path",
        str(trend),
        *extra,
    ]
    code = int(eval_main(argv))
    records = read_records(trend)
    alerts = compare(records, engine=records[-1].engine if records else "")
    return code, alerts


def report(*, trend: Path, alerts: Sequence[Any], window_n: int) -> None:
    """打印趋势表与提醒。表在前、提醒在后：先看形状，再看结论。"""
    from kubedocs.evaluation.monitor import format_trend, read_records

    records = read_records(trend)
    _emit(f"\n=== 趋势（{trend}，共 {len(records)} 轮）")
    _emit(format_trend(records, n=window_n))
    if not records:
        return
    _emit("\n=== 这一轮的结论")
    if not alerts:
        _emit("  ✓ 与最近几轮相比没有异常（既没触发安全不变量，也没有指标越出历史区间）")
        return
    for alert in alerts:
        _emit(alert.render())
    hard = [a for a in alerts if a.severity == "alert"]
    soft = [a for a in alerts if a.severity != "alert"]
    if hard:
        _emit(f"\n  → {len(hard)} 条要立即看；{len(soft)} 条是趋势提示（默认不改退出码）")
    else:
        _emit(f"\n  → 只有 {len(soft)} 条趋势提示：模型行为的波动，默认不改退出码（要硬拦加 --strict）")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="clinic 域的长期监听（评测 → 趋势 → 提醒）")
    p.add_argument("--status", action="store_true", help="只看历史趋势，不跑评测（不花钱、不联网）")
    p.add_argument("--run", action="store_true", help="跑一轮评测并留档（默认动作）")
    p.add_argument("--follow", action="store_true", help="常驻：每 --interval 秒跑一轮")
    p.add_argument("--interval", type=float, default=3600.0, help="常驻模式的间隔秒数（默认 3600）")
    p.add_argument("--rounds", type=int, default=0, help="常驻模式最多跑几轮（0 = 不限）")
    p.add_argument("--arm", default="both", choices=("both", "rule", "llm", "tool", "reasoner"))
    p.add_argument("--gold", default=str(REPO_ROOT / "data" / "eval" / "clinic_gold.jsonl"))
    p.add_argument("--trend-path", default=str(TREND_FILE))
    p.add_argument("--archive", default=str(ARCHIVE_DIR))
    p.add_argument("--window", type=int, default=5, help="比对窗口（最近几轮，默认 5）")
    p.add_argument("--show", type=int, default=8, help="趋势表打印几行（默认 8）")
    p.add_argument("--strict", action="store_true", help="趋势提示也让退出码变 1（给硬拦的流水线用）")
    p.add_argument("--extra", default="", help="透传给评测脚本的额外参数（空格分隔）")
    args = p.parse_args(argv)

    trend = Path(args.trend_path)
    if args.status:
        # 只看历史**也要给结论**：运维打开它是想问"现在这样算不算异常"。
        # 这里用已存的最后一条当"当前"，与它之前的历史比对——不产生新数据，所以永远退 0。
        from kubedocs.evaluation.monitor import compare, read_records

        records = read_records(trend)
        alerts = compare(records, window=args.window, engine=records[-1].engine if records else "")
        report(trend=trend, alerts=alerts, window_n=args.show)
        return EXIT_OK

    if args.interval <= 0:
        _emit("参数错误：--interval 必须为正数（常驻模式的间隔秒数）")
        return EXIT_USAGE

    eval_main = _load_eval_main()
    extra = [chunk for chunk in args.extra.split(" ") if chunk]
    rounds = 0
    code = EXIT_OK
    while True:
        rounds += 1
        _emit(f"\n=== 第 {rounds} 轮（{time.strftime('%Y-%m-%d %H:%M:%S')}）")
        eval_code, alerts = run_round(
            eval_main=eval_main,
            gold=args.gold,
            trend=trend,
            archive=args.archive,
            arm=args.arm,
            extra=extra,
        )
        report(trend=trend, alerts=alerts, window_n=args.show)
        hard = any(a.severity == "alert" for a in alerts)
        # 退出码：不变量/未跑成/有题没判断 ⇒ 1；`--strict` 时趋势提示也算。
        code = EXIT_ALERT if (eval_code != 0 or hard or (args.strict and alerts)) else EXIT_OK
        if not args.follow:
            return code
        if args.rounds and rounds >= args.rounds:
            _emit(f"\n已跑满 {rounds} 轮，退出（退出码 {code}）")
            return code
        _emit(f"\n（下一轮在 {args.interval:.0f} 秒后）")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
