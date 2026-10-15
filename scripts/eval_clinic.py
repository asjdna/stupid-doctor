"""clinic 域评测：红线硬闸门 + 挂号意图 + 分诊。

用法：
    python scripts/eval_clinic.py --gate --json      # 跑评测；闸门不通过则退出码 2
    python scripts/eval_clinic.py --write-baseline   # 用当前结果重建冻结基线

前置条件：`--gold` 默认 `data/eval/clinic_gold.jsonl`；分诊依赖 clinic 受限卡片语料，
缺语料时记为「未评测」、闸门不通过。
输出：红线召回率/误报率、挂号意图准确率、分诊 top-3 命中率与证据键覆盖率、各错误明细。
退出码：0 通过 / 1 参数错误 / 2 闸门不通过 / 3 前置条件不满足。
`--json` 只改格式（旁白改道 stderr），**不**绕过闸门，退出码仍由 `--gate` 决定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.domains.clinic.pack import wants_booking  # noqa: E402
from kubedocs.evaluation.clinic import (  # noqa: E402
    clinic_gold_sha256,
    evaluate_redline,
    evaluate_triage,
    gate_failures,
    load_clinic_gold,
)

GOLD = REPO_ROOT / "data" / "eval" / "clinic_gold.jsonl"
BASELINE = REPO_ROOT / "data" / "eval" / "clinic_baseline.json"

GATE_FP_TOLERANCE = 0.05  # 相对基线允许的误报率上升幅度；红线召回容差为 0（安全属性）

TRIAGE_RED_NOTE = """分诊为什么是红的：冻结基线里记着的就是 0 分。
  · data/eval/clinic_baseline.json 的 triage.accuracy = 0.0（hit 0 / total 5、evaluated=true）；
  · 根因是受限卡片语料与症状词典两套词汇不重合（证据键覆盖率 1/6）；
  · 基线是冻结的、**不允许为了让闸门变绿而改**（改基线等于把红灯涂成绿灯）；
  · 要让它变绿只有一条路：真的把分诊做对（补口语->体征映射或换语料），而不是放宽门限。
  门限取值理由见 src/kubedocs/evaluation/clinic.py 的 DEFAULT_TRIAGE_MIN_ACCURACY。"""


def _emit(text: str, *, json_mode: bool) -> None:
    """`--json` 时旁白改道 stderr，stdout 留给纯 JSON（否则 `--gate --json | jq` 会被旁白打断）。"""
    print(text, file=sys.stderr if json_mode else sys.stdout)


def _triage_candidates(*, json_mode: bool = False) -> Any:
    """构造"文本 -> 候选分数"。没有语料就返回 None（指标标为未评测）。"""
    try:
        from kubedocs.agent import hypothesis
        from kubedocs.domains.clinic.domain import ClinicPack
        from kubedocs.domains.clinic.pack import extract_evidence

        links = ClinicPack().triage_links()
    except FileNotFoundError as e:
        _emit(f"（分诊指标未评测：{e.args[0].splitlines()[0]}）", json_mode=json_mode)
        return None
    if not links:
        _emit("（分诊指标未评测：症状-科室关联表为空）", json_mode=json_mode)
        return None

    def candidates_for(text: str) -> dict[str, float] | None:
        return None  # 占位：见下方按用例逐一计算

    return links, extract_evidence, hypothesis, candidates_for


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="clinic 域评测（红线硬闸门）")
    p.add_argument("--gold", default=str(GOLD))
    p.add_argument("--baseline", default=str(BASELINE))
    p.add_argument("--gate", action="store_true")
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cases = load_clinic_gold(Path(args.gold))
    report = evaluate_redline(cases)

    # 挂号意图（确定性词表，不依赖语料）
    for case in cases:
        if case.expect_booking is None:
            continue
        report.booking_total += 1
        if wants_booking(case.text) == case.expect_booking:
            report.booking_correct += 1
        else:
            report.booking_wrong.append(f"{case.id} 期望 {case.expect_booking}：{case.text}")

    # 分诊（依赖受限语料；缺语料时 evaluated=False，闸门**不通过**）
    triage_cases = [c for c in cases if c.expect_department]
    built = _triage_candidates(json_mode=args.json)
    if built is not None:
        links, extract_evidence, hypothesis, _ = built
        # 与 `evaluation/clinic.evaluate_triage()` 同口径：有语料但无分诊用例同样是"没测"。
        triage_total, triage_hit, evaluated, missed = 0, 0, bool(triage_cases), []
        cov_num: int = 0
        cov_den: int = 0
        for case in triage_cases:
            ev = extract_evidence(case.text, 1)
            # 覆盖率诊断：抽出的症状有几个真在关联表里做键。两套词汇不重合时命中率必然为 0，
            # 而这个数字能区分"数据还支撑不了该指标"与"系统试过但失败了"。
            cov_den += len(ev)
            cov_num += sum(1 for e in ev if e.symptom in links)
            scores = hypothesis.score(ev, links)
            top3 = [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:3]]
            triage_total += 1
            if case.expect_department in top3:
                triage_hit += 1
            else:
                missed.append(f"{case.id} 期望 {case.expect_department}，实际 {top3}")
        report.triage_total, report.triage_hit = triage_total, triage_hit
        report.triage_evaluated, report.triage_missed = evaluated, missed
        _ = evaluate_triage  # 保留导出以便单测直接调用

    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, indent=2))
    else:
        payload = report.to_json()
        raw_redline = payload["redline"]
        assert isinstance(raw_redline, dict)
        r = raw_redline
        print(
            f"用例总数    : {report.n_cases}（红线） + {report.booking_total}（挂号意图）"
            f" + {len(triage_cases)}（分诊）"
        )
        print(
            f"红线召回    : "
            f"{'未评测（用例集里没有 expect_red_line）' if report.redline_recall is None else f'{report.redline_recall:.2%}'}"
            f"  (TP={report.redline_tp} FN={report.redline_fn})"
            f"   ← 硬闸门，要求 100%"
        )
        print(
            f"红线误报率  : {report.redline_false_positive_rate:.2%}"
            f"  (FP={report.redline_fp} TN={report.redline_tn})"
        )
        acc_booking = report.booking_accuracy
        print(
            f"挂号意图    : "
            f"{'未评测' if acc_booking is None else f'{acc_booking:.2%}'}"
            f"  ({report.booking_correct}/{report.booking_total})"
        )
        acc = report.triage_accuracy
        print(f"分诊科室    : {'未评测（语料缺失）' if acc is None else f'{acc:.2%} top3 命中'}")
        if report.triage_evaluated and cov_den:
            print(f"  证据键覆盖率: {cov_num}/{cov_den} —— 抽出的症状有几个真在关联表里做键")
            if cov_num == 0:
                print("  ⚠ 覆盖率 0：**两套词汇不重合**。词典认的是口语主诉（咳嗽/发热），")
                print("    卡片键是临床体征（纤毛结构缺陷/氧分压低）。这不是调参能解决的，")
                print("    需要一层口语->体征映射或换语料（记入 P5）。")
        for label, items in (
            ("漏报", r["missed"]),
            ("误报", r["false_alarms"]),
            ("挂号判错", report.booking_wrong),
            ("分诊未命中", report.triage_missed),
        ):
            for item in list(items)[:10]:
                print(f"  ✗ {label}: {item}")

    if args.write_baseline:
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            # 金标指纹必须一起冻结：只存指标的话，"改了尺子"与"变好了"看起来一模一样。
            "gold_sha256": clinic_gold_sha256(cases),
            "n_cases": report.n_cases,
            "metrics": report.to_json(),
        }
        Path(args.baseline).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit(f"\n已写入基线：{args.baseline}", json_mode=args.json)
        return 0

    if not args.gate:
        return 0

    fails = gate_failures(report)
    base_path = Path(args.baseline)
    if base_path.exists():
        base_raw = json.loads(base_path.read_text(encoding="utf-8"))
        base = base_raw["metrics"]
        current_sha = clinic_gold_sha256(cases)
        if "gold_sha256" not in base_raw:
            # 旧格式基线缺指纹即拒绝比较：跳过校验会放过"改了尺子"的情况。
            fails.append(
                "基线没有 gold_sha256（旧格式）：无法判断这把尺子是否被改过 ⇒ 拒绝比较。"
                "确认无误后用 `--write-baseline` 重建基线"
            )
        elif base_raw["gold_sha256"] != current_sha:
            fails.append(
                f"标注集已变更（gold_sha256 {base_raw['gold_sha256']} -> {current_sha}）——"
                "跨标注集的分数不可比，请确认后用 `--write-baseline` 重建基线"
            )
        else:
            if int(base_raw.get("n_cases", -1)) != report.n_cases:
                fails.append(
                    f"红线用例数不一致（基线 {base_raw.get('n_cases')}，本次 {report.n_cases}）："
                    "指纹相同却用例数不同，说明基线格式异常，拒绝比较"
                )
            base_fp = float(base["redline"]["false_positive_rate"])
            delta = report.redline_false_positive_rate - base_fp
            if delta > GATE_FP_TOLERANCE:
                fails.append(f"红线误报率相对基线上升 {delta:.2%}（上限 {GATE_FP_TOLERANCE:.0%}）")
    else:
        _emit("（没有基线文件；跳过基线比较）", json_mode=args.json)

    for f in fails:
        _emit(("  ✗ " if not f.startswith("  ") else "") + f, json_mode=args.json)
    if any("分诊" in f and not f.startswith("  ") for f in fails):
        _emit("\n" + TRIAGE_RED_NOTE, json_mode=args.json)
    if fails:
        _emit("\n闸门：不通过", json_mode=args.json)
        return 2
    _emit("\n闸门：通过", json_mode=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
