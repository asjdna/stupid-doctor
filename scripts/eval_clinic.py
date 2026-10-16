"""clinic 域评测：**两个臂分开报**——规则臂（线索提供者）+ LLM 臂（真正的判断者）。

用法：
    python scripts/eval_clinic.py --arm both --gate --json   # 默认：两个臂都跑，LLM 臂是硬闸门
    python scripts/eval_clinic.py --arm rule --gate          # 离线只跑规则臂（不花钱）
    python scripts/eval_clinic.py --arm llm                  # 只跑 LLM 臂（会调模型、会花钱）
    python scripts/eval_clinic.py --write-baseline           # 重建冻结基线（含 LLM 臂数值）

## 为什么是两个臂

架构已把全部判断交给大模型：急诊红线词表 + 三维极性、症状词表、症状→科室票数都
**降级为证据提供者**（见 `clinical/reference.py` 的分工表）。所以闸门不能只测规则轨——
那测的是一个**已经不再做决定**的组件，它绿着说明不了产品。本脚本因此分两臂：

- **规则臂**（确定性、免费）：红线召回（喂给模型的最高优先级线索一条都不许漏）
  + 相对基线的误报率回归比较。它的**分诊数字只报不卡**——规则臂已经是线索提供者，
  拿它卡门限等于用错尺子。
- **LLM 臂**（`ClinicalReasoner`，每题一次调用）：**硬闸门**，红线召回 100%（零容差）、
  误报率 ≤20%、分诊 ≥60%。这些门限是固定常量，**不拿基线做比较**（模型不确定，
  重跑分数会波动）；基线里只记录它"当时是什么样子"。没有密钥时标为「未评测」，
  而**未评测 = 闸门失败**（没测不等于测了没问题）。

`--arm rule` 是**离线复现历史规则轨**的模式：此时没有 LLM 判断者，规则臂自己的口径
（含分诊门限）仍然卡。一旦 LLM 臂参与（`both`/`llm`），判断权归 LLM，
规则臂的分诊数字就只剩"线索质量"的含义，只报不卡。

前置条件：`--gold` 默认 `data/eval/clinic_gold.jsonl`；规则臂的分诊依赖 clinic 受限卡片
语料，缺语料时记为「未评测」。
输出：两个臂各自的红线召回率/误报率、分诊 top-3 命中率与错误明细；LLM 臂另有
失败调用数 / 调用次数 / 花费。
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
    ClinicCase,
    ClinicReport,
    clinic_gold_sha256,
    evaluate_redline,
    evaluate_triage,
    gate_failures,
    load_clinic_gold,
    redline_gate_failures,
)
from kubedocs.evaluation.clinic_llm import (  # noqa: E402
    LLM_FALSE_POSITIVE_LIMIT,
    LLM_TRIAGE_MIN_ACCURACY,
    LlmClinicReport,
    evaluate_llm_arm,
    llm_gate_failures,
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

LLM_TRIAGE_RED_NOTE = """LLM 臂分诊为什么是红的：分诊现在是模型的结论（`assessment.departments` 前三名）。
  · 门限 60% 是**固定常量**（`clinic_llm.LLM_TRIAGE_MIN_ACCURACY`），不看基线：
    基线只记录它重跑时的数值，模型不是确定性的，拿基线卡它会把波动当成回归；
  · 要让它变绿只有一条路：让模型真的答对科室（改提示词/给更好的候选线索/补语料），
    而不是放宽门限或换一把尺子。"""

LLM_BASELINE_NOTE = """LLM 臂的分数**每次重跑都会有波动**（模型不是确定性的）：把它写进基线是为了
"记录当时是什么样子"，**不是**当回归门限用。回归门限只看规则臂
（metrics.redline.false_positive_rate 与基线比较）；LLM 臂的门限是固定常量
（红线召回 100% / 误报率 ≤20% / 分诊 ≥60%，见 clinic_llm.llm_gate_failures）。"""


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


def _run_rule_arm(cases: list[ClinicCase], *, json_mode: bool) -> tuple[ClinicReport, int, int]:
    """规则臂：红线 + 挂号意图 + 分诊。**行为与改动前的脚本逐字一致**。

    它现在是线索提供者的质量记录，不再是产品结论的来源；但记录口径不变，
    历史基线才比得下去。
    """
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

    # 分诊（依赖受限语料；缺语料时 evaluated=False）
    cov_num, cov_den = 0, 0
    triage_cases = [c for c in cases if c.expect_department]
    built = _triage_candidates(json_mode=json_mode)
    if built is not None:
        links, extract_evidence, hypothesis, _ = built
        # 与 `evaluation/clinic.evaluate_triage()` 同口径：有语料但无分诊用例同样是"没测"。
        triage_total, triage_hit, evaluated, missed = 0, 0, bool(triage_cases), []
        for case in triage_cases:
            # `links` 一起传进去：抽取与打分必须用同一张词表，否则抽出的键在关联表里
            # 查不到、分数全空——那正是这条指标一直是 0 的形态。
            ev = extract_evidence(case.text, 1, links=links)
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
    return report, cov_num, cov_den


def _llm_settings() -> Any:
    """LLM 臂用的 `Settings`：与 `scripts/eval_medical_cot.py` 同一套（套上 clinic 域默认值）。"""
    from kubedocs.cli import apply_domain_defaults
    from kubedocs.config import Settings

    return apply_domain_defaults(Settings(domain="clinic"))


def _llm_arm_status() -> tuple[bool, str]:
    """LLM 臂能不能跑：`.env` 里有没有**真的**密钥。返回 `(可用, 不可用时的原因)`。

    缺密钥必须在这里就停下并标为「未评测」：让它跑下去的话，每一题都会走降级路径，
    指标全是 0 却长得像"模型表现差"——那是把"没测"伪装成"测了"。
    """
    from kubedocs.config import load_env_file
    from kubedocs.generate import has_api_key, required_api_key_env

    # 先把 .env 导进 os.environ：`has_api_key()` 与 litellm 读的都是它，而 pydantic-settings
    # 只填 Settings 对象；漏了这步整轮评测会全走降级路径，却照样打印指标。
    load_env_file()
    model = str(_llm_settings().llm_model)
    if has_api_key(model):
        return True, ""
    return False, f"未配置大模型密钥（{required_api_key_env(model)} 未设置或形如占位符）"


def _build_llm_reasoner() -> tuple[Any, Any]:
    """装配 LLM 臂：`Settings` → `DailyBudget` → `BudgetedClient` → `ClinicalReasoner`。

    **与 `scripts/eval_medical_cot.py` 是同一套装配形状**（同一本日预算账 `llm_spend.json`），
    不另发明一套：新增一条产品路径却绕过预算闸，等于把已有的花费上限作废。
    返回 `(reasoner, budget)`；`budget` 交给评测读本次花费。
    """
    from kubedocs.clinical.reasoning import BudgetedClient, ClinicalReasoner, LiteLLMAdapter
    from kubedocs.generate import DailyBudget, LiteLLMClient

    settings = _llm_settings()
    budget = DailyBudget(
        settings.llm_spend_path,
        settings.llm_daily_budget_usd,
        unpriced_usd=settings.llm_unpriced_call_usd,
    )
    client = BudgetedClient(LiteLLMClient(settings), budget)
    return ClinicalReasoner(LiteLLMAdapter(client)), budget


def _run_llm_arm(cases: list[ClinicCase]) -> LlmClinicReport:
    """LLM 臂：真正做判断的那条路。**任何失败都收敛成"未评测/失败调用"，不许让脚本崩。**"""
    n_cases = sum(1 for c in cases if c.expect_red_line is not None or c.expect_department)
    ok, why = _llm_arm_status()
    if not ok:
        return LlmClinicReport(evaluated=False, reason=why, n_cases=n_cases)
    try:
        reasoner, budget = _build_llm_reasoner()
    except Exception as e:  # noqa: BLE001 - 密钥/依赖/网络配置问题都只报告，不崩评测
        return LlmClinicReport(
            evaluated=False, reason=f"LLM 臂装配失败（{type(e).__name__}: {e}）", n_cases=n_cases
        )
    return evaluate_llm_arm(cases, reasoner=reasoner, budget=budget)


def _print_rule_arm(report: ClinicReport, triage_cases: list[ClinicCase], cov_num: int, cov_den: int) -> None:
    print("规则臂（确定性，是线索提供者，不再是判断者）")
    print(
        f"用例总数    : {report.n_cases}（红线） + {report.booking_total}（挂号意图）"
        f" + {len(triage_cases)}（分诊）"
    )
    recall = report.redline_recall
    print(
        f"红线召回    : "
        f"{'未评测（用例集里没有 expect_red_line）' if recall is None else f'{recall:.2%}'}"
        f"  (TP={report.redline_tp} FN={report.redline_fn})"
        f"   ← 喂给模型的最高优先级线索，一条都不许漏"
    )
    print(
        f"红线误报率  : {report.redline_false_positive_rate:.2%}"
        f"  (FP={report.redline_fp} TN={report.redline_tn})"
        f"   ← 与基线比较（判据不因臂而异）"
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
        ("漏报", report.missed),
        ("误报", report.false_alarms),
        ("挂号判错", report.booking_wrong),
        ("分诊未命中", report.triage_missed),
    ):
        for item in list(items)[:10]:
            print(f"  ✗ {label}: {item}")


def _print_llm_arm(report: LlmClinicReport) -> None:
    print("LLM 臂（真正的判断者）")
    if not report.evaluated:
        print(f"  未评测      : {report.reason}")
        print(
            "  · **未评测不等于通过**：红线召回 100% / 误报率 ≤20% / 分诊 ≥60% 这三条门限一条都没有被检验。"
        )
        return
    recall = report.redline_recall
    print(
        f"  红线召回    : "
        f"{'未评测（用例集里没有 expect_red_line）' if recall is None else f'{recall:.2%}'}"
        f"  (TP={report.redline_tp} FN={report.redline_fn})   ← 硬闸门，要求 100%"
    )
    print(
        f"  红线误报率  : {report.redline_false_positive_rate:.2%}"
        f"  (FP={report.redline_fp} TN={report.redline_tn})"
        f"   ← 上限 {LLM_FALSE_POSITIVE_LIMIT:.0%}"
    )
    acc = report.triage_accuracy
    print(
        f"  分诊科室    : "
        f"{'未评测（没有 expect_department 用例）' if acc is None else f'{acc:.2%} top3 命中'}"
        f"  (hit {report.triage_hit}/{report.triage_total})   ← 门限 {LLM_TRIAGE_MIN_ACCURACY:.0%}"
    )
    print(f"  失败调用    : {report.n_failed_calls}   ← 没有得到模型判断的题（未评测不等于通过）")
    print(f"  调用次数    : {report.llm_calls}")
    print(f"  花费（美元）: {report.cost_usd:.6f}")
    print(
        "  · 门限是固定常量（红线召回 100% / 误报率 ≤20% / 分诊 ≥60%），**不拿基线做比较**："
        "模型不是确定性的，基线只记录它当时的样子。"
    )
    for label, items in (
        ("漏报", report.redline_missed),
        ("误报", report.redline_false_alarms),
        ("分诊未命中", report.triage_missed),
        ("未判断", report.failed_calls),
    ):
        for item in list(items)[:10]:
            print(f"  ✗ {label}: {item}")


def _baseline_failures(
    report: ClinicReport, cases: list[ClinicCase], baseline: Path, *, json_mode: bool
) -> list[str]:
    """规则臂与冻结基线的比较（现有逻辑：指纹绑定 + 误报率回归）。

    只有规则臂进这里：LLM 臂的分数每次重跑都会波动（模型不确定），拿它做回归门限
    等于把波动当回归。它的门限是固定常量。
    """
    fails: list[str] = []
    if not baseline.exists():
        _emit("（没有基线文件；跳过基线比较）", json_mode=json_mode)
        return fails
    base_raw = json.loads(baseline.read_text(encoding="utf-8"))
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
    return fails


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="clinic 域评测（规则臂 + LLM 臂，LLM 臂是硬闸门）")
    p.add_argument("--gold", default=str(GOLD))
    p.add_argument("--baseline", default=str(BASELINE))
    p.add_argument(
        "--arm",
        choices=("both", "rule", "llm"),
        default="both",
        help="跑哪个臂（默认 both）：rule = 离线只跑规则轨（不花钱）；llm = 只跑真正的判断者（会花钱）",
    )
    p.add_argument("--gate", action="store_true")
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    run_rule = args.arm in ("both", "rule")
    run_llm = args.arm in ("both", "llm")
    json_mode = bool(args.json)

    if args.write_baseline and not run_rule:
        # 基线的主体是规则臂指标（回归门限只看它）；只跑 LLM 臂时没有可写的东西。
        _emit(
            "参数错误：`--write-baseline` 需要规则臂（`--arm rule` 或 `both`）——基线的主体是规则臂指标",
            json_mode=json_mode,
        )
        return 1

    cases = load_clinic_gold(Path(args.gold))
    triage_cases = [c for c in cases if c.expect_department]

    rule_report: ClinicReport | None = None
    cov_num, cov_den = 0, 0
    if run_rule:
        rule_report, cov_num, cov_den = _run_rule_arm(cases, json_mode=json_mode)

    llm_report: LlmClinicReport | None = None
    if run_llm:
        llm_report = _run_llm_arm(cases)

    # ------------------------------------------------------------------ 输出
    if json_mode:
        # 顶层仍是规则臂的形状（既有契约，别动）：LLM 臂作为新键追加。
        payload: dict[str, object] = rule_report.to_json() if rule_report is not None else {}
        payload["arm"] = args.arm
        payload["llm"] = None if llm_report is None else llm_report.to_json()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if rule_report is not None:
            _print_rule_arm(rule_report, triage_cases, cov_num, cov_den)
            if args.arm == "rule":
                print(
                    "  · `--arm rule` 是离线复现历史规则轨：这一轮没有 LLM 判断者，"
                    "规则臂自己的分诊门限仍然卡。"
                )
            else:
                print(
                    "  · 规则臂的分诊数字**只报不卡**：它已经是线索提供者，"
                    "拿它卡门限等于用错尺子（判断者是下面的 LLM 臂）。"
                )
        if llm_report is not None:
            _print_llm_arm(llm_report)
        else:
            print("（`--arm rule`：本轮没有跑 LLM 臂。LLM 臂才是真正的判断者，闸门应以它为准。）")
        if rule_report is None:
            print("（`--arm llm`：本轮没有跑规则臂，基线的误报率回归比较也一并跳过。）")

    # ------------------------------------------------------------------ 基线
    if args.write_baseline:
        assert rule_report is not None  # 上面已拒绝 --arm llm
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, object] = rule_report.to_json()
        if llm_report is not None:
            # 新键（既有键的形状不动）：LLM 臂的分数只作**记录**，不当回归门限。
            metrics["llm"] = llm_report.to_json()
        payload = {
            "schema": 1,
            # 金标指纹必须一起冻结：只存指标的话，"改了尺子"与"变好了"看起来一模一样。
            "gold_sha256": clinic_gold_sha256(cases),
            "n_cases": rule_report.n_cases,
            "llm_note": LLM_BASELINE_NOTE,
            "metrics": metrics,
        }
        Path(args.baseline).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit(f"\n已写入基线：{args.baseline}", json_mode=json_mode)
        return 0

    if not args.gate:
        return 0

    # ------------------------------------------------------------------ 闸门
    rule_fails: list[str] = []
    llm_fails: list[str] = []

    if rule_report is not None:
        if args.arm == "rule":
            # 规则单臂 = 历史口径（红线 + 分诊）。没有 LLM 判断者时，
            # 规则臂自己的分诊门限是唯一可用的尺子，所以这里仍用它。
            rule_fails.extend(gate_failures(rule_report))
        else:
            # 判断权归 LLM 臂：规则臂只保证红线线索不漏 + 误报不回归，分诊只报不卡。
            rule_fails.extend(redline_gate_failures(rule_report))
        rule_fails.extend(_baseline_failures(rule_report, cases, Path(args.baseline), json_mode=json_mode))
    if llm_report is not None:
        llm_fails.extend(llm_gate_failures(llm_report))

    fails = rule_fails + llm_fails
    for f in fails:
        _emit(("  ✗ " if not f.startswith("  ") else "") + f, json_mode=json_mode)
    if any("分诊" in f and not f.startswith("  ") for f in rule_fails):
        _emit("\n" + TRIAGE_RED_NOTE, json_mode=json_mode)
    if any("分诊" in f and not f.startswith("  ") for f in llm_fails):
        _emit("\n" + LLM_TRIAGE_RED_NOTE, json_mode=json_mode)
    if fails:
        _emit("\n闸门：不通过", json_mode=json_mode)
        return 2
    _emit("\n闸门：通过", json_mode=json_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
