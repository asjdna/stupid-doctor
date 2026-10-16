"""clinic 域评测：**两个臂分开报**——规则臂（线索提供者）+ LLM 臂（真正的判断者）。

用法：
    python scripts/eval_clinic.py --arm both --gate --json   # 默认：两臂都跑，闸门只卡安全不变量
    python scripts/eval_clinic.py --arm rule --gate          # 离线只跑规则臂（不花钱）
    python scripts/eval_clinic.py --arm llm                  # 只跑 LLM 臂（会调模型、会花钱）
    python scripts/eval_clinic.py --write-baseline           # 重建冻结基线（含 LLM 臂数值）

## 指标是记录，闸门只卡确定性安全不变量

危险信号召回、误报率、分诊命中率都是**要看的数**：它们来自自建的小样本用例集，
能说明趋势与回归，不能承担"这条回答安全"的责任。把 100% 召回写成门限，只会得到两种
结果——为了让闸门变绿而放宽尺子，或者一个永远红着、没人看的闸门。

所以 `--gate` 现在只失败于两件事（见 `evaluation/clinic.py` 的 `governance_failures`
与 `evaluation/clinic_llm.py` 的 `llm_governance_failures`）：

1. **这次运行不能作为证据**：一条用例都没跑、或有题目没拿到模型判断；
2. **触发了确定性安全不变量**：无依据给医学内容、编造挂号动作、判了急症却不给急救动作
   —— 这三条由代码强制执行，触发就是真问题，与样本量无关。

规则臂与冻结基线不可比（标注集指纹变了）时**只写档案注记，不判失败**：跨标注集的分数
本来就不该拿来比，注明即可。

真正确保安全的是三层防护（语义理解 / 主动问诊 / 确定性安全门，见 `clinical/safety.py`）
与治理审计，不是这里的分数。

## 为什么是两个臂

架构已把全部判断交给大模型：危险信号关键词 + 三维极性、症状词表、症状→科室票数都
**降级为证据提供者**（见 `clinical/reference.py` 的分工表）。所以只测规则轨没有意义——
那测的是一个**已经不再做决定**的组件。规则臂的价值在于它是最便宜、最稳定的
**线索质量读数**（危险信号词表漏一条仍然是回归）；LLM 臂才是产品的判断者。

前置条件：`--gold` 默认 `data/eval/clinic_gold.jsonl`；规则臂的分诊依赖 clinic 受限卡片
语料，缺语料时记为「未评测」。
输出：两个臂各自的危险信号召回率/误报率、分诊 top-3 命中率与错误明细；LLM 臂另有
失败调用数 / 调用次数 / 花费 / 安全不变量触发记录 / 置信度校准报告。
退出码：0 通过 / 1 参数错误 / 2 闸门不通过 / 3 前置条件不满足。
`--json` 只改格式（旁白改道 stderr），**不**绕过闸门，退出码仍由 `--gate` 决定。
`--archive` 把这一轮的完整报告写成治理档案（默认写到 `data/eval/clinic_archive/`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    governance_failures,
    load_clinic_gold,
    reporting_targets,
)
from kubedocs.evaluation.clinic_llm import (  # noqa: E402
    LLM_FALSE_POSITIVE_LIMIT,
    LLM_TRIAGE_MIN_ACCURACY,
    LlmClinicReport,
    evaluate_llm_arm,
    llm_governance_failures,
)

GOLD = REPO_ROOT / "data" / "eval" / "clinic_gold.jsonl"
BASELINE = REPO_ROOT / "data" / "eval" / "clinic_baseline.json"
ARCHIVE_DIR = REPO_ROOT / "data" / "eval" / "clinic_archive"

TRIAGE_RED_NOTE = """分诊为什么是 0 分：冻结基线里记着的就是 0。
  · data/eval/clinic_baseline.json 的 triage.accuracy = 0.0（hit 0 / total 5、evaluated=true）；
  · 根因是受限卡片语料与症状词典两套词汇不重合（证据键覆盖率 1/6）；
  · 这是**记录**不是闸门：分数低不代表系统不安全（安全由三层防护保证），
    但它确实是"分诊这条线索目前不可用"的读数，别把它当成可以忽略的数字。
  · 要让它变好只有一条路：真的把分诊做对（补口语->体征映射或换语料）。"""

LLM_TRIAGE_RED_NOTE = """LLM 臂分诊是模型的结论（`assessment.departments` 前三名）。
  · 它是**记录**不是门限：模型不是确定性的，拿固定分数卡退出码只会把正常波动当成回归；
  · 要让它变好只有一条路：让模型真的答对科室（改提示词/给更好的候选线索/补语料）。"""

LLM_BASELINE_NOTE = """LLM 臂的分数**每次重跑都会有波动**（模型不是确定性的）：把它写进基线是为了
"记录当时是什么样子"，**不是**当回归门限用。基线只有规则臂的线索质量读数
（危险信号词表漏报、误报率与基线比较）用于回归判断——那部分是确定性的，值得比。"""


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
    """规则臂：危险信号词表 + 挂号意图 + 分诊。**口径与之前逐字一致**。

    它现在是线索提供者的质量读数，不再是产品结论的来源；但口径不变，历史基线才比得下去。
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
        f"用例总数    : {report.n_cases}（危险信号） + {report.booking_total}（挂号意图）"
        f" + {len(triage_cases)}（分诊）"
    )
    recall = report.redline_recall
    print(
        f"危险信号召回: "
        f"{'未评测（用例集里没有 expect_red_line）' if recall is None else f'{recall:.2%}'}"
        f"  (TP={report.redline_tp} FN={report.redline_fn})"
        f"   ← 喂给模型的线索，漏一条就是线索质量回归"
    )
    print(
        f"危险信号误报: {report.redline_false_positive_rate:.2%}"
        f"  (FP={report.redline_fp} TN={report.redline_tn})"
        f"   ← 关键词表判据（记录；判断由 LLM 臂给出）"
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
        print("  · **未评测不等于通过**：这次一条用例都没跑成，召回/误报/分诊三个数一个都没测到。")
        return
    recall = report.redline_recall
    print(
        f"  危险信号召回: "
        f"{'未评测（用例集里没有 expect_red_line）' if recall is None else f'{recall:.2%}'}"
        f"  (TP={report.redline_tp} FN={report.redline_fn})   ← 参考目标 100%（记录，不判通过/不通过）"
    )
    print(
        f"  危险信号误报: {report.redline_false_positive_rate:.2%}"
        f"  (FP={report.redline_fp} TN={report.redline_tn})"
        f"   ← 参考目标 ≤{LLM_FALSE_POSITIVE_LIMIT:.0%}；判据是「判成急症或还有未排除的危险信号」"
    )
    acc = report.triage_accuracy
    print(
        f"  分诊科室    : "
        f"{'未评测（没有 expect_department 用例）' if acc is None else f'{acc:.2%} top3 命中'}"
        f"  (hit {report.triage_hit}/{report.triage_total})   ← 参考目标 {LLM_TRIAGE_MIN_ACCURACY:.0%}"
    )
    print(f"  失败调用    : {report.n_failed_calls}   ← 没有得到模型判断的题（未评测不等于通过）")
    print(f"  调用次数    : {report.llm_calls}")
    print(f"  花费（美元）: {report.cost_usd:.6f}")
    print(
        "  · 上面的分数是**记录**：参考目标 "
        f"召回 100% / 误报 ≤{LLM_FALSE_POSITIVE_LIMIT:.0%} / 分诊 ≥{LLM_TRIAGE_MIN_ACCURACY:.0%}，"
        "但**不判通过/不通过**；闸门只看「有没有拿到判断」与「有没有触发确定性安全不变量」。"
    )
    print(
        f"  安全不变量  : "
        f"{'未触发（0 条）' if not report.invariant_violations else f'触发 {len(report.invariant_violations)} 条'}"
    )
    cal = report.calibration()
    if cal.n:
        print(f"  置信度校准  : n={cal.n} Brier={cal.brier:.4f}（只报告，不上门限）；{cal.note}")
        for b in cal.bins:
            if b.n:
                print(
                    f"    {b.lo:.1f}-{b.hi:.1f}: n={b.n} 平均置信 {b.mean_confidence:.2f}"
                    f" 实际判对 {b.accuracy:.2%}"
                )
    else:
        print(f"  置信度校准  : {cal.note}")
    for label, items in (
        ("漏报", report.redline_missed),
        ("误报", report.redline_false_alarms),
        ("分诊未命中", report.triage_missed),
        ("未判断", report.failed_calls),
        ("安全不变量", report.invariant_violations),
    ):
        for item in list(items)[:10]:
            print(f"  ✗ {label}: {item}")


def _baseline_notes(
    report: ClinicReport, cases: list[ClinicCase], baseline: Path, *, json_mode: bool
) -> list[str]:
    """规则臂与冻结基线的比较 → **档案注记**（不改退出码）。

    只有规则臂进这里：它是确定性的，与基线比较才有意义（LLM 臂每次重跑都会波动）。
    注记的用途是让档案可读——"这次的误报率比基线高了多少""尺子有没有换过"。
    它**不再判失败**：指标是记录，闸门只卡确定性安全不变量。
    """
    notes: list[str] = []
    if not baseline.exists():
        _emit("（没有基线文件；跳过基线比较）", json_mode=json_mode)
        return notes
    base_raw = json.loads(baseline.read_text(encoding="utf-8"))
    base = base_raw["metrics"]
    current_sha = clinic_gold_sha256(cases)
    if "gold_sha256" not in base_raw:
        notes.append("基线没有 gold_sha256（旧格式）：无法判断这把尺子是否被改过 ⇒ 本次不与历史分数比较")
    elif base_raw["gold_sha256"] != current_sha:
        notes.append(
            f"标注集已变更（gold_sha256 {base_raw['gold_sha256']} -> {current_sha}）："
            "跨标注集的分数不可比，本次只作独立记录"
        )
    else:
        base_fp = float(base["redline"]["false_positive_rate"])
        delta = report.redline_false_positive_rate - base_fp
        notes.append(f"相对冻结基线的误报率变化 {delta:+.2%}（基线 {base_fp:.2%}）")
    return notes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="clinic 域评测（规则臂 + LLM 臂；闸门只卡确定性安全不变量）")
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
    p.add_argument(
        "--archive",
        default="",
        help=f"把这一轮的完整报告写成治理档案（默认不写；建议 {ARCHIVE_DIR}）",
    )
    args = p.parse_args(argv)

    run_rule = args.arm in ("both", "rule")
    run_llm = args.arm in ("both", "llm")
    json_mode = bool(args.json)

    if args.write_baseline and not run_rule:
        # 基线的主体是规则臂指标（确定性的那部分）；只跑 LLM 臂时没有可写的东西。
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

    baseline_notes = (
        _baseline_notes(rule_report, cases, Path(args.baseline), json_mode=json_mode)
        if rule_report is not None
        else []
    )

    # ------------------------------------------------------------------ 输出
    payload: dict[str, object] = rule_report.to_json() if rule_report is not None else {}
    payload["arm"] = args.arm
    payload["targets"] = reporting_targets()
    payload["llm"] = None if llm_report is None else llm_report.to_json()
    payload["baseline_notes"] = baseline_notes
    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if rule_report is not None:
            _print_rule_arm(rule_report, triage_cases, cov_num, cov_den)
            print(
                "  · 规则臂的分数是**线索质量读数**（它已经是线索提供者，不再是判断者）："
                "只记不卡，判断者是下面的 LLM 臂。"
            )
        if llm_report is not None:
            _print_llm_arm(llm_report)
        else:
            print("（`--arm rule`：本轮没有跑 LLM 臂，所以没有「这次判断得怎么样」的读数。）")
        for note in baseline_notes:
            print(f"  · 基线：{note}")

    # ------------------------------------------------------------------ 档案
    if args.archive:
        # 治理档案：**评测结果的留痕**。指标不上门限，但"某天跑出来是什么样"必须可追，
        # 否则趋势与异常都只能靠人记。文件名带 UTC 时间戳，永不覆盖。
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(args.archive)
        if out.suffix != ".json":
            out = out / f"clinic_eval_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        archive = {
            "ts": stamp,
            "gold_sha256": clinic_gold_sha256(cases),
            "gold_path": str(args.gold),
            "arm": args.arm,
            "report": payload,
        }
        out.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit(f"\n已写入治理档案：{out}", json_mode=json_mode)

    # ------------------------------------------------------------------ 基线
    if args.write_baseline:
        assert rule_report is not None  # 上面已拒绝 --arm llm
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, object] = rule_report.to_json()
        if llm_report is not None:
            # 新键（既有键的形状不动）：LLM 臂的分数只作**记录**。
            metrics["llm"] = llm_report.to_json()
        payload_base = {
            "schema": 1,
            # 金标指纹一起冻结：只存指标的话，"改了尺子"与"变好了"看起来一模一样。
            "gold_sha256": clinic_gold_sha256(cases),
            "n_cases": rule_report.n_cases,
            "llm_note": LLM_BASELINE_NOTE,
            "metrics": metrics,
        }
        Path(args.baseline).write_text(
            json.dumps(payload_base, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _emit(f"\n已写入基线：{args.baseline}", json_mode=json_mode)
        return 0

    if not args.gate:
        return 0

    # ------------------------------------------------------------------ 闸门
    # **只有两类事会让它失败**：这次运行不能作为证据（用例空集 / 有题没拿到判断）、
    # 以及确定性安全不变量被触发。指标一个字都不参与（见模块文档）。
    rule_fails: list[str] = []
    llm_fails: list[str] = []
    if rule_report is not None:
        rule_fails.extend(governance_failures(rule_report, label="规则臂"))
    if llm_report is not None:
        llm_fails.extend(llm_governance_failures(llm_report))
    elif run_llm:
        # 要求跑 LLM 臂却拿不到报告：装配阶段就失败了，这属于"没测"。
        llm_fails.append("LLM 臂没有产生报告（未评测）——没测不等于通过")

    fails = rule_fails + llm_fails
    for f in fails:
        _emit("  ✗ " + f, json_mode=json_mode)
    if any("分诊" in f for f in rule_fails):
        _emit("\n" + TRIAGE_RED_NOTE, json_mode=json_mode)
    if any("分诊" in f for f in llm_fails):
        _emit("\n" + LLM_TRIAGE_RED_NOTE, json_mode=json_mode)
    if fails:
        _emit("\n闸门：不通过", json_mode=json_mode)
        return 2
    _emit("\n闸门：通过（只代表没有触发确定性安全不变量；指标请按记录自己判断）", json_mode=json_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
