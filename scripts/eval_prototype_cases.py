"""用原型抽取的用例测本项目的红线判定与抗噪（离线、免费、确定）。

用例在 `data/eval/prototype/`，只用两份与本项目语料无关、可直接迁移的：
`symptom_status_gold.jsonl`（35 条症状实体三维判定 + 难例标签）与
`noise_ood_gold.jsonl`（30 条非医疗寒暄 + 30 条真实主诉）。
`rag_gold.jsonl` 不用：它的 `expect_titles` 指向原型那套 OpenCMKG 派生卡片，与本项目 clinic 语料不同源。

三条指标，权重不同：

| 指标 | 含义 | 门槛 |
|---|---|---|
| **急诊假阳性** | 原型标了"否定/他人/过去/假设"，红线仍命中 | **必须为 0**：误报会让人不再相信红线提示 |
| **急诊漏报** | 原型标了本人当下的红线症状，我们没命中 | 报告，不设门槛（15 词表之外的说法本就可能漏） |
| **三维一致率** | status / experiencer / temporality 逐条比对 | 报告；低一致率要点名到标签 |

用法：
    python scripts/eval_prototype_cases.py
    python scripts/eval_prototype_cases.py --verbose   # 逐条打印不一致的用例

退出码：0 通过（假阳性 0 且 non_medical 无误触发）/ 1 未通过；前置条件：`data/eval/prototype/` 三个文件都在。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kubedocs.domains.clinic.polarity import assertion, is_reportable  # noqa: E402
from kubedocs.domains.clinic.redline import EMERGENCY_RED_LINE, RiskGuard  # noqa: E402

CASES = REPO / "data" / "eval" / "prototype"
DIMENSIONS = ("status", "experiencer", "temporality")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"用例文件缺失：{path}（见该目录 README 的出处说明）")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expected_reportable(entity: dict[str, Any]) -> bool:
    """原型标签 → "可报告"：present/possible + 本人 + 现在，与 `polarity.is_reportable` 同定义。"""
    return (
        entity.get("status") in ("present", "possible")
        and entity.get("experiencer") == "self"
        and entity.get("temporality") == "present"
    )


def eval_symptom_status(guard: RiskGuard, *, verbose: bool) -> dict[str, Any]:
    cases = _load_jsonl(CASES / "symptom_status_gold.jsonl")
    dim_ok: Counter[str] = Counter()
    dim_total: Counter[str] = Counter()
    bad_by_tag: defaultdict[str, int] = defaultdict(int)
    mismatches: list[str] = []
    false_positive: list[str] = []
    missed: list[str] = []

    for case in cases:
        text = case["text"]
        tags = case.get("tags") or []
        entities = case.get("entities") or []

        for entity in entities:
            mention = entity["mention"]
            ours = assertion(text, mention)
            for dim in DIMENSIONS:
                dim_total[dim] += 1
                if ours[dim] == entity.get(dim):
                    dim_ok[dim] += 1
            wrong = [d for d in DIMENSIONS if ours[d] != entity.get(d)]
            if wrong:
                detail = f"{case['id']} 「{text}」 {mention}: " + "; ".join(
                    f"{d} 期望 {entity.get(d)} 实际 {ours[d]}" for d in wrong
                )
                mismatches.append(detail)
                for tag in tags or ["(无标签)"]:
                    bad_by_tag[tag] += 1

        hit = guard.check(text)
        contains_red_line = any(word in text for word in EMERGENCY_RED_LINE)
        says_asserted = any(_expected_reportable(e) for e in entities)
        # "漏报"要看**红线词本身**是否可报告，而不是文本里有没有可报告的症状：
        # 「没有胸痛，就是有点胸闷」里可报告的是"胸闷"（不在 15 词表内），
        # 红线词"胸痛"恰被否认，不命中才是对的。
        red_line_reportable = any(word in text and is_reportable(text, word) for word in EMERGENCY_RED_LINE)

        # 假阳性：文本里有红线词，但原型没把任何症状标成"本人当下存在"，我们却命中了。
        if contains_red_line and not says_asserted and hit is not None:
            false_positive.append(f"{case['id']} 「{text}」 命中「{hit.word}」（tags={tags}）")
        # 漏报：红线词本身被断言为"本人当下存在"，我们却没命中。
        if red_line_reportable and hit is None:
            missed.append(f"{case['id']} 「{text}」（tags={tags}）")

    if verbose:
        print("\n逐条不一致：")
        for line in mismatches:
            print("  -", line)
        for line in false_positive:
            print("  ! 假阳性", line)
        for line in missed:
            print("  ! 漏报  ", line)

    return {
        "n_cases": len(cases),
        "n_entities": sum(len(c.get("entities") or []) for c in cases),
        "dim_ok": dict(dim_ok),
        "dim_total": dict(dim_total),
        "bad_by_tag": dict(bad_by_tag),
        "false_positive": false_positive,
        "missed": missed,
    }


def eval_ood(guard: RiskGuard, *, verbose: bool) -> dict[str, Any]:
    """`noise_ood_gold.jsonl` 是区分集：`non_medical`（30 条寒暄）不得触发红线（硬门槛），
    `medical`（30 条真实主诉）触发红线不是错；其中含红线词却没命中的记为潜在漏报并列出，
    暴露的是 15 词红线表的召回缺口。补词表要有依据，所以不设门槛、只如实报告。
    """
    cases = _load_jsonl(CASES / "noise_ood_gold.jsonl")
    false_trigger: list[str] = []
    medical_hit = 0
    medical_missed: list[str] = []
    medical_nohit: list[str] = []

    for case in cases:
        text = case["text"]
        label = case.get("label")
        hit = guard.check(text)
        if label == "non_medical":
            if hit is not None:
                false_trigger.append(f"「{text}」 命中「{hit.word}」")
            continue
        if hit is not None:
            medical_hit += 1
        else:
            medical_nohit.append(f"「{text}」")
            if any(word in text for word in EMERGENCY_RED_LINE):
                medical_missed.append(f"「{text}」（含红线词但未命中）")

    if verbose and medical_missed:
        print("\nmedical 组的潜在漏报（含红线词却没命中）：")
        for line in medical_missed:
            print("  -", line)
    if verbose and medical_nohit:
        # 这一份必须人工看：没有红线词不等于不紧急（"胸口疼放射到左肩"是典型心梗表现，
        # 字面却不在 15 词表里）。列出来是供人判断，不是自动判定。
        print("\nmedical 组未命中（需人工复核是否属于漏召回）：")
        for line in medical_nohit:
            print("  -", line)

    return {
        "n_cases": len(cases),
        "false_trigger": false_trigger,
        "medical_hit": medical_hit,
        "medical_missed": medical_missed,
        "medical_nohit": medical_nohit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", action="store_true", help="逐条打印不一致与误触发")
    args = parser.parse_args(argv)

    guard = RiskGuard()
    stress = eval_symptom_status(guard, verbose=args.verbose)
    ood = eval_ood(guard, verbose=args.verbose)
    departments = json.loads((CASES / "departments.json").read_text(encoding="utf-8"))

    print("=" * 72)
    print("原型用例回归（离线、免费、确定）")
    print("=" * 72)

    print(f"\n【1】症状三维判定：{stress['n_cases']} 条用例 / {stress['n_entities']} 个实体")
    print(f"{'维度':<16}{'一致':>6}{'总数':>6}{'一致率':>10}")
    print("-" * 40)
    for dim in DIMENSIONS:
        ok, total = stress["dim_ok"].get(dim, 0), stress["dim_total"].get(dim, 0)
        rate = ok / total if total else 0.0
        print(f"{dim:<16}{ok:>6}{total:>6}{rate:>9.1%}")

    if stress["bad_by_tag"]:
        print("\n  不一致集中在这些标签上（标签=原型的难例分类）：")
        for tag, count in sorted(stress["bad_by_tag"].items(), key=lambda kv: -kv[1]):
            print(f"    {tag:<28} {count} 次")
    else:
        print("\n  没有任何维度不一致。")

    print("\n【2】急诊红线（安全底线）")
    fp = len(stress["false_positive"])
    print(f"{'假阳性（必须 0）':<24}{fp:>4}")
    print(f"{'漏报（报告，不设门槛）':<24}{len(stress['missed']):>4}")
    for line in stress["false_positive"]:
        print("    !", line)

    print(f"\n【3】医疗/非医疗区分集：{ood['n_cases']} 条（30 non_medical + 30 medical）")
    print(f"{'non_medical 误触发（必须 0）':<24}{len(ood['false_trigger']):>4}")
    print(f"{'medical 命中红线（正常）':<24}{ood['medical_hit']:>4}")
    print(f"{'medical 含红线词但未命中':<24}{len(ood['medical_missed']):>4}")
    for line in ood["false_trigger"]:
        print("    ! 误触发", line)
    for line in ood["medical_missed"]:
        print("    - 潜在漏报", line)

    print(f"\n【4】科室词表：{len(departments)} 个科室名（供分诊用，含真实脏数据）")

    ok = fp == 0 and not ood["false_trigger"]
    print("\n" + ("通过：安全底线守住（假阳性 0、non_medical 0 误触发）" if ok else "不通过：见上面的 ! 行"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
