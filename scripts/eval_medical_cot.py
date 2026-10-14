"""用他人开源的中文医疗 CoT 数据集，给「可能病症 / 检查建议 / 用药建议」建**第一份**评测。

## 数据与用途边界（先读这段）

参考数据在 `E:\\Users\\86189\\Agent_In_Action-main` 里（**没有复制进本仓库**）：

- `medical_o1_sft_Chinese_alpaca_cot.jsonl`，20,171 条 `{prompt, response}`，
  `response` 含 `<think>…</think>` 链式推理；
- 来源 **Apache-2.0** 的 `FreedomIntelligence/medical-o1-reasoning-SFT`（+ HuatuoGPT-o1）；
- 上游 README 明写"**仅供科研与模型训练，不得用于诊断/治疗**"。

**我们不把它当语料**（不 ingest），只用它当**评测题**：抽样 N 条，跑我们的产品链路，
看输出与"参考答"的术语是否重合。这与本仓库对 clinic 卡片语料的处置一致（仅内部测试）。

## 指标是什么、**不是什么**（这条比数字重要）

**是**：

| 指标 | 含义 |
|---|---|
| `有依据作答` | 检索到了知识依据 ⇒ 契约允许给检查/用药建议 |
| **`无依据拒答`** | 没检索到依据 ⇒ **按契约不给**检查/用药。**这个比例是最有价值的发现**：它衡量我们的 clinic 语料能不能覆盖这类问题 |
| `术语重合率` | 我们的输出里是否出现了参考答里的中文长词（≥3 字，逐字匹配） |
| `最长重合词` | 重合词的长度，越长越可能是真正的临床术语 |

**不是**：医学正确性。参考答是**考试标准答案（单一）**，而我们是"多轮问诊 + 证据约束 +
不给剂量"的产品；两者口径不同。**任何"我们答对了 X%"的说法都必须带上这句限定。**

## 用法

    python scripts/eval_medical_cot.py --dry-run          # 只加载并抽样，不调模型（免费）
    python scripts/eval_medical_cot.py --limit 20         # 真跑 20 条（会花钱）
    python scripts/eval_medical_cot.py --limit 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SOURCE = Path(
    r"E:\Users\86189\Agent_In_Action-main\05-agent-model-finetuning\easy-dataset"
    r"\dataset\medical_o1_sft_Chinese_alpaca_cot\medical_o1_sft_Chinese_alpaca_cot.jsonl"
)
"""评测题来源。**不在仓库里**：一是 62 MB 的体量，二是它只作评测用、不入语料。"""

SEED = 42
"""抽样种子。固定种子 = 同样的样本、可比的数字；换样本必须显式改这个数并说明。"""

MIN_TERM_LEN = 3
"""判定"术语重合"的最短中文词长。取 3 是有意的：2 字词（"患者""治疗"）太普遍，
命中不代表任何东西，会把指标变成噪声。"""

_CJK_RUN = re.compile(rf"[\u4e00-\u9fa5]{{{MIN_TERM_LEN},}}")


def load_cases(limit: int) -> list[dict[str, str]]:
    """按固定种子抽样。`response` 只取 `</think>` **之后**的最终答案部分。"""
    if not SOURCE.is_file():
        raise SystemExit(
            f"评测题源文件不存在：{SOURCE}\n"
            "它是参考项目附带的数据集；本仓库不复制它。见本文件顶部的来源说明。"
        )
    rows: list[dict[str, str]] = []
    with SOURCE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt") or "").strip()
            response = str(item.get("response") or "")
            answer = response.split("</think>")[-1].strip() if "</think>" in response else response
            if prompt and answer:
                rows.append({"prompt": prompt, "answer": answer})
    if not rows:
        raise SystemExit(f"没能从 {SOURCE} 解析出任何用例")
    # S311：这里**要的就是确定性伪随机**——固定种子保证"同一批样本、可比的数字"，
    # 与密码学用途无关；换成真随机会让两次评测不可比。
    return random.Random(SEED).sample(rows, min(limit, len(rows)))  # noqa: S311


def answer_terms(answer: str) -> list[str]:
    """参考答里的中文长词（≥3 字）。逐字匹配，不做同义词归一——**这是粗口径**。"""
    return sorted(set(_CJK_RUN.findall(answer)), key=len, reverse=True)


def _output_text(assessment: Any) -> str:
    """把一次主判的产物拼成一段可比对的文本。"""
    parts = [assessment.patient_summary]
    parts += [d.condition for d in assessment.differentials]
    parts += [e.name for e in assessment.exams]
    parts += [m.name for m in assessment.medications]
    parts += [d.name for d in assessment.departments]
    return " ".join(parts)


def _build_backend() -> Any:
    """clinic 域的检索后端（与 API 里 `_build_retriever` 同一套：优先用域包自己的）。"""
    from kubedocs.cli import apply_domain_defaults, open_store
    from kubedocs.clinical.rag import LocalIndexBackend
    from kubedocs.config import Settings
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry
    from kubedocs.embed import build_embedder
    from kubedocs.retrieve.hybrid import HybridRetriever

    load_all()
    settings = apply_domain_defaults(Settings(domain="clinic"))
    store = open_store(settings)
    embedder = build_embedder(settings)
    try:
        retriever = DomainRegistry.get(settings.domain).retriever(store, embedder, settings)
    except KeyError:
        retriever = HybridRetriever(store, embedder, settings)
    return LocalIndexBackend(retriever), settings


def _build_reasoner(settings: Any) -> Any:
    """**走同一本日预算账**的 reasoner：与 CLI `ask` / `/ask` / `/chat` 共用 `llm_spend.json`。"""
    from kubedocs.clinical.reasoning import BudgetedClient, ClinicalReasoner, LiteLLMAdapter
    from kubedocs.generate import DailyBudget, LiteLLMClient

    budget = DailyBudget(
        settings.llm_spend_path,
        settings.llm_daily_budget_usd,
        unpriced_usd=settings.llm_unpriced_call_usd,
    )
    client = BudgetedClient(LiteLLMClient(settings), budget)
    return ClinicalReasoner(LiteLLMAdapter(client)), budget


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=20, help="抽样条数（每条一次模型调用，会花钱）")
    parser.add_argument("--dry-run", action="store_true", help="只抽样与建后端，不调模型（免费）")
    parser.add_argument("--json", type=Path, default=None, help="把结果落盘")
    args = parser.parse_args(argv)

    cases = load_cases(args.limit)
    print(f"抽样 {len(cases)} 条（种子 {SEED}）；示例：{cases[0]['prompt'][:60]}…")

    # **先把 .env 导进 os.environ**：`has_api_key()` 与 litellm 读的都是 os.environ，
    # 而 pydantic-settings 只把 .env 填进 Settings 对象。漏了这一步，整轮评测会
    # **全部走降级路径**，花费 0、指标还照样打印出来——实测踩过，20/20 全降级。
    from kubedocs.config import load_env_file

    load_env_file()

    backend, settings = _build_backend()
    print(f"clinic 检索后端就绪（域 {settings.domain}，模型 {settings.embedding_model}）")

    if args.dry_run:
        print("\n--dry-run：不调用模型。上面已证明数据可加载、后端可构造。")
        return 0

    from kubedocs.generate import has_api_key

    if not has_api_key(settings.llm_model):
        # 快速失败：没有密钥时"跑完 20 条"得到的是**降级产物的分数**，
        # 而那与模型能力无关。宁可拒绝运行，也不要产出一个会被误读的数字。
        raise SystemExit(
            f"未配置 {settings.llm_model} 所需密钥 ⇒ 本次评测会全部走降级路径，数字没有意义。\n"
            "请在 .env 里填好密钥（见 .env.example），或先用 --dry-run 验证数据与后端。"
        )

    from kubedocs.agent.tools import ToolRegistry  # noqa: F401  仅确认导入链正常
    from kubedocs.clinical import build_evidence
    from kubedocs.clinical.contracts import validate_against
    from kubedocs.clinical.intake import IntakeState, RuleSlotExtractor
    from kubedocs.domains.clinic.redline import RiskGuard

    reasoner, budget = _build_reasoner(settings)
    guard = RiskGuard()
    extractor = RuleSlotExtractor()
    spent_before = budget.spent_today()

    stats: dict[str, Any] = {
        "n": len(cases),
        "asked_followup": 0,
        "refused_no_evidence": 0,
        "answered_with_evidence": 0,
        "term_hit": 0,
        "longest_terms": [],
        "degraded": 0,
        "cases": [],
    }

    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        state = IntakeState()
        state.merge(case["prompt"], extractor)
        evidence = build_evidence(case["prompt"][:300], backend=backend, top_k=8, guard=guard)
        result = reasoner.reason(case["prompt"], evidence, state)
        assessment = result.assessment
        elapsed = time.perf_counter() - t0
        if assessment is None:  # pragma: no cover - reasoning 层保证降级产物
            continue

        has_evidence = bool(evidence.items)
        problems = validate_against(assessment, evidence.ids())
        terms = answer_terms(case["answer"])
        produced = _output_text(assessment)
        matched = [t for t in terms if t in produced]
        hit = bool(matched)
        longest = max((len(t) for t in matched), default=0)

        if has_evidence:
            stats["answered_with_evidence"] += 1
        else:
            stats["refused_no_evidence"] += 1
        if hit:
            stats["term_hit"] += 1
            stats["longest_terms"].append(longest)
        if result.degraded:
            stats["degraded"] += 1

        stats["cases"].append(
            {
                "prompt": case["prompt"][:120],
                "evidence": len(evidence.items),
                "urgency": assessment.urgency,
                "exams": [e.name for e in assessment.exams],
                "medications": [m.name for m in assessment.medications],
                "differentials": [d.condition for d in assessment.differentials],
                "longest_match": longest,
                "problems": problems,
                "**degraded**": result.degraded,
                "raw_chars": len(result.raw),
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"  [{i}/{len(cases)}] 依据 {len(evidence.items):>2} 词 "
            f"检查 {len(assessment.exams)} 用药 {len(assessment.medications)} "
            f"最长重合 {longest} | {elapsed:.0f}s"
        )

    spent = budget.spent_today() - spent_before
    answered = max(1, stats["answered_with_evidence"])

    if stats["degraded"] == stats["n"]:
        # 全部降级 ⇒ 分数来自降级产物，与模型能力无关。**不打印指标**，
        # 否则它会被当成"模型表现"记进文档（实测踩过：20/20 全降级仍打印了表格）。
        print(f"\n全部 {stats['n']} 条都走了降级路径 ⇒ 指标无效，不予报告。")
        print("最常见原因：密钥没进 os.environ（漏调 load_env_file）或密钥无效。")
        return 1

    print("\n" + "=" * 72)
    print(f"中文医疗 CoT 评测（抽样 {stats['n']} 条，种子 {SEED}）")
    print("=" * 72)
    print(f"{'有依据作答':<22}{stats['answered_with_evidence']:>6}")
    print(f"{'无依据（按契约不给检查/用药）':<22}{stats['refused_no_evidence']:>6}")
    print(f"{'术语重合（≥1 个 ≥3 字词）':<22}{stats['term_hit']:>6}  / {stats['n']}")
    if stats["longest_terms"]:
        terms = stats["longest_terms"]
        print(f"{'最长重合词长度 中位/最大':<22}{sorted(terms)[len(terms) // 2]:>6} / {max(terms)}")
    print(f"{'降级次数':<22}{stats['degraded']:>6}")
    if stats["degraded"]:
        # **降级原因必须分类打印**：只报"35% 降级"是不可行动的数字，
        # 而原因分类直接指向该修什么（截断 / 解析 / 网络 / 契约）。
        reasons: dict[str, int] = {}
        for c in stats["cases"]:
            key = str(c.get("**degraded**") or "")
            if key:
                reasons[key[:60]] = reasons.get(key[:60], 0) + 1
        print("\n  降级原因分布（按前 60 字归类）：")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>3} × {reason}")
    print(f"{'本次花费（美元）':<22}{spent:>10.6f}")
    print(f"{'当日累计（美元）':<22}{budget.spent_today():>10.6f} / 上限 {budget.limit}")
    print("\n注意：**术语重合 ≠ 医学正确性**。参考答是考试标准答案（单一），我们是")
    print(f"多轮问诊 + 证据约束 + 不给剂量的产品；口径不同。分母是 {answered} 条有依据的用例。")

    if args.json:
        args.json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
