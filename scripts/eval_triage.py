"""分诊评测：把"症状 → 科室"的准确率测成一个**可复现的数字**。

## 为什么需要派生集（以及它为什么可以被接受）

人工金标只有 **5 条**（`data/eval/clinic_gold.jsonl` 的 `triage_department`）。
在 5 条上谈"≥80%"是数字游戏。而 `data/clinic/cards.json` 给了我们大规模测法的依据：

- 31,592 张卡，**14,440 张（45.7%）带 `departments`**；
- 卡片正文里「建议就诊科室：X、Y。」可稳定解析（前 3000 条与字段零不一致）。

于是可以派生：**用卡片自己的内容当查询、用它的 `departments` 当期望**。
这是"查询症状名应能召回自己的卡片"那类**语料自洽性**测法 —— 与原型 `rag_gold.jsonl` 同一思路。

## 四个集，口径由松到紧（**必须分开报，不许混成一个数**）

| 集 | 查询 | 性质 |
|---|---|---|
| A 规范名 | 卡片的 `key`（如 `纤毛结构缺陷`） | **上限**：规范名与索引里的键几乎逐字相同，最容易被召回 |
| **B 去答案描述** | 卡片正文**剥掉「建议就诊科室」那句** | **主口径**：症状描述里不含答案，比 A 接近真实 |
| C 人工金标 | 5 条真实问法（"我咳嗽还有点发热"） | 人手写、最接近真实，但**只有 5 条**，不足以支撑结论 |
| **C2 标准派生** | 71 条口语变体（"一直咳"、"拉黑便"） | 由 `scripts/build_triage_gold.py` 按原型 `department_scope.md` 标准派生，**未经医学复核** |

**C2 是本节新增的主力口径**：它把样本量从 5 提到 71，且查询是**口语**（这是 C 的意义所在）。
但它的答案来自标准文档的规则 + 挖掘来的口语变体表，`human_verified=false`——
**它测的是"对原型标准的符合度"，不是诊断正确性，也不是上线准确率。**
构造过程中的三道质量闸与它们各自拦下的错例，见 `build_triage_gold.py` 的模块文档。

**B 也不是真实问法的替代品**：真实患者会说"咳了三天有点烧"，而 B 用的仍是规范表述。
所以本脚本报的是**语料自洽性与可测基线**，不是"上线准确率"。任何引用这些数字的地方
都必须带上这句限定。

## 用法

    python scripts/eval_triage.py                      # A/B 各 200 条 + C，全离线免费
    python scripts/eval_triage.py --limit 50           # 少抽一点，快
    python scripts/eval_triage.py --gate 0.8           # 主口径低于 80% 即退出码 1
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

from kubedocs.clinical.triage import DEFAULT_PRECISION_K  # noqa: E402

"""`--precision-k` 的默认值**取自模块**，不在这里另写一个数字。

踩过的坑：脚本里写成 `default=10` 而模块是 `DEFAULT_PRECISION_K = 0`（不截断），
于是同一个旋钮有两个默认值——脚本一直按"截断到 10"跑，而直连探针不截断，
两边结果一个 40% 一个 80%，我为此白查了一轮"不可复现"。
**一个旋钮只能有一个默认值。**"""

CARDS = REPO / "data" / "clinic" / "cards.json"
GOLD = REPO / "data" / "eval" / "clinic_gold.jsonl"
TRIAGE_GOLD = REPO / "data" / "eval" / "triage_gold.jsonl"
"""标准派生的分诊金标（`scripts/build_triage_gold.py` 生成，`human_verified=false`）。"""
SEED = 42
_DEPT_CLAUSE = re.compile(r"建议就诊科室：[^。]+。")
"""剥掉答案用。与 `clinical.triage` 里那条同源——两处都依赖 `clinic_build_cards.py` 的格式。"""


def _load_cards() -> list[dict[str, Any]]:
    if not CARDS.is_file():
        raise SystemExit(f"卡片语料缺失：{CARDS}")
    cards: list[dict[str, Any]] = json.loads(CARDS.read_text(encoding="utf-8"))
    return [c for c in cards if c.get("departments")]


def _build_backend() -> Any:
    """clinic 检索后端（与 API 的 `_build_retriever` 同一套：优先用域包自己的）。"""
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
    return LocalIndexBackend(retriever)


def _evaluate(
    engine: Any, backend: Any, cases: list[tuple[str, list[str]]], *, top_k: int = 5
) -> dict[str, Any]:
    """跑一批 (查询, 期望科室) 用例。命中定义：期望科室出现在 **top-1** / **top-3** 里。"""
    top1 = top3 = 0
    misses: list[str] = []
    for query, expected in cases:
        hits = backend.search(query, top_k=top_k)
        ranked = engine.rank(query, hits)
        names = [d.name for d in ranked]
        if names[:1] and names[0] in expected:
            top1 += 1
        if set(names[:3]) & set(expected):
            top3 += 1
        else:
            misses.append(
                f"「{query[:40]}」 期望 {'/'.join(expected)} 实际 {'/'.join(names[:3]) or '（空）'}"
            )
    return {"n": len(cases), "top1": top1, "top3": top3, "misses": misses}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=200, help="每个派生集抽样条数")
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="**粗排**候选池大小（分诊是投票，池子大才有票可投；精排在引擎内做）",
    )
    parser.add_argument(
        "--precision-k",
        type=int,
        default=DEFAULT_PRECISION_K,
        help="**精排**之后参与投票的候选条数（粗排池大保召回、投票池小保精度）",
    )
    parser.add_argument("--gate", type=float, default=0.0, help="主口径（B）top-3 的下限，超过即退出码 1")
    parser.add_argument(
        "--department-map",
        choices=("off", "high", "medium"),
        default="off",
        help="启用语料补全映射表：high=图谱关系/别名，medium=再纳入模糊归并",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", help="逐条打印未命中")
    args = parser.parse_args(argv)

    from kubedocs.clinical.triage import (
        DEFAULT_DEPARTMENT_MAP,
        RetrievalTriage,
        load_department_map,
    )

    cards = _load_cards()
    # S311：要的是确定性伪随机（固定种子 ⇒ 同一批样本、可比的数字），与密码学无关。
    rng = random.Random(SEED)  # noqa: S311
    sample = rng.sample(cards, min(args.limit, len(cards)))
    print(f"带科室标签的卡片 {len(cards)} 张；抽样 {len(sample)} 张（种子 {SEED}）")

    backend = _build_backend()
    dep_map: dict[str, list[str]] = {}
    if args.department_map != "off":
        dep_map = load_department_map(min_confidence=args.department_map)
        if not dep_map:
            # **响亮失败**：`data/clinic/` 是 .gitignore 的，换台机器就没有这张表。
            # 若在这里静默放行，跑出来的其实是 "off" 的数字，却会被当成 "high" 记档——
            # 那正是"看起来测了、其实没测"的假证据。
            print(
                f"[x] --department-map={args.department_map} 但映射表为空"
                f"（{DEFAULT_DEPARTMENT_MAP} 不存在或该档位无条目）。\n"
                "    先跑 scripts/clinic_derive_departments.py；"
                "不想用映射表就显式传 --department-map off。"
            )
            return 2
        print(f"补全映射表：{args.department_map} 档，{len(dep_map):,} 条")
    engine = RetrievalTriage(known_departments=None, precision_k=args.precision_k, department_map=dep_map)
    """**不约束科室清单**：派生集的期望科室来自卡片数据（42 个），
    与号源系统的科室名未必一致；约束会把正确答案误判成"不在系统里"。"""

    set_a = [(str(c["key"]), list(c["departments"])) for c in sample]
    set_b = [(_DEPT_CLAUSE.sub("", str(c["text"])).strip(), list(c["departments"])) for c in sample]

    gold_cases: list[tuple[str, list[str]]] = []
    if GOLD.is_file():
        for line in GOLD.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "triage_department" and row.get("expect_department"):
                gold_cases.append((str(row["text"]), [str(row["expect_department"])]))

    # C2：按原型标准派生的口语金标（71 条）。首行是 kind=meta 头，自然被下面的过滤跳过。
    derived_cases: list[tuple[str, list[str]]] = []
    if TRIAGE_GOLD.is_file():
        for line in TRIAGE_GOLD.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "triage_department" and row.get("expect_department"):
                derived_cases.append((str(row["text"]), [str(row["expect_department"])]))
        print(f"标准派生金标 C2：{len(derived_cases)} 条（human_verified=false）")

    results: dict[str, Any] = {}
    for name, cases in (
        ("A 规范名", set_a),
        ("B 去答案描述", set_b),
        ("C 人工金标", gold_cases),
        ("C2 标准派生", derived_cases),
    ):
        if not cases:
            continue
        t0 = time.perf_counter()
        engine.map_hits = 0
        res = _evaluate(engine, backend, cases, top_k=args.top_k)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        res["map_hits"] = engine.map_hits
        results[name] = res

    print("\n" + "=" * 74)
    print("分诊评测（离线、免费、确定）")
    print("=" * 74)
    print(f"{'集':<16}{'条数':>6}{'top-1':>9}{'top-3':>9}{'top-3 率':>10}{'耗时':>9}{'映射命中':>10}")
    print("-" * 74)
    for name, res in results.items():
        rate = res["top3"] / res["n"] if res["n"] else 0.0
        hits = f"{res['map_hits']:,}" if res.get("map_hits") else "-"
        print(
            f"{name:<16}{res['n']:>6}{res['top1']:>9}{res['top3']:>9}"
            f"{rate:>9.1%}{res['seconds']:>8.0f}s{hits:>10}"
        )

    if args.verbose:
        for name, res in results.items():
            if res["misses"]:
                print(f"\n{name} 未命中（最多 20 条）：")
                for line in res["misses"][:20]:
                    print("  -", line)

    print(
        "\n口径说明：A 是**上限**（规范名与索引键几乎逐字相同）；B 是**主口径**"
        "（症状描述里已剥掉答案），但仍用规范表述；\nC 是手工写的真实问法，"
        f"只有 {len(gold_cases)} 条；C2 是按原型标准派生的口语金标，"
        f"{len(derived_cases)} 条但 **human_verified=false**（测的是对标准的符合度）。\n"
        "**这些都不是「上线准确率」。**"
    )

    if dep_map:
        print(
            f"\n补全映射表命中：**{engine.map_hits:,}** 次（映射表 {len(dep_map):,} 条）。"
            "0 命中说明映射表根本没被用上，此时的数字差异与映射表无关。"
        )

    if args.json:
        payload = {"department_map": args.department_map, "map_hits": engine.map_hits, **results}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json}")

    if args.gate > 0:
        primary = results.get("B 去答案描述")
        if primary is None:
            print("主口径缺失，无法卡闸门")
            return 1
        rate = primary["top3"] / primary["n"] if primary["n"] else 0.0
        if rate < args.gate:
            print(f"\n闸门不通过：主口径 top-3 {rate:.1%} < {args.gate:.0%}")
            return 1
        print(f"\n闸门通过：主口径 top-3 {rate:.1%} ≥ {args.gate:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
