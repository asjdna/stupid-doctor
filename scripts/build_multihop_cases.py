"""多跳用例工厂：从语料**机械生成**候选用例 + 第一轮确定性筛选。

## 为什么机械生成，而不是让 LLM 写题

手写或用 LLM 生成题目有两个问题：

1. **量上不去**：要 50 条有区分度的用例，得先有几百条候选，手写不现实；
2. **偏**：我知道自己想验证什么（多跳有用），写题时会不自觉地把子目标写得
   更"不相交"——那是把结论写进题里。

机械生成没有这个问题：锚点来自语料的 IDF 统计，文档对来自词面重叠度，
模板是固定的。**题目长什么样不由我决定，由语料决定。**

代价是题目读起来会有点生硬（"What does X cover regarding A, and what does Y cover
regarding B?"）。但对"在小检索预算下，一次检索能不能装下两个不相交子目标"
这个问题，**生硬反而是优点**：它去掉了一切口径与措辞的干扰。

## 两轮筛选里的第一轮（本脚本负责）

- **可检索性**：每跳的锚点必须能从该跳自己的查询里检索到（`top_k` 个候选内）。
  检索不到说明语料里没有，那会把"语料没有"伪装成"系统没答对"。
- **文档不相交**：两跳的 top-1 文档必须**不同**。这是让测试台具备区分能力的
  关键——若两跳证据在同一篇，一次检索必然同时拿到，对照没有信息量。

第二轮（单轮必须漏 ≥1 跳要点）要花钱，交给 `ab_multihop.py --select`。

用法：
    python scripts/build_multihop_cases.py --pairs 400 --out data/eval/multihop_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.embed import HashingEmbedder  # noqa: E402
from kubedocs.retrieve.hybrid import HybridRetriever  # noqa: E402
from kubedocs.store.sqlite_store import from_settings  # noqa: E402

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{3,}")
# 文档里到处都是的词没有区分度，锚点必须是"这一篇特有的"
_GENERIC = frozenset(["kubernetes", "cluster", "container", "containers", "pod", "pods", "node", "nodes", "object", "objects", "field", "fields", "value", "values", "default", "example", "examples", "note", "notes", "must", "should", "which", "there", "their", "this", "that", "when", "where", "with", "been", "have", "from", "will", "your", "more", "than", "also", "only", "into", "some", "such", "other", "others", "spec", "metadata", "name", "names", "type", "types", "true", "false", "null", "list", "item", "items", "path", "paths"])


def doc_anchors(store: Any, *, per_doc: int = 6, max_docs: int = 1200,
                ) -> dict[str, list[str]]:
    """给每篇文档挑几个**高 IDF 锚点词**（该文档特有、且在全语料里罕见）。

    纯统计，不看内容语义——这正是"机械生成"的意思。
    """
    with store._lock:  # noqa: SLF001
        rows = store._con.execute(  # noqa: SLF001
            "SELECT doc_id, body FROM chunks ORDER BY doc_id").fetchall()
    df: Counter[str] = Counter()
    per_doc_tokens: dict[str, Counter[str]] = {}
    for r in rows:
        toks = {t.lower() for t in _WORD.findall(str(r["body"]))}
        toks -= _GENERIC
        if not toks:
            continue
        per_doc_tokens.setdefault(str(r["doc_id"]), Counter()).update(toks)
        df.update(toks)
    n_docs = max(1, len(per_doc_tokens))

    out: dict[str, list[str]] = {}
    for doc_id, counts in list(per_doc_tokens.items())[:max_docs]:
        scored = []
        for tok, tf in counts.items():
            idf = math.log((n_docs + 1) / (df[tok] + 1))
            # 锚点要"罕见"且"在这篇里出现多次"——后者说明它确实是这篇的主题词
            scored.append((idf * math.log(1 + tf), tok))
        scored.sort(reverse=True)
        picks = [t for _, t in scored[:per_doc]]
        if len(picks) >= 2:
            out[doc_id] = picks
    return out


def disjoint_pairs(anchors: dict[str, list[str]], *, limit: int,
                   max_overlap: float = 0.1) -> list[tuple[str, str]]:
    """挑词面重叠度低的文档对——重叠低才可能"两跳证据不在同一篇"。"""
    docs = sorted(anchors)
    sets = {d: set(anchors[d]) for d in docs}
    pairs: list[tuple[str, str]] = []
    step = max(1, len(docs) // max(1, int(math.sqrt(limit)) or 1))
    for i in range(0, len(docs), step):
        for j in range(i + step, len(docs), step):
            a, b = docs[i], docs[j]
            inter = len(sets[a] & sets[b])
            if inter / max(1, min(len(sets[a]), len(sets[b]))) <= max_overlap:
                pairs.append((a, b))
            if len(pairs) >= limit:
                return pairs
    return pairs


def build_cases(pairs: list[tuple[str, str]], anchors: dict[str, list[str]],
                ) -> list[dict[str, Any]]:
    """套固定模板出题。每跳的期望要点 = 该文档的锚点（取其前 2 个）。"""
    cases: list[dict[str, Any]] = []
    for n, (d1, d2) in enumerate(pairs, start=1):
        a1 = anchors[d1][:2]
        a2 = anchors[d2][:2]
        t1 = d1.rsplit("/", 1)[-1].replace("-", " ")
        t2 = d2.rsplit("/", 1)[-1].replace("-", " ")
        cases.append({
            "id": f"g{n:04d}",
            "q": (f"What does the Kubernetes documentation on \"{t1}\" cover "
                  f"regarding {a1[0]}, and what does the page on \"{t2}\" cover "
                  f"regarding {a2[0]}?"),
            "hops": [
                {"q": f"{d1} {a1[0]}", "expect_points": a1, "doc": d1},
                {"q": f"{d2} {a2[0]}", "expect_points": a2, "doc": d2},
            ],
            "why": "机械生成：两个高 IDF 锚点分属词面不相交的两篇文档",
        })
    return cases


def screen(cases: list[dict[str, Any]], retriever: Any, *, top_k: int = 5,
           ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """第一轮确定性筛选：锚点可检索 + 两跳 top-1 文档不同。"""
    keep: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for case in cases:
        ok = True
        tops = []
        for hop in case["hops"]:
            res = retriever.search(str(hop["q"]), top_k=top_k)
            items = list(res.results)
            tops.append(items[0].doc_id if items else "")
            blob = " ".join(r.body for r in items).lower()
            if not all(str(p).lower() in blob for p in hop["expect_points"]):
                ok = False
                stats["要点检索不到"] += 1
                break
        if not ok:
            continue
        if len(set(tops)) < 2:
            stats["两跳同文档"] += 1
            continue
        case["_tops"] = tops
        keep.append(case)
        stats["通过"] += 1
    return keep, dict(stats)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="多跳用例工厂（机械生成 + 确定性筛选）")
    p.add_argument("--pairs", type=int, default=400, help="生成多少个文档对")
    p.add_argument("--screen-top-k", type=int, default=5)
    p.add_argument("--out", default="data/eval/multihop_cases.jsonl")
    p.add_argument("--report", default="data/eval/multihop_cases_report.json")
    args = p.parse_args(argv)

    settings = Settings()
    store = from_settings(settings)
    try:
        retriever = HybridRetriever(store, HashingEmbedder(dim=settings.embedding_dim),
                                   settings)
        print("统计 IDF 锚点 …", flush=True)
        anchors = doc_anchors(store)
        print(f"  有可用锚点的文档：{len(anchors)}")
        pairs = disjoint_pairs(anchors, limit=args.pairs)
        print(f"  词面不相交的文档对：{len(pairs)}")
        cases = build_cases(pairs, anchors)
        print(f"  生成候选：{len(cases)}")
        keep, stats = screen(cases, retriever, top_k=args.screen_top_k)
        print(f"第一轮筛选通过：{len(keep)}  {stats}")
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for c in keep:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        (REPO_ROOT / args.report).write_text(json.dumps({
            "anchored_docs": len(anchors), "pairs": len(pairs),
            "candidates": len(cases), "screened": len(keep), "stats": stats,
            "screen_top_k": args.screen_top_k,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {out}（{len(keep)} 条）")
        print("下一步（花钱）：python scripts/ab_multihop.py --gold "
              f"{args.out} --select --limit {len(keep)}")
        return 0 if keep else 3
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
