"""多跳 vs 单轮对照：先筛出有区分度的用例，再比逐跳覆盖率。

用例两轮筛出：初次筛选（确定性、零 LLM 成本）只留每跳要点都能从该跳查询检索到的用例
——要点在语料里找不到的用例会把"语料没有"伪装成"系统没答对"；精选（只跑单轮）只留单轮
至少漏一跳要点的用例，否则两路都满分、对照没有信息量；精选不跑多跳，避免事后调集合。

判定逐跳进行：每跳的 `expect_points` **全部**命中才算该跳覆盖；指标为 `full_coverage`
（全跳）与 `hop_coverage`（逐跳），两路同一套判定，因此可比。

用法：
    python scripts/ab_multihop.py --screen            # 只做初次筛选（不花钱）
    python scripts/ab_multihop.py --select --limit 6  # 初次筛选 + 精选（只跑单轮）
    python scripts/ab_multihop.py --limit 6           # 两轮 + 对照

前置条件：`--gold`（默认 `data/eval/multihop_gold.jsonl`）指向多跳集，索引已建好；
`--select` 与对照还需 `DEEPSEEK_API_KEY`。

输出：逐条打印每跳命中，末尾按模式对照全跳覆盖 / 逐跳覆盖 / 依赖率 / LLM 调用数 /
p50 延迟（JSON 写入 `--out`）。退出码：0 正常结束；3 缺少 `DEEPSEEK_API_KEY`。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.config import Settings, load_env_file  # noqa: E402
from kubedocs.embed import build_embedder  # noqa: E402
from kubedocs.retrieve.hybrid import HybridRetriever  # noqa: E402
from kubedocs.store.sqlite_store import from_settings  # noqa: E402

GOLD = "data/eval/multihop_gold.jsonl"

SYSTEM = (
    "你是 Kubernetes 官方文档助手。只能依据提供的【文档片段】作答，"
    "不得使用片段之外的知识；片段不足以回答时明确说明，不要猜测；"
    "参数名、字段名、命令必须逐字保留；简洁作答。"
)

HOP_SYSTEM = "你是检索助手。依据提供的【文档片段】简要回答这一跳的问题，只写与问题直接相关的要点，不要展开。"


def load_multihop(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


class Planner:
    """真实 LLM 的规划/逐跳作答/合成。"""

    def __init__(self, settings: Settings) -> None:
        import litellm

        self.litellm = litellm
        self.model = settings.llm_model
        self.calls = 0
        self.cost = 0.0

    def _chat(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 600) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "timeout": 60,
            "num_retries": 2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.litellm.completion(**kwargs)
        self.calls += 1
        try:
            self.cost += float(self.litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001 - 成本算不出不阻断
            self.cost = self.cost
        return str(resp.choices[0].message.content or "").strip()

    def plan(self, question: str, max_hops: int) -> list[dict[str, Any]]:
        raw = self._chat(
            "把问题拆成一条**有依赖**的多跳链。若后一跳需要用到前一跳的答案，"
            "在它的 question 里写 {{prev}} 占位符并置 needs_previous=true。"
            '只输出 JSON：{"hops":[{"question":"...","needs_previous":false}, ...]}',
            f"问题：{question}\n最多 {max_hops} 跳。",
            json_mode=True,
        )
        try:
            return list(json.loads(raw).get("hops", []))
        except json.JSONDecodeError:
            return []

    def answer_hop(self, question: str, context: str) -> str:
        return self._chat(HOP_SYSTEM, f"【文档片段】\n{context}\n\n【这一跳的问题】\n{question}")

    def compose(self, question: str, hops: Any) -> str:
        blocks = "\n\n".join(f"### 子问题：{h.query}\n{h.answer}" for h in hops)
        return self._chat(SYSTEM, f"【各跳结果】\n{blocks}\n\n【原问题】\n{question}\n\n请综合回答原问题。")


def run_single_shot(question: str, retriever: Any, planner: Planner, *, top_k: int = 8) -> dict[str, Any]:
    """单轮：一次检索 + 一次生成（与多跳用同一模型、同一 SYSTEM 提示）。"""
    res = retriever.search(question, top_k=top_k)
    items = list(res.results)
    blocks, cites = [], []
    for i, r in enumerate(items, start=1):
        label = f"{r.title} › {r.breadcrumb}" if r.breadcrumb else r.title
        blocks.append(f"[{i}] {label}\n{r.body[:900]}")
        cites.append(
            {
                "index": i,
                "chunk_id": r.chunk_id,
                "title": r.title,
                "breadcrumb": r.breadcrumb,
                "source_url": r.source_url,
            }
        )
    if not cites:
        return {"answer": "", "n_citations": 0, "llm_calls": 0}
    text = planner._chat(
        SYSTEM, "【文档片段】\n" + "\n\n".join(blocks) + f"\n\n【问题】\n{question}\n\n请依据片段作答。"
    )
    return {"answer": text, "n_citations": len(cites), "llm_calls": 1}


def hop_coverage(answer: str, hops: list[dict[str, Any]]) -> tuple[bool, list[bool]]:
    """逐跳判定：该跳的 `expect_points` 全部命中才算覆盖（大小写不敏感）。"""
    low = answer.lower()
    # 每跳的要点必须**全部**命中：否则"覆盖"只等于"碰到了这个话题"。
    per_hop = [all(str(p).lower() in low for p in h.get("expect_points", [])) for h in hops]
    return all(per_hop), per_hop


def _retrievable(retriever: Any, query: str, points: list[str], *, top_k: int = 20) -> tuple[bool, list[str]]:
    """初次筛选的判据：该跳的要点能否从**这条查询**的候选里找到。"""
    res = retriever.search(query, top_k=top_k)
    blob = " ".join(r.body for r in res.results).lower()
    missing = [p for p in points if str(p).lower() not in blob]
    return (not missing), missing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="多跳对照 + 两轮筛选（真实 LLM）")
    p.add_argument("--gold", default=GOLD)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--screen", action="store_true", help="只做初次筛选（不花钱）")
    p.add_argument("--select", action="store_true", help="做完初次筛选后再精选（只跑单轮）")
    p.add_argument("--out", default="data/eval/ab_multihop.json")
    args = p.parse_args(argv)

    rows = load_multihop(REPO_ROOT / args.gold)
    load_env_file()
    settings = Settings()
    store = from_settings(settings)
    try:
        retriever = HybridRetriever(store, build_embedder(settings), settings)
        retriever.search("warmup", top_k=1)

        print("【初次筛选】每跳期望要点必须能从该跳查询检索到（确定性，不花钱）")
        screened: list[dict[str, Any]] = []
        for row in rows:
            ok, detail = True, []
            for h in row["hops"]:
                q = str(h["q"]).replace("{prev}", "")
                good, missing = _retrievable(retriever, q, list(h.get("expect_points", [])))
                if not good:
                    ok = False
                    detail.append(f"跳『{q}』要点检索不到：{missing}")
            if ok:
                screened.append(row)
                print(f"  ✓ {row['id']}")
            else:
                print(f"  ✗ {row['id']} 淘汰：{'; '.join(detail)}")
        print(f"初次筛选用例：{len(screened)}/{len(rows)}")
        if args.screen:
            return 0

        need_llm = args.select or not args.screen
        if need_llm and not os.environ.get("DEEPSEEK_API_KEY"):
            print("没有 DEEPSEEK_API_KEY，无法精选或对照", file=sys.stderr)
            return 3

        planner = Planner(settings)
        selected: list[dict[str, Any]] = []
        if args.select:
            print("\n【精选】只跑单轮；单轮必须至少漏一跳要点才算有区分度")
            for row in screened[: args.limit]:
                r = run_single_shot(row["q"], retriever, planner, top_k=args.top_k)
                full, per_hop = hop_coverage(str(r["answer"]), row["hops"])
                row["_single_shot_covered"] = full
                row["_single_shot_per_hop"] = per_hop
                mark = "保留（单轮失手）" if not full else "淘汰（单轮已全中）"
                print(f"  {row['id']} 单轮逐跳={per_hop} -> {mark}")
                if not full:
                    selected.append(row)
            Path(args.out).write_text(
                json.dumps({"screened": len(screened), "selected": selected}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n精选出 {len(selected)}/{len(screened)} 条有区分度的用例 -> {args.out}")
            print(f"本轮花费 ${planner.cost:.4f}（{planner.calls} 次调用，只跑了单轮）")
            return 0

        print("\n【对照】多跳 vs 单轮（逐跳覆盖率）")
        out: dict[str, Any] = {"n": len(screened[: args.limit]), "modes": {}}
        for mode in ("single_shot", "multi_hop"):
            per_row = []
            print(f"跑 {mode} …", flush=True)
            for row in screened[: args.limit]:
                t0 = time.perf_counter()
                if mode == "single_shot":
                    r = run_single_shot(row["q"], retriever, planner, top_k=args.top_k)
                    answer, calls = str(r["answer"]), int(r["llm_calls"])
                    deps = False
                else:
                    from kubedocs.agent.multihop import multi_hop

                    before = planner.calls
                    mh = multi_hop(
                        row["q"],
                        search=lambda q: retriever.search(q, top_k=args.top_k),
                        llm=planner,
                        max_hops=3,
                    )
                    answer, calls = mh.answer, planner.calls - before
                    deps = mh.has_dependency
                full, per_hop = hop_coverage(answer, row["hops"])
                ms = (time.perf_counter() - t0) * 1000
                per_row.append(
                    {
                        "id": row["id"],
                        "full": full,
                        "per_hop": per_hop,
                        "llm_calls": calls,
                        "latency_ms": round(ms, 1),
                        "has_dependency": deps,
                    }
                )
                print(f"  {row['id']} full={full} per_hop={per_hop} calls={calls} {ms:.0f}ms", flush=True)
            n = len(per_row)
            all_hops = [b for r in per_row for b in r["per_hop"]]
            out["modes"][mode] = {
                "full_coverage": round(sum(r["full"] for r in per_row) / n, 4),
                "hop_coverage": round(sum(all_hops) / len(all_hops), 4) if all_hops else 0.0,
                "llm_calls_total": sum(r["llm_calls"] for r in per_row),
                "latency_p50_ms": round(statistics.median([r["latency_ms"] for r in per_row]), 1),
                "dependency_rate": round(sum(r["has_dependency"] for r in per_row) / n, 4),
                "rows": per_row,
            }
        out["cost_usd"] = round(planner.cost, 6)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{'模式':<14}{'全跳覆盖':>10}{'逐跳覆盖':>10}{'依赖率':>8}{'LLM调用':>9}{'p50ms':>8}")
        print("-" * 62)
        for mode, m in out["modes"].items():
            print(
                f"{mode:<14}{m['full_coverage']:>10.2%}{m['hop_coverage']:>10.2%}"
                f"{m['dependency_rate']:>8.0%}{m['llm_calls_total']:>9}"
                f"{m['latency_p50_ms']:>8.0f}"
            )
        print(f"\n花费 ${out['cost_usd']:.4f} -> {args.out}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
