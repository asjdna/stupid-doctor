"""A/B：self-ask vs 单轮 RAG，同一批问题、同一套检索、同一个模型。

single-shot 检索 top-k 后直接生成；self-ask 先分解为子问题、各自检索再合成。
指标除成本外全部确定性、可复现：`term_coverage`（答案是否含该条金标的
`must_contain` 关键词，弱接地检查：含关键词不等于答对，但不含几乎一定没答到点上）、
`cited`、`citation_valid`（引用编号是否都指向真实片段）、`llm_calls`、`latency_ms`。

用法：
    python scripts/ab_selfask.py --limit 12                  # 小样本先跑
    python scripts/ab_selfask.py --kinds paraphrase,concept  # 只跑指定类型
    python scripts/ab_selfask.py --json                      # 打印结果 JSON

前置条件：`--gold`（默认 `data/eval/retrieval_gold.jsonl`，需带 `must_contain`）与索引
已就绪，且有 `DEEPSEEK_API_KEY`；结果 JSON 落盘到 `--out`。

输出/解读：逐条打印覆盖与耗时，末尾按模式对照一张表。self-ask 的 `term_coverage` 应
不低于 single-shot，代价是 `llm_calls` 约翻倍。退出码：0 正常结束；3 缺少密钥。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.config import Settings, load_env_file  # noqa: E402
from kubedocs.embed import build_embedder  # noqa: E402
from kubedocs.evaluation.harness import load_gold  # noqa: E402
from kubedocs.retrieve.hybrid import HybridRetriever  # noqa: E402
from kubedocs.store.sqlite_store import from_settings  # noqa: E402

_MARKER = re.compile(r"\[(\d{1,3})\]")

SYSTEM = """你是 Kubernetes 官方文档助手。严格依据提供的【文档片段】回答：

1. 只能使用片段内容，不得使用片段之外的知识；
2. 每个论断后必须紧跟引用标记 [n]，编号对应片段编号；
3. 片段不足以回答时明确说"提供的片段中没有说明"，不要猜测；
4. 参数名、字段名、命令、错误码等标识符必须逐字保留，不得翻译或缩写；
5. 用与提问相同的语言，简洁分点。"""


class LiteLLMDecomposer:
    """真实 LLM 的分解/合成实现（结构化输出，不解析自由文本的语义）。"""

    def __init__(self, settings: Settings) -> None:
        import litellm

        self.litellm = litellm
        self.model = settings.llm_model
        self.calls = 0
        self.cost = 0.0

    def _chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": 700,
            "timeout": 60,
            "num_retries": 2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.litellm.completion(**kwargs)
        self.calls += 1
        try:
            self.cost += float(self.litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001 - 成本算不出不该让流程失败
            self.cost = self.cost
        return str(resp.choices[0].message.content or "").strip()

    def decompose(self, question: str, max_sub: int) -> list[str]:
        raw = self._chat(
            '把用户的技术问题拆成 2-4 个可以独立检索的子问题。只输出 JSON：{"sub_questions": ["...", "..."]}',
            f"问题：{question}\n最多 {max_sub} 个子问题。",
            json_mode=True,
        )
        try:
            subs = json.loads(raw).get("sub_questions", [])
        except json.JSONDecodeError:
            subs = []
        return [str(s).strip() for s in subs if str(s).strip()][:max_sub]

    def compose(self, question: str, parts: Any) -> str:
        blocks = "\n\n".join(f"### 子问题：{p.question}\n{p.answer}" for p in parts)
        return self._chat(
            SYSTEM,
            f"【子问题的检索与作答】\n{blocks}\n\n【原问题】\n{question}\n\n"
            "请综合上面内容回答原问题，并保留 [n] 引用。",
        )


def _citations_valid(text: str, n_available: int) -> bool:
    marks = [int(m) for m in _MARKER.findall(text)]
    return bool(marks) and all(1 <= m <= n_available for m in marks)


def run_single_shot(
    question: str, retriever: Any, llm: LiteLLMDecomposer, *, top_k: int = 5
) -> dict[str, Any]:
    """单轮：检索一次 → 生成。"""
    from kubedocs.generate import build_context

    res = retriever.search(question, top_k=top_k)
    hits = list(res.results)
    ctx, citations = build_context(
        [  # 复用 generate 需要的 ChunkHit 形状
            __import__("kubedocs.store.base", fromlist=["ChunkHit"]).ChunkHit(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                ordinal=h.ordinal,
                title=h.title,
                breadcrumb=h.breadcrumb,
                source_url=h.source_url,
                body=h.body,
            )
            for h in hits
        ],
        max_chars=6000,
    )
    if not citations:
        return {"text": "", "n_citations": 0, "llm_calls": 0}
    text = llm._chat(SYSTEM, f"【文档片段】\n{ctx}\n\n【问题】\n{question}\n\n请依据片段作答并带 [n] 引用。")
    return {"text": text, "n_citations": len(citations), "llm_calls": 1}


def run_self_ask(question: str, retriever: Any, llm: LiteLLMDecomposer, *, top_k: int = 5) -> dict[str, Any]:
    """self-ask：分解 → 各自检索 → 合成。"""
    from kubedocs.agent.selfask import self_ask

    def search(q: str) -> Any:
        return retriever.search(q, top_k=top_k)

    before = llm.calls
    result = self_ask(question, search=search, llm=llm, max_sub=3)
    return {
        "text": result.answer,
        "n_citations": len(result.citations),
        "llm_calls": llm.calls - before,
        "decomposed": result.decomposed,
        "notes": result.notes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="A/B：self-ask vs 单轮 RAG（真实 LLM）")
    p.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    p.add_argument("--limit", type=int, default=12, help="跑多少条（控制时间与花费）")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", default="data/eval/ab_selfask.json")
    p.add_argument("--kinds", default="", help="只跑这些类型，逗号分隔（如 paraphrase,concept）")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    added = load_env_file()
    settings = Settings()
    if "DEEPSEEK_API_KEY" not in added and not __import__("os").environ.get("DEEPSEEK_API_KEY"):
        print("没有可用的 LLM 密钥（DEEPSEEK_API_KEY），A/B 需要真实模型", file=sys.stderr)
        return 3

    rows = load_gold(REPO_ROOT / args.gold)
    if args.kinds:
        wanted = {k.strip() for k in args.kinds.split(",") if k.strip()}
        rows = [r for r in rows if r.kind in wanted]
    rows = rows[: args.limit]
    store = from_settings(settings)
    try:
        retriever = HybridRetriever(store, build_embedder(settings), settings)
        retriever.search("warmup", top_k=1)  # 排除嵌入冷启动
        llm = LiteLLMDecomposer(settings)

        out: dict[str, Any] = {"n": len(rows), "top_k": args.top_k, "modes": {}}
        for mode, fn in (("single_shot", run_single_shot), ("self_ask", run_self_ask)):
            rows_out: list[dict[str, Any]] = []
            print(f"跑 {mode} …", flush=True)
            for i, row in enumerate(rows, 1):
                term = next(iter(row.must_contain.values()))
                t0 = time.perf_counter()
                r = fn(row.q, retriever, llm, top_k=args.top_k)
                ms = (time.perf_counter() - t0) * 1000
                text = str(r.get("text", ""))
                rows_out.append(
                    {
                        "id": row.id,
                        "kind": row.kind,
                        "term": term,
                        "term_covered": term.lower() in text.lower(),
                        "cited": bool(_MARKER.search(text)),
                        "citation_valid": _citations_valid(text, int(r.get("n_citations", 0))),
                        "llm_calls": int(r.get("llm_calls", 0)),
                        "latency_ms": round(ms, 1),
                    }
                )
                print(f"  {i}/{len(rows)} {row.id} cov={rows_out[-1]['term_covered']} {ms:.0f}ms", flush=True)
            out["modes"][mode] = {
                "term_coverage": round(sum(x["term_covered"] for x in rows_out) / len(rows_out), 4),
                "cited_rate": round(sum(x["cited"] for x in rows_out) / len(rows_out), 4),
                "citation_valid_rate": round(sum(x["citation_valid"] for x in rows_out) / len(rows_out), 4),
                "llm_calls_total": sum(x["llm_calls"] for x in rows_out),
                "latency_p50_ms": round(statistics.median([x["latency_ms"] for x in rows_out]), 1),
                "rows": rows_out,
            }
        out["cost_usd"] = round(llm.cost, 6)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"\n{'模式':<14}{'关键词覆盖':>10}{'带引用':>8}{'引用合规':>10}{'LLM调用':>9}{'p50ms':>8}")
            print("-" * 62)
            for mode, m in out["modes"].items():
                print(
                    f"{mode:<14}{m['term_coverage']:>10.2%}{m['cited_rate']:>8.2%}"
                    f"{m['citation_valid_rate']:>10.2%}{m['llm_calls_total']:>9}"
                    f"{m['latency_p50_ms']:>8.0f}"
                )
            print(f"\n模型花费合计：${out['cost_usd']:.4f}（{len(rows)} 条 × 2 模式）")
            print(f"结果已写入 {args.out}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
