"""复查检索标注质量：打印未命中条目**实际返回**了什么。

文档级单目标标注的固有风险是替代答案被记成未命中（问"Pod 一直重启"时
`pod-lifecycle` 与 `debug-running-pod` 都算合格，标注只写后者就记一次失败）。
本脚本把实际返回打出来，供人判断替代答案是否同样正确，再决定要不要扩标注。

用法：
    python scripts/diagnose_misses.py --judge-k 10 --show 3   # 只看未命中
    python scripts/diagnose_misses.py --all                   # 连命中的一起打印

前置条件：`--gold` 指向标注文件，索引已按 `--provider` 建好（或用 `--index-dir` 指定）。
输出：每条列出标注目标、前 `--show` 名（`<-标注` 标出命中项）与名次；`--json` 改为逐行 JSON（含 `ranked` / `hit`）；末行是未命中总数。
`--judge-k` 决定"算不算命中"，**必须与评测的 `--top-k` 取同一个值**，否则两者未命中数不一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.embed import build_embedder  # noqa: E402
from kubedocs.evaluation.harness import load_gold  # noqa: E402
from kubedocs.retrieve.hybrid import HybridRetriever  # noqa: E402
from kubedocs.store.sqlite_store import SqliteStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="诊断标注未命中：看实际返回了什么")
    p.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    p.add_argument("--index-dir", default=None)
    p.add_argument("--provider", default=None, choices=[None, "bge", "hashing"])
    p.add_argument("--hf-offline", action="store_true")
    p.add_argument(
        "--judge-k", type=int, default=10, help="判定命中的 k，必须与评测的 --top-k 一致，否则口径不同"
    )
    p.add_argument("--show", type=int, default=3, help="每个未命中显示前几条")
    p.add_argument("--all", action="store_true", help="连命中的也一起打印")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.index_dir:
        overrides["index_dir"] = args.index_dir
    if args.provider:
        overrides["embedding_provider"] = args.provider
    if args.hf_offline:
        overrides["hf_offline"] = True
    settings = Settings(**overrides)  # type: ignore[arg-type]

    gold_path = Path(args.gold)
    if not gold_path.is_absolute():
        gold_path = REPO_ROOT / gold_path
    rows = load_gold(gold_path)

    store = SqliteStore(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
    )
    try:
        retriever = HybridRetriever(store, build_embedder(settings), settings)
        retriever.search("warmup", top_k=1)  # 排除冷启动
        misses = 0
        for row in rows:
            # 判定用 judge_k、显示只取前 show 条：判定口径与评测一致，才不会出现两套未命中数。
            res = retriever.search(row.q, top_k=args.judge_k)
            ranked = [r.doc_id for r in res.results]
            hit = bool(set(ranked) & row.relevant)
            if hit and not args.all:
                continue
            if not hit:
                misses += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "id": row.id,
                            "kind": row.kind,
                            "q": row.q,
                            "expected": sorted(row.relevant),
                            "ranked": ranked,
                            "hit": hit,
                            "judge_k": args.judge_k,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            flag = "✓" if hit else "✗"
            rank_note = ""
            if hit:
                first = next(i for i, d in enumerate(ranked, 1) if d in row.relevant)
                rank_note = f"  (名次 {first})"
            print(f"\n{flag} {row.id} [{row.kind}] {row.q}{rank_note}")
            print(f"    标注目标: {', '.join(sorted(row.relevant))}")
            for i, r in enumerate(res.results[: args.show], 1):
                mark = "  <-标注" if r.doc_id in row.relevant else ""
                print(f"    {i}. {r.doc_id}{mark}")
                print(f"       {r.title} › {r.breadcrumb}")
            if row.note:
                print(f"    备注: {row.note}")
        print(f"\n未命中 {misses} / {len(rows)}（判定 k={args.judge_k}）")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
