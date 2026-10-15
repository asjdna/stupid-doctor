"""存储层检索延迟对照：在**真实索引**上测各后端的 p50/p95，只测存储层。

只计 `store.dense_search()` / `store.keyword_search()` 本身；端到端口径里的查询编码
占 30~45ms，会把后端之间的差异淹掉。

用法：
    python scripts/bench_store_latency.py                      # sqlite + sqlite-vec
    python scripts/bench_store_latency.py --backends all       # 加上 pgvector（需已灌数据）
    python scripts/bench_store_latency.py --provider hashing   # 不加载真实模型（仅冒烟）

前置条件：`--index-dir` / `--domain` 下已按 `--provider` 建好索引（块数或维度不符的后端会被
跳过），`--gold` 提供查询文本；查询编码只做一次、不计入延迟。

输出：每后端一行块数与 dense/keyword 的 p50/p95/样本数，随后是相对第一个后端的 top-k 一致性（向量路、全文路各一行）；`--json-out` 落盘，不含逐查询明细。

数字怎么读：一致性比延迟更重要——换后端不该改检索结果。`identical_topk`（名次逐个相同）最强；
`same_hit_set`（集合相同、只是排序不同）是 `ts_rank` 与 BM25 的正常差异，只看前者会把
"排序变了"误读成"召回坏了"。退出码：0 正常结束；1 无后端跑出结果。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.embed import build_embedder  # noqa: E402
from kubedocs.evaluation.harness import load_gold  # noqa: E402
from kubedocs.evaluation.metrics import percentile  # noqa: E402
from kubedocs.store import from_settings  # noqa: E402
from kubedocs.store.base import Store  # noqa: E402


@dataclass
class BenchResult:
    backend: str
    n_chunks: int = 0
    open_ms: float = 0.0
    n_queries: int = 0
    n_samples: int = 0
    dense_ms: dict[str, float] = field(default_factory=dict)
    keyword_ms: dict[str, float] = field(default_factory=dict)
    dense_top: list[list[str]] = field(default_factory=list)
    keyword_top: list[list[str]] = field(default_factory=list)
    agreement: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        """落盘用：不含逐查询明细（那个只有跨后端对照时才用，体积大且无独立价值）。"""
        return {
            "backend": self.backend,
            "n_chunks": self.n_chunks,
            "open_ms": self.open_ms,
            "n_queries": self.n_queries,
            "n_samples": self.n_samples,
            "dense_ms": self.dense_ms,
            "keyword_ms": self.keyword_ms,
            "agreement": self.agreement,
        }


def bench_backend(
    name: str, store: Store, queries: list[str], qvecs: np.ndarray, top_k: int, repeats: int
) -> BenchResult:
    """分别测两路，**不做交错**：交错跑时两路的页访问会互相挤占 SQLite 页缓存，
    测出的既不是向量检索延迟也不是全文检索延迟。一个 pass 只测一件事。
    """
    res = BenchResult(backend=name, n_queries=len(queries))
    # 预热一轮（不计入）：第一次查询要走过冷页缓存，把它算进分位数会污染结果
    for i, q in enumerate(queries[: min(10, len(queries))]):
        store.dense_search(qvecs[i], top_k)
        store.keyword_search(q, top_k)

    dense_lat: list[float] = []
    kw_lat: list[float] = []
    for _ in range(repeats):
        for i in range(len(queries)):
            t0 = time.perf_counter()
            hits = store.dense_search(qvecs[i], top_k)
            dense_lat.append((time.perf_counter() - t0) * 1000.0)
            res.dense_top.append([c for c, _ in hits])
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            khits = store.keyword_search(q, top_k)
            kw_lat.append((time.perf_counter() - t0) * 1000.0)
            res.keyword_top.append([c for c, _ in khits])
    res.n_samples = len(dense_lat)
    res.dense_ms = _summary(dense_lat)
    res.keyword_ms = _summary(kw_lat)
    return res


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def _agreement(base: list[list[str]], other: list[list[str]], top_k: int) -> dict[str, str]:
    """跨后端 top-k 一致性的两个口径（都要报，它们说的不是一回事）。

    `identical_topk` 是名次逐个相同（最强）；`same_hit_set` 是集合相同、只是排序不同，
    正是 `ts_rank` 与 BM25 的差异，只看前一个会把"排序变了"误读成"召回坏了"。
    """
    total = len(base)
    identical = sum(1 for a, b in zip(base, other, strict=True) if a == b)
    same_set = sum(1 for a, b in zip(base, other, strict=True) if set(a) == set(b))
    overlap = (
        statistics.fmean(
            len(set(a) & set(b)) / max(min(len(a), len(b)), 1) for a, b in zip(base, other, strict=True)
        )
        if total
        else 0.0
    )
    return {
        "n_queries": str(total),
        "identical_topk": f"{identical}/{total}",
        "same_hit_set": f"{same_set}/{total}",
        "mean_set_overlap": f"{overlap:.4f}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="三个存储后端的检索延迟对照")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--domain", default="k8s_docs")
    ap.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    ap.add_argument("--provider", default="bge", choices=["bge", "hashing"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="每个查询重复轮数（样本数 = 轮数 × 查询数）")
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["sqlite", "sqlite-vec"],
        help="要测的后端；也可以写 all（含 pgvector）",
    )
    ap.add_argument("--json-out", default="data/eval/store_bench/latency.json")
    args = ap.parse_args(argv)

    backends = ["sqlite", "sqlite-vec", "pgvector"] if "all" in args.backends else list(args.backends)

    root = Path(__file__).resolve().parents[1]
    gold = Path(args.gold)
    gold = gold if gold.is_absolute() else root / gold
    queries = [r.q for r in load_gold(gold)]
    print(f"金标查询：{len(queries)} 条（来自 {gold}）")

    settings = Settings(index_dir=args.index_dir, domain=args.domain, embedding_provider=args.provider)
    embedder = build_embedder(settings)
    t0 = time.perf_counter()
    qvecs = embedder.encode(queries, is_query=True)
    qvecs = np.asarray(qvecs, dtype=np.float32)
    print(f"查询编码：{time.perf_counter() - t0:.1f}s（不计入延迟统计），形状 {qvecs.shape}")

    results: list[BenchResult] = []
    stores: dict[str, Store] = {}
    try:
        for name in backends:
            # `# type: ignore` 必须与 mypy 报错的参数同物理行，所以参数逐行展开、
            # 把 ignore 钉在 store=name 上；ruff format 拆行会让它失效。
            cfg = Settings(
                index_dir=args.index_dir,
                domain=args.domain,
                embedding_provider=args.provider,
                store=name,  # type: ignore[arg-type]
            )
            t0 = time.perf_counter()
            try:
                store = from_settings(cfg)
            except Exception as e:  # noqa: BLE001 - 某个后端不可用不该让整轮跑不出结果
                print(f"[{name}] 不可用：{type(e).__name__}: {e}")
                continue
            open_ms = (time.perf_counter() - t0) * 1000
            st = store.stats()
            if st.n_chunks == 0 or st.embedding_dim != qvecs.shape[1]:
                print(
                    f"[{name}] 跳过：块 {st.n_chunks}，维度 {st.embedding_dim}（查询向量 {qvecs.shape[1]}）"
                )
                store.close()
                continue
            print(f"[{name}] {st.n_chunks} 块，打开耗时 {open_ms:.0f}ms，开始计时…")
            res = bench_backend(name, store, queries, qvecs, args.top_k, args.repeats)
            res.open_ms = round(open_ms, 1)
            res.n_chunks = st.n_chunks
            results.append(res)
            stores[name] = store

        if not results:
            print("没有任何后端跑出结果", file=sys.stderr)
            return 1

        print(
            f"\n{'后端':<12}{'块数':>8}{'dense p50':>11}{'dense p95':>11}"
            f"{'kw p50':>9}{'kw p95':>9}{'样本':>8}"
        )
        print("-" * 68)
        for r in results:
            print(
                f"{r.backend:<12}{r.n_chunks:>8}{r.dense_ms['p50']:>11.2f}"
                f"{r.dense_ms['p95']:>11.2f}{r.keyword_ms['p50']:>9.2f}"
                f"{r.keyword_ms['p95']:>9.2f}{r.n_samples:>8}"
            )

        base = results[0]
        print(f"\n跨后端 top-{args.top_k} 一致性（基准 = {base.backend}）")
        for r in results[1:]:
            dn = _agreement(base.dense_top, r.dense_top, args.top_k)
            kw = _agreement(base.keyword_top, r.keyword_top, args.top_k)
            r.agreement = {"vs": base.backend, "dense": dn, "keyword": kw}
            print(
                f"  {r.backend:<12} 向量路 名次相同 {dn['identical_topk']:>8}，"
                f"集合相同 {dn['same_hit_set']:>8}（平均重合 {dn['mean_set_overlap']}）"
            )
            print(
                f"  {'':<12} 全文路 名次相同 {kw['identical_topk']:>8}，"
                f"集合相同 {kw['same_hit_set']:>8}（平均重合 {kw['mean_set_overlap']}）"
            )

        out = Path(args.json_out)
        out = out if out.is_absolute() else root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "top_k": args.top_k,
            "repeats": args.repeats,
            "provider": args.provider,
            "n_queries": len(queries),
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": [r.to_json() for r in results],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")
        return 0
    finally:
        for s in stores.values():
            s.close()


if __name__ == "__main__":
    sys.exit(main())
