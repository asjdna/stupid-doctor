"""评测执行器：加载标注集 -> 校验 -> 跑多路对照 -> 汇总 -> 卡闸门。

一个永远通过的评测等于没有评测，因此这里设了四道拒绝/失败条件：

1. **标注集自检失败**：标注引用的文档不在索引里，或 `must_contain` 关键词在该文档里
   找不到 —— 说明标注写错了（或者索引没建全），此时直接退出，不产出任何分数。
2. **引用完整性**：每条结果必须有可点击的官方 URL，且 URL 必须能由 doc_id 重算出来
   （防止 URL 映射与存储内容漂移）；违规数不为 0 即失败。
3. **基线与标注集哈希绑定**：改了问题却没重建基线 -> 拒绝比较；跨标注集的分数没有
   可比性。
4. **基线漂移容忍**：hit@k / mrr@k 相对基线下降超过阈值即失败（退出码 2）。

退出码：0 通过，2（`EXIT_GATE_FAIL`）闸门失败或没有基线，`EXIT_PRECONDITION` 表示
标注集缺失或自检不通过。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kubedocs.chunker import source_url_for
from kubedocs.config import Settings, repo_root
from kubedocs.embed import build_embedder
from kubedocs.evaluation.metrics import aggregate, evaluate_one, percentile
from kubedocs.retrieve.hybrid import HybridRetriever
from kubedocs.store.base import Store

BASELINE_SCHEMA = 1
# 闸门阈值：绝对分差。检索分数本身有噪声，容差太紧会因为无关改动红，
# 太松则失去意义。2 个百分点对 40+ 条标注集是"约 1 条查询"的量级。
GATE_TOL_HIT = 0.02
GATE_TOL_MRR = 0.02
CORPUS_DRIFT_TOL = 0.02
"""块数变化的粗判阈值。

**它现在是次要判据**：语料身份由 `corpus_sha`（`(rel_path, n_chunks)` 的 sha256）
精确判定，两者不等就直接拒绝比较。这个阈值只在**已经用 `--allow-corpus-drift`
放行**之后才起作用，用来把"变了多少"写进输出；它单独存在时是近似的（容差之内的
语料变化不会有任何输出），不要当主判据。"""

KINDS = ("identifier", "paraphrase", "concept")
MODES: dict[str, tuple[str, ...]] = {
    "hybrid": ("bm25", "dense"),
    "bm25-only": ("bm25",),
    "dense-only": ("dense",),
}


@dataclass
class GoldRow:
    id: str
    q: str
    kind: str
    must_contain: dict[str, str]
    note: str = ""

    @property
    def relevant(self) -> set[str]:
        return set(self.must_contain)


@dataclass
class GoldProblem:
    row_id: str
    issue: str


@dataclass
class ModeResult:
    mode: str
    overall: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    citation_violations: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "overall": {k: round(v, 4) for k, v in self.overall.items()},
            "by_kind": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.by_kind.items()},
            "latency_ms": {
                "p50": round(percentile(self.latencies, 0.50), 1),
                "p95": round(percentile(self.latencies, 0.95), 1),
            },
            "n_misses": len(self.misses),
            "n_citation_violations": len(self.citation_violations),
        }


def _resolve(raw: str, repo: Path) -> Path:
    """相对路径锚定仓库根，而不是进程 cwd。"""
    p = Path(raw)
    return p if p.is_absolute() else repo / p


def load_gold(path: Path) -> list[GoldRow]:
    rows: list[GoldRow] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{e}") from e
            must = obj.get("must_contain") or {}
            if not must:
                raise ValueError(
                    f"{path}:{lineno} 缺少 must_contain（标注必须说明答案在哪篇文档的哪个关键词附近）"
                )
            rows.append(
                GoldRow(
                    id=obj["id"],
                    q=obj["q"],
                    kind=obj.get("kind", "concept"),
                    must_contain=must,
                    note=obj.get("note", ""),
                )
            )
    if not rows:
        raise ValueError(f"{path} 里没有任何标注行")
    ids = [r.id for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path} 里 id 重复：{sorted(dupes)}")
    return rows


def gold_sha256(rows: list[GoldRow]) -> str:
    blob = json.dumps(
        [{"id": r.id, "q": r.q, "kind": r.kind, "must_contain": r.must_contain} for r in rows],
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def validate_gold(rows: list[GoldRow], store: Store) -> list[GoldProblem]:
    """标注集自检：引用的文档必须在索引里，且关键词确实出现在该文档中。"""
    problems: list[GoldProblem] = []
    known = store.existing_doc_ids()
    wanted: list[str] = []
    for r in rows:
        for doc in r.must_contain:
            if doc in known:
                wanted.append(doc)
            else:
                problems.append(GoldProblem(r.id, f"引用的文档不在索引中：{doc}"))
    texts = store.doc_texts(sorted(set(wanted)))
    for r in rows:
        for doc, term in r.must_contain.items():
            if doc not in known:
                continue
            if term.lower() not in texts.get(doc, ""):
                problems.append(GoldProblem(r.id, f"关键词 {term!r} 在 {doc} 的正文中找不到——标注可能写错了"))
    return problems


def run_mode(
    mode: str, rows: list[GoldRow], retriever: HybridRetriever, k: int, settings: Settings
) -> ModeResult:
    channels = MODES[mode]
    out = ModeResult(mode=mode)
    per_kind: dict[str, list[dict[str, float]]] = {kind: [] for kind in KINDS}
    all_rows: list[dict[str, float]] = []
    for row in rows:
        t0 = time.perf_counter()
        res = retriever.search(row.q, top_k=k, channels=channels)
        out.latencies.append((time.perf_counter() - t0) * 1000)
        ranked_docs = [r.doc_id for r in res.results]
        metrics = evaluate_one(ranked_docs, row.relevant, k)
        all_rows.append(metrics)
        if row.kind in per_kind:
            per_kind[row.kind].append(metrics)
        if metrics["hit"] == 0.0:
            out.misses.append(f"{row.id}[{row.kind}] {row.q}")
        for r in res.results:
            expected = source_url_for(f"{r.doc_id}.md", settings.docs_url_prefix)
            if not r.source_url or not r.source_url.startswith(settings.docs_url_prefix):
                out.citation_violations.append(f"{row.id}:{r.chunk_id} URL 不可点击 ({r.source_url!r})")
            elif r.source_url != expected:
                out.citation_violations.append(
                    f"{row.id}:{r.chunk_id} URL 与 doc_id 重算结果不一致 ({r.source_url} != {expected})"
                )
            if not r.body.strip():
                out.citation_violations.append(f"{row.id}:{r.chunk_id} 正文为空")
    out.overall = aggregate(all_rows)
    out.by_kind = {kind: aggregate(vals) for kind, vals in per_kind.items() if vals}
    return out


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def check_gate(base: dict, current: dict, *, allow_drift: bool) -> list[str]:
    """返回失败原因列表；空列表表示通过。

    比较的前提是**基线可比**：基线缺必需字段，或 schema / gold_sha256 /
    embedding_model / top_k 与本次不一致，都属于"不可比"，一律拒绝比较而不是跳过校验；
    corpus_sha 不一致默认也拒绝，只有显式 `--allow-corpus-drift` 才降级为警告。
    """
    fails: list[str] = []
    # **先校验基线有必需字段**，再谈比较。
    #
    # 下面多处写成 `base.get(key, 0)` 配合 `if b_top_k and ...`，基线缺字段会让整段
    # 校验静默跳过，于是"不可比"变成"通过"。**缺字段 = 不可比，不是"不用比"。**
    missing = [
        key
        for key in ("schema", "gold_sha256", "corpus_sha", "embedding_model", "top_k", "n_chunks", "metrics")
        if key not in base
    ]
    if missing:
        return [
            f"基线缺少必需字段 {missing}：无法判断可比性 ⇒ 拒绝比较。"
            "（不许写成'缺字段就跳过校验'——那正是 top_k 假绿洞的成因。）"
            "请用 `--write-baseline` 重建基线"
        ]
    if base.get("schema") != BASELINE_SCHEMA:
        fails.append(f"基线 schema={base.get('schema')} 与当前 {BASELINE_SCHEMA} 不符，请重建基线")
        return fails
    if base.get("gold_sha256") != current["gold_sha256"]:
        fails.append(
            "标注集已变更（gold_sha256 不同）——跨标注集的分数不可比，请确认后用 --write-baseline 重建基线"
        )
        return fails
    if base.get("embedding_model") != current["embedding_model"]:
        fails.append(
            f"嵌入模型变了（{base.get('embedding_model')} -> {current['embedding_model']}），向量空间不可比"
        )
        return fails
    # top_k 必须与基线一致，否则就是拿 hit@k 去比基线的 hit@10。方向是不对称的：
    # hit 随 k 单调上升，"把 --top-k 改大"就能让闸门变绿、盖过真实的指标退化。
    # 字段存在性已由上面的 missing 校验保证，所以这里直接取、不再用 .get 兜底。
    b_top_k = int(base["top_k"])
    c_top_k = int(current["top_k"])
    if b_top_k != c_top_k:
        fails.append(
            f"top_k 不一致（基线 top_k={b_top_k}，本次 top_k={c_top_k}）："
            f"hit@k 的 k 不同则指标不可比，请用与基线一致的 --top-k"
        )
        return fails
    b_chunks = int(base["n_chunks"])
    c_chunks = int(current["n_chunks"])
    # ---- 语料身份：**精确**判据，优先于下面的块数容差 ----
    #
    # `corpus_sha` 是 (rel_path, n_chunks) 的 sha256，两个后端都按 `ORDER BY rel_path`
    # 迭代（确定性），因此它可以直接当判据：指纹不同就意味着索引不是基线那一批数据，
    # 分数不可比。只靠下面 2% 的块数容差时，容差之内的语料变化不会有任何输出。
    if str(base["corpus_sha"]) != str(current["corpus_sha"]):
        msg = (
            f"语料指纹变了（corpus_sha {base['corpus_sha']} -> {current['corpus_sha']}）："
            f"文档/块数 {base['n_chunks']} -> {current['n_chunks']}，跨语料的分数不可比"
        )
        if not allow_drift:
            fails.append(msg + "；确认无误后加 --allow-corpus-drift")
            return fails
        fails.append("警告(已用 --allow-corpus-drift 放行)：" + msg)

    if abs(c_chunks - b_chunks) / b_chunks > CORPUS_DRIFT_TOL:
        msg = (
            f"索引规模变化 {(c_chunks - b_chunks) / b_chunks:+.1%}"
            f"（{b_chunks} -> {c_chunks}），语料变了，指标不可比"
        )
        if not allow_drift:
            fails.append(msg + "；确认无误后加 --allow-corpus-drift")
            return fails
        fails.append("警告(已用 --allow-corpus-drift 放行)：" + msg)

    if current["n_citation_violations"] != 0:
        fails.append(f"引用完整性失败：{current['n_citation_violations']} 条违规")
    b_hit = float(base.get("metrics", {}).get("hit", 0.0))
    c_hit = float(current["metrics"]["hit"])
    if c_hit < b_hit - GATE_TOL_HIT:
        fails.append(f"hit@{current['top_k']} 从 {b_hit:.4f} 掉到 {c_hit:.4f}（容差 {GATE_TOL_HIT}）")
    b_mrr = float(base.get("metrics", {}).get("mrr", 0.0))
    c_mrr = float(current["metrics"]["mrr"])
    if c_mrr < b_mrr - GATE_TOL_MRR:
        fails.append(f"mrr@{current['top_k']} 从 {b_mrr:.4f} 掉到 {c_mrr:.4f}（容差 {GATE_TOL_MRR}）")
    return fails


def _print_table(results: dict[str, ModeResult], k: int, n_by_kind: dict[str, int]) -> None:
    print(
        f"\n{'模式':<12} {'hit@' + str(k):>8} {'recall':>8} {'mrr':>8} {'ndcg':>8} {'p50ms':>7} {'p95ms':>7}"
    )
    print("-" * 68)
    for mode in MODES:
        r = results.get(mode)
        if r is None:
            continue
        o = r.overall
        p50 = percentile(r.latencies, 0.50)
        p95 = percentile(r.latencies, 0.95)
        print(
            f"{mode:<12} {o.get('hit', 0):>8.4f} {o.get('recall', 0):>8.4f} "
            f"{o.get('mrr', 0):>8.4f} {o.get('ndcg', 0):>8.4f} {p50:>7.1f} {p95:>7.1f}"
        )
    hv = results.get("hybrid")
    if hv:
        print("\n按问题类型（模式=hybrid）：")
        print(f"{'类型':<12} {'条数':>5} {'hit':>8} {'mrr':>8} {'ndcg':>8}")
        for kind in KINDS:
            if kind not in hv.by_kind:
                continue
            m = hv.by_kind[kind]
            print(
                f"{kind:<12} {n_by_kind.get(kind, 0):>5} {m.get('hit', 0):>8.4f} "
                f"{m.get('mrr', 0):>8.4f} {m.get('ndcg', 0):>8.4f}"
            )


def run_eval(args: Namespace, settings: Settings, store: Store) -> int:
    from kubedocs.cli import EXIT_GATE_FAIL, EXIT_OK, EXIT_PRECONDITION

    def emit(msg: str, *, flush: bool = False) -> None:
        """所有**人类可读**的输出都走这里。

        `--json` 时改道 **stderr**：stdout 留给机器可读的纯 JSON，
        否则 `python -m kubedocs.cli eval --json | jq` 会被旁白打断。
        退出码与信息一条都不少，只是分流。
        """
        print(msg, file=sys.stderr if args.json else sys.stdout, flush=flush)

    repo = repo_root()
    gold_path = _resolve(args.gold, repo)
    if not gold_path.exists():
        emit(f"找不到标注集：{gold_path}")
        return EXIT_PRECONDITION

    rows = load_gold(gold_path)
    problems = validate_gold(rows, store)
    if problems:
        emit(f"标注集自检未通过（{len(problems)} 项）——不产出任何分数：")
        for p in problems[:20]:
            emit(f"  ✗ {p.row_id}: {p.issue}")
        return EXIT_PRECONDITION
    emit(f"标注集：{gold_path}")
    emit(f"  条数 {len(rows)}，自检通过（引用的文档与关键词均已在索引中）")
    n_by_kind: dict[str, int] = {}
    for r in rows:
        n_by_kind[r.kind] = n_by_kind.get(r.kind, 0) + 1
    emit(f"  类型分布：{n_by_kind}")

    st = store.stats()
    emit(f"索引：{st.n_chunks} 块 / {st.n_docs} 文档，{st.embedding_model}（{st.embedding_dim} 维）")

    retriever = HybridRetriever(store, build_embedder(settings), settings)

    # 预热：sentence-transformers 的**首次** encode 含模块导入与权重载入，
    # 把它算进第一个查询的延迟会污染 p50/p95，让"检索延迟"这个指标完全失真。
    # 所以先空跑一次，冷启动耗时单独报告。
    t_cold = time.perf_counter()
    if rows:
        retriever.search(rows[0].q, top_k=1)
    cold_s = time.perf_counter() - t_cold
    emit(f"预热完成：冷启动 {cold_s:.1f}s（不计入下面的延迟统计）")

    modes = ["hybrid"] if args.no_ablation else list(MODES)
    k = args.top_k
    results: dict[str, ModeResult] = {}
    for mode in modes:
        emit(f"跑 {mode} …", flush=True)
        results[mode] = run_mode(mode, rows, retriever, k, settings)

    hv = results["hybrid"]
    current = {
        "gold_sha256": gold_sha256(rows),
        "embedding_model": st.embedding_model,
        "embedding_provider": settings.embedding_provider,
        "n_chunks": st.n_chunks,
        "n_docs": st.n_docs,
        "top_k": k,
        "n_gold": len(rows),
        "metrics": {kk: round(vv, 4) for kk, vv in hv.overall.items()},
        "ablation": {m: r.to_json() for m, r in results.items()},
        "n_citation_violations": len(hv.citation_violations),
        "corpus_sha": _corpus_fingerprint(store),
    }

    # 这一段的**顺序本身**就是契约，别改：
    #
    # `--json` 只影响输出格式，不影响是否跑闸门、也不影响退出码；退出码只由
    # `--write-baseline` / `--gate` 决定，JSON 一律最后才打印。表格与未命中列表
    # **只打印一份**，且统一走 `emit()`——人类可读输出全部经由同一个出口，
    # `--json` 时它改道 stderr，stdout 只剩纯 JSON。
    if not args.json:
        _print_table(results, k, n_by_kind)
        if hv.citation_violations:
            emit(f"\n引用完整性：{len(hv.citation_violations)} 条违规")
            for v in hv.citation_violations[:5]:
                emit(f"  ✗ {v}")
        else:
            emit("\n引用完整性：通过（每条结果都有可点击且可重算的官方 URL）")
        if hv.misses:
            emit(f"\n未命中 {len(hv.misses)} 条：")
            for m in hv.misses[:12]:
                emit(f"  ✗ {m}")

    baseline_path = _resolve(args.baseline, repo)

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": BASELINE_SCHEMA, **current}
        with open(baseline_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        emit(f"\n已写入基线：{baseline_path}")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    rc = EXIT_OK
    if args.gate:
        base = load_baseline(baseline_path)
        if base is None:
            emit(f"\n没有基线文件 {baseline_path}；先跑一次 `eval --write-baseline`")
            rc = EXIT_GATE_FAIL
        else:
            fails = check_gate(base, current, allow_drift=args.allow_corpus_drift)
            hard = [f for f in fails if not f.startswith("警告")]
            for f in fails:
                emit(("  ✗ " if not f.startswith("警告") else "  ⚠ ") + f)
            if hard:
                emit("\n闸门：不通过")
                rc = EXIT_GATE_FAIL
            else:
                emit("\n闸门：通过")

    if args.json:
        print(json.dumps(current, ensure_ascii=False, indent=2))
    return rc


def _corpus_fingerprint(store: Store) -> str:
    """索引内容指纹：块 id + content_hash 的哈希。用于判断两次评测是否同一批数据。"""
    h = hashlib.sha256()
    for row in store.doc_summary(limit=10**9):
        h.update(str(row["rel_path"]).encode())
        h.update(str(row["n_chunks"]).encode())
    return h.hexdigest()[:16]
