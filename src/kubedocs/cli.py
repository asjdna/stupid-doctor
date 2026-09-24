"""命令行入口：`python -m kubedocs.cli <子命令>`。

存在的理由很直接：**业务逻辑必须先能在本机跑起来，才谈得上包装成服务**。
所有能力（摄取 / 检索 / 索引体检 / 评测）都从这里可执行，HTTP 层只是它的一个
薄包装——这也保证了"能跑通"这件事可以在没有容器、没有数据库、没有网络的情况下复现。

退出码约定（供 CI 与脚本判断，而不是靠解析 stdout）：
    0 成功      1 参数/用户错误      2 评测闸门不通过      3 索引为空等前置条件不满足
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from kubedocs.config import Settings
from kubedocs.store.sqlite_store import SqliteStore

EXIT_OK = 0
EXIT_USER = 1
EXIT_GATE_FAIL = 2
EXIT_PRECONDITION = 3


# --------------------------------------------------------------------------- 辅助
def make_settings(args: argparse.Namespace) -> Settings:
    """CLI 参数覆盖配置。直接构造 Settings 而不是改环境变量，避免污染进程环境。"""
    overrides: dict[str, object] = {}
    if getattr(args, "provider", None):
        overrides["embedding_provider"] = args.provider
    if getattr(args, "index_dir", None):
        overrides["index_dir"] = args.index_dir
    if getattr(args, "corpus_dir", None):
        overrides["corpus_dir"] = args.corpus_dir
    if getattr(args, "hf_endpoint", None):
        overrides["hf_endpoint"] = args.hf_endpoint
    if getattr(args, "hf_offline", False):
        overrides["hf_offline"] = True
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level
    return Settings(**overrides)  # type: ignore[arg-type]


def open_store(settings: Settings, *, require_data: bool = True) -> SqliteStore:
    from kubedocs.store.sqlite_store import from_settings

    store = from_settings(settings)
    if require_data and store.stats().n_chunks == 0:
        print(f"索引为空：{settings.sqlite_path}\n先跑 `python -m kubedocs.cli ingest`",
              file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)
    return store


def _snippet(text: str, width: int = 160) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[:width] + "…"


# --------------------------------------------------------------------------- ingest
def cmd_ingest(args: argparse.Namespace) -> int:
    from kubedocs.embed import build_embedder
    from kubedocs.ingest import ingest

    settings = make_settings(args)
    store = open_store(settings, require_data=False)
    try:
        embedder = build_embedder(settings)
        print(f"嵌入器：{settings.embedding_provider} / {getattr(embedder, 'name', '?')} "
              f"({settings.embedding_dim} 维)")
        print(f"语料：{settings.corpus_path}")
        print(f"索引：{settings.sqlite_path}")
        report = ingest(store, embedder, settings, rebuild=args.rebuild,
                        limit_files=args.limit_files, batch_files=args.batch_files,
                        progress=lambda m: print(m, flush=True))
        print("-" * 72)
        print(report.summary())
        for note in report.notes:
            print(f"注意：{note}")
        print(f"索引规模：{store.stats().n_chunks} 块 / {store.stats().n_docs} 文档")
        return EXIT_OK
    finally:
        store.close()


# --------------------------------------------------------------------------- search
def cmd_search(args: argparse.Namespace) -> int:
    from kubedocs.embed import build_embedder
    from kubedocs.retrieve.hybrid import HybridRetriever

    settings = make_settings(args)
    store = open_store(settings)
    try:
        retriever = HybridRetriever(store, build_embedder(settings), settings)
        res = retriever.search(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps({
                "query": res.query,
                "diagnostics": res.diagnostics,
                "results": [
                    {"chunk_id": r.chunk_id, "doc_id": r.doc_id, "title": r.title,
                     "breadcrumb": r.breadcrumb, "source_url": r.source_url,
                     "score": r.score, "ranks": r.ranks, "rerank_score": r.rerank_score,
                     "body": r.body}
                    for r in res.results
                ],
            }, ensure_ascii=False, indent=2))
            return EXIT_OK
        if not res.results:
            print("（无结果）")
        for i, r in enumerate(res.results, 1):
            print(f"\n[{i}] {r.citation}")
            print(f"    {r.source_url}")
            print(f"    {_snippet(r.body, 220)}")
        diag = res.diagnostics
        print(f"\n--- 诊断：融合池 {diag.get('n_fused')}，两路重合 {diag.get('overlap')}，"
              f"耗时 {diag.get('total_ms')}ms")
        for name, d in (diag.get("channels_detail") or {}).items():  # type: ignore[attr-defined]
            print(f"    {name}: {d}")
        for msg in (diag.get("degraded") or []):  # type: ignore[attr-defined]
            print(f"    ⚠ {msg}")
        if args.explain:
            print("\n--- 排序解释")
            print(res.explain_all())
        return EXIT_OK
    finally:
        store.close()


# --------------------------------------------------------------------------- stats
def cmd_stats(args: argparse.Namespace) -> int:
    settings = make_settings(args)
    store = open_store(settings, require_data=False)
    try:
        st = store.stats()
        meta = store.meta_dump()
        payload = {
            "corpus_dir": str(settings.corpus_path),
            "db": str(settings.sqlite_path),
            "vectors": str(settings.vector_path),
            "n_docs": st.n_docs, "n_chunks": st.n_chunks,
            "mean_chars": round(st.mean_chars, 1),
            "n_with_code": st.n_with_code,
            "embedding_model": st.embedding_model,
            "embedding_dim": st.embedding_dim,
            "indexed_at": st.indexed_at,
            "meta": meta,
        }
        if args.verify:
            payload["fts_integrity"] = store.verify_fts()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"语料目录   : {payload['corpus_dir']}")
            print(f"索引库     : {payload['db']}")
            print(f"向量矩阵   : {payload['vectors']}")
            print(f"文档/块数  : {st.n_docs} / {st.n_chunks}")
            print(f"块均长度   : {st.mean_chars:.1f} 字符")
            print(f"含代码块   : {st.n_with_code}（{100.0 * st.n_with_code / max(st.n_chunks, 1):.1f}%）")
            print(f"嵌入模型   : {st.embedding_model}（{st.embedding_dim} 维）")
            print(f"索引时间   : {st.indexed_at}")
            if args.verify:
                print(f"FTS 一致性 : {payload['fts_integrity']}")
                if payload["fts_integrity"] != "ok":
                    return EXIT_PRECONDITION
            if st.n_chunks == 0:
                print("\n索引为空，请先跑 ingest", file=sys.stderr)
                return EXIT_PRECONDITION
        return EXIT_OK
    finally:
        store.close()


# --------------------------------------------------------------------------- eval
def cmd_eval(args: argparse.Namespace) -> int:
    from kubedocs.evaluation import run_eval

    settings = make_settings(args)
    store = open_store(settings)
    try:
        return run_eval(args, settings, store)
    finally:
        store.close()


# --------------------------------------------------------------------------- 解析
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kubedocs",
        description="Kubernetes 官方文档检索问答服务（本地可跑，无需容器）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型流程：\n"
            "  1) python scripts/fetch_corpus.py          # 拉语料（一次就够）\n"
            "  2) python -m kubedocs.cli ingest           # 建索引\n"
            "  3) python -m kubedocs.cli search \"...\"     # 检索\n"
            "  4) python -m kubedocs.cli eval --gate      # 评测并卡闸门\n"
        ),
    )
    p.add_argument("--index-dir", help="索引目录（默认 data/index）")
    p.add_argument("--corpus-dir", help="语料目录（默认 data/raw/k8s-docs）")
    p.add_argument("--provider", choices=["bge", "hashing"],
                   help="嵌入器（bge=真实向量，hashing=离线降级）")
    p.add_argument("--hf-endpoint", help="模型下载源（受限网络下指向镜像）")
    p.add_argument("--hf-offline", action="store_true",
                   help="模型已缓存时跳过联网校验（受限网络下必需，否则载入会挂死）")
    p.add_argument("--log-level", help="日志级别")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="把语料切块、编码并写入索引（增量）")
    pi.add_argument("--rebuild", action="store_true", help="先清空索引再全量重建")
    pi.add_argument("--limit-files", type=int, default=None, help="只处理前 N 个文件（冒烟用）")
    pi.add_argument("--batch-files", type=int, default=200, help="每批提交的文件数")
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("search", help="混合检索")
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int, default=None)
    ps.add_argument("--explain", action="store_true", help="打印每条的排序依据")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_search)

    pt = sub.add_parser("stats", help="索引体检")
    pt.add_argument("--verify", action="store_true", help="顺便做 FTS 一致性自检")
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(func=cmd_stats)

    pe = sub.add_parser("eval", help="在标注集上评测检索质量")
    pe.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    pe.add_argument("--baseline", default="data/eval/retrieval_baseline.json")
    pe.add_argument("--top-k", type=int, default=10)
    pe.add_argument("--gate", action="store_true", help="与冻结基线比较，不达标则退出码 2")
    pe.add_argument("--write-baseline", action="store_true", help="把本次结果写成新基线")
    pe.add_argument("--allow-corpus-drift", action="store_true",
                    help="索引规模变化时仍允许与基线比较（默认拒绝）")
    pe.add_argument("--no-ablation", action="store_true", help="跳过单路对照（更快）")
    pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    try:
        rc = int(args.func(args))
    except SystemExit as e:
        return int(e.code or 0)
    except FileNotFoundError as e:
        print(f"找不到文件：{e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    if rc == EXIT_OK:
        print(f"\n完成，用时 {time.perf_counter() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
