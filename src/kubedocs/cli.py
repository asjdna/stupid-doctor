"""命令行入口：`python -m kubedocs.cli <子命令>`。

所有能力（摄取 / 检索 / 索引体检 / 评测）都能从这里跑起来，HTTP 层只是它的薄包装，
所以"能跑通"这件事可以在没有容器、没有数据库、没有网络的情况下复现。

退出码约定（供脚本与闸门判断，而不是靠解析 stdout）：
    0 成功      1 参数/用户错误      2 评测闸门不通过      3 索引为空等前置条件不满足
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING

from kubedocs.config import Settings
from kubedocs.store import BACKENDS, Store, StoreError

if TYPE_CHECKING:
    from kubedocs.domains.base import DomainPack

EXIT_OK = 0
EXIT_USER = 1
EXIT_GATE_FAIL = 2
EXIT_PRECONDITION = 3


# --------------------------------------------------------------------------- 辅助
def make_settings(args: argparse.Namespace) -> Settings:
    """CLI 参数覆盖配置。直接构造 Settings 而不是改环境变量，避免污染进程环境。

    **域自报的默认值**（这是"双域"真正成立的地方）：`Settings` 的 `normalizer` /
    `embedding_model` / `embedding_dim` 是域相关的——clinic 域用 jieba 预分词 + 中文
    模型（512 维），k8s 域用原样归一化 + 英文模型（384 维）。这组取值由 `DomainPack`
    自报，本函数在用户**没有显式指定**时套用。

    判据是 `model_fields_set`：环境变量与 `.env` 写过的字段会出现在里面，所以
    `KUBEDOCS_EMBEDDING_MODEL=xxx` 仍然优先于域默认值——域提供的是**默认**，不是**强制**。
    """
    overrides: dict[str, object] = {}
    if getattr(args, "domain", None):
        overrides["domain"] = args.domain
    if getattr(args, "store", None):
        overrides["store"] = args.store
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
    settings = Settings(**overrides)  # type: ignore[arg-type]
    return apply_domain_defaults(settings)


def apply_domain_defaults(settings: Settings) -> Settings:
    """把域包自报的归一化/嵌入默认值套到 `Settings` 上（**不覆盖**显式取值）。

    只换索引目录是不够的：语料、归一化器与嵌入模型都得跟着域一起换。
    否则 `--domain clinic ingest` 会把英文文档按 identity 归一化、用英文模型编码，
    灌进 clinic 的索引。
    """
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry

    load_all()
    try:
        pack = DomainRegistry.get(settings.domain)
    except KeyError:
        # 未知域：这里不报错，交给真正用到域的地方（open_store / ingest）给出可读信息。
        return settings

    patch: dict[str, object] = {}
    explicit = settings.model_fields_set
    if "normalizer" not in explicit:
        patch["normalizer"] = pack.normalizer_name
    if "embedding_model" not in explicit:
        patch["embedding_model"] = pack.embedding_model
    if "embedding_dim" not in explicit:
        patch["embedding_dim"] = pack.embedding_dim
    return settings.model_copy(update=patch) if patch else settings


def _get_pack(settings: Settings) -> DomainPack:
    """按 `settings.domain` 取域包。调用方需先确保 `load_all()` 已注册。"""
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry

    load_all()
    try:
        return DomainRegistry.get(settings.domain)
    except KeyError as e:
        # 未注册的域是**用户错误**，不该以 traceback 形式抛出：
        # 调用方需要看到的是"有哪些域可选"。
        print(f"域错误：{e}", file=sys.stderr)
        raise SystemExit(EXIT_USER) from e


def open_store(settings: Settings, *, require_data: bool = True) -> Store:
    """按 `KUBEDOCS_STORE` 打开存储后端。

    这里**不做任何后端回退**：后端不可用（sqlite-vec 扩展加载失败、Postgres 连不上）
    就带着原因报错。一个"悄悄换回另一种实现"的检索服务，其指标变化将无法解释。
    """
    from kubedocs.store import from_settings

    store = from_settings(settings)
    if require_data and store.stats().n_chunks == 0:
        where = (
            str(settings.sqlite_path)
            if settings.store != "pgvector"
            else f"Postgres schema {settings.domain}（{settings.database_url}）"
        )
        print(f"索引为空：{where}\n先跑 `python -m kubedocs.cli ingest`", file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)
    return store


def _snippet(text: str, width: int = 160) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[:width] + "…"


def _index_label(settings: Settings) -> str:
    """索引位置的人类可读描述：不同后端存在的位置本来就不同。"""
    if settings.store == "pgvector":
        return f"{settings.database_url}（schema={settings.domain}）"
    return str(settings.sqlite_path)


# --------------------------------------------------------------------------- ingest
def cmd_ingest(args: argparse.Namespace) -> int:
    """把当前**域**的语料灌进索引。

    走 `DomainPack.build_index()` 而不是直接调通用 `ingest()`：
    语料从哪来、怎么切、用哪个归一化器与嵌入模型，都是域自己的事。
    内核（本函数）只负责打开存储、打印报告。
    """
    from kubedocs.embed import build_embedder

    settings = make_settings(args)
    pack = _get_pack(settings)
    store = open_store(settings, require_data=False)
    try:
        embedder = build_embedder(settings)
        print(f"域：{settings.domain}（归一化器 {settings.normalizer}）")
        print(
            f"嵌入器：{settings.embedding_provider} / {getattr(embedder, 'name', '?')} "
            f"({settings.embedding_dim} 维)"
        )
        print(f"语料：{pack.corpus_root(settings)}")
        print(f"索引：{settings.store} @ {_index_label(settings)}")
        report = pack.build_index(
            store,
            embedder,
            settings,
            rebuild=args.rebuild,
            limit_files=args.limit_files,
            batch_files=args.batch_files,
            progress=lambda m: print(m, flush=True),
        )
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

    settings = make_settings(args)
    pack = _get_pack(settings)
    store = open_store(settings)
    try:
        retriever = pack.retriever(store, build_embedder(settings), settings)
        res = retriever.search(args.query, top_k=args.top_k)
        if args.json:
            print(
                json.dumps(
                    {
                        "query": res.query,
                        "diagnostics": res.diagnostics,
                        "results": [
                            {
                                "chunk_id": r.chunk_id,
                                "doc_id": r.doc_id,
                                "title": r.title,
                                "breadcrumb": r.breadcrumb,
                                "source_url": r.source_url,
                                "score": r.score,
                                "ranks": r.ranks,
                                "rerank_score": r.rerank_score,
                                "body": r.body,
                            }
                            for r in res.results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return EXIT_OK
        if not res.results:
            print("（无结果）")
        for i, r in enumerate(res.results, 1):
            print(f"\n[{i}] {r.citation}")
            print(f"    {r.source_url}")
            print(f"    {_snippet(r.body, 220)}")
        diag = res.diagnostics
        print(
            f"\n--- 诊断：融合池 {diag.get('n_fused')}，两路重合 {diag.get('overlap')}，"
            f"耗时 {diag.get('total_ms')}ms"
        )
        for name, d in (diag.get("channels_detail") or {}).items():  # type: ignore[attr-defined]
            print(f"    {name}: {d}")
        for msg in diag.get("degraded") or []:  # type: ignore[attr-defined]
            print(f"    ⚠ {msg}")
        if args.explain:
            print("\n--- 排序解释")
            print(res.explain_all())
        return EXIT_OK
    finally:
        store.close()


# --------------------------------------------------------------------------- ask
def cmd_ask(args: argparse.Namespace) -> int:
    from kubedocs.embed import build_embedder
    from kubedocs.generate import build_answerer

    settings = make_settings(args)
    pack = _get_pack(settings)
    store = open_store(settings)
    try:
        retriever = pack.retriever(store, build_embedder(settings), settings)
        ans = build_answerer(retriever, settings).answer(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(ans.to_json(), ensure_ascii=False, indent=2))
            return EXIT_OK
        print(ans.render())
        if args.show_context and ans.retrieval is not None:
            print("\n--- 提供给模型的上下文 ---")
            from kubedocs.generate import build_context
            from kubedocs.store.base import ChunkHit

            ctx, _ = build_context(
                [
                    ChunkHit(
                        chunk_id=r.chunk_id,
                        doc_id=r.doc_id,
                        ordinal=r.ordinal,
                        title=r.title,
                        breadcrumb=r.breadcrumb,
                        source_url=r.source_url,
                        body=r.body,
                    )
                    for r in ans.retrieval.results
                ],
                max_chars=settings.max_context_chars,
            )
            print(ctx)
        return EXIT_OK
    finally:
        store.close()


# --------------------------------------------------------------------------- domains
def cmd_domains(args: argparse.Namespace) -> int:
    """列出可用域。内核不认具体域，只认 DomainPack 协议——这里是唯一能看到全景的地方。"""
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry

    load_all()
    items = DomainRegistry.describe()
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"已注册 {len(items)} 个域：\n")
    for item in items:
        sets = item["eval_sets"]
        sets_text = ", ".join(str(x) for x in sets) if isinstance(sets, list) else "（无）"
        print(f"  {item['name']}")
        print(f"    说明      : {item['description']}")
        print(f"    归一化器  : {item['normalizer']}")
        print(f"    嵌入模型  : {item['embedding_model']}（{item['embedding_dim']} 维）")
        print(f"    金标集    : {sets_text or '（无）'}")
    print("\n用 --domain <名字> 选择域；每个域的索引彼此独立（<index_dir>/<domain>/）。")
    print("域自报的语料/归一化/嵌入会在 ingest 时自动套用（显式配置优先）。")
    return EXIT_OK


# --------------------------------------------------------------------------- stats
def cmd_stats(args: argparse.Namespace) -> int:
    settings = make_settings(args)
    pack = _get_pack(settings)
    store = open_store(settings, require_data=False)
    try:
        st = store.stats()
        meta = store.meta_dump()
        payload = {
            "domain": settings.domain,
            "store_backend": settings.store,
            "normalizer": st.normalizer,
            # 语料位置由域自报：k8s 是一个目录，clinic 是一个 JSON 文件。
            "corpus": str(pack.corpus_root(settings)),
            "db": str(settings.sqlite_path),
            "vectors": str(settings.vector_path),
            "n_docs": st.n_docs,
            "n_chunks": st.n_chunks,
            "mean_chars": round(st.mean_chars, 1),
            "n_with_code": st.n_with_code,
            "embedding_model": st.embedding_model,
            "embedding_dim": st.embedding_dim,
            "indexed_at": st.indexed_at,
            "meta": meta,
        }
        if args.verify:
            payload["fts_integrity"] = store.verify_fts()
            # 向量侧体检：**各后端的方法名不一样**——`sqlite-vec` 叫
            # `verify_vec`，`pgvector` 叫 `verify_hnsw`，两个都要试，否则 pgvector
            # 那一路的向量体检永远是空的。注意**不要**预先塞一个 None：那会让
            # "这个后端没有向量体检"打印成「向量一致性 : None」，比不打印更误导。
            for name in ("verify_vec", "verify_hnsw"):
                checker = getattr(store, name, None)
                if callable(checker):
                    payload["vector_integrity"] = checker()
                    payload["vector_verify_method"] = name
                    break
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"域         : {settings.domain}（归一化器 {st.normalizer}）")
            print(f"语料       : {payload['corpus']}")
            print(f"存储后端   : {settings.store}")
            print(f"索引位置   : {_index_label(settings)}")
            print(f"文档/块数  : {st.n_docs} / {st.n_chunks}")
            print(f"块均长度   : {st.mean_chars:.1f} 字符")
            print(f"含代码块   : {st.n_with_code}（{100.0 * st.n_with_code / max(st.n_chunks, 1):.1f}%）")
            print(f"嵌入模型   : {st.embedding_model}（{st.embedding_dim} 维）")
            print(f"索引时间   : {st.indexed_at}")
            if args.verify:
                print(f"FTS 一致性 : {payload['fts_integrity']}")
                if "vector_integrity" in payload:
                    print(f"向量一致性 : {payload['vector_integrity']}")
                bad = payload["fts_integrity"] != "ok" or (
                    payload.get("vector_integrity") not in (None, "ok")
                )
                if bad:
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


# --------------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    """起 HTTP 服务（uvicorn）。

    `kubedocs.api.app` 是**延迟导入**的：没有装 FastAPI/uvicorn 或只想用 CLI 时，
    其它子命令不该因为一个 import 就全部不可用。
    """
    import uvicorn

    from kubedocs.api.app import create_app
    from kubedocs.observability import setup_tracing

    settings = make_settings(args)
    tracer = setup_tracing(settings)
    app = create_app(settings)
    host = args.host or "127.0.0.1"
    port = int(args.port or settings.api_port)
    print(f"服务地址：http://{host}:{port}")
    print(f"域：{settings.domain}    索引：{settings.sqlite_path}")
    print(
        f"可观测：日志级别 {settings.log_level}／指标 /metrics／"
        f"OTel {settings.otel_exporter if tracer else '未启用'}"
    )
    print("首页（前端演示）：/    接口文档：/docs")
    uvicorn.run(app, host=host, port=port, log_level=(args.log_level or "info").lower())
    return EXIT_OK


# --------------------------------------------------------------------------- mcp
def cmd_mcp(args: argparse.Namespace) -> int:
    """把工具经 MCP 协议（stdio）暴露给外部客户端，阻塞到客户端断开。

    **只是转调**：参数解析在 `kubedocs.mcp.server.main`，工具装配与协议细节也在那里——
    CLI 这边多写一份装配，就会多出一个"CLI 能列的工具有 MCP 没有"的机会。
    传 `settings` 而不是让 MCP 层自己再造一份配置：`--index-dir` / `--provider` /
    `--hf-offline` 这些全局参数已经在这里解析好了，域自报的嵌入模型与维度也由
    `make_settings` 套好（clinic 512 维中文模型、k8s_docs 384 维英文模型，不能混用）。

    注意 stdout：MCP 的 stdio 传输把它当协议通道，所以这条命令**不往 stdout 写任何人类可读文本**
    （启动信息走 MCP 层自己的日志、收尾计时在 `main` 里改道 stderr）。
    """
    from kubedocs.mcp.server import main as mcp_main

    return mcp_main(
        [],
        settings=make_settings(args),
        domain=args.domain or "",
        allow_side_effects=args.allow_side_effects,
    )


# --------------------------------------------------------------------------- 解析
def _add_globals(p: argparse.ArgumentParser, default: object) -> None:
    """全局参数（放在主解析器与每个子解析器上，两处都能写）。

    两边都要加：只挂在主解析器上时，`kubedocs ingest --index-dir X` 会被判为
    "无法识别的参数"。子解析器上的副本用 `default=SUPPRESS`，不写就**不设属性**，
    从而不会用 None 覆盖掉主解析器已经解析到的值。
    """
    p.add_argument("--index-dir", default=default, help="索引根目录（默认 data/index）")
    p.add_argument("--corpus-dir", default=default, help="语料目录（默认 data/raw/k8s-docs）")
    p.add_argument("--domain", default=default, help="域（默认 k8s_docs；可用 `domains` 子命令列出）")
    p.add_argument(
        "--store",
        choices=list(BACKENDS),
        default=default,
        help="存储后端（默认 sqlite-vec；sqlite=旧实现，pgvector=生产）",
    )
    p.add_argument(
        "--provider",
        choices=["bge", "hashing"],
        default=default,
        help="嵌入器（bge=真实向量，hashing=离线降级）",
    )
    p.add_argument("--hf-endpoint", default=default, help="模型下载源（受限网络下指向镜像）")
    p.add_argument(
        "--hf-offline",
        action="store_true",
        default=default,
        help="模型已缓存时跳过联网校验（受限网络下必需，否则载入会挂死）",
    )
    p.add_argument("--log-level", default=default, help="日志级别")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kubedocs",
        description="Kubernetes 官方文档检索问答服务（本地可跑，无需容器）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型流程：\n"
            "  1) python scripts/fetch_corpus.py          # 拉语料（一次就够）\n"
            "  2) python -m kubedocs.cli ingest           # 建索引\n"
            '  3) python -m kubedocs.cli search "..."     # 检索\n'
            '  4) python -m kubedocs.cli ask "..."        # 带引用回答（需 LLM 密钥）\n'
            "  5) python -m kubedocs.cli eval --gate      # 评测并卡闸门\n"
            "  6) python -m kubedocs.cli mcp --domain k8s_docs   # 以 MCP server 暴露工具（stdio）\n\n"
            "全局参数（--index-dir/--provider 等）放在子命令前后都可以。\n"
        ),
    )
    _add_globals(p, None)
    common = argparse.ArgumentParser(add_help=False)
    _add_globals(common, argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="把当前域的语料切块、编码并写入索引（增量）", parents=[common])
    pi.add_argument("--rebuild", action="store_true", help="先清空索引再全量重建")
    pi.add_argument("--limit-files", type=int, default=None, help="只处理前 N 个文件/卡片（冒烟用）")
    pi.add_argument(
        "--batch-files",
        type=int,
        default=60,
        help="k8s 域=每批提交的文件数；clinic 域=每次编码的卡片数（默认 60，越小越不容易因中断丢工作）",
    )
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("search", help="混合检索", parents=[common])
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int, default=None)
    ps.add_argument("--explain", action="store_true", help="打印每条的排序依据")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_search)

    pd = sub.add_parser("domains", help="列出可用域（双域平台：内核只认 DomainPack 协议）", parents=[common])
    pd.add_argument("--json", action="store_true")
    pd.set_defaults(func=cmd_domains)

    psrv = sub.add_parser("serve", help="起 HTTP 服务 + 前端演示页", parents=[common])
    psrv.add_argument("--host", default=None, help="监听地址（默认 127.0.0.1）")
    psrv.add_argument("--port", type=int, default=None, help="端口（默认取配置 api_port）")
    psrv.set_defaults(func=cmd_serve)

    pm = sub.add_parser(
        "mcp",
        help="把工具经 MCP 协议（stdio）暴露给 Claude Desktop / Cursor 等外部客户端",
        description=(
            "把 ToolRegistry 里的工具经 MCP 协议（stdio）暴露出去，长驻到客户端断开。\n"
            "\n"
            "两个域都支持：\n"
            "  --domain k8s_docs   检索 k8s 文档：search_docs + check_citations\n"
            "  --domain clinic     号源三件套 + search_docs + check_citations\n"
            "\n"
            "副作用工具（取号）默认**不暴露**；开 --allow-side-effects 后仍必须传 confirm=true。\n"
            "启动不加载嵌入模型：只有 search_docs 真被调用时才装配检索器（索引不可用时\n"
            "返回可读的 isError，而不是把服务打挂）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    pm.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="同时暴露有副作用的工具（**调用时仍必须显式 confirm=true**）",
    )
    pm.set_defaults(func=cmd_mcp)

    pt = sub.add_parser("stats", help="索引体检", parents=[common])
    pt.add_argument("--verify", action="store_true", help="顺便做 FTS 一致性自检")
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(func=cmd_stats)

    pa = sub.add_parser("ask", help="带引用回答（需要 LLM 密钥；无密钥时只返回检索结果）", parents=[common])
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int, default=None)
    pa.add_argument("--show-context", action="store_true", help="打印实际送给模型的上下文")
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_ask)

    pe = sub.add_parser("eval", help="在标注集上评测检索质量", parents=[common])
    pe.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    pe.add_argument("--baseline", default="data/eval/retrieval_baseline.json")
    pe.add_argument("--top-k", type=int, default=10)
    pe.add_argument("--gate", action="store_true", help="与冻结基线比较，不达标则退出码 2")
    pe.add_argument("--write-baseline", action="store_true", help="把本次结果写成新基线")
    pe.add_argument(
        "--allow-corpus-drift", action="store_true", help="索引规模变化时仍允许与基线比较（默认拒绝）"
    )
    pe.add_argument("--no-ablation", action="store_true", help="跳过单路对照（更快）")
    pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    # 必须在任何第三方库（LiteLLM / huggingface_hub）读到 os.environ **之前**导出 .env，
    # 否则 .env 里的密钥到不了模型调用（原因见 `config.load_env_file`）。
    from kubedocs.config import load_env_file
    from kubedocs.observability import configure_logging

    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    # `--log-level` / `KUBEDOCS_LOG_LEVEL` 在这里生效：configure_logging() 设的是
    # 本项目自己的 root logger 级别（uvicorn 另有自己的级别参数，由 cmd_serve 传入）。
    configure_logging(make_settings(args))
    t0 = time.perf_counter()
    try:
        rc = int(args.func(args))
    except SystemExit as e:
        # `SystemExit.code` 可能是**字符串**（例如域没注册时由内核给出的那句人话）：
        # 直接 `int()` 会在这里抛出一个看不懂的 ValueError，把真正的原因盖掉。
        if e.code is None:
            return EXIT_OK
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return EXIT_USER
    except FileNotFoundError as e:
        print(f"找不到文件：{e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except StoreError as e:
        # 存储层的可预期错误（维度/模型/后端不一致、扩展缺失、库连不上）是**用户错误**，
        # 不该以 traceback 形式抛出：调用方需要看到的是"该跑哪条命令"。
        print(f"存储层错误：{e}", file=sys.stderr)
        return EXIT_USER
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    if rc == EXIT_OK:
        # `--json` 的契约是"stdout 上只有一份机器可读的 JSON"，所以这条人类可读的
        # 收尾计时在 `--json` 时改道 stderr。否则 `json.loads` 会因为多出这一行而
        # 整体失败，机器读不到任何东西；search / ask / stats 的 `--json` 分支同理。
        # `mcp` 同样改道：那条命令的 stdout 是 MCP 协议通道，多一行文本就会让客户端解析失败。
        quiet = getattr(args, "json", False) or getattr(args, "cmd", "") == "mcp"
        stream = sys.stderr if quiet else sys.stdout
        print(f"\n完成，用时 {time.perf_counter() - t0:.1f}s", file=stream)
    return rc


if __name__ == "__main__":
    sys.exit(main())
