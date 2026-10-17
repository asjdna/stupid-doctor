"""MCP server：把 `ToolRegistry` 里的工具按 MCP 协议经 stdio 暴露出去（也是 `--help` 的说明文本）。

工具定义全部来自 `ToolRegistry`，本模块不含任何工具的业务描述。四条实现约束：

1. **副作用默认不外露**：未开 `--allow-side-effects` 时过滤掉 `side_effect=True` 的工具；
   开启后仍要入参传 `confirm=true` 才执行。
2. **参数校验走 pydantic**：与 agent 内部调用同一条路径，MCP 不是绕过校验的后门。
3. **两个域都装配真实工具**：`register_domain_tools` 填充空的 `ToolRegistry`——clinic 得到
   号源四个只读工具（`list_departments` / `get_availability` / `booking_window` / `my_appointments`）
   + `search_docs` + `check_citations`，k8s_docs 得到 `search_docs` + `check_citations`。
   列出来的工具就是能跑的工具，不存在"注册了但没有执行体"的空壳。
4. **检索器由后台线程预热**：`build_server` / `tools/list` 本身不加载嵌入模型（bge 首次要几十秒，
   装在这里会让客户端握手超时）；默认在握手之后起一个线程 `warm()` 一次（`--no-warmup` 关掉），
   所以"第一次 `search_docs` 不必现付模型加载"。装配失败返回 `isError` + 一句"先跑 ingest"，
   不是 traceback。

用法：

    kubedocs mcp --domain k8s_docs                     # 推荐：CLI 子命令（一条命令就能用）
    kubedocs mcp --domain clinic --allow-side-effects
    python -m kubedocs.mcp.server --domain k8s_docs     # 等价的模块入口

MCP 客户端（stdio）配置大致是：

    {"mcpServers": {"kubedocs": {
        "command": "F:/86189/Documents/kubedocs/.venv/Scripts/python.exe",
        "args": ["-m", "kubedocs", "mcp", "--domain", "k8s_docs"]}}}
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from kubedocs.agent.tools import ToolError, ToolRegistry

if TYPE_CHECKING:  # 只为标注：运行时按需导入，避免"导入 MCP 层就拖进配置/模型"
    from collections.abc import Callable

    from kubedocs.config import Settings

logger = logging.getLogger(__name__)

_SIDE_EFFECT_CONFIRM_FIELD = "confirm"
"""副作用工具必须显式给出的确认字段名，**不在** `ToolSpec.params` 里。

确认是 MCP 边界特有的要求（agent 内部走图上的 `interrupt`）；塞进 `BookParams` 会改变那条已被测试覆盖的调用路径。
"""


# ===========================================================================
# 检索器：惰性装配，失败变成可读的 isError
# ===========================================================================
def _assemble_retriever(settings: Settings) -> Any:
    """按域装配真检索器：store → 嵌入器 → 域包自己的 `retriever()`。

    这条路径与 CLI `search` / `serve` 完全一致（同一个 `from_settings` + `build_embedder` +
    `DomainPack.retriever`），所以 MCP 检索到的与 CLI/HTTP 检索到的是同一份东西，
    不会出现"从 MCP 问到的答案对不上"。

    只负责装配、不负责措辞：异常原样抛出，由 `_LazyRetriever` 翻译成一句人话。
    这样单测可以 monkeypatch 本函数制造"装配失败"，不必真去准备一个坏索引。
    """
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry
    from kubedocs.embed import build_embedder
    from kubedocs.store import from_settings

    load_all()
    store = from_settings(settings)
    try:
        n_chunks = store.stats().n_chunks
        if n_chunks == 0:
            where = (
                f"Postgres schema {settings.domain}（{settings.database_url}）"
                if settings.store == "pgvector"
                else str(settings.sqlite_path)
            )
            raise RuntimeError(f"索引为空（0 块）：{where}")
        embedder = build_embedder(settings)
        return DomainRegistry.get(settings.domain).retriever(store, embedder, settings)
    except BaseException:
        # 失败路径也要关连接：Windows 上不关会锁住索引目录（删不掉、换不了）。
        store.close()
        raise


class _LazyRetriever:
    """惰性检索器：第一次 `search()` 才装配 store + embedder + retriever。

    **为什么必须惰性**：装配里最贵的一步是加载 bge（首次几十秒）。`build_server()` 在 MCP
    进程启动时就会跑、`tools/list` 每次握手也要跑，那时装配会让客户端以为服务卡死；而列工具
    只需要知道"有 `search_docs` 这个工具"，不需要真向量。

    **为什么把失败挡在这里**：索引没建、维度不符、模型没缓存都是可预期的部署状态，不该以
    traceback 形式回给客户端。异常转成 `ToolError`（一句话 + 该跑哪条命令），`call_tool`
    会把它变成 `isError=True`。失败会被记住，避免每次调用都重试一次几十秒的加载。
    """

    def __init__(self, settings: Settings, factory: Callable[[Settings], Any]) -> None:
        self.settings = settings
        self._factory = factory
        self._retriever: Any | None = None
        self._error = ""
        self._ready = False
        self._lock = threading.Lock()

    def search(self, query: str, top_k: int = 5) -> Any:
        """`HybridRetriever.search` 的同形接口（`make_search_tool` 只按鸭子类型用它）。"""
        retriever = self._ensure()
        self._ensure_ready(retriever)
        return retriever.search(query, top_k=top_k)

    def warm(self) -> None:
        """把"第一次调用才付的那笔账"提前付掉：**真的编码一次**。

        `_ensure()` 只是装配——`BgeEmbedder` 的权重是惰性加载的，所以装配只要几秒。
        真正贵的是第一次 `encode()`：它才去导入 `sentence_transformers` / torch 并读权重。
        """
        self._ensure_ready(self._ensure())

    def _ensure_ready(self, retriever: Any) -> None:
        """保证"第一次编码"在**锁内只发生一次**。

        这一条是被真实故障逼出来的：预热线程与工具调用的工作线程各自去做第一次
        `encode()`（也就是各自去导入 ML 栈、各自去读权重），两个线程卡在彼此身上，
        实测超过 10 分钟没有任何一方返回——客户端看到的是"工具调用没有响应"。
        反过来，单独在任一线程里做这件事都只要约 40 秒（已实测）。

        代价只有第一次调用要排队；`_ready` 之后完全不再加锁，不影响并发检索。
        """
        if self._ready:
            return
        with self._lock:
            if self._ready:  # 双检：等锁期间别人可能已经做完了
                return
            retriever.search("warmup", top_k=1)
            self._ready = True

    def _ensure(self) -> Any:
        with self._lock:  # 工具调用可能来自不同线程（mcp 的会话任务），只装配一次
            if self._retriever is not None:
                return self._retriever
            if self._error:
                raise ToolError(self._error)
            try:
                self._retriever = self._factory(self.settings)
            except Exception as e:  # noqa: BLE001 - 装配失败要变成可读的 isError，而不是崩服务
                self._error = (
                    f"索引不可用（域 {self.settings.domain}）：{type(e).__name__}: {e}；"
                    f"先跑 `python -m kubedocs.cli ingest --domain {self.settings.domain}` 建索引"
                )
                logger.warning("%s", self._error)
                raise ToolError(self._error) from e
            return self._retriever


# ===========================================================================
# 装配
# ===========================================================================
def _resolve_settings(domain: str, settings: Settings | None) -> Settings:
    """定出本次装配用的 `Settings`，并校验域是已注册的。

    **域自报的默认值必须跟着域走**：clinic 是 512 维中文模型、k8s_docs 是 384 维英文模型，
    混用会直接撞 `StoreError`（向量维度不符）。所以这里统一过一遍
    `cli.apply_domain_defaults`（它**不覆盖**显式配置：环境变量/`.env` 写了模型的仍然优先）。
    """
    from kubedocs.cli import apply_domain_defaults

    if settings is None:
        from kubedocs.cli import make_settings

        # `make_settings` 只读属性，缺的属性走 getattr 默认值，所以一个光秃秃的
        # Namespace 就够（不必在这里复制一份 CLI 的参数定义）。
        settings = make_settings(argparse.Namespace(domain=domain or None))
    elif domain:
        settings = settings.model_copy(update={"domain": domain})
    settings = apply_domain_defaults(settings)

    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry

    load_all()
    known = DomainRegistry.names()
    if settings.domain not in known:
        # 未知域必须响亮失败：静默按某个默认域装配，会让客户端以为"查的就是我指定的那个域"，
        # 拿到的却是另一份语料的结果——这种错误比启动失败难发现得多。
        raise SystemExit(f"未知的域 {settings.domain!r}；可用域：{', '.join(known)}（用 --domain 指定）")
    return settings


def _wire_tools(domain: str, *, settings: Settings, booking_service: Any = None) -> _LazyRetriever:
    """按域装配工具：`ToolRegistry` 初始为空，工具只在装配时才注册进来。

    与 HTTP 层 `create_app` 同一个装配入口；漏掉它，MCP 会列出空清单——看起来像"这个项目没有能力"。
    检索器位置传的是 `_LazyRetriever`：`search_docs` 这一刻就出现在清单里，但一个模型权重都没加载。

    `booking_service` 是给测试用的注入缝（默认 `None` = 按 settings 装配真号源）。
    """
    from kubedocs.agent.bootstrap import register_domain_tools, unregister_domain
    from kubedocs.domains import load_all

    load_all()
    # 注册表按**工具名**索引，不支持同名多域（`search_docs` / `check_citations` 是两个域共用的名字）。
    # MCP server 一个进程只服务一个域，所以装配前把**别的域**留下的注册摘掉；否则同一进程里
    # 先装配 k8s_docs 再装配 clinic 会直接撞"工具已注册"（测试按文件顺序跑时就会踩到）。
    for item in ToolRegistry.describe():
        other = str(item.get("domain") or "")
        if other and other != domain:
            unregister_domain(other)

    retriever = _LazyRetriever(settings, _assemble_retriever)
    if domain == "clinic":
        from kubedocs.domains.clinic.assembly import build_booking_service

        # 与 HTTP 三条路径（`/consult`、`/chat`）走**同一个构造点**：Postgres 号源优先，
        # 连不上才降级进程内。各写一份的话，MCP 里查到的号源与网页面上的会是两套数据。
        # `service` 是注入缝：测试用进程内号源，免得一行单测去消耗真实号源库存。
        service = booking_service if booking_service is not None else build_booking_service(settings)
        register_domain_tools(domain, service=service, retriever=retriever)
    else:
        # k8s_docs（以及未来任何只做检索的域）：检索 + 引用核对。
        register_domain_tools(domain, retriever=retriever)
    return retriever


def _start_warmup(retriever: _LazyRetriever, domain: str) -> threading.Thread:
    """在后台线程里把检索器装配好（最贵的一步：导入 ML 栈 + 加载嵌入模型）。

    **为什么必须预热**：第一次 `search_docs` 会在 MCP 的**工作线程**里现付这套导入与模型加载。
    实测在这台机器上，同样的导入在主线程只要约 45 秒，但放进 MCP 的工作线程后
    10 分钟仍未返回（faulthandler 显示它一直停在 `scipy` → `sklearn` → `sentence_transformers`
    的导入链里），客户端看到的就是"工具调用没有响应"。把它挪到启动后的独立线程，
    客户端体验到的是"启动稍慢"，而不是"调用卡死"；预热失败也不改行为——真正的错误
    仍然在调用时以可读的 `isError` 回给客户端。
    """

    def work() -> None:
        started = time.perf_counter()
        logger.info("预热检索器（域 %s）：导入依赖 + 加载嵌入模型，首次约 1 分钟…", domain)
        try:
            retriever.warm()  # 装配 + 真编码一次；只装配不够（权重是惰性加载的）
        except Exception as e:  # noqa: BLE001 - 预热失败不该影响服务；调用时会再试并如实回报
            logger.warning("预热失败（不影响列工具，调用时会重试）：%s: %s", type(e).__name__, e)
        else:
            logger.info("预热完成，用时 %.1fs", time.perf_counter() - started)

    thread = threading.Thread(target=work, name="kubedocs-mcp-warmup", daemon=True)
    thread.start()
    return thread


def _tool_schema(spec: Any, *, allow_side_effects: bool) -> dict[str, Any]:
    schema: dict[str, Any] = dict(spec.params.model_json_schema())
    if spec.side_effect and allow_side_effects:
        props = dict(schema.get("properties") or {})
        props[_SIDE_EFFECT_CONFIRM_FIELD] = {
            "type": "boolean",
            "description": (
                "必须为 true 才会真正执行。这个工具**有副作用**，调用方必须先取得当事人明确同意。"
            ),
        }
        schema["properties"] = props
        schema["required"] = sorted({*(schema.get("required") or []), _SIDE_EFFECT_CONFIRM_FIELD})
    return schema


def build_server(
    domain: str = "",
    *,
    allow_side_effects: bool = False,
    settings: Settings | None = None,
    warmup: bool = False,
    patient_id: str = "anon",
    booking_service: Any = None,
) -> Any:
    """造一个 MCP server：按 `domain` 装配工具，并挂好 `list_tools` / `call_tool`。

    `domain` 留空表示用 `settings.domain`（未传 `settings` 时即配置里的当前域）。
    `allow_side_effects=False` 时清单不含 `side_effect=True` 的工具，直接调用也会被拒。
    **本函数不打开索引、不加载嵌入模型**（检索器惰性，见 `_LazyRetriever`）。
    `warmup=True` 时在启动后另起线程预热检索器（见 `_start_warmup`）；默认关，
    这样单测与"只列工具"的客户端不会为一个模型付代价。

    `patient_id` 是**这个 MCP 会话代表谁**（`--patient-id`，默认 `anon`）。经 `context=`
    注入到工具执行体，**不进工具 schema**——客户端改不了它，模型也伪造不了。
    它只影响按人查询的工具（`my_appointments`）与取号记账：MCP 是本地/内网运维通道，
    没有"登录用户"这个概念，所以身份必须由启动方在命令行上声明。

    `booking_service` 是注入缝：默认按 settings 装配真号源（Postgres 优先）。测试要注入
    进程内号源——否则一行单测会真的占掉真实号源的库存，几天后那条用例会因为"某个时段
    挂满了"而红，而它验的其实只是协议。
    """
    from mcp.server import Server
    from mcp.types import CallToolResult, TextContent, Tool

    cfg = _resolve_settings(domain, settings)
    domain = cfg.domain
    retriever = _wire_tools(domain, settings=cfg, booking_service=booking_service)
    if warmup:
        _start_warmup(retriever, domain)
    server = Server("kubedocs")

    def _visible() -> list[Any]:
        specs = ToolRegistry.for_domain(domain)
        if allow_side_effects:
            return specs
        return [s for s in specs if not s.side_effect]

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [
            Tool(
                name=s.name,
                description=f"{s.description}\n不适用：{s.when_not}",
                inputSchema=_tool_schema(s, allow_side_effects=allow_side_effects),
            )
            for s in _visible()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        args = dict(arguments or {})

        try:
            spec = ToolRegistry.get(name)
        except Exception as e:  # noqa: BLE001 - 工具不存在要变成可读的失败，而不是崩服务
            return CallToolResult(
                content=[TextContent(type="text", text=f"没有名为 {name!r} 的工具：{e}")],
                isError=True,
            )

        if spec.side_effect and not allow_side_effects:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(f"工具 {name!r} 有副作用，本服务未开启 --allow-side-effects，拒绝执行。"),
                    )
                ],
                isError=True,
            )

        if spec.side_effect:
            confirmed = args.pop(_SIDE_EFFECT_CONFIRM_FIELD, None)
            if confirmed is not True:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"拒绝执行 {name!r}：这是有副作用的操作，"
                                f"必须先取得当事人明确同意并传 {_SIDE_EFFECT_CONFIRM_FIELD}=true。"
                            ),
                        )
                    ],
                    isError=True,
                )

        if spec.handler is None:
            return CallToolResult(
                content=[TextContent(type="text", text=f"工具 {name!r} 只声明未实现（无 handler）")],
                isError=True,
            )

        try:
            validated = spec.params.model_validate(args)
            # 执行体的形状有两种：`rag_tools` 是 `handler(params)`（收一个 pydantic 模型），
            # `clinic.tools` 是 `handler(**kwargs)`。**复用 agent 工具循环里那一次分派**，
            # 而不是在这里再猜一遍签名：MCP 自己猜错的结果是"工具列得出来、一调就
            # TypeError"——那才是真正"无谓占用空间"的形态。
            from kubedocs.agent.tool_loop import _invoke

            # `context` 是**启动方**声明的身份（`--patient-id`），不是客户端传参：
            # `my_appointments` 与取号记账都靠它，而模型/客户端无法改写。
            result = _invoke(spec, validated, {"patient_id": patient_id})
        except Exception as e:  # noqa: BLE001 - 工具执行失败要结构化返回，不许把栈抛给客户端
            # 索引不可用、检索失败、号源已满都走这里：`isError=True` + 一句人话。
            logger.warning("MCP 工具 %s 执行失败：%s: %s", name, type(e).__name__, e)
            return CallToolResult(
                content=[TextContent(type="text", text=f"{type(e).__name__}: {e}")], isError=True
            )

        import json

        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return CallToolResult(content=[TextContent(type="text", text=text)])

    return server


async def _serve(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def _pin_offline_env(settings: Settings) -> None:
    """把"离线加载"钉到环境变量上，并在日志里说明当前模式。

    **为什么不能只依赖 settings**：`SentenceTransformer` 是在**第一次编码**时才构造的，
    而"没开离线"在这类受限网络下的表现不是报错，而是**连 huggingface.co 挂死**
    （实测：不带 `HF_HUB_OFFLINE` 时加载模型 7 分钟没有返回；带上只要 1.4 秒）。
    MCP 子进程的第一次工具调用正好踩在这条路上，客户端看到的就是"调用没有响应"。
    把开关在服务启动时就写进环境变量，比指望下游每个组件都去读一遍 settings 可靠。
    """
    import os

    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    if settings.hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logger.info(
        "模型加载模式：%s（HF_HUB_OFFLINE=%s）",
        "离线" if settings.hf_offline else "允许联网",
        os.environ.get("HF_HUB_OFFLINE", "未设"),
    )


def _build_parser() -> argparse.ArgumentParser:
    """模块入口自己的参数解析器。

    全局参数（`--index-dir` / `--provider` / `--hf-offline` …）直接复用 `kubedocs.cli` 的那一份
    定义：两处各写一遍迟早会漂移，而 `--hf-offline` 这类开关漂移的后果是"离线机器上挂死"。
    """
    from kubedocs.cli import _add_globals

    parser = argparse.ArgumentParser(
        prog="kubedocs mcp",
        description=(
            "把 ToolRegistry 里的工具经 MCP 协议（stdio）暴露给外部客户端。"
            "支持 k8s_docs 与 clinic 两个域；启动后台预热检索器（不拖慢握手，"
            "但让第一次 search_docs 不必现付模型加载）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="同时暴露有副作用的工具（**调用时仍必须显式 confirm=true**）",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="不在启动后预热检索器（第一次 search_docs 会现付导入与模型加载，通常 1 分钟以上）",
    )
    parser.add_argument(
        "--patient-id",
        default="anon",
        help="这个 MCP 会话代表哪位就诊人（影响「我的挂号」与取号记账；默认 anon）",
    )
    _add_globals(parser, None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    domain: str = "",
    allow_side_effects: bool = False,
    no_warmup: bool = False,
    patient_id: str = "",
) -> int:
    """MCP server 入口：解析参数 → 配日志 → 起 stdio 服务（阻塞到客户端断开）。

    两种调用方式：

    - 命令行：`python -m kubedocs.mcp.server [--domain …]`，参数由这里解析（`argv=None`）；
    - 程序内：CLI 的 `mcp` 子命令传 `argv=[]` + 已经定好的 `settings`
      （`--index-dir` / `--provider` / `--hf-offline` 由 CLI 解析，域自报的嵌入模型与维度也在
      `cli.make_settings` 里套好了，这里不再重复构造，避免两份配置漂移）。
    """
    if argv is None and settings is not None:
        # 程序内调用没给 argv：参数已经由调用方解析好了，绝不能再去读 sys.argv——
        # 那里面是外层 CLI 自己的参数（`mcp --domain x`），会被当成 MCP server 的参数而报错。
        argv = []
    args = _build_parser().parse_args(argv)
    target = domain or args.domain or ""
    allow = bool(allow_side_effects or args.allow_side_effects)
    who = patient_id or args.patient_id or "anon"

    from kubedocs.config import load_env_file
    from kubedocs.observability import configure_logging

    load_env_file()
    if settings is None:
        from kubedocs.cli import make_settings

        args.domain = target or None
        settings = make_settings(args)
    configure_logging(settings)
    _pin_offline_env(settings)

    server = build_server(
        target,
        allow_side_effects=allow,
        settings=settings,
        warmup=not (args.no_warmup or no_warmup),
        patient_id=who,
    )
    logger.info(
        "MCP server 启动（域 %s，就诊人 %s，副作用工具 %s；传输 stdio）",
        settings.domain,
        who,
        "开启" if allow else "未暴露",
    )
    asyncio.run(_serve(server))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
