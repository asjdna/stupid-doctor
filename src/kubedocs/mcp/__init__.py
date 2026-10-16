"""MCP 适配层：把 `ToolRegistry` 里注册的工具暴露给外部 MCP 客户端。

MCP（Model Context Protocol）是跨客户端标准，接上之后 Claude Desktop / Cursor、院内系统或别的
agent 都能直接调用本项目已有的能力，而不必再实现一遍工具定义。工具定义只有一个来源：
`ToolRegistry` 里的 `ToolSpec`（name / description / when_not / params 的 JSON Schema / handler）；
本层只把它翻译成 `mcp.types.Tool`，并把调用翻译回 `ToolSpec.handler`。

## 怎么用

    kubedocs mcp --domain k8s_docs                 # 一条命令（CLI 子命令，推荐）
    kubedocs mcp --domain clinic --allow-side-effects
    python -m kubedocs.mcp.server --domain k8s_docs # 等价的模块入口

两个域都支持：`k8s_docs` 得到 `search_docs` + `check_citations`，`clinic` 再加上号源三件套。
**检索器惰性装配**：启动与 `tools/list` 不加载嵌入模型，只有 `search_docs` 真被调用时才建
store + embedder + retriever；索引不可用时返回可读的 `isError`（"先跑 ingest"）而不是抛栈。

## 安全边界

- **有副作用的工具默认不暴露**（取号会真实占号）。必须显式开 `--allow-side-effects`；开启后服务端
  会给这类工具的入参 schema **注入必填的 `confirm: boolean`**，调用时该字段必须为 `true`，否则拒绝
  执行并返回可读原因。`BookParams` 里没有确认字段——agent 内部走图上的 `interrupt`，MCP 这边用
  schema 强制确认，两条路径都不存在"默认已确认"。
- **患者身份不经 MCP 传入**：`BookParams` 没有 `patient_id`，pydantic 校验也会丢掉多余入参，所以
  MCP 侧取号一律记为匿名 `"anon"`。需要绑定真实患者的调用走 `kubedocs.api`。

## 扩展

新增一个域 = 在 `server._wire_tools` 里补一条装配分支（并让域包声明自己的嵌入模型与检索器）；
域里新注册的工具会自动出现在 MCP 清单中，本层无需改动。选项见 `kubedocs.mcp.server`。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 只为标注：见下面的惰性再导出
    from kubedocs.mcp.server import build_server, main

__all__ = ["build_server", "main"]


def __getattr__(name: str) -> Any:
    """惰性再导出 `build_server` / `main`（PEP 562）。

    不在包导入时就 import 子模块：`python -m kubedocs.mcp.server` 会先导入包、再由 runpy
    执行子模块，提前 import 会让解释器打出"found in sys.modules after import of package"
    的 RuntimeWarning——这条模块入口是文档里写明的用法，不该自带一行警告。
    """
    if name in __all__:
        from kubedocs.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
