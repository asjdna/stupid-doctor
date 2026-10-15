"""MCP 适配层：把 `ToolRegistry` 里注册的工具暴露给外部 MCP 客户端。

MCP（Model Context Protocol）是跨客户端标准，接上之后 Claude Desktop / Cursor、院内系统或别的
agent 都能直接调用本项目已有的能力，而不必再实现一遍工具定义。工具定义只有一个来源：
`ToolRegistry` 里的 `ToolSpec`（name / description / when_not / params 的 JSON Schema / handler）；
本层只把它翻译成 `mcp.types.Tool`，并把调用翻译回 `ToolSpec.handler`。

## 安全边界

- **有副作用的工具默认不暴露**（取号会真实占号）。必须显式开 `--allow-side-effects`；开启后服务端
  会给这类工具的入参 schema **注入必填的 `confirm: boolean`**，调用时该字段必须为 `true`，否则拒绝
  执行并返回可读原因。`BookParams` 里没有确认字段——agent 内部走图上的 `interrupt`，MCP 这边用
  schema 强制确认，两条路径都不存在"默认已确认"。
- **患者身份不经 MCP 传入**：`BookParams` 没有 `patient_id`，pydantic 校验也会丢掉多余入参，所以
  MCP 侧取号一律记为匿名 `"anon"`。需要绑定真实患者的调用走 `kubedocs.api`。

## 启用与扩展

目前暴露 `clinic` 域的工具：`list_departments` / `get_availability` / `book_appointment`（副作用，
默认隐藏）+ `check_citations`。启动方式：

    python -m kubedocs.mcp.server --domain clinic [--allow-side-effects]

选项见 `kubedocs.mcp.server`。新增一个域 = 在 `server._wire_tools` 里补一条装配分支；
域里新注册的工具会自动出现在 MCP 清单中，本层无需改动。
"""

from __future__ import annotations

from kubedocs.mcp.server import build_server, main

__all__ = ["build_server", "main"]
