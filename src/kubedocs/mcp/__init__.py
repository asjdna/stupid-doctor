"""把本项目的工具暴露成 **MCP server**（借用参考项目 `01-agent-tool-mcp` 的做法）。

## 为什么值得做

我们自己的工具协议只服务于"本项目内的 agent"。MCP（Model Context Protocol）是
**跨客户端**的标准：把它接上，院内系统、Claude Desktop / Cursor 这类客户端、
或者别的 agent 都能直接调用我们已有的能力——而**不必再实现一遍工具定义**。

## 这一层刻意不做的事

**不另写一份工具定义**。`ToolRegistry` 里的 `ToolSpec` 已经是单一真相源
（name / description / when_not / params 的 JSON Schema / handler），MCP 只是把它
翻译成 `mcp.types.Tool`，再把调用翻译回去。两处各写一份定义，早晚会不一致——
而那正是本项目一直在删的"假的第二真相源"。

## 安全边界（这一层的重点）

**有副作用的工具默认不暴露**。挂号会真的占号，而 `BookParams` **没有确认字段**；
原样暴露出去，等于让任何 MCP 客户端跳过我们那条"未确认绝不取号"的性质
（见 `clinical/skills/booking.py` 的设计说明）。

所以要暴露副作用工具必须显式开 `--allow-side-effects`，**而且**：服务端会给这类工具
的入参 schema **注入一个必填的 `confirm: boolean`**，调用时该字段必须为 `true`，
否则拒绝执行并返回可读原因。这样"确认"这道关在 MCP 边界上仍然存在，
而不是被写进一句说明里靠调用方自觉。
"""

from __future__ import annotations

from kubedocs.mcp.server import build_server, main

__all__ = ["build_server", "main"]
