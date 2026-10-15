"""MCP server：把 `ToolRegistry` 里的工具按 MCP 协议经 stdio 暴露出去（也是 `--help` 的说明文本）。

工具定义全部来自 `ToolRegistry`，本模块不含任何工具的业务描述。三条实现约束：

1. **副作用默认不外露**：未开 `--allow-side-effects` 时过滤掉 `side_effect=True` 的工具；开启后仍需入参传 `confirm=true`。
2. **参数校验走 pydantic**：与 agent 内部调用同一条路径，MCP 不是绕过校验的后门。
3. **工具由域装配**：`register_domain_tools` 填充空的 `ToolRegistry`。目前只支持 `clinic` 域，暴露
   `list_departments` / `get_availability` / `book_appointment`（副作用）+ `check_citations`；MCP 层不持有真实检索器。

用法：

    python -m kubedocs.mcp.server                      # clinic 域，只读工具
    python -m kubedocs.mcp.server --domain clinic --allow-side-effects
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from kubedocs.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

_SIDE_EFFECT_CONFIRM_FIELD = "confirm"
"""副作用工具必须显式给出的确认字段名，**不在** `ToolSpec.params` 里。

确认是 MCP 边界特有的要求（agent 内部走图上的 `interrupt`）；塞进 `BookParams` 会改变那条已被测试覆盖的调用路径。
"""


def _wire_tools(domain: str) -> None:
    """按域装配工具：`ToolRegistry` 初始为空，工具只在装配时才注册进来。

    与 HTTP 层 `create_app` 同一个装配入口；漏掉它，MCP 会列出空清单——看起来像"这个项目没有能力"。
    """
    from kubedocs.agent.bootstrap import register_domain_tools
    from kubedocs.domains import load_all

    load_all()
    if domain == "clinic":
        from kubedocs.domains.clinic.booking import BookingService
        from kubedocs.domains.clinic.mock import MockAppointmentProvider

        register_domain_tools("clinic", service=BookingService(MockAppointmentProvider()))
    else:
        # `k8s_docs` 的检索工具需要真实检索器（store + embedder），MCP 层不持有它们。
        # 传 None 注册出来的 search 工具调不通，所以直接拒绝，而不是注册一个一调用就报错的空壳。
        raise SystemExit(
            f"域 {domain!r} 的检索工具需要检索器，MCP server 暂不持有；"
            "目前只支持 clinic（号源三件套 + 引用核对）。"
        )


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


def build_server(domain: str = "clinic", *, allow_side_effects: bool = False) -> Any:
    """造一个 MCP server：按 `domain` 装配工具，并挂好 `list_tools` / `call_tool`。
    `allow_side_effects=False` 时清单不含 `side_effect=True` 的工具，直接调用也会被拒。
    """
    from mcp.server import Server
    from mcp.types import CallToolResult, TextContent, Tool

    _wire_tools(domain)
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
            result = spec.handler(**validated.model_dump())
        except Exception as e:  # noqa: BLE001 - 工具执行失败要结构化返回，不许把栈抛给客户端
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--domain", default="clinic", help="暴露哪个域的工具（默认 clinic）")
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="同时暴露有副作用的工具（**调用时仍必须显式 confirm=true**）",
    )
    args = parser.parse_args(argv)

    from kubedocs.config import get_settings, load_env_file
    from kubedocs.observability import configure_logging

    load_env_file()
    configure_logging(get_settings())

    server = build_server(args.domain, allow_side_effects=args.allow_side_effects)
    logger.info(
        "MCP server 启动（域 %s，副作用工具 %s）",
        args.domain,
        "开启" if args.allow_side_effects else "未暴露",
    )
    asyncio.run(_serve(server))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
