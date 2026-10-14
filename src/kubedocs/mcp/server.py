"""MCP server：把 `ToolRegistry` 里的工具按 MCP 协议暴露出去。

设计取舍见包文档（`kubedocs/mcp/__init__.py`）。这里只强调三条实现约束：

1. **单一真相源**：工具定义来自 `ToolRegistry`，本文件不含任何工具的业务描述；
2. **副作用默认不外露**，且开启后强制 `confirm=true`（见 `_SIDE_EFFECT_CONFIRM_FIELD`）；
3. **参数校验走 pydantic**：与 agent 内部调用同一条路径，MCP 不该是一条绕过校验的后门。

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
"""副作用工具被要求显式给出的确认字段。

它**不在** `ToolSpec.params` 里——因为它是 MCP 边界特有的要求（agent 内部走的是
图上的 `interrupt` 确认，而不是这个字段）。加在这里而不是加进 `BookParams`，
是为了不改变 agent 内部那条已经被测试覆盖的调用路径。
"""


def _wire_tools(domain: str) -> None:
    """按域注册工具（与 `create_app` 用的是同一个 `register_domain_tools`）。

    为什么必须做这一步：`ToolRegistry` 是**空的**，工具由域包在装配时注册。
    漏了它，MCP 会列出一个空清单——而"空清单"看起来像"这个项目没有能力"，
    是个安静且误导的失败。
    """
    from kubedocs.agent.bootstrap import register_domain_tools
    from kubedocs.domains import load_all

    load_all()
    if domain == "clinic":
        from kubedocs.domains.clinic.booking import BookingService
        from kubedocs.domains.clinic.mock import MockAppointmentProvider

        register_domain_tools("clinic", service=BookingService(MockAppointmentProvider()))
    else:
        # `k8s_docs` 域的检索工具需要真实检索器（要 store + embedder），
        # 而 MCP 这一层不持有它们。传 None 注册出来的 search 工具**调不通**，
        # 所以这里直接说清楚，而不是注册一个一调用就报错的空壳。
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
    """造一个 MCP server。返回的对象已挂好 `list_tools` / `call_tool`。"""
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
