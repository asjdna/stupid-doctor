"""MCP server 测试：工具定义来自注册表，副作用边界在 MCP 边界上仍然成立。

不启 stdio，直接调 `build_server` 挂的 `list_tools` / `call_tool`——验的是边界语义，传输层归 `mcp` 包。
"""

from __future__ import annotations

import asyncio

import pytest

from kubedocs.mcp.server import build_server


def _call(server: object, name: str, arguments: dict[str, object] | None = None) -> object:
    from mcp.types import CallToolRequest, CallToolRequestParams

    async def run() -> object:
        return await server.request_handlers[CallToolRequest](  # type: ignore[attr-defined]
            CallToolRequest(params=CallToolRequestParams(name=name, arguments=arguments))
        )

    return asyncio.run(run())


def _list(server: object) -> list[object]:
    from mcp.types import ListToolsRequest

    async def run() -> object:
        result = await server.request_handlers[ListToolsRequest](  # type: ignore[attr-defined]
            ListToolsRequest(method="tools/list")
        )
        return result.root.tools

    return asyncio.run(run())


def _text(result: object) -> str:
    return str(result.root.content[0].text)  # type: ignore[attr-defined]


class TestMcpToolExposure:
    def test_read_only_tools_are_listed(self) -> None:
        server = build_server("clinic")
        names = {t.name for t in _list(server)}
        assert {"list_departments", "get_availability"}.issubset(names), names

    def test_side_effect_tools_are_hidden_by_default(self) -> None:
        """**安全边界**：挂号会真的占号，默认不暴露。"""
        server = build_server("clinic")
        names = {t.name for t in _list(server)}
        assert "book_appointment" not in names, "默认就暴露了有副作用的工具"

    def test_side_effect_tools_require_an_explicit_flag(self) -> None:
        server = build_server("clinic", allow_side_effects=True)
        tools = {t.name: t for t in _list(server)}
        assert "book_appointment" in tools
        schema = tools["book_appointment"].inputSchema
        assert "confirm" in (schema.get("required") or []), "开了副作用也要强制 confirm 字段"
        assert "confirm" in (schema.get("properties") or {})

    def test_schema_comes_from_the_registry(self) -> None:
        """工具定义必须来自 `ToolRegistry`，不是这里另写的一份。"""
        from kubedocs.agent.tools import ToolRegistry

        server = build_server("clinic")
        spec = ToolRegistry.get("list_departments")
        tool = {t.name: t for t in _list(server)}["list_departments"]
        assert tool.inputSchema == spec.params.model_json_schema()


class TestMcpCalls:
    def test_read_only_call_works(self) -> None:
        server = build_server("clinic")
        result = _call(server, "list_departments", {})
        assert result.root.isError is False  # type: ignore[attr-defined]
        assert "科室" in _text(result) or "内科" in _text(result)

    def test_unknown_tool_is_a_readable_error(self) -> None:
        server = build_server("clinic")
        result = _call(server, "no_such_tool", {})
        assert result.root.isError is True  # type: ignore[attr-defined]
        assert "没有名为" in _text(result)

    def test_hidden_side_effect_tool_is_refused_even_if_called(self) -> None:
        """**不列出 ≠ 拒绝**：直接点名调用也必须被挡住。"""
        server = build_server("clinic")
        result = _call(server, "book_appointment", {"department": "内科"})
        assert result.root.isError is True  # type: ignore[attr-defined]
        assert "副作用" in _text(result)

    def test_side_effect_missing_confirm_is_rejected_by_the_framework(self) -> None:
        """`confirm` 是 schema 必填项，所以 mcp 框架自己就挡下了（多一层防护）；提示语在 schema 的 `description` 里。"""
        server = build_server("clinic", allow_side_effects=True)
        result = _call(server, "book_appointment", {"department": "内科"})
        assert result.root.isError is True  # type: ignore[attr-defined]
        assert "confirm" in _text(result)

    def test_side_effect_with_confirm_false_is_refused_with_reason(self) -> None:
        """显式传 `confirm=false` 时才轮到我们的处理器，**必须给出可读的理由**。"""
        server = build_server("clinic", allow_side_effects=True)
        result = _call(server, "book_appointment", {"department": "内科", "confirm": False})
        assert result.root.isError is True  # type: ignore[attr-defined]
        assert "明确同意" in _text(result)

    def test_bad_arguments_are_rejected_by_pydantic(self) -> None:
        """参数校验必须走与 agent 内部同一条 pydantic 路径（MCP 不是绕过校验的后门）。"""
        server = build_server("clinic")
        result = _call(server, "get_availability", {"department": ""})
        assert result.root.isError is True  # type: ignore[attr-defined]

    def test_unknown_domain_fails_loudly(self) -> None:
        with pytest.raises(SystemExit, match="只支持 clinic"):
            build_server("k8s_docs")
