"""MCP server 测试：工具定义来自注册表，副作用边界在 MCP 边界上仍然成立。

用 `mcp` 官方客户端的**进程内**会话（`create_connected_server_and_client_session`）：
起子进程 + stdio 会把测试变成"偶尔挂死"的东西，而这里要验的是边界语义，不是传输层。
**全程离线**：检索器由假替身提供，任何一步都不会去加载嵌入模型或打开真索引。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, Tool

import kubedocs.mcp.server as mcp_server
from kubedocs.agent.tools import ToolRegistry
from kubedocs.mcp.server import build_server


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """注册表是**进程级全局状态**：不清干净，两个域的用例会互相看见对方的工具。"""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


@pytest.fixture
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> Any:
    """假检索器：返回一条固定命中，证明"检索器真的被接到了 `search_docs` 上"。

    monkeypatch 必须在 `build_server` **之前**生效：装配器是在那一步被捕获进
    `_LazyRetriever` 的（真正的 `search()` 时才调用它）。
    """

    class _Hit:
        chunk_id = "c1"
        doc_id = "d1"
        ordinal = 0
        title = "Pods"
        breadcrumb = "工作负载 / Pod"
        source_url = "https://kubernetes.io/docs/concepts/workloads/pods/"
        body = "Pod 是可以在 Kubernetes 中创建和管理的最小可部署计算单元。"

    class _Result:
        results = [_Hit()]

    class _Retriever:
        def search(self, query: str, top_k: int = 5) -> Any:
            return _Result()

    monkeypatch.setattr(mcp_server, "_assemble_retriever", lambda settings: _Retriever())
    return _Retriever()


def _run(coro: Any) -> Any:
    """一个测试内跑一段会话（每次新建事件循环，避免跨用例复用已关闭的 session）。"""
    return asyncio.run(coro)


def _list(server: Any) -> list[Tool]:
    async def run() -> list[Tool]:
        async with create_connected_server_and_client_session(server) as session:
            return (await session.list_tools()).tools

    return _run(run())


def _call(server: Any, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
    async def run() -> CallToolResult:
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(name, arguments)

    return _run(run())


def _text(result: Any) -> str:
    return str(result.content[0].text)


def _mock_booking() -> Any:
    """进程内号源服务：**这一组用例不碰真实号源库**。

    默认装配会连 Postgres 号源，于是"取号"这条用例会真的占掉真实库存——某天某个时段
    被占满之后，它就会以"号源已满"的形式变红，而它验的其实只是协议形状。
    """
    from kubedocs.domains.clinic.booking import BookingService
    from kubedocs.domains.clinic.mock import MockAppointmentProvider

    return BookingService(MockAppointmentProvider())


class TestMcpToolExposure:
    def test_read_only_tools_are_listed(self) -> None:
        names = {t.name for t in _list(build_server("clinic", booking_service=_mock_booking()))}
        assert {"list_departments", "get_availability"}.issubset(names), names

    def test_rag_tools_are_exposed(self, fake_retriever: Any) -> None:
        """**RAG 工具真的在清单里**：只有号源三件套的 MCP 没有价值。"""
        names = {t.name for t in _list(build_server("clinic", booking_service=_mock_booking()))}
        assert {"list_departments", "search_docs", "check_citations"}.issubset(names), names

    def test_k8s_docs_domain_is_supported(self, fake_retriever: Any) -> None:
        """k8s_docs 域过去直接 SystemExit；现在它装配检索 + 引用核对，且没有号源工具。"""
        names = {t.name for t in _list(build_server("k8s_docs"))}
        assert {"search_docs", "check_citations"}.issubset(names), names
        assert "book_appointment" not in names, "k8s 域不该出现挂号工具"

    def test_domain_defaults_are_not_mixed_up(self) -> None:
        """两个域的嵌入模型/维度必须各自就位（中文 512 维 vs 英文 384 维）。"""
        clinic = mcp_server._resolve_settings("clinic", None)
        k8s = mcp_server._resolve_settings("k8s_docs", None)
        assert (clinic.embedding_dim, k8s.embedding_dim) == (512, 384)
        assert clinic.embedding_model != k8s.embedding_model
        assert clinic.normalizer != k8s.normalizer

    def test_side_effect_tools_are_hidden_by_default(self) -> None:
        """**安全边界**：挂号会真的占号，默认不暴露。"""
        names = {t.name for t in _list(build_server("clinic", booking_service=_mock_booking()))}
        assert "book_appointment" not in names, "默认就暴露了有副作用的工具"

    def test_side_effect_tools_require_an_explicit_flag(self) -> None:
        tools = {
            t.name: t
            for t in _list(build_server("clinic", allow_side_effects=True, booking_service=_mock_booking()))
        }
        assert "book_appointment" in tools
        schema = tools["book_appointment"].inputSchema
        assert "confirm" in (schema.get("required") or []), "开了副作用也要强制 confirm 字段"
        assert "confirm" in (schema.get("properties") or {})

    def test_schema_comes_from_the_registry(self) -> None:
        """工具定义必须来自 `ToolRegistry`，不是这里另写的一份。"""
        server = build_server("clinic", booking_service=_mock_booking())
        spec = ToolRegistry.get("list_departments")
        tool = {t.name: t for t in _list(server)}["list_departments"]
        assert tool.inputSchema == spec.params.model_json_schema()


class TestMcpCalls:
    def test_read_only_call_works(self) -> None:
        result = _call(build_server("clinic", booking_service=_mock_booking()), "list_departments", {})
        assert result.isError is False
        assert "科室" in _text(result) or "内科" in _text(result)

    def test_search_docs_calls_the_wired_retriever(self, fake_retriever: Any) -> None:
        """检索工具在 MCP 上真的能取到东西（用的是装配进来的那个检索器）。"""
        result = _call(build_server("k8s_docs"), "search_docs", {"query": "pod", "top_k": 2})
        assert result.isError is False
        assert "Pods" in _text(result)

    def test_listing_tools_does_not_assemble_a_retriever(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**惰性**：只装配/列工具时，一次检索器装配都不该发生（否则握手要等模型加载）。"""
        calls: list[str] = []

        def _boom(settings: Any) -> Any:  # pragma: no cover - 走到这里就是失败
            calls.append("assembled")
            raise AssertionError("列工具不该装配检索器")

        monkeypatch.setattr(mcp_server, "_assemble_retriever", _boom)
        names = {t.name for t in _list(build_server("k8s_docs"))}
        assert "search_docs" in names
        assert calls == []

    def test_retriever_assembly_failure_is_a_readable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """索引装不起来（没建索引 / 模型缺失）→ 可读的 isError，不是崩服务、不是栈。"""

        def _fail(settings: Any) -> Any:
            raise RuntimeError("索引为空（0 块）：data/index/k8s_docs/kubedocs.db")

        monkeypatch.setattr(mcp_server, "_assemble_retriever", _fail)
        result = _call(build_server("k8s_docs"), "search_docs", {"query": "pod"})
        assert result.isError is True
        text = _text(result)
        assert "索引不可用" in text
        assert "ingest" in text, "失败说明里要给出该跑哪条命令"

    def test_unknown_tool_is_a_readable_error(self) -> None:
        result = _call(build_server("clinic", booking_service=_mock_booking()), "no_such_tool", {})
        assert result.isError is True
        assert "没有名为" in _text(result)

    def test_hidden_side_effect_tool_is_refused_even_if_called(self) -> None:
        """**不列出 ≠ 拒绝**：直接点名调用也必须被挡住。"""
        result = _call(
            build_server("clinic", booking_service=_mock_booking()),
            "book_appointment",
            {"department": "内科"},
        )
        assert result.isError is True
        assert "副作用" in _text(result)

    def test_side_effect_missing_confirm_is_rejected(self) -> None:
        """开了 `--allow-side-effects` 也还不够：`confirm` 缺失一律拒绝。"""
        result = _call(
            build_server("clinic", allow_side_effects=True, booking_service=_mock_booking()),
            "book_appointment",
            {"department": "内科"},
        )
        assert result.isError is True
        assert "confirm" in _text(result)

    def test_side_effect_with_confirm_false_is_refused_with_reason(self) -> None:
        """显式传 `confirm=false` 时**必须给出可读的理由**。"""
        result = _call(
            build_server("clinic", allow_side_effects=True, booking_service=_mock_booking()),
            "book_appointment",
            {"department": "内科", "confirm": False},
        )
        assert result.isError is True
        assert "明确同意" in _text(result)

    def test_side_effect_with_confirm_true_executes(self) -> None:
        """开了开关 + 显式 `confirm=true` 才真的执行（这条路径必须仍然可用）。"""
        result = _call(
            build_server("clinic", allow_side_effects=True, booking_service=_mock_booking()),
            "book_appointment",
            {"department": "心内科", "time_slot": "上午", "confirm": True},
        )
        assert result.isError is False, _text(result)
        assert "ticket_no" in _text(result)

    def test_bad_arguments_are_rejected_by_pydantic(self) -> None:
        """参数校验必须走与 agent 内部同一条 pydantic 路径（MCP 不是绕过校验的后门）。"""
        result = _call(
            build_server("clinic", booking_service=_mock_booking()), "get_availability", {"department": ""}
        )
        assert result.isError is True

    def test_unknown_domain_fails_loudly(self) -> None:
        with pytest.raises(SystemExit, match="未知的域"):
            build_server("no_such_domain")

    def test_unknown_domain_through_the_cli_is_a_readable_user_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`kubedocs mcp --domain <拼错>` 要退出码 1 + 一句人话，不是 traceback。"""
        from kubedocs.cli import EXIT_USER
        from kubedocs.cli import main as cli_main

        rc = cli_main(["mcp", "--domain", "no_such_domain"])
        assert rc == EXIT_USER
        assert "未知的域" in capsys.readouterr().err


class TestMcpCliWiring:
    def test_cli_forwards_domain_settings_and_the_side_effect_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`kubedocs mcp --domain clinic --hf-offline --allow-side-effects` 的转发契约。

        CLI 只负责解析参数并把**定好的 settings** 交给 MCP 层（不在两边各造一份配置），
        所以这里挡住的是"参数解析了但没传下去"这类静默失效：
        `--hf-offline` 丢了会在离线机器上挂死，域默认值丢了会拿英文模型查中文索引。
        """
        seen: dict[str, Any] = {}

        def _fake_main(argv: Any = None, **kwargs: Any) -> int:
            seen["argv"] = argv
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(mcp_server, "main", _fake_main)
        from kubedocs.cli import main as cli_main

        rc = cli_main(["mcp", "--domain", "clinic", "--hf-offline", "--allow-side-effects"])

        assert rc == 0
        assert seen["argv"] == [], "程序内调用不该再去解析外层 CLI 的 sys.argv"
        assert seen["domain"] == "clinic"
        assert seen["allow_side_effects"] is True
        settings = seen["settings"]
        assert settings.domain == "clinic"
        assert settings.hf_offline is True
        assert (settings.embedding_model, settings.embedding_dim) == ("BAAI/bge-small-zh-v1.5", 512)
