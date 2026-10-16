"""提示词承诺的工具集合 == 实际注册的工具集合。

`ClinicReactRuntime` 在 `/chat` 路径上不注入检索器，于是 `search_docs` 不注册；若系统
提示仍承诺"知识库检索"，模型每一步都会拿回"没有这个工具"，白烧一轮预算。

两个方向都是缺陷：提示多做（承诺未注册的）让模型空转，提示少做（注册了没承诺）让模型
不知道自己能查什么。所以断言集合相等——清单由 `build_system_prompt(tools)` 从注册表
生成，两边是同一数据源的两种视图；任何一侧退回写死的文本，这里立刻红。
"""

from __future__ import annotations

from typing import Any

import pytest

from kubedocs.agent.clinic_react import (
    TOOL_LIST_HEADER,
    ClinicReactRuntime,
    build_system_prompt,
)
from kubedocs.agent.tools import ToolRegistry
from kubedocs.config import Settings
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.mock import MockAppointmentProvider


def promised_tools(prompt: str) -> set[str]:
    """从系统提示里抠出「被承诺可用的工具名」集合。

    约定：`TOOL_LIST_HEADER` 那行之后连续的 `- 名字` 行就是清单；找不到标记返回空集
    而不是抛异常，让"提示词退回写死的常量"表现为一次干净的集合断言失败。
    """
    lines = prompt.splitlines()
    if TOOL_LIST_HEADER not in lines:
        return set()
    names: set[str] = set()
    for line in lines[lines.index(TOOL_LIST_HEADER) + 1 :]:
        if not line.startswith("- "):
            break
        names.add(line[2:].strip())
    return names


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        index_dir=str(tmp_path),
        domain="clinic",
        embedding_provider="hashing",
        embedding_dim=32,
        llm_model="deepseek/deepseek-chat",
    )


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """注册表是**全局**的，测试之间必须自己收尾，否则会污染同会话的其它用例。"""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


def make_runtime(settings: Settings, retriever: Any = None) -> ClinicReactRuntime:
    return ClinicReactRuntime(settings, BookingService(MockAppointmentProvider()), retriever=retriever)


class _StubRetriever:
    """为"装配时传了 retriever"这条路径提供鸭子类型的检索器。

    注册工具不会碰它，所以不必实现真实检索——真被调用就说明装配跑了不该跑的东西。
    """

    def search(self, query: str, top_k: int = 5) -> Any:
        raise AssertionError("本测试只验证装配，不该真的检索")


class TestChatPathParity:
    """`/chat` 的实际装配：没有 retriever，所以提示词里不该出现检索工具。"""

    def test_prompt_promises_exactly_what_is_registered(self, settings):
        rt = make_runtime(settings)
        registered = {t.name for t in rt.tools}
        assert registered == {
            "list_departments",
            "get_availability",
            "booking_window",
            "my_appointments",
            "book_appointment",
            "check_citations",
        }, "clinic 域的注册集合变了：确认是有意为之，再同步这条断言"
        assert promised_tools(rt.system_prompt) == registered, (
            "系统提示承诺的工具与注册表不一致——模型会去调一个它没有的工具"
        )

    def test_proposal_tool_only_exists_when_a_pending_store_is_wired(self, settings):
        """没有待确认单存储时**不注册** `propose_booking`：那张单子无处安放，
        注册一个必然失败的提议工具只会让模型反复重试。"""
        assert "propose_booking" not in {t.name for t in make_runtime(settings).tools}

        from kubedocs.domains.clinic.pending import PendingBookings

        rt = ClinicReactRuntime(
            settings,
            BookingService(MockAppointmentProvider()),
            pending=PendingBookings(),
        )
        assert "propose_booking" in {t.name for t in rt.tools}
        assert promised_tools(rt.system_prompt) == {t.name for t in rt.tools}

    def test_prompt_does_not_promise_the_unregistered_search_tool(self, settings):
        """`search_docs` 没注册，提示词就不许提它（含中文能力名的等价说法）。"""
        rt = make_runtime(settings)
        assert "search_docs" not in rt.system_prompt
        assert "知识库检索" not in rt.system_prompt
        assert "search_docs" not in ToolRegistry.names()

    def test_every_promised_name_is_actually_callable(self, settings):
        """反向：清单里点名的每一个都必须真在注册表里（否则模型调它必失败）。"""
        promised = promised_tools(make_runtime(settings).system_prompt)
        assert promised <= set(ToolRegistry.names())


class TestWithRetriever:
    """传了 retriever 的装配：`search_docs` 注册了，提示词也必须跟上。"""

    def test_search_tool_is_both_registered_and_promised(self, settings):
        rt = make_runtime(settings, retriever=_StubRetriever())
        registered = {t.name for t in rt.tools}
        assert "search_docs" in registered, (
            "传了 retriever 却没注册 search_docs：bootstrap 的装配条件被改坏了"
        )
        assert promised_tools(rt.system_prompt) == registered
        assert "search_docs" in rt.system_prompt


class TestBuildSystemPrompt:
    """生成器本身的边界：清单永远与入参一一对应。"""

    def test_empty_tool_list_promises_nothing(self):
        parsed = promised_tools(build_system_prompt([]))
        assert parsed == set(), (
            "没有工具时提示词仍然承诺了东西；注意空清单的占位文案不能以 '- ' 开头，"
            "否则会被读成'有一个叫（…）的工具'"
        )

    def test_names_are_sorted_and_complete(self):
        from kubedocs.agent.tools import EmptyParams, ToolSpec

        def spec(name: str, domain: str = "clinic") -> ToolSpec:
            return ToolSpec(
                name=name, description="（测试用）", params=EmptyParams, domain=domain, when_not="（测试用）"
            )

        prompt = build_system_prompt([spec("zeta"), spec("alpha"), spec("mid")])
        assert promised_tools(prompt) == {"alpha", "mid", "zeta"}
