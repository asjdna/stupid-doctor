"""提示词承诺的工具集合 == 实际注册的工具集合。

## 这条测试钉的是哪个真实缺陷

`ClinicReactRuntime` 在 `/chat` 路径上是 `ClinicReactRuntime(settings, service)` 构造的
（`api/app.py` 刻意不注入检索器：注入就要在启动时加载嵌入模型，与"资源惰性构造"冲突），
于是 `bootstrap.register_domain_tools` 里的 `if retriever is not None` 不成立、
**`search_docs` 根本没注册**（`GET /tools` 里也看不到它）。

而系统提示当时是一段**写死的常量**，第 2 条对模型说：
"需要资料时用工具查（科室列表、号源余量、**知识库检索**）"——
**模型被告知有一个它根本调不到的工具**。它会照做，然后每一步都拿回
"没有名为 search_docs 的工具"，白白烧掉一轮预算。

## 为什么用"集合相等"而不是"包含"来钉

两个方向都是缺陷：

* 提示**多做**了（承诺了没注册的）→ 模型空转，就是上面这个 bug；
* 提示**少做**了（注册了没承诺）→ 工具白注册，模型不知道自己能查号源。

所以断言的是相等，而提示词里那份清单由 `build_system_prompt(tools)` 从注册表生成，
两边是同一个数据源的两种视图——任何一侧被改回写死的文本，这里立刻红。
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

    约定：`TOOL_LIST_HEADER` 那行之后、连续的 `- 名字` 行就是清单。
    找不到标记就返回**空集**而不是抛异常——这样"提示词退回写死的常量"表现为一次
    干净的断言失败（`set() != {…}`），而不是一个 `StopIteration` 让人以为是测试写坏了。
    """
    lines = prompt.splitlines()
    if TOOL_LIST_HEADER not in lines:
        return set()
    names: set[str] = set()
    for line in lines[lines.index(TOOL_LIST_HEADER) + 1:]:
        if not line.startswith("- "):
            break
        names.add(line[2:].strip())
    return names


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(index_dir=str(tmp_path), domain="clinic",
                    embedding_provider="hashing", embedding_dim=32,
                    llm_model="deepseek/deepseek-chat")


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """注册表是**全局**的，测试之间必须自己收尾，否则会污染同会话的其它用例。"""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


def make_runtime(settings: Settings, retriever: Any = None) -> ClinicReactRuntime:
    return ClinicReactRuntime(settings, BookingService(MockAppointmentProvider()),
                              retriever=retriever)


class _StubRetriever:
    """只为"装配时传了 retriever"这条路径提供一个鸭子类型的检索器。

    注册工具**不会**碰它（`make_search_tool` 只在执行时才 `getattr(retriever, 'search')`），
    所以这里不必实现真实检索——真的被调用就说明装配把不该跑的东西跑了。
    """

    def search(self, query: str, top_k: int = 5) -> Any:
        raise AssertionError("本测试只验证装配，不该真的检索")


class TestChatPathParity:
    """`/chat` 的实际装配：没有 retriever，所以提示词里不该出现检索工具。"""

    def test_prompt_promises_exactly_what_is_registered(self, settings):
        rt = make_runtime(settings)
        registered = {t.name for t in rt.tools}
        assert registered == {"list_departments", "get_availability",
                              "book_appointment", "check_citations"}, (
            "clinic 域的注册集合变了：确认是有意为之，再同步这条断言"
        )
        assert promised_tools(rt.system_prompt) == registered, (
            "系统提示承诺的工具与注册表不一致——模型会去调一个它没有的工具"
        )

    def test_prompt_does_not_promise_the_unregistered_search_tool(self, settings):
        """**这条直接对应缺陷本身**：`search_docs` 没注册，提示词就不许提它。

        只看集合相等还不够直观：这个断言把"模型被告知有一个它没有的工具"写成了
        一句能被读懂的失败信息。
        """
        rt = make_runtime(settings)
        assert "search_docs" not in rt.system_prompt
        # 老提示词用的是中文能力名而不是工具名，一并钉住，防止"换个说法继续承诺"
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
            return ToolSpec(name=name, description="（测试用）", params=EmptyParams,
                            domain=domain, when_not="（测试用）")

        prompt = build_system_prompt([spec("zeta"), spec("alpha"), spec("mid")])
        assert promised_tools(prompt) == {"alpha", "mid", "zeta"}
