"""工具编排（`clinical/orchestration.py`）的回归测试。

这里钉住的是**架构决定**：底层约束是大模型 ⇒ 它自己决定调哪些模块取证。
代码只保留三件非医学的事：轮次上限、工具白名单、以及把工具返回的证据编号累积起来
以便事后校验引用。下面每条测试对应其中一件，外加"模型犯错时不要崩"。

全部用假客户端与假工具，不联网、不花钱、不需要语料。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from kubedocs.agent.tools import ToolError, ToolRegistry, ToolSpec
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.orchestration import (
    MAX_TOOL_ROUNDS,
    ClinicalToolLoop,
    describe_tools,
    tool_schemas,
)


class _SearchParams(BaseModel):
    query: str = ""


class _BookParams(BaseModel):
    department: str = ""


def _hits(*titles: str) -> dict[str, Any]:
    return {
        "query": "q",
        "n": len(titles),
        "hits": [{"title": t, "body": f"{t} 的正文", "score": 0.5, "source_url": ""} for t in titles],
    }


@pytest.fixture
def registry() -> Any:
    """干净的注册表：注册一个检索工具（无副作用）与一个挂号工具（有副作用）。"""
    ToolRegistry.clear()
    ToolRegistry.register(
        ToolSpec(
            name="search_docs",
            description="检索",
            params=_SearchParams,
            domain="clinic",
            when_not="不要重复检索",
            side_effect=False,
            handler=lambda p: _hits("急性支气管炎", "咳嗽"),
        )
    )
    ToolRegistry.register(
        ToolSpec(
            name="book_appointment",
            description="挂号",
            params=_BookParams,
            domain="clinic",
            when_not="患者还没确认要挂号时不要调用",
            side_effect=True,
            handler=lambda p: {"ok": True},
        )
    )
    yield ToolRegistry
    ToolRegistry.clear()


@dataclass
class _TurnSpec:
    """假客户端的一步：要么要工具，要么给最终答复。"""

    content: str = ""
    calls: list[tuple[str, str]] = field(default_factory=list)


class _FakeClient:
    def __init__(self, steps: list[_TurnSpec], *, cost: float = 0.001) -> None:
        self.steps = list(steps)
        self.seen: list[list[dict[str, Any]]] = []
        self.tools_seen: list[int] = []
        self.cost = cost

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        self.seen.append([dict(m) for m in messages])
        self.tools_seen.append(len(tools))
        step = self.steps.pop(0) if self.steps else _TurnSpec(content="{}")

        @dataclass
        class _Call:
            id: str
            name: str
            arguments: str

        @dataclass
        class _Turn:
            content: str
            tool_calls: list[Any]
            cost_usd: float = 0.0
            cost_priced: bool = True

        return _Turn(
            content=step.content,
            tool_calls=[_Call(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(step.calls)],
            cost_usd=self.cost,
        )


def _run(client: _FakeClient, **kw: Any) -> Any:
    loop = ClinicalToolLoop(client, **kw)
    return loop.run(system="SYS", question="我咳嗽", intake=IntakeState())


# --------------------------------------------------------------- 工具可见性
def test_side_effect_tools_are_hidden_by_default(registry: Any) -> None:
    """**副作用工具默认不给模型**（挂号会占号源）。与 MCP server 同一套约定。"""
    names = [s["function"]["name"] for s in tool_schemas(domain="clinic")]
    assert "search_docs" in names
    assert "book_appointment" not in names


def test_side_effect_tools_need_explicit_opt_in(registry: Any) -> None:
    names = [s["function"]["name"] for s in tool_schemas(domain="clinic", allow_side_effects=True)]
    assert "book_appointment" in names


def test_tool_catalogue_is_rendered_into_the_prompt(registry: Any) -> None:
    """`when_not` 必须进提示词：function calling 的 description 常被截断，而"什么时候别用"最关键。"""
    text = describe_tools(domain="clinic")
    assert "search_docs" in text
    assert "不要重复检索" in text
    assert "book_appointment" not in text


# --------------------------------------------------------------- 主流程
def test_model_drives_the_calls_and_evidence_is_numbered_by_us(registry: Any) -> None:
    """**核心**：模型要工具 → 我们执行 → 证据编号由我们分配 → 模型出最终结论。"""
    client = _FakeClient(
        [
            _TurnSpec(calls=[("search_docs", '{"query": "咳嗽"}')]),
            _TurnSpec(content='{"urgency": "routine"}'),
        ]
    )
    result = _run(client)
    assert result.final_text == '{"urgency": "routine"}'
    assert [c.name for c in result.calls] == ["search_docs"]
    assert result.calls[0].ok is True
    # 编号是 K1、K2…（**不信任工具返回的 id 形状**：契约校验要的是单一来源）
    assert [i.id for i in result.evidence.items] == ["K1", "K2"]
    assert result.evidence.items[0].text == "急性支气管炎 的正文"
    assert result.rounds == 2


def test_messages_carry_the_tool_result_back(registry: Any) -> None:
    """工具结果必须以 `role=tool` 回给模型，否则它下一轮等于没拿到东西。"""
    client = _FakeClient([_TurnSpec(calls=[("search_docs", '{"query":"x"}')]), _TurnSpec(content="{}")])
    _run(client)
    second = client.seen[1]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second[3]["tool_call_id"] == "c0"


def test_final_answer_without_tools_stops_immediately(registry: Any) -> None:
    """模型选择不检索也必须能一轮结束——**这正是"它自己决定"的意思**。"""
    client = _FakeClient([_TurnSpec(content='{"urgency":"routine"}')])
    result = _run(client)
    assert result.calls == []
    assert result.evidence.items == []
    assert result.rounds == 1


def test_no_tools_available_is_not_an_error(registry: Any) -> None:
    """注册表为空时也要能跑（例如 k8s 域没有号源工具）。"""
    ToolRegistry.clear()
    client = _FakeClient([_TurnSpec(content="{}")])
    assert _run(client).final_text == "{}"


# --------------------------------------------------------------- 模型犯错
def test_unknown_tool_becomes_a_message_not_a_crash(registry: Any) -> None:
    """模型点名一个不存在的工具 ⇒ 把错误回给它，让它换策略，而不是抛出去。"""
    client = _FakeClient([_TurnSpec(calls=[("nope", "{}")]), _TurnSpec(content="{}")])
    result = _run(client)
    assert result.calls[0].ok is False
    assert "没有名为" in result.calls[0].summary


def test_invalid_json_arguments_are_reported_back(registry: Any) -> None:
    """参数不是合法 JSON 是模型常犯的错：回一句人话让它重试。"""
    client = _FakeClient([_TurnSpec(calls=[("search_docs", "{不是 json")]), _TurnSpec(content="{}")])
    result = _run(client)
    assert result.calls[0].ok is False
    assert "合法 JSON" in result.calls[0].summary


def test_schema_violation_is_reported_back(registry: Any) -> None:
    client = _FakeClient([_TurnSpec(calls=[("search_docs", '{"query": 123}')]), _TurnSpec(content="{}")])
    result = _run(client)
    assert result.calls[0].ok is False
    assert "schema" in result.calls[0].summary


def test_tool_error_is_swallowed_into_the_conversation(registry: Any) -> None:
    """工具自己抛 `ToolError`（可预期失败）不该让整轮问诊挂掉。"""
    ToolRegistry.clear()
    ToolRegistry.register(
        ToolSpec(
            name="search_docs",
            description="检索",
            params=_SearchParams,
            domain="clinic",
            when_not="患者还没确认要挂号时不要调用",
            handler=lambda p: (_ for _ in ()).throw(ToolError("索引是空的")),
        )
    )
    client = _FakeClient([_TurnSpec(calls=[("search_docs", "{}")]), _TurnSpec(content="{}")])
    result = _run(client)
    assert result.calls[0].ok is False
    assert "索引是空的" in result.calls[0].summary


# --------------------------------------------------------------- 硬上限
def test_round_limit_is_enforced_and_the_model_is_told(registry: Any) -> None:
    """**硬上限**：模型不能靠"再查一次"把预算烧穿；到顶要**明说**，不静默截断。"""
    client = _FakeClient([_TurnSpec(calls=[("search_docs", "{}")]) for _ in range(MAX_TOOL_ROUNDS + 1)])
    result = _run(client)
    assert result.rounds == MAX_TOOL_ROUNDS + 1
    # 最后一次调用不再带工具，且最后一条 user 消息明确要求"用现有依据作答"
    assert client.tools_seen[-1] == 0
    last_user = [m for m in client.seen[-1] if m["role"] == "user"][-1]
    assert "上限" in last_user["content"]
    assert "已经拿到的依据" in last_user["content"]


def test_evidence_is_capped(registry: Any) -> None:
    """证据条数有上界：模型不能靠狂调检索把上下文撑爆。"""
    ToolRegistry.clear()
    ToolRegistry.register(
        ToolSpec(
            name="search_docs",
            description="检索",
            params=_SearchParams,
            domain="clinic",
            when_not="患者还没确认要挂号时不要调用",
            handler=lambda p: _hits(*[f"卡{i}" for i in range(50)]),
        )
    )
    client = _FakeClient(
        [_TurnSpec(calls=[("search_docs", "{}")]), _TurnSpec(content="{}")],
    )
    result = _run(client, max_evidence=5)
    assert len(result.evidence.items) == 5


# --------------------------------------------------------------- 计费
def test_every_tool_round_is_billed(registry: Any) -> None:
    """工具循环里**每一次模型调用都要计费**，否则多轮取证是绕过日预算的口子。"""
    from kubedocs.clinical.reasoning import BudgetedClient

    class _Budget:
        def __init__(self) -> None:
            self.records: list[tuple[float, bool]] = []
            self.reserves = 0

        def reserve(self) -> Any:
            import contextlib

            self.reserves += 1
            return contextlib.nullcontext()

        def record(self, cost: float, *, priced: bool) -> None:
            self.records.append((cost, priced))

    budget = _Budget()
    client = _FakeClient([_TurnSpec(calls=[("search_docs", "{}")]), _TurnSpec(content="{}")], cost=0.002)
    wrapped = BudgetedClient(client, budget)
    loop = ClinicalToolLoop(wrapped)  # type: ignore[arg-type]
    loop.run(system="S", question="q", intake=IntakeState())
    assert budget.reserves == 2, "两轮模型调用就该预留两次"
    assert budget.records == [(0.002, True), (0.002, True)]
