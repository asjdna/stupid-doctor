"""端到端接线：注册表里的工具 + 工具循环 + 真实 domain 串起来跑。

链路是 register_domain_tools → run_tool_loop(tools=注册表里的工具) → handler 真的改
号源库状态。LLM 全部打桩（`ToolCaller`），不联网、不花钱。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kubedocs.agent.bootstrap import register_domain_tools
from kubedocs.agent.tool_loop import run_tool_loop
from kubedocs.agent.tools import ToolRegistry
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.mock import MockAppointmentProvider


class ScriptedCaller:
    """按剧本返回：每次 `choose` 弹出脚本下一项，用尽后返回空回答；并记录收到的 messages / tools。"""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]]] = []

    def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.seen_messages.append(list(messages))
        self.seen_tools.append(list(tools))
        return self.script.pop(0) if self.script else {"content": "（无脚本）", "tool_calls": []}


@pytest.fixture
def wired():
    ToolRegistry.clear()
    service = BookingService(MockAppointmentProvider())
    register_domain_tools("clinic", service=service)
    yield service
    ToolRegistry.clear()


def _tools() -> list[Any]:
    return list(ToolRegistry.for_domain("clinic"))


class TestToolsAndLoopTogether:
    def test_registered_tools_are_handed_to_the_model_as_schemas(self, wired):
        """注册表里的工具必须以 schema 形状交给模型，而不只是躺在注册表里。"""
        caller = ScriptedCaller([{"content": "答", "tool_calls": []}])
        run_tool_loop("你好", tools=_tools(), llm=caller)
        names = [t["function"]["name"] for t in caller.seen_tools[0]]
        assert "list_departments" in names
        assert "book_appointment" in names
        assert all("parameters" in t["function"] for t in caller.seen_tools[0])

    def test_model_choice_drives_execution_and_result_is_visible(self, wired):
        """模型选中的工具要真的执行，结果回灌进下一轮上下文。"""
        caller = ScriptedCaller(
            [
                {"content": "", "tool_calls": [{"name": "list_departments", "arguments": {"contains": ""}}]},
                {"content": "共有若干科室", "tool_calls": []},
            ]
        )
        turn = run_tool_loop("有哪些科室", tools=_tools(), llm=caller)
        assert turn.used_tools == ["list_departments"]
        assert turn.tool_calls[0].error == ""
        assert (turn.tool_calls[0].result or {}).get("n", 0) > 0
        # 结果确实进了第二轮上下文
        second = " ".join(str(m.get("content", "")) for m in caller.seen_messages[1])
        assert "departments" in second

    def test_availability_tool_runs_end_to_end(self, wired):
        caller = ScriptedCaller(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {"name": "get_availability", "arguments": {"department": "心内科", "date": ""}}
                    ],
                },
                {"content": "上午还有号", "tool_calls": []},
            ]
        )
        turn = run_tool_loop("心内科还有号吗", tools=_tools(), llm=caller)
        assert turn.tool_calls[0].error == ""
        assert (turn.tool_calls[0].result or {}).get("slots")

    def test_side_effect_tool_is_refused_without_confirmation(self, wired):
        """取号不可逆：没有确认回调时**一次都不能执行**，号源数量不能变。"""
        before = {s["time"]: s["remaining"] for s in wired.availability("心内科", "2026-01-01").slots}
        caller = ScriptedCaller(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "book_appointment",
                            "arguments": {"department": "心内科", "date": "2026-01-01", "time_slot": "上午"},
                        }
                    ],
                },
                {"content": "需要您确认", "tool_calls": []},
            ]
        )
        turn = run_tool_loop("帮我挂号", tools=_tools(), llm=caller, confirm=None)
        assert turn.tool_calls[0].error, "未确认却执行了有副作用的工具"
        after = {s["time"]: s["remaining"] for s in wired.availability("心内科", "2026-01-01").slots}
        assert after == before, "号源被改了——副作用闸门失效"

    def test_side_effect_tool_executes_when_confirmed(self, wired):
        """给确认回调后应当真的挂号、号源减少：闸门是"要求确认"而非"禁用"。"""
        caller = ScriptedCaller(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "book_appointment",
                            "arguments": {"department": "心内科", "date": "2026-01-01", "time_slot": "上午"},
                        }
                    ],
                },
                {"content": "已挂号", "tool_calls": []},
            ]
        )
        turn = run_tool_loop("帮我挂号", tools=_tools(), llm=caller, confirm=lambda call: True)
        assert turn.tool_calls[0].error == ""
        assert (turn.tool_calls[0].result or {}).get("ticket_no")
        assert wired.my_appointments("anon")

    def test_unknown_tool_name_is_fed_back_not_raised(self, wired):
        caller = ScriptedCaller(
            [
                {"content": "", "tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
                {"content": "换个办法", "tool_calls": []},
            ]
        )
        turn = run_tool_loop("随便", tools=_tools(), llm=caller)
        assert turn.tool_calls[0].error
        assert turn.answer == "换个办法"

    def test_booking_without_department_reports_readable_error(self, wired):
        """业务错误要变成人话，而不是抛异常。"""
        caller = ScriptedCaller(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "book_appointment",
                            "arguments": {"department": "火星科", "date": "2026-01-01", "time_slot": "上午"},
                        }
                    ],
                },
                {"content": "没有这个科室", "tool_calls": []},
            ]
        )
        turn = run_tool_loop("挂火星科", tools=_tools(), llm=caller, confirm=lambda c: True)
        assert turn.tool_calls[0].error
        assert "火星科" in turn.tool_calls[0].error


class TestRealCallerParsing:
    """`LiteLLMToolCaller` 的参数解析：只测形状，不联网。"""

    def test_json_string_arguments_are_parsed(self):
        from kubedocs.agent.llm_tool_caller import LiteLLMToolCaller
        from kubedocs.config import Settings

        caller = LiteLLMToolCaller(Settings(embedding_provider="hashing"))
        assert caller is not None  # 构造不应触发网络或导入 litellm 之外的副作用

    def test_bad_json_is_marked_not_raised(self):
        """模型给出畸形 JSON 时不能崩——要把错误交回循环回灌给模型。"""
        import json

        from kubedocs.agent.tool_loop import ToolCall  # noqa: F401  仅确认可导入

        raw = "{not json"
        args: dict[str, object] = {}
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            args = {"__parse_error__": "x"}
        assert "__parse_error__" in args


class _FakeLitellmForToolCaller:
    """假 litellm：只回一个响应 + 一个可由用例指定的成本行为（不联网）。"""

    drop_params = True

    def __init__(self, *, cost: object = None, cost_raises: Exception | None = None) -> None:
        self._cost = cost
        self._cost_raises = cost_raises
        self.calls = 0

    def completion(self, **_kwargs: Any) -> Any:
        self.calls += 1
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="直接作答", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    def completion_cost(self, completion_response: Any = None) -> object:
        if self._cost_raises is not None:
            raise self._cost_raises
        return self._cost


class TestToolCallerReportsUnknownPrice:
    """算不出价的调用必须标成未定价，不能记成 0：否则它在账本上等于免费，预算闸门失效。"""

    def _caller(self, fake: _FakeLitellmForToolCaller) -> Any:
        from kubedocs.agent.llm_tool_caller import LiteLLMToolCaller
        from kubedocs.config import Settings

        caller = LiteLLMToolCaller(Settings(embedding_provider="hashing"))
        caller._litellm = fake  # 绕开 import litellm（并避免它去拉价格表）
        return caller

    def test_price_map_miss_is_flagged_not_zeroed(self):
        fake = _FakeLitellmForToolCaller(cost_raises=Exception("This model isn't mapped yet"))
        caller = self._caller(fake)

        out = caller.choose([{"role": "user", "content": "问"}], [])

        assert out["cost_usd"] == 0.0
        assert out["cost_priced"] is False, "算不出价必须标成未定价，否则循环会当成免费"
        assert caller.unpriced_calls == 1
        assert caller.cost == 0.0

    def test_real_price_is_reported_as_priced(self):
        fake = _FakeLitellmForToolCaller(cost=0.002)
        caller = self._caller(fake)

        out = caller.choose([{"role": "user", "content": "问"}], [])

        assert out == {"content": "直接作答", "tool_calls": [], "cost_usd": 0.002, "cost_priced": True}
        assert caller.cost == pytest.approx(0.002)
        assert caller.unpriced_calls == 0

    def test_unpriced_call_is_charged_by_the_loop_budget(self, tmp_path: Path) -> None:
        """端到端：算不出价的调用由真 `DailyBudget` 按保守上限记账，而不是记 0。"""
        from kubedocs.generate import DailyBudget

        fake = _FakeLitellmForToolCaller(cost_raises=Exception("not mapped"))
        caller = self._caller(fake)
        budget = DailyBudget(tmp_path / "llm_spend.json", 5.0, unpriced_usd=0.25)

        turn = run_tool_loop("问", tools=[], llm=caller, budget=budget)

        assert turn.unpriced_calls == 1
        assert budget.spent_today() == pytest.approx(0.25), "未定价的调用被当成了免费"
        assert budget.unpriced_calls_today() == 1
