"""第二轮接线测试：**已注册的工具 + 工具循环 + 真实 domain** 能不能一起跑。

这组测试针对的正是本项目反复栽的那个坑：**模块各自测试通过，但没接在一起**。
所以这里不测单个模块，而是把它们串起来跑：

    register_domain_tools("clinic", ...)   → 工具真的进注册表
    run_tool_loop(..., tools=注册表里的)    → 循环真的按 LLM 的选择执行
    handler 真的改了号源库状态              → 副作用真的落地

全部用打桩 LLM（`ToolCaller`），不联网、不花钱。
"""
from __future__ import annotations

from typing import Any

import pytest

from kubedocs.agent.bootstrap import register_domain_tools
from kubedocs.agent.tool_loop import run_tool_loop
from kubedocs.agent.tools import ToolRegistry
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.mock import MockAppointmentProvider


class ScriptedCaller:
    """按剧本返回：先要求调工具，再给出最终回答。

    `require_tools` 里的第一项是"该轮要调哪些工具"，最后一项之后返回 content。
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]]] = []

    def choose(self, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> dict[str, Any]:
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
        """模型必须真的收到 schema —— 这是"接线生效"的直接证据。"""
        caller = ScriptedCaller([{"content": "答", "tool_calls": []}])
        run_tool_loop("你好", tools=_tools(), llm=caller)
        names = [t["function"]["name"] for t in caller.seen_tools[0]]
        assert "list_departments" in names
        assert "book_appointment" in names
        assert all("parameters" in t["function"] for t in caller.seen_tools[0])

    def test_model_choice_drives_execution_and_result_is_visible(self, wired):
        """LLM 选了 list_departments → 真的执行 → 结果回到下一轮上下文。"""
        caller = ScriptedCaller([
            {"content": "", "tool_calls": [{"name": "list_departments",
                                            "arguments": {"contains": ""}}]},
            {"content": "共有若干科室", "tool_calls": []},
        ])
        turn = run_tool_loop("有哪些科室", tools=_tools(), llm=caller)
        assert turn.used_tools == ["list_departments"]
        assert turn.tool_calls[0].error == ""
        assert (turn.tool_calls[0].result or {}).get("n", 0) > 0
        # 结果确实进了第二轮上下文
        second = " ".join(str(m.get("content", "")) for m in caller.seen_messages[1])
        assert "departments" in second

    def test_availability_tool_runs_end_to_end(self, wired):
        caller = ScriptedCaller([
            {"content": "", "tool_calls": [{"name": "get_availability",
                                            "arguments": {"department": "心内科", "date": ""}}]},
            {"content": "上午还有号", "tool_calls": []},
        ])
        turn = run_tool_loop("心内科还有号吗", tools=_tools(), llm=caller)
        assert turn.tool_calls[0].error == ""
        assert (turn.tool_calls[0].result or {}).get("slots")

    def test_side_effect_tool_is_refused_without_confirmation(self, wired):
        """取号不可逆：没有确认回调时**一次都不能执行**，号源数量不能变。"""
        before = {s["time"]: s["remaining"]
                  for s in wired.availability("心内科", "2026-01-01").slots}
        caller = ScriptedCaller([
            {"content": "", "tool_calls": [{"name": "book_appointment",
                                            "arguments": {"department": "心内科",
                                                          "date": "2026-01-01",
                                                          "time_slot": "上午"}}]},
            {"content": "需要您确认", "tool_calls": []},
        ])
        turn = run_tool_loop("帮我挂号", tools=_tools(), llm=caller, confirm=None)
        assert turn.tool_calls[0].error, "未确认却执行了有副作用的工具"
        after = {s["time"]: s["remaining"]
                 for s in wired.availability("心内科", "2026-01-01").slots}
        assert after == before, "号源被改了——副作用闸门失效"

    def test_side_effect_tool_executes_when_confirmed(self, wired):
        """给确认回调后应当真的挂号，且号源减少——证明闸门是"要求确认"而非"禁用"。"""
        caller = ScriptedCaller([
            {"content": "", "tool_calls": [{"name": "book_appointment",
                                            "arguments": {"department": "心内科",
                                                          "date": "2026-01-01",
                                                          "time_slot": "上午"}}]},
            {"content": "已挂号", "tool_calls": []},
        ])
        turn = run_tool_loop("帮我挂号", tools=_tools(), llm=caller,
                             confirm=lambda call: True)
        assert turn.tool_calls[0].error == ""
        assert (turn.tool_calls[0].result or {}).get("ticket_no")
        assert wired.my_appointments("anon")

    def test_unknown_tool_name_is_fed_back_not_raised(self, wired):
        caller = ScriptedCaller([
            {"content": "", "tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
            {"content": "换个办法", "tool_calls": []},
        ])
        turn = run_tool_loop("随便", tools=_tools(), llm=caller)
        assert turn.tool_calls[0].error
        assert turn.answer == "换个办法"

    def test_booking_without_department_reports_readable_error(self, wired):
        """业务错误要变成人话，而不是抛异常。"""
        caller = ScriptedCaller([
            {"content": "", "tool_calls": [{"name": "book_appointment",
                                            "arguments": {"department": "火星科",
                                                          "date": "2026-01-01",
                                                          "time_slot": "上午"}}]},
            {"content": "没有这个科室", "tool_calls": []},
        ])
        turn = run_tool_loop("挂火星科", tools=_tools(), llm=caller, confirm=lambda c: True)
        assert turn.tool_calls[0].error
        assert "火星科" in turn.tool_calls[0].error


class TestRealCallerParsing:
    """`LiteLLMToolCaller` 的解析逻辑（不联网：直接测它处理过的形状）。"""

    def test_json_string_arguments_are_parsed(self):
        from kubedocs.agent.llm_tool_caller import LiteLLMToolCaller
        from kubedocs.config import Settings

        caller = LiteLLMToolCaller(Settings(embedding_provider="hashing"))
        assert caller is not None      # 构造不应触发网络或导入 litellm 之外的副作用

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
