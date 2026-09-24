"""`run_tool_loop` 的单测：全打桩，不联网、不调真实模型。

## 为什么必须有"错误回灌"的断言

工具循环最容易写成两种错法："报错就抛"和"报错就静默跳过"。它们在真实模型下都测不出来
——模型不会抱怨，评测也只看最终回答。所以这里直接抓 `choose` 收到的 messages，断言错误
文本**真的**进了下一轮的上下文：这是"模型能自我纠正"这个说法的唯一硬证据。

## 为什么每个安全用例都断言 handler 没被调用

"参数不合法 → 不执行""副作用未确认 → 不执行"都是安全属性。只看回答文本看不出工具有没有
被偷偷执行过，所以这些用例都用记录调用的桩 handler 兜底。

## 为什么打桩要覆盖两种 handler 形状

仓库里 `rag_tools` 写成 `handler(params)`、`clinic.tools` 写成 `handler(**kwargs)`。
循环的分发逻辑必须同时支持两者，否则"接线"只接得上一半的工具。
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from kubedocs.agent.rag_tools import make_search_tool
from kubedocs.agent.tool_loop import AgentTurn, ToolCall, run_tool_loop
from kubedocs.agent.tools import SearchDocsParams, ToolError, ToolSpec


class ScriptExhausted(BaseException):
    """打桩脚本用尽。

    刻意继承 `BaseException` 而不是 `Exception`：循环会用 `except Exception` 把 LLM 的任何
    异常都降级成"没答完"，于是测试桩自己的"调用次数超出预期"断言会被**静默吞掉**，用例
    假装通过。这是本文件真实踩过的坑——`max_steps=0` 那次就是被它掩盖成 "DID NOT RAISE"。
    """


class StubLLM:
    """按脚本逐次返回的打桩 LLM，同时记下每轮收到的 messages / tools 供断言。

    脚本里可以塞异常对象：用于验证"LLM 自己炸了"的降级路径。
    """

    def __init__(self, *responses: dict[str, Any] | Exception) -> None:
        self._responses = list(responses)
        self.messages: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]]] = []

    def choose(self, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.messages.append(copy.deepcopy(messages))
        self.tools.append(tools)
        if not self._responses:
            raise ScriptExhausted("打桩脚本已用尽：循环调用 LLM 的次数超出了用例预期")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name: str, **arguments: Any) -> dict[str, Any]:
    """构造一步"模型要求调工具"的响应。"""
    return {"content": "", "tool_calls": [{"name": name, "arguments": dict(arguments)}]}


def answer(text: str) -> dict[str, Any]:
    """构造一步"模型给出最终回答"的响应。"""
    return {"content": text, "tool_calls": []}


class Recorder:
    """桩 handler（`handler(params)` 形状）：记下参数并返回固定结果。"""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = {"ok": True} if result is None else result

    def __call__(self, params: SearchDocsParams) -> Any:
        self.calls.append(params.model_dump())
        return self.result


def spec_of(name: str, handler: Any, *, params: type[BaseModel] = SearchDocsParams,
            side_effect: bool = False) -> ToolSpec:
    """造一个最小的 `ToolSpec`；`when_not` 是项目约定，这里也照填。"""
    return ToolSpec(name=name, description=f"测试用工具 {name}", params=params,
                    when_not="测试场景里不涉及", side_effect=side_effect, handler=handler)


# ===========================================================================
# 正常路径：模型选工具、拿结果、再作答
# ===========================================================================
class TestHappyPath:
    def test_llm_picks_tool_then_answers(self) -> None:
        """LLM 先选工具、拿到结果后再作答：used_tools 与 steps 都要准确。"""
        recorder = Recorder({"hits": ["c1"]})
        llm = StubLLM(call("search_docs", query="pod 重启", top_k=2), answer("据文档…[1]"))

        turn = run_tool_loop("pod 一直重启怎么办", tools=[spec_of("search_docs", recorder)],
                             llm=llm)

        assert turn.answer == "据文档…[1]"
        assert turn.used_tools == ["search_docs"]
        assert turn.steps == 2
        assert turn.llm_calls == 2
        assert turn.tool_calls[0].result == {"hits": ["c1"]}
        assert turn.tool_calls[0].error == ""
        assert turn.tool_calls[0].ok is True
        assert recorder.calls == [{"query": "pod 重启", "top_k": 2}]
        # schema 是真的交给了 LLM（而不是靠工具名或关键词猜）
        assert llm.tools[0][0]["function"]["name"] == "search_docs"
        assert "parameters" in llm.tools[0][0]["function"]
        # 工具结果回灌进了第二次 choose 的上下文
        assert llm.messages[1][-1]["role"] == "tool"
        assert "c1" in llm.messages[1][-1]["content"]

    @pytest.mark.parametrize("chosen", ["search_docs", "check_citations"])
    def test_different_tool_names_take_different_paths(self, chosen: str) -> None:
        """两次打桩返回不同工具名 → 执行的是各自 handler，另一个一次都没跑。"""
        search, cite = Recorder({"n": 3}), Recorder({"ok": True})
        tools = [spec_of("search_docs", search), spec_of("check_citations", cite)]
        llm = StubLLM(call(chosen, query="同一个问题"), answer("done"))

        turn = run_tool_loop("问", tools=tools, llm=llm)

        assert turn.used_tools == [chosen]
        executed = search if chosen == "search_docs" else cite
        untouched = cite if chosen == "search_docs" else search
        assert executed.calls == [{"query": "同一个问题", "top_k": 5}]
        assert untouched.calls == []

    def test_plain_question_answers_in_one_step(self) -> None:
        """无需工具的普通问题：一步作答、tool_calls 为空、没有多余 LLM 调用。"""
        recorder = Recorder()
        llm = StubLLM(answer("你好，我是 kubedocs 助手。"))

        turn = run_tool_loop("你好", tools=[spec_of("search_docs", recorder)], llm=llm,
                             system="你是 K8s 文档助手")

        assert turn.answer == "你好，我是 kubedocs 助手。"
        assert turn.tool_calls == []
        assert turn.used_tools == []
        assert turn.steps == 1
        assert turn.llm_calls == 1
        assert recorder.calls == []
        assert llm.messages[0][0] == {"role": "system", "content": "你是 K8s 文档助手"}
        assert llm.messages[0][1] == {"role": "user", "content": "你好"}

    def test_multiple_tool_calls_in_one_step_count_as_one_step(self) -> None:
        """一步里模型要求调两个工具 → 都执行，但只算一步（max_steps 是 LLM 轮次上限）。"""
        search, cite = Recorder({"n": 1}), Recorder({"ok": False})
        llm = StubLLM({"content": "", "tool_calls": [
            {"name": "search_docs", "arguments": {"query": "q"}},
            {"name": "check_citations", "arguments": {"query": "答案 [9]"}},
        ]}, answer("好"))

        turn = run_tool_loop("问", tools=[spec_of("search_docs", search),
                                          spec_of("check_citations", cite)], llm=llm)

        assert turn.steps == 2
        assert turn.used_tools == ["search_docs", "check_citations"]
        assert search.calls != [] and cite.calls != []

    def test_kwargs_style_handler_gets_validated_values(self) -> None:
        """另一种执行体形状（clinic 的 `handler(**kwargs)`）也能被正确调用。"""
        seen: list[dict[str, Any]] = []

        def kwargs_handler(**kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs)
            return {"echo": kwargs}

        llm = StubLLM(call("search_docs", query=" 前导空格"), answer("ok"))
        turn = run_tool_loop("问", tools=[spec_of("search_docs", kwargs_handler)], llm=llm)

        assert seen == [{"query": " 前导空格", "top_k": 5}]  # 默认值由 schema 补齐
        assert turn.tool_calls[0].result == {"echo": {"query": " 前导空格", "top_k": 5}}


# ===========================================================================
# 错误回灌：不存在 / 参数非法 / handler 失败
# ===========================================================================
class TestErrorFeedback:
    def test_unknown_tool_is_fed_back_not_raised(self) -> None:
        """工具名不存在 → 不抛异常、error 非空，且该错误被回灌给下一轮 LLM。"""
        llm = StubLLM(call("no_such_tool", query="x"), answer("换个说法"))

        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm)

        assert turn.tool_calls[0].error != ""
        assert "no_such_tool" in turn.tool_calls[0].error
        # 错误里带上可用工具名，模型下一轮才知道能改成什么
        assert "search_docs" in turn.tool_calls[0].error
        assert turn.answer == "换个说法"  # 循环继续，没有中断
        fed_back = llm.messages[1][-1]
        assert fed_back["role"] == "tool"
        assert "no_such_tool" in fed_back["content"]

    def test_unknown_tool_twice_in_a_row_is_noted(self) -> None:
        """同一个工具名连续两次不存在 → 在 notes 里记录（提示词/schema 可能在误导模型）。"""
        llm = StubLLM(call("ghost", query="a"), call("ghost", query="b"), answer("ok"))

        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm)

        assert any("ghost" in n and "不存在" in n for n in turn.notes)

    @pytest.mark.parametrize("arguments", [
        {"query": "x", "top_k": 0},          # 越出 ge=1
        {"top_k": 3},                        # 缺必填 query
        {"query": "x", "top_k": "很多"},      # 类型不对
    ])
    def test_invalid_arguments_never_reach_the_handler(self, arguments: dict[str, Any]) -> None:
        """参数不合法 → error 含校验信息、handler 一次都不许被调用、错误回灌。"""
        recorder = Recorder()
        llm = StubLLM(call("search_docs", **arguments), answer("改好了"))

        turn = run_tool_loop("问", tools=[spec_of("search_docs", recorder)], llm=llm)

        assert "参数校验未通过" in turn.tool_calls[0].error
        assert turn.tool_calls[0].ok is False
        assert recorder.calls == []
        assert "参数校验未通过" in llm.messages[1][-1]["content"]

    def test_handler_tool_error_is_recorded_and_loop_continues(self) -> None:
        """handler 抛 ToolError → error 记录（不冒 traceback），循环继续走到回答。"""

        def boom(params: SearchDocsParams) -> Any:
            raise ToolError("检索器连不上")

        llm = StubLLM(call("search_docs", query="x"), answer("检索暂时不可用，先说结论…"))
        turn = run_tool_loop("问", tools=[spec_of("search_docs", boom)], llm=llm)

        assert "检索器连不上" in turn.tool_calls[0].error
        assert turn.tool_calls[0].result is None
        assert turn.steps == 2
        assert turn.answer == "检索暂时不可用，先说结论…"
        assert "检索器连不上" in llm.messages[1][-1]["content"]

    def test_handler_unexpected_exception_carries_type_name(self) -> None:
        """handler 抛任意异常 → 同样不崩，error 里带异常类型名（便于排查）。"""

        def boom(params: SearchDocsParams) -> Any:
            raise ZeroDivisionError("division by zero")

        llm = StubLLM(call("search_docs", query="x"), answer("ok"))
        turn = run_tool_loop("问", tools=[spec_of("search_docs", boom)], llm=llm)

        assert "ZeroDivisionError" in turn.tool_calls[0].error
        assert turn.answer == "ok"

    def test_malformed_arguments_json_is_fed_back(self) -> None:
        """arguments 不是合法 JSON（有些适配器原样透传字符串）→ 记错误并回灌，不崩。"""
        llm = StubLLM({"content": "", "tool_calls": [
            {"name": "search_docs", "arguments": "{不是 JSON"}]}, answer("我按 schema 重来"))
        recorder = Recorder()

        turn = run_tool_loop("问", tools=[spec_of("search_docs", recorder)], llm=llm)

        assert "不是合法 JSON" in turn.tool_calls[0].error
        assert recorder.calls == []
        assert "不是合法 JSON" in llm.messages[1][-1]["content"]


# ===========================================================================
# 有界、安全、降级
# ===========================================================================
class TestBoundSafetyDegrade:
    def test_max_steps_bounds_the_loop(self) -> None:
        """max_steps 生效：模型一直要工具也停得下来，notes 写明达到步数上限。"""
        recorder = Recorder()
        llm = StubLLM(*[call("search_docs", query=f"q{i}") for i in range(5)])

        turn = run_tool_loop("问", tools=[spec_of("search_docs", recorder)], llm=llm, max_steps=3)

        assert turn.steps == 3
        assert turn.steps <= 3
        assert turn.llm_calls == 3
        assert len(turn.tool_calls) == 3
        assert turn.answer == ""
        assert any("步数上限" in n for n in turn.notes)

    def test_zero_max_steps_is_rejected(self) -> None:
        """max_steps < 1 是编程错误：直接拒绝，而不是跑出一个没有意义的空轨迹。"""
        with pytest.raises(ValueError, match="max_steps"):
            run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=StubLLM(),
                          max_steps=0)

    def test_side_effect_tool_without_confirm_is_refused(self) -> None:
        """side_effect=True 且未提供 confirm → 拒绝执行、handler 未被调用、理由回灌。"""
        booked = Recorder({"ticket": "A001"})
        llm = StubLLM(call("book_appointment", query="内科"), answer("需要您确认后我再取号"))

        turn = run_tool_loop("帮我挂个号", llm=llm,
                             tools=[spec_of("book_appointment", booked, side_effect=True)])

        assert booked.calls == []  # 安全属性：默认不执行不可逆操作
        assert "确认" in turn.tool_calls[0].error
        assert turn.tool_calls[0].result is None
        assert "确认" in llm.messages[1][-1]["content"]
        assert turn.steps == 2
        assert any("副作用" in n for n in turn.notes)

    def test_side_effect_tool_runs_only_after_confirm(self) -> None:
        """提供 confirm 且回调同意 → 副作用工具才真的执行；确认看到的是校验后的参数。"""
        booked = Recorder({"ticket": "A001"})
        seen: list[ToolCall] = []

        def approve(proposal: ToolCall) -> bool:
            seen.append(proposal)
            return True

        llm = StubLLM(call("book_appointment", query="内科"), answer("已取号"))
        turn = run_tool_loop("挂号", llm=llm, confirm=approve,
                             tools=[spec_of("book_appointment", booked, side_effect=True)])

        assert booked.calls == [{"query": "内科", "top_k": 5}]
        assert len(seen) == 1
        assert seen[0].name == "book_appointment"
        assert seen[0].arguments["top_k"] == 5  # 默认值已补齐，用户确认的是真正会执行的调用
        assert turn.tool_calls[0].result == {"ticket": "A001"}

    def test_side_effect_tool_refused_when_confirm_says_no(self) -> None:
        """confirm 回调拒绝 → 同样不执行，并把"用户拒绝"回灌给模型。"""
        booked = Recorder({"ticket": "A001"})

        def refuse(proposal: ToolCall) -> bool:
            return False

        llm = StubLLM(call("book_appointment", query="内科"), answer("好的，先不挂号"))
        turn = run_tool_loop("挂号", llm=llm, confirm=refuse,
                             tools=[spec_of("book_appointment", booked, side_effect=True)])

        assert booked.calls == []
        assert "拒绝" in turn.tool_calls[0].error
        assert turn.answer == "好的，先不挂号"

    def test_llm_failure_degrades_without_crashing(self) -> None:
        """llm.choose 抛异常 → 不崩、notes 写明原因、回答为空（不编造内容）。"""
        llm = StubLLM(RuntimeError("供应商 502"))

        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm)

        assert isinstance(turn, AgentTurn)
        assert turn.answer == ""
        assert turn.tool_calls == []
        assert any("RuntimeError" in n and "供应商 502" in n for n in turn.notes)
        assert turn.steps == 1
        assert turn.llm_calls == 1

    def test_llm_failure_after_a_tool_call_keeps_results(self) -> None:
        """LLM 在拿到工具结果之后才失败 → 结果仍留在轨迹里（不丢证据），回答为空。"""
        recorder = Recorder({"hits": []})
        llm = StubLLM(call("search_docs", query="x"), RuntimeError("超时"))

        turn = run_tool_loop("问", tools=[spec_of("search_docs", recorder)], llm=llm)

        assert turn.tool_calls[0].result == {"hits": []}
        assert turn.answer == ""
        assert any("LLM 调用失败" in n for n in turn.notes)

    def test_cost_is_accumulated_when_the_adapter_reports_it(self) -> None:
        """适配器上报 cost_usd 时累加（含数字字符串）；脏值按 0 处理，不影响主流程。"""
        llm = StubLLM({"content": "", "cost_usd": 0.25, "tool_calls": [
            {"name": "search_docs", "arguments": {"query": "x"}}]},
            {"content": "答", "cost_usd": "0.5"})
        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm)

        assert turn.cost_usd == pytest.approx(0.75)
        assert turn.answer == "答"

        dirty = StubLLM({"content": "答", "cost_usd": "免费", "tool_calls": []})
        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=dirty)

        assert turn.cost_usd == 0.0


# ===========================================================================
# 真实接线：rag_tools 产出的 ToolSpec 必须能被循环直接驱动
# ===========================================================================
class TestRagToolsWiring:
    def test_search_tool_from_rag_tools_is_driven_by_the_loop(self) -> None:
        """真实接线：`make_search_tool` 的 ToolSpec 能被循环按 schema 驱动（命中被回灌）。"""
        class FakeRetriever:
            """最小检索器桩：只实现 `search(query, top_k=...)`。"""

            def __init__(self) -> None:
                self.queries: list[tuple[str, int]] = []

            def search(self, query: str, top_k: int) -> Any:
                self.queries.append((query, top_k))
                return SimpleNamespace(results=[SimpleNamespace(
                    chunk_id="c1", doc_id="d1", title="Pod 生命周期", breadcrumb="概念",
                    source_url="https://kubernetes.io/docs/concepts/workloads/pods/",
                    body="Pod 的 restartPolicy 决定容器退出后是否重启。")])

        retriever = FakeRetriever()
        llm = StubLLM(call("search_docs", query="pod 重启策略", top_k=9), answer("见 [1]"))

        turn = run_tool_loop("pod 的重启策略是什么？",
                             tools=[make_search_tool(retriever, top_k=3)], llm=llm)

        # 工具自己的检索预算（top_k=3）生效：模型要 9 条也只给 3 条
        assert retriever.queries == [("pod 重启策略", 3)]
        assert turn.used_tools == ["search_docs"]
        assert turn.tool_calls[0].result["n"] == 1
        assert "Pod 生命周期" in llm.messages[1][-1]["content"]

    def test_large_tool_result_is_truncated_before_refeed(self) -> None:
        """工具结果过大 → 回灌时截断（否则一次检索就能把对话历史挤出上下文）。"""
        recorder = Recorder({"body": "x" * 20000})
        llm = StubLLM(call("search_docs", query="x"), answer("ok"))

        run_tool_loop("问", tools=[spec_of("search_docs", recorder)], llm=llm)

        refed = llm.messages[1][-1]["content"]
        assert len(refed) < 20000
        assert "已截断" in refed
