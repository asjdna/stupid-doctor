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


# ===========================================================================
# 多轮上下文：history 必须**真的**进模型，且不带 history 时序列逐字不变
# ===========================================================================
class TestHistoryReachesTheModel:
    """这一组是"多轮上下文有没有丢"的唯一硬证据。

    为什么不能只断言"history 参数被接了"：那是个静态事实，改动接线（把 history 传进
    循环却仍不拼进 messages）照样能通过。这里直接抓 `choose` 收到的 messages 列表——
    模型拿到什么，就断言什么。
    """

    HISTORY: list[dict[str, Any]] = [
        {"role": "user", "content": "我最近总是头晕"},
        {"role": "assistant", "content": "头晕多久了？有没有测过血压？"},
    ]

    def _captured(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """跑一轮并把 `choose` 收到的每一份 messages 快照返回。"""
        llm = StubLLM(answer("据上次说的…"))
        run_tool_loop("昨天量了 150/95", tools=[spec_of("search_docs", Recorder())],
                      llm=llm, **kwargs)
        return llm.messages

    def test_history_is_sent_in_order_between_system_and_the_current_question(self) -> None:
        """顺序必须是 system → history（原序，角色原样）→ 本轮 user。"""
        messages = self._captured(system="你是导诊助手", history=list(self.HISTORY))

        assert messages[0] == [
            {"role": "system", "content": "你是导诊助手"},
            {"role": "user", "content": "我最近总是头晕"},
            {"role": "assistant", "content": "头晕多久了？有没有测过血压？"},
            {"role": "user", "content": "昨天量了 150/95"},
        ], "history 没有按 system → history → 本轮 user 的顺序进模型"

    def test_history_without_system_still_lands_before_the_question(self) -> None:
        """没有 system 提示时同样成立：history 在前、本轮问题在最后。"""
        messages = self._captured(history=list(self.HISTORY))

        assert messages[0] == [*self.HISTORY, {"role": "user", "content": "昨天量了 150/95"}]

    def test_history_trailing_user_message_is_not_merged_or_reordered(self) -> None:
        """history 以 user 结尾时**不做合并**（诚实边界：连续两条 user 原样发出）。

        这条不是在主张"两条连续 user 是对的"，而是钉住"循环不替调用方猜意图"：
        合并 content 会改写用户原话，重排会颠倒他说话的先后。
        """
        history: list[dict[str, Any]] = [{"role": "user", "content": "在吗"}]

        messages = self._captured(history=history)

        assert messages[0] == [{"role": "user", "content": "在吗"},
                               {"role": "user", "content": "昨天量了 150/95"}]

    def test_extra_keys_in_history_entries_are_not_forwarded(self) -> None:
        """`tool_calls` 之类多余字段不透传：没有配对的 tool 消息，半个工具调用会污染上下文。"""
        history: list[dict[str, Any]] = [
            {"role": "assistant", "content": "稍等", "tool_calls": [{"id": "call_1"}],
             "name": "search_docs", "extra": 1}]

        messages = self._captured(history=history)

        assert messages[0][0] == {"role": "assistant", "content": "稍等"}

    def test_caller_history_list_is_not_mutated(self) -> None:
        """不改调用方的列表：`api/app.py` 把同一份 history 存在 `app.state` 上复用。"""
        history = [dict(item) for item in self.HISTORY]
        before = copy.deepcopy(history)

        self._captured(history=history)

        assert history == before

    def test_history_is_refed_back_into_later_rounds_too(self) -> None:
        """第一轮就带 history 时，第二次 `choose`（工具结果之后）同样能看到它。"""
        llm = StubLLM(call("search_docs", query="血压"), answer("见 [1]"))

        run_tool_loop("昨天量了 150/95", tools=[spec_of("search_docs", Recorder())], llm=llm,
                      system="S", history=list(self.HISTORY))

        second = llm.messages[1]
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user",
                                               "assistant", "tool"]
        assert second[1]["content"] == "我最近总是头晕"
        assert second[2]["content"] == "头晕多久了？有没有测过血压？"


class TestHistoryDefaultsAreUnchanged:
    """回归：不传 history 时的消息序列必须与加这个参数之前**逐字相同**。"""

    def test_default_none_sends_exactly_system_and_question(self) -> None:
        llm = StubLLM(answer("答"))

        run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                      system="S")

        assert llm.messages[0] == [{"role": "system", "content": "S"},
                                   {"role": "user", "content": "问"}]

    def test_default_none_without_system_sends_exactly_one_user_message(self) -> None:
        llm = StubLLM(answer("答"))

        run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm)

        assert llm.messages[0] == [{"role": "user", "content": "问"}]

    def test_none_empty_and_missing_history_are_indistinguishable(self) -> None:
        """`None` / `[]` / 不传 三种写法必须产出**同一份** messages。"""
        runs = []
        for kwargs in ({}, {"history": None}, {"history": []}):
            llm = StubLLM(answer("答"))
            run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                          system="S", **kwargs)
            runs.append(llm.messages[0])

        assert runs[0] == runs[1] == runs[2]
        assert runs[0] == [{"role": "system", "content": "S"},
                           {"role": "user", "content": "问"}]


class TestInvalidHistoryIsRejectedNotSilentlyDropped:
    """非法 history 必须**报错**：静默丢弃在日志里与"模型没记住"无法区分。

    每个用例都断言 `choose` **一次都没被调用**——校验必须发生在请求发出去之前，
    否则"报错"就变成了"先花一次钱再报错"。
    """

    @pytest.mark.parametrize("bad, why", [
        ({"role": "system", "content": "偷偷改掉的系统提示"}, "system 由 system= 独占"),
        ({"role": "tool", "content": "无主工具结果"}, "tool 消息没有配对"),
        ({"role": "developer", "content": "未知角色"}, "角色不在白名单"),
        ({"role": "", "content": "空角色"}, "空角色"),
        ({"content": "缺 role"}, "缺 role"),
        ({"role": "user", "content": 42}, "content 不是字符串"),
        ({"role": "user", "content": None}, "content 为 None"),
        ({"role": "assistant"}, "缺 content"),
        ("我不是对象", "条目不是 dict"),
    ])
    def test_bad_history_raises_before_any_llm_call(self, bad: Any, why: str) -> None:
        llm = StubLLM(answer("答"))

        with pytest.raises(ValueError):
            run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                          history=[bad])

        assert llm.messages == [], f"非法 history（{why}）却在报错前调用了模型"

    def test_error_message_names_the_bad_index(self) -> None:
        """报错要说清是第几条：history 是多轮拼出来的，只说"不合法"等于让人自己找。"""
        llm = StubLLM(answer("答"))

        with pytest.raises(ValueError, match=r"history\[1\]"):
            run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                          history=[{"role": "user", "content": "ok"},
                                   {"role": "system", "content": "不行"}])


# ===========================================================================
# 预算闸门（`budget=`）：缺陷回归组
#
# 缺陷形态（SECURITY.md S-2）：`/chat` 的工具循环里，成本只被 `LiteLLMToolCaller`
# 累加到 `self.cost`、从不与预算比较，而 caller 每轮新建 ⇒ `/chat` 的花费没有上限。
# 现在的契约：**每次调用模型之前 check()、之后 record()**；check() 抛出的异常
# **原样上抛**（不许被"LLM 是外部服务"那个 except 降级成一条 note——note 在 /chat
# 的响应里根本到不了调用方）。
# ===========================================================================
class BudgetExceeded(Exception):
    """打桩用的"超限"异常：循环**不认识**这个类型，必须照样上抛。"""


class StubBudget:
    """只数调用次数与金额的假闸门（`SpendGuard` 是鸭子类型）。"""

    def __init__(self, *, limit: float = 1.0, fail_after: int | None = None) -> None:
        self.limit = limit
        self.fail_after = fail_after
        self.checks = 0
        self.records: list[tuple[float, bool]] = []

    def check(self, estimated_usd: float = 0.0) -> None:
        if self.fail_after is not None and self.checks >= self.fail_after:
            raise BudgetExceeded(f"已达上限（第 {self.checks} 次检查）")
        self.checks += 1

    def record(self, usd: float, *, priced: bool = True) -> float:
        self.records.append((usd, priced))
        return sum(u for u, _ in self.records)


class TestBudgetGateInTheLoop:
    def test_budget_is_checked_before_every_model_call(self) -> None:
        """两轮对话 = 两次调用 = 两次检查；检查次数与调用次数同步。"""
        budget = StubBudget()
        llm = StubLLM(call("search_docs", query="q"), answer("答"))
        run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                      budget=budget)
        assert len(llm.messages) == 2
        assert budget.checks == 2

    def test_default_is_no_budget_check(self) -> None:
        """不给 `budget=` 时行为与从前逐字相同（打桩循环、离线自检不受影响）。"""
        llm = StubLLM(answer("答"))
        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm)
        assert turn.answer == "答"

    def test_over_budget_stops_calling_the_model(self) -> None:
        """**关键断言**：超限之后一次模型调用都不发生（不是"先花了再拒"）。"""
        budget = StubBudget(fail_after=1)
        llm = StubLLM(call("search_docs", query="q"), answer("答"), answer("多余的"))

        with pytest.raises(BudgetExceeded):
            run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                          budget=budget)

        assert len(llm.messages) == 1, "第二次调用前就该被预算拦住"
        assert budget.checks == 1

    def test_budget_exceeded_is_not_swallowed_into_a_note(self) -> None:
        """超限异常必须上抛，不能变成 `turn.notes` 里的一句（`/chat` 看不到 notes）。"""
        llm = StubLLM(answer("答"))
        with pytest.raises(BudgetExceeded):
            run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                          budget=StubBudget(fail_after=0))
        assert llm.messages == []

    def test_cost_is_recorded_after_each_call(self) -> None:
        """调用之后按响应里的 `cost_usd` 累计（两轮 = 两笔账）。"""
        budget = StubBudget()
        first = call("search_docs", query="q")
        first["cost_usd"] = 0.25
        second = answer("答")
        second["cost_usd"] = 0.5
        llm = StubLLM(first, second)

        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                             budget=budget)

        assert len(llm.messages) == 2
        assert budget.records == [(0.25, True), (0.5, True)]
        assert turn.cost_usd == pytest.approx(0.75)

    def test_unpriced_cost_is_reported_and_not_treated_as_free(self) -> None:
        """`cost_priced=False` ⇒ 闸门按**未定价**记账，并计入 `turn.unpriced_calls`。"""
        budget = StubBudget()
        llm = StubLLM({"content": "答", "tool_calls": [], "cost_usd": 0.0,
                       "cost_priced": False})
        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                             budget=budget)

        assert budget.records == [(0.0, False)], "未定价的调用必须带 priced=False 交给闸门"
        assert turn.unpriced_calls == 1
        assert any("算不出来" in n for n in turn.notes)
        assert turn.cost_usd == 0.0          # 上报的数字仍是 0，但语义由 priced 区分

    def test_priced_zero_is_not_counted_as_unpriced(self) -> None:
        """真免费（本地模型定价就是 0）与"算不出价"必须分得开。"""
        budget = StubBudget()
        llm = StubLLM({"content": "答", "tool_calls": [], "cost_usd": 0.0,
                       "cost_priced": True})
        turn = run_tool_loop("问", tools=[spec_of("search_docs", Recorder())], llm=llm,
                             budget=budget)
        assert budget.records == [(0.0, True)]
        assert turn.unpriced_calls == 0

