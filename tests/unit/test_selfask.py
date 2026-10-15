"""self-ask 的单测：确定性分解决策、引用重编、降级与有界性；全程离线（假检索器 + 假 Decomposer）。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pytest

from kubedocs.agent.selfask import (
    MAX_SUB_QUESTIONS,
    SelfAskResult,
    SubAnswer,
    needs_decomposition,
    self_ask,
)
from kubedocs.retrieve.hybrid import RetrievedChunk, SearchResult

SHORT = "kubelet 是什么？"
"""短问题：装不下两个子目标，不该分解。"""

COMPOUND = "Pod 处于 Pending 状态时应该怎么排查，和节点资源不足以及调度器配置错误有什么关系？"
"""长复合问题：够长 + 并列连词，是分解的目标场景。"""

TWO_MARKS = "kubelet 怎么启动？它和 API server 怎么通信？"
"""一次问了两个问题（两个问号）。"""

LONG_SINGLE = "请详细说明 kubelet 在节点上启动时如何读取配置文件并注册到 API server 的完整流程是什么？"
"""长，但只有一个目标、没有并列连词：拆它是白花钱。"""


# --------------------------------------------------------------------------- 测试替身
def _chunk(cid: str, body: str) -> RetrievedChunk:
    """造一条检索命中。标题/面包屑都带上，方便断言引用内容。"""
    return RetrievedChunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        ordinal=0,
        title=f"文档 {cid}",
        breadcrumb="小节",
        source_url=f"https://kubernetes.io/docs/{cid}",
        body=body,
    )


class FakeSearch:
    """假检索器：按 query 回放预置结果，并记录被问过哪些 query。"""

    def __init__(
        self,
        table: dict[str, list[RetrievedChunk]] | None = None,
        default: list[RetrievedChunk] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.table = table or {}
        self.default = default if default is not None else []
        self.errors = errors or {}
        self.queries: list[str] = []

    def __call__(self, query: str) -> SearchResult:
        self.queries.append(query)
        if query in self.errors:
            raise self.errors[query]
        return SearchResult(query=query, results=list(self.table.get(query, self.default)))


@dataclass
class FakeDecomposer:
    """假 Decomposer：只回放预置结果，绝不调模型。

    `decompose` 故意不按 max_sub 截断，这样 `self_ask` 自己的截断逻辑才会被测到。
    """

    subs: object = field(default_factory=list)
    composed: str = "合成后的答案"
    decompose_error: Exception | None = None
    compose_error: Exception | None = None
    decompose_calls: int = 0
    compose_calls: int = 0
    seen_max_sub: int = 0
    seen_parts: list[list[SubAnswer]] = field(default_factory=list)

    def decompose(self, question: str, max_sub: int) -> list[str]:
        self.decompose_calls += 1
        self.seen_max_sub = max_sub
        if self.decompose_error is not None:
            raise self.decompose_error
        return cast("list[str]", self.subs)

    def compose(self, question: str, parts: Sequence[SubAnswer]) -> str:
        self.compose_calls += 1
        self.seen_parts.append(list(parts))
        if self.compose_error is not None:
            raise self.compose_error
        return self.composed


def _markers(text: str) -> list[int]:
    """取出正文里的引用编号，按出现顺序。"""
    return [int(m) for m in re.findall(r"\[(\d{1,3})\]", text)]


def _cited_ids(result: SelfAskResult, text: str) -> list[str]:
    """把正文里的 [n] 翻译成它**真正指向**的 chunk_id——引用重编是否正确的判据。"""
    return [str(result.citations[n - 1]["chunk_id"]) for n in _markers(text)]


# --------------------------------------------------------------------------- 分解决策
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (SHORT, False),
        ("", False),
        ("   ", False),
        ("Pod 和 Node 有什么区别？", False),
        (COMPOUND, True),
        (TWO_MARKS, True),
        (LONG_SINGLE, False),
        (COMPOUND.replace("和", "跟").replace("以及", "还有"), False),
    ],
)
def test_needs_decomposition_heuristics(question: str, expected: bool) -> None:
    """分解决策是确定性启发式：长度是闸门，问号个数与并列连词才是信号。

    短问题（哪怕带连词）、长但单目标都不分解（拆它是白花钱）；长 + 表内连词、
    一次多个问号 -> 分解；连词不在表内（"跟"/"还有"）不算信号。
    """
    assert needs_decomposition(question) is expected


def test_min_words_param_is_a_gate() -> None:
    """min_words 可调：门槛调高后同一个复合问题不再分解（闸门优先于连词）。"""
    assert needs_decomposition(COMPOUND) is True
    assert needs_decomposition(COMPOUND, min_words=40) is False


# --------------------------------------------------------------------------- 主流程
def test_short_question_skips_decomposition() -> None:
    """短问题不分解：只检索一次、只调一次合成，llm_calls == 1（绝不白发分解调用）。"""
    llm = FakeDecomposer(subs=["不该被用到的子问题"], composed="kubelet 是节点代理[1]。")
    search = FakeSearch(default=[_chunk("d::0000", "kubelet 是节点上运行的代理")])
    result = self_ask(SHORT, search=search, llm=llm)

    assert result.decomposed is False
    assert result.sub_questions == []
    assert result.sub_answers == []
    assert search.queries == [SHORT]
    assert llm.decompose_calls == 0
    assert llm.compose_calls == 1
    assert result.llm_calls == 1
    assert [c["chunk_id"] for c in result.citations] == ["d::0000"]
    assert result.answer == "kubelet 是节点代理[1]。"
    assert _cited_ids(result, result.answer) == ["d::0000"]


def test_compound_question_decomposes_and_searches_each_sub() -> None:
    """需要分解的复合问题：拆出 2 个子问题、各自检索一次，llm_calls == 2，且无降级记录。"""
    subs = ["节点资源不足会导致什么", "调度器配置错误会导致什么"]
    llm = FakeDecomposer(subs=subs, composed="资源不足会驱逐 Pod[1]；调度错误会 Pending[2]。")
    search = FakeSearch(
        table={
            subs[0]: [_chunk("a::0000", "节点资源不足时 kubelet 会驱逐 Pod")],
            subs[1]: [_chunk("b::0000", "调度器配置错误时 Pod 一直 Pending")],
        }
    )
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.decomposed is True
    assert result.sub_questions == subs
    assert len(result.sub_answers) == 2
    assert search.queries == subs
    assert llm.decompose_calls == 1
    assert llm.compose_calls == 1
    assert result.llm_calls == 2
    assert [c["chunk_id"] for c in result.citations] == ["a::0000", "b::0000"]
    assert _cited_ids(result, result.answer) == ["a::0000", "b::0000"]
    assert result.notes == []


def test_citations_are_renumbered_deduped_and_markers_rewritten() -> None:
    """引用重编号（重点用例）：合并去重 + 正文标记跟着改。

    合并后 A=1、B=2、C=3 且 B 只留一份；子答案 2 与最终答案里的局部标记必须改指全局编号。
    """
    q1, q2 = "子问题一", "子问题二"
    llm = FakeDecomposer(subs=[q1, q2], composed="结论：B 与 C 都相关[2][3]。")
    search = FakeSearch(
        table={
            q1: [_chunk("A::0000", "A 的正文"), _chunk("B::0000", "B 的正文")],
            q2: [_chunk("B::0000", "B 的正文"), _chunk("C::0000", "C 的正文")],
        }
    )
    result = self_ask(COMPOUND, search=search, llm=llm)

    # 1) 引用列表：按出现顺序、去重、编号从 1 连续
    assert [c["index"] for c in result.citations] == [1, 2, 3]
    assert [c["chunk_id"] for c in result.citations] == ["A::0000", "B::0000", "C::0000"]
    assert len({str(c["chunk_id"]) for c in result.citations}) == 3

    # 2) 子答案 1：A=[1]、B=[2] 与全局一致；子答案 2 不许再有 [1]（B->[2]、C->[3]）
    assert _markers(result.sub_answers[0].answer) == [1, 2]
    assert _cited_ids(result, result.sub_answers[0].answer) == ["A::0000", "B::0000"]

    assert _markers(result.sub_answers[1].answer) == [2, 3]
    assert _cited_ids(result, result.sub_answers[1].answer) == ["B::0000", "C::0000"]
    assert [str(c["chunk_id"]) for c in result.sub_answers[1].citations] == ["B::0000", "C::0000"]
    assert _cited_ids(result, result.answer) == ["B::0000", "C::0000"]

    # 3) hit_ids 与引用一一对应；SubAnswer 能直接转成状态里流动的 CitedAnswer
    assert result.sub_answers[1].hit_ids == ["B::0000", "C::0000"]
    cited = result.sub_answers[1].to_cited_answer()
    assert cited.text == result.sub_answers[1].answer
    assert cited.sub_question == q2
    assert cited.citations == result.sub_answers[1].citations


def test_out_of_range_marker_in_composed_answer_is_stripped() -> None:
    """合成答案里越界的 [n] 被剥掉：打不开的引用比没有引用更误导。"""
    llm = FakeDecomposer(composed="结论[1]，另外[9]是模型瞎编的。")
    search = FakeSearch(default=[_chunk("A::0000", "正文")])
    result = self_ask(SHORT, search=search, llm=llm)

    assert result.answer == "结论[1]，另外是模型瞎编的。"
    assert any("越界" in n for n in result.notes)


# --------------------------------------------------------------------------- 降级
def test_decompose_failure_degrades_to_direct_search() -> None:
    """llm 抛异常时不崩：退化成"直接检索一次 + 合成"，原因写进 notes。"""
    llm = FakeDecomposer(decompose_error=RuntimeError("模型超时（测试用）"), composed="直接回答[1]。")
    search = FakeSearch(default=[_chunk("A::0000", "直接检索到的正文")])
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.decomposed is False
    assert result.sub_questions == []
    assert result.sub_answers == []
    assert search.queries == [COMPOUND]
    assert llm.compose_calls == 1
    assert result.llm_calls == 1  # 抛异常的分解调用不计入（见 SelfAskResult 文档）
    assert result.answer == "直接回答[1]。"
    assert _cited_ids(result, result.answer) == ["A::0000"]
    assert any("分解失败" in n and "模型超时" in n for n in result.notes)


def test_compose_failure_falls_back_to_evidence() -> None:
    """合成失败也不崩：退回已重编引用的证据原文，引用编号仍正确；llm_calls 只计成功调用。"""
    llm = FakeDecomposer(compose_error=RuntimeError("上游 500（测试用）"))
    search = FakeSearch(default=[_chunk("A::0000", "A 的正文")])
    result = self_ask(SHORT, search=search, llm=llm)

    assert result.answer.startswith("[1] 文档 A::0000 › 小节")
    assert "A 的正文" in result.answer
    assert result.llm_calls == 0
    assert any("合成失败" in n for n in result.notes)


def test_search_failure_on_one_sub_is_isolated() -> None:
    """单个子问题检索抛异常只让那一路没证据，不影响其它子问题与最终合成。"""
    q1, q2 = "正常子问题", "会炸的子问题"
    llm = FakeDecomposer(subs=[q1, q2], composed="回答[1]。")
    search = FakeSearch(
        table={q1: [_chunk("A::0000", "A 的正文")]}, errors={q2: ConnectionError("检索后端不可达（测试用）")}
    )
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.decomposed is True
    assert [c["chunk_id"] for c in result.citations] == ["A::0000"]
    assert result.sub_answers[1].citations == []
    assert result.sub_answers[1].answer == ""
    assert result.answer == "回答[1]。"
    assert any("检索失败" in n for n in result.notes)


def test_empty_sub_result_still_composes() -> None:
    """某个子问题检索为空时仍能合成：不能因为一路没证据就整体失败。"""
    q1, q2 = "有证据的子问题", "没证据的子问题"
    llm = FakeDecomposer(subs=[q1, q2], composed="只能依据已知证据回答[1]。")
    search = FakeSearch(table={q1: [_chunk("A::0000", "A 的正文")]})
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.decomposed is True
    assert len(result.sub_answers) == 2
    assert result.sub_answers[1].citations == []
    assert result.sub_answers[1].answer == ""
    assert [c["chunk_id"] for c in result.citations] == ["A::0000"]
    assert result.answer == "只能依据已知证据回答[1]。"
    assert result.llm_calls == 2
    assert llm.compose_calls == 1
    assert any("没有命中" in n for n in result.notes)


# --------------------------------------------------------------------------- 有界性
def test_sub_questions_truncated_to_max_sub() -> None:
    """子问题数超过 max_sub 时被截断：每个子问题都要花一次检索，必须设界。"""
    six = [f"子问题{i}" for i in range(6)]
    llm = FakeDecomposer(subs=six)
    search = FakeSearch(default=[_chunk("x::0000", "正文")])
    result = self_ask(COMPOUND, search=search, llm=llm, max_sub=4)

    assert result.sub_questions == six[:4]
    assert len(search.queries) == 4
    assert llm.seen_max_sub == 4
    assert any("超过上限" in n for n in result.notes)


def test_max_sub_over_hard_cap_is_clamped() -> None:
    """max_sub 再大也被硬上限钉住：宽度直接决定延迟与成本，不给调用方放开的口子。"""
    llm = FakeDecomposer(subs=[f"子问题{i}" for i in range(9)])
    search = FakeSearch(default=[_chunk("x::0000", "正文")])
    result = self_ask(COMPOUND, search=search, llm=llm, max_sub=99)

    assert len(result.sub_questions) == MAX_SUB_QUESTIONS
    assert llm.seen_max_sub == MAX_SUB_QUESTIONS
    assert any(f"上限 {MAX_SUB_QUESTIONS}" in n for n in result.notes)


def test_max_sub_below_two_skips_decompose_call() -> None:
    """max_sub < 2 时连分解调用都不发：装不下两个子问题就没有分解可言，发了是白花钱。"""
    llm = FakeDecomposer(subs=["a", "b"])
    search = FakeSearch(default=[_chunk("x::0000", "正文")])
    result = self_ask(COMPOUND, search=search, llm=llm, max_sub=1)

    assert result.decomposed is False
    assert llm.decompose_calls == 0
    assert search.queries == [COMPOUND]
    assert result.llm_calls == 1
    assert any("跳过分解" in n for n in result.notes)


def test_single_sub_question_is_treated_as_undecomposed() -> None:
    """只拆出一个子问题等于没拆：按未分解处理，但仍如实计入那次已经发生的分解调用。"""
    llm = FakeDecomposer(subs=["只有一个"], composed="回答[1]。")
    search = FakeSearch(default=[_chunk("A::0000", "正文")])
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.decomposed is False
    assert result.sub_questions == []
    assert search.queries == [COMPOUND]
    assert result.llm_calls == 2  # 分解成功返回（+1）+ 合成（+1）
    assert any("不足 2 个子问题" in n for n in result.notes)


def test_duplicate_and_empty_sub_questions_are_dropped() -> None:
    """分解结果里的空串与重复子问题被去掉：重复子问题等于重复检索。"""
    llm = FakeDecomposer(subs=["同一个子问题", "", "同一个子问题", "另一个子问题"])
    search = FakeSearch(default=[_chunk("A::0000", "正文")])
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.sub_questions == ["同一个子问题", "另一个子问题"]
    assert search.queries == ["同一个子问题", "另一个子问题"]
    assert any("重复" in n for n in result.notes)


def test_string_decompose_result_is_not_split_into_characters() -> None:
    """模型只返回一个字符串时按单个子问题处理，绝不能被逐字符拆成几十个子问题。"""
    llm = FakeDecomposer(subs="只有一个子问题")
    search = FakeSearch(default=[_chunk("A::0000", "正文")])
    result = self_ask(COMPOUND, search=search, llm=llm)

    assert result.decomposed is False
    assert result.sub_questions == []
    assert search.queries == [COMPOUND]
    assert any("字符串" in n for n in result.notes)


# --------------------------------------------------------------------------- 检索解耦
def test_dict_shaped_search_result_is_tolerated() -> None:
    """检索返回 dict 形状（而非 SearchResult）也能用——内核不认识检索实现。"""

    def fake_search(query: str) -> dict[str, object]:
        return {
            "results": [
                {
                    "chunk_id": "A::0000",
                    "title": "标题",
                    "breadcrumb": "小节",
                    "source_url": "https://example.com/a",
                    "body": "正文",
                }
            ]
        }

    llm = FakeDecomposer(composed="回答[1]。")
    result = self_ask(SHORT, search=fake_search, llm=llm)

    assert [c["chunk_id"] for c in result.citations] == ["A::0000"]
    assert result.citations[0]["source_url"] == "https://example.com/a"
    assert llm.seen_parts[0][0].answer == "[1] 标题 › 小节\n正文"


def test_hit_without_chunk_id_is_skipped_without_shifting_numbering() -> None:
    """缺 chunk_id 的命中被跳过，且不让后面命中与引用编号错位一格。

    错位是最隐蔽的引用 bug：正文写着 [1] A、[2] B，引用列表却是 B、C。
    """

    class NoId:
        title = "没有 id 的块"
        breadcrumb = ""
        source_url = "https://example.com/x"
        body = "正文"

    def fake_search(query: str) -> dict[str, object]:
        return {
            "results": [
                NoId(),
                {
                    "chunk_id": "A::0000",
                    "title": "标题",
                    "breadcrumb": "",
                    "source_url": "https://example.com/a",
                    "body": "正文",
                },
            ]
        }

    llm = FakeDecomposer(composed="回答[1]。")
    result = self_ask(SHORT, search=fake_search, llm=llm)

    assert [c["chunk_id"] for c in result.citations] == ["A::0000"]
    evidence = llm.seen_parts[0][0].answer
    assert _markers(evidence) == [1]
    assert _cited_ids(result, evidence) == ["A::0000"]
    assert any("缺 chunk_id" in n for n in result.notes)
