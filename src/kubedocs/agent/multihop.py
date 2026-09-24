"""带依赖的多跳推理（multi-hop）：第 N 跳的查询要用第 N-1 跳的**答案**。

## 与 self-ask 的根本区别

`selfask.py` 是**并行分解**：把问题拆成互不依赖的子问题，各自检索，再合成。
它对"单轮 top-k 装不下两组证据"有用，但对**真正的多跳问题无能为力**：

    问：哪个 API 版本引入了 sidecar 容器，它的默认行为是什么？
    跳1：sidecar 容器是哪个 API 版本引入的？        → 答案 A
    跳2：<用 A 作为检索词> 这个版本的默认行为？      → 答案 B 依赖 A

`selfask` 无法表达"跳2 的问题由跳1 的答案决定"，因为它在分解时还不知道 A。

## 这里的依赖怎么表达

规划器给出的每一跳可以带 `{prev}` 占位符：

    [{"question": "which API version introduced sidecar containers"},
     {"question": "{prev} default behavior for sidecar containers", "needs_previous": true}]

执行时把跳1 的答案填进跳2 的 `{prev}`。**填进去的是答案原文，不是改写**——
不额外调 LLM 去"改写查询"，因为那会再加一次调用且引入不可控的改写偏差。

## 有界性

- `max_hops` 硬上限（默认 3）：每跳至少 1 次 LLM + 1 次检索，无界多跳的延迟与
  成本是乘法增长的；
- 只做**单向链**，不做分支与回溯：分支会让"跳数"失去意义，也让成本无法预估。
- 跳1 的答案为空时，依赖它的跳**退化为用原问题检索**并在 notes 记录——
  宁可少一跳的信息，也不能让整条链断掉。
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# 查询里引用上一跳答案的占位符
PREV_PLACEHOLDER = "{prev}"

# 引用标记（与 generate.validate_citations 同一套约定）
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


class HopPlanner(Protocol):
    """多跳规划与作答的 LLM 接口（结构化输出，不解析自由文本的语义）。"""

    def plan(self, question: str, max_hops: int) -> list[dict[str, Any]]:
        """返回 [{question, needs_previous}, ...]，长度 1~max_hops。"""

    def answer_hop(self, question: str, context: str) -> str:
        """依据检索到的上下文回答**这一跳**的问题。"""

    def compose(self, question: str, hops: Sequence[HopResult]) -> str:
        """综合各跳结果回答原问题，保留 [n] 引用。"""


@dataclass(frozen=True)
class Hop:
    index: int
    template: str
    needs_previous: bool = False


@dataclass
class HopResult:
    hop: Hop
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)

    def describe(self) -> str:
        dep = "←依赖上一跳" if self.hop.needs_previous else ""
        return f"跳{self.hop.index} {dep}\n    查询: {self.query}\n    作答: {self.answer[:120]}"


@dataclass
class MultiHopResult:
    question: str
    hops: list[HopResult]
    answer: str
    citations: list[dict]
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    @property
    def has_dependency(self) -> bool:
        return any(h.hop.needs_previous for h in self.hops)

    def hop_answers(self) -> list[str]:
        return [h.answer for h in self.hops]


def parse_plan(raw: Sequence[Mapping[str, Any]], *, max_hops: int = 3) -> list[Hop]:
    """把规划器的原始输出规范化成 `Hop` 列表。

    第一跳**强制不依赖**上一跳（没有上一跳可用）——规划器偶尔会给出
    `needs_previous=true` 的第一跳，静默接受会让执行时拿到空占位符。
    """
    hops: list[Hop] = []
    for i, item in enumerate(raw[:max(1, max_hops)], start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        needs_prev = bool(item.get("needs_previous")) and i > 1
        if PREV_PLACEHOLDER in question and i > 1:
            needs_prev = True
        hops.append(Hop(index=len(hops) + 1, template=question, needs_previous=needs_prev))
    return hops


def resolve_query(hop: Hop, previous_answer: str) -> str:
    """把 `{prev}` 换成上一跳的答案；没有占位符时原样返回。"""
    if PREV_PLACEHOLDER in hop.template:
        return hop.template.replace(PREV_PLACEHOLDER, previous_answer).strip()
    if hop.needs_previous and previous_answer:
        # 规划器说"依赖"但没写占位符：把上一跳答案拼在后面，让它进入检索
        return f"{hop.template} {previous_answer}".strip()
    return hop.template


def _renumber(citations_by_hop: Sequence[Sequence[Mapping[str, Any]]],
              ) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    """各跳的局部引用合并成全局编号，按 chunk_id 去重。

    返回 (全局引用列表, {(跳下标, 局部编号): 全局编号})。
    """
    merged: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for hop_i, cites in enumerate(citations_by_hop):
        for local, c in enumerate(cites, start=1):
            key = str(c.get("chunk_id", ""))
            if not key:
                continue
            if key not in seen:
                seen[key] = len(merged) + 1
                merged.append({**c, "index": seen[key]})
            mapping[(hop_i, local)] = seen[key]
    return merged, mapping


def multi_hop(question: str, *, search: Callable[[str], Any], llm: HopPlanner,
              max_hops: int = 3, per_hop_top_k: int = 4) -> MultiHopResult:
    """执行多跳：规划 → 逐跳（带依赖）检索作答 → 合成。"""
    result = MultiHopResult(question=question, hops=[], answer="", citations=[])
    try:
        raw = llm.plan(question, max_hops)
        result.llm_calls += 1
    except Exception as e:  # noqa: BLE001 - 规划失败不该让整条链崩掉
        result.notes.append(f"规划失败，退化为单轮检索：{type(e).__name__}: {e}")
        raw = [{"question": question, "needs_previous": False}]

    hops = parse_plan(raw, max_hops=max_hops)
    if not hops:
        hops = [Hop(index=1, template=question)]
        result.notes.append("规划结果为空，退化为单跳")

    prev_answer = ""
    cites_per_hop: list[list[dict[str, Any]]] = []
    for hop in hops:
        query = resolve_query(hop, prev_answer)
        if hop.needs_previous and not prev_answer:
            result.notes.append(f"跳{hop.index} 依赖上一跳但上一跳无答案，退化为原问题检索")
            query = hop.template.replace(PREV_PLACEHOLDER, "").strip() or question
        try:
            raw_hits = search(query)
            items = list(getattr(raw_hits, "results", raw_hits) or [])[:per_hop_top_k]
        except Exception as e:  # noqa: BLE001 - 单跳检索失败只损失这一跳
            result.notes.append(f"跳{hop.index} 检索失败：{type(e).__name__}: {e}")
            items = []
        contexts, cites = _contexts_and_citations(items)
        cites_per_hop.append(cites)
        try:
            answer = llm.answer_hop(query, contexts) if contexts else ""
            if contexts:
                result.llm_calls += 1
        except Exception as e:  # noqa: BLE001
            result.notes.append(f"跳{hop.index} 作答失败：{type(e).__name__}: {e}")
            answer = ""
        result.hops.append(HopResult(hop=hop, query=query, answer=answer, citations=cites))
        prev_answer = answer or prev_answer

    merged, mapping = _renumber(cites_per_hop)
    result.citations = merged
    # 各跳答案里的局部 [n] 换成全局编号，这样合成的答案引用才是对的
    for hop_i, hr in enumerate(result.hops):
        def _sub(m: re.Match[str], hop_i: int = hop_i) -> str:
            new = mapping.get((hop_i, int(m.group(1))))
            return f"[{new}]" if new else ""

        hr.answer = _sub_all(_MARKER_RE, hr.answer, _sub)

    try:
        if len(result.hops) > 1:
            result.answer = llm.compose(question, result.hops)
            result.llm_calls += 1
        else:
            result.answer = result.hops[0].answer
    except Exception as e:  # noqa: BLE001
        result.notes.append(f"合成失败，返回各跳答案：{type(e).__name__}: {e}")
        result.answer = "\n\n".join(f"{h.query} -> {h.answer}" for h in result.hops)
    return result


def _sub_all(pattern: re.Pattern[str], text: str,
             fn: Callable[[re.Match[str]], str]) -> str:
    return pattern.sub(fn, text)


def _contexts_and_citations(items: Sequence[Any]) -> tuple[str, list[dict[str, Any]]]:
    """把检索结果拼成上下文，并返回局部引用（编号从 1）。"""
    blocks: list[str] = []
    cites: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        get = (item.get if isinstance(item, Mapping) else
               lambda k, d=None, _it=item: getattr(_it, k, d))
        body = str(get("body", "") or "")
        title = str(get("title", "") or "")
        crumb = str(get("breadcrumb", "") or "")
        label = f"{title} › {crumb}" if crumb else title
        blocks.append(f"[{i}] {label}\n{body[:900]}")
        cites.append({"index": i, "chunk_id": get("chunk_id", ""), "title": title,
                      "breadcrumb": crumb, "source_url": get("source_url", "")})
    return "\n\n".join(blocks), cites
