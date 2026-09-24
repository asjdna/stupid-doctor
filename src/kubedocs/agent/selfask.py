"""self-ask：先把复合问题拆成子问题分别检索，再合成一个带引用的答案。

## 什么时候值得分解

self-ask 的收益来自**检索的针对性**：一个复合问题（"节点资源不足和调度器配置错误
分别会导致什么现象"）整体丢给检索器时，两半信息互相稀释，召回常常只覆盖其中一半。
拆成两个子问题各检索一次，每一路都能召回自己那份证据。

代价是固定的：多一次 LLM 调用 + N 次检索，延迟与费用随 N 线性上涨。所以"要不要分解"
必须是**确定性可判**的（见 `needs_decomposition`，不调 LLM）：把这个决策交给模型，
等于让模型自己决定要不要多花钱，而且不可复现、没法单测。

## 只做一层分解，不做递归

子问题**不再**交给 self-ask 自己拆第二层，这是刻意的取舍：每多一层，LLM 调用与检索
次数按子问题数成倍增长（第二层就是 N×M 次检索），延迟与成本失去上界；而收益衰减很快
——第二层的"子问题"通常已经是可以直接检索的原子问题了，再拆只增加噪声。
递归分解的系统出故障时表现为"延迟莫名翻倍、成本失控"，还很难定位是哪一层炸的。
因此宁可用 `max_sub`（硬上限 4）把**宽度**限死，也不放开**深度**。

## 引用编号必须重编（本项目最容易出错的地方）

每个子问题的引用天然是**局部编号**（各自从 [1] 开始，与 `generate.build_context` 一致）。
把 N 个子答案合并时冲突是必然的：两个子答案各有一个 `[1]`，指向不同的块。如果只重编
引用列表、不改正文里的 `[n]`，用户点开 `[1]` 会跳到另一条来源；肉眼读答案基本看不出来，
但评测闸门的引用核对会成片失败。所以合并时**同一份 (局部 -> 全局) 映射同时作用于
引用列表与正文标记**，并按 chunk_id 去重（同一块只保留首次出现的那一号）。

## 降级必须是"有记录的降级"

`Decomposer` 是外部 LLM 适配器，随时可能超时/报错。任何一次 LLM 或检索失败都
**不允许**把异常抛给上层：退化成"直接检索一次 + 合成"，并把原因写进 `notes`。
静默降级的推理链路在评测里只表现为"分数莫名掉了几个点"，排查成本极高。
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from kubedocs.agent.state import CitedAnswer

MAX_SUB_QUESTIONS = 4
"""子问题数硬上限。宽度直接决定检索次数与端到端延迟，不能让调用方随意放大。"""

_MIN_SEGMENT_TOKENS = 2
"""并列连词两侧各自至少要有的词数（见 `_has_substantive_conjunction`）。"""

_MAX_BODY_CHARS = 1200
"""单块证据进入子答案文本的长度上限。检索条数已经由检索器限流，正文再限一次长，
提示词才不会随语料大小无界膨胀。"""

_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
"""正文里的引用标记。与 `generate.validate_citations` 同一口径，避免两处解析不一致。"""

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*|[\u4e00-\u9fff]")
"""词元切分：标识符整体算一个词，**中文按字**算一个词。

中文没有空格，按空白切词会得到 1，于是所有中文长问题都被判成"短问题"而永不分解；
反过来按字计又会让门槛偏高——所以 `min_words` 是可调参数，默认 12 个字。
"""

_QUESTION_MARK_RE = re.compile(r"[?？]")
"""半角与全角问号都算。"""

_CONJUNCTION_RE = re.compile(r"as well as|versus|and|plus|以及|同时|并且|和")
"""并列连词。只收这几个：覆盖"一句话里塞了两件事"的主要写法，误判可控。"""


# --------------------------------------------------------------------------- 数据结构
@dataclass(frozen=True)
class SubAnswer:
    """一个子问题及其证据。`citations` 与 `CitedAnswer.citations` 同形状。

    `answer` 是**证据文本**（带 `[n]` 标记），不是模型写的一段话——原因见 `_answer_sub`。
    """

    question: str
    answer: str
    citations: list[dict]
    hit_ids: list[str]

    def to_cited_answer(self) -> CitedAnswer:
        """转成图状态里流动的 `CitedAnswer`，下游节点不必认识 self-ask 的内部类型。"""
        return CitedAnswer(text=self.answer, citations=list(self.citations),
                           sub_question=self.question)


class Decomposer(Protocol):
    """分解与合成的 LLM 接口。用结构化输出，不解析自由文本。

    ## compose 的引用编号约定

    传给 `compose` 的 `parts` 已经**全局重编**过：`parts[i].citations[j]["index"]`
    就是该来源在最终 `SelfAskResult.citations` 里的编号。因此 `compose` 返回的文本里
    写 `[n]` 时必须按**全局**编号写（不要按某个子答案内部的顺序重新数）。
    越界标记会被 `self_ask` 剥掉并记进 `notes`：给一个打不开的引用比没有引用更误导。
    """

    def decompose(self, question: str, max_sub: int) -> list[str]:
        """把复合问题拆成至多 `max_sub` 个自包含、可独立检索的子问题。"""
        ...

    def compose(self, question: str, parts: Sequence[SubAnswer]) -> str:
        """把各子问题的证据合成一段最终回答（引用编号约定见类文档）。"""
        ...


@dataclass
class SelfAskResult:
    """一次 self-ask 的全部产物。

    - `decomposed`：本次**实际**是否走了分解（分解失败降级后为 False）。
    - `sub_questions` / `sub_answers`：未分解时都是空列表——不伪造"子问题"，
      免得下游把一次直接检索误读成分解产物；那种情况下的证据在 `citations` 里。
    - `llm_calls`：**成功返回**的 LLM 调用数（分解成功 +1、合成成功 +1）。
      抛异常的调用不计入：`Decomposer` 协议不暴露用量，无法确知失败调用是否已计费，
      所以只把原因写进 `notes`，不用一个猜出来的数字去污染成本口径。
    - `cost_usd`：本模块恒为 0.0。协议里没有用量/token 信息，成本由上层
      （litellm 回调 / 预算器）记账；这里**不猜**，猜出来的成本比没有成本更危险。
    """

    question: str
    decomposed: bool
    sub_questions: list[str]
    sub_answers: list[SubAnswer]
    answer: str
    citations: list[dict]
    llm_calls: int
    cost_usd: float
    notes: list[str]


# --------------------------------------------------------------------------- 分解决策
def needs_decomposition(question: str, *, min_words: int = 12) -> bool:
    """判断一个问题是否值得做 self-ask 分解（纯启发式，**绝不调 LLM**）。

    为什么必须确定性：这个判定决定整条链路的花费与延迟（多 1 次 LLM + N 次检索）。
    交给模型等于让它自己决定要不要多花钱，且结果不可复现、无法单测；线上排障也
    需要一个能讲清楚的理由——这里只用长度、问号个数、并列连词三样东西。

    判定规则（长度是**闸门**，结构才是**信号**）：

    1. 词数 < `min_words` -> False。太短的问题装不下两个子目标，拆它只会白付一次
       LLM 调用与多次检索，收益为负。中文按**字**计（见 `_TOKEN_RE`）。
    2. 出现 >= 2 个问号 -> True：用户事实上一次问了多个问题。
    3. 并列连词（和/以及/同时/并且）两侧**都有实质内容** -> True。只判"出现了连词"
       会误伤大量单目标问题（"Pod 和 Node 有什么区别"一次检索就能答完），
       所以要求连词前后各自都还有内容（见 `_has_substantive_conjunction`）。

    只靠长度分解是错的：一段 30 字、只描述一个事实的问题并不含多个子目标，
    拆开只是把一次检索变成三次。
    """
    text = question.strip()
    if not text:
        return False
    if len(_TOKEN_RE.findall(text)) < max(1, min_words):
        return False
    if len(_QUESTION_MARK_RE.findall(text)) >= 2:
        return True
    return _has_substantive_conjunction(text)


def _has_substantive_conjunction(text: str) -> bool:
    """并列连词两侧是否都还有实质内容。

    "和"在中文里出现频率极高（"…和…的区别"），只看连词存在会把大量单目标问题
    误判成复合问题。要求连词前、后各自至少 `_MIN_SEGMENT_TOKENS` 个词，
    才能认为它真的连了两个子目标。
    """
    segments = _CONJUNCTION_RE.split(text)
    if len(segments) < 2:
        return False
    head = len(_TOKEN_RE.findall(segments[0]))
    rest = len(_TOKEN_RE.findall("".join(segments[1:])))
    return head >= _MIN_SEGMENT_TOKENS and rest >= _MIN_SEGMENT_TOKENS


# --------------------------------------------------------------------------- 主流程
def self_ask(
    question: str,
    *,
    search: Callable[[str], object],
    llm: Decomposer,
    max_sub: int = 4,
    min_words: int = 12,
) -> SelfAskResult:
    """跑一次 self-ask：判定 -> 分解 -> 逐个子问题检索 -> 合并重编引用 -> 合成。

    `search` 只要求"给一个查询、返回带 `.results` 的结果对象"，用 duck typing 取用，
    因此这里**不** import `HybridRetriever`：agent 内核依赖检索实现会形成循环依赖，
    也让单测无法塞一个假检索器进来。`search` 抛异常或返回空都只影响该次检索。
    """
    notes: list[str] = []
    llm_calls = 0

    cap = min(max_sub, MAX_SUB_QUESTIONS)
    if max_sub > MAX_SUB_QUESTIONS:
        notes.append(
            f"max_sub={max_sub} 超过上限 {MAX_SUB_QUESTIONS}，已截断："
            "子问题数直接决定检索次数与端到端延迟，必须硬性设界"
        )

    sub_questions: list[str] = []
    if cap < 2:
        # 装不下两个子问题就没有分解可言；连分解调用都不发，免得白花一次 LLM
        notes.append(f"max_sub={cap} 不足以产生两个子问题，跳过分解，直接检索一次")
    elif needs_decomposition(question, min_words=min_words):
        sub_questions, decomposed_ok = _decompose(question, cap=cap, llm=llm, notes=notes)
        if decomposed_ok:
            llm_calls += 1
    # 两个分支都不满足时就按"不分解"走：启发式判定不值得分解是正常路径，不写 notes
    # ——那不是降级，是省下一次 LLM 调用与 N 次检索。

    citations: list[dict]
    merged_parts: list[SubAnswer]
    compose_parts: list[SubAnswer]
    if sub_questions:
        raw_parts = [_answer_sub(q, search=search, notes=notes) for q in sub_questions]
        merged_parts, citations = _merge_citations(raw_parts, notes)
        compose_parts = merged_parts
    else:
        # 降级/不分解路径：只检索一次原始问题，证据同样带 [1..n] 的局部（即全局）编号
        hits = _search(question, search, notes)
        text, citations, hit_ids = _evidence_and_citations(hits, notes)
        compose_parts = [SubAnswer(question=question, answer=text,
                                   citations=citations, hit_ids=hit_ids)]
        merged_parts = []

    answer, composed_ok = _compose(question, compose_parts, llm, notes)
    if composed_ok:
        llm_calls += 1
        # 合成文本按全局编号解释（见 Decomposer 的类文档），这里只做越界校验：
        # 打不开的引用比没有引用更误导，越界标记一律剥除并记账。
        answer, dropped = _rewrite_markers(
            answer, {i: i for i in range(1, len(citations) + 1)})
        if dropped:
            notes.append(f"合成答案里有 {dropped} 处引用标记越界，已剥除")

    return SelfAskResult(
        question=question,
        decomposed=bool(sub_questions),
        sub_questions=sub_questions,
        sub_answers=merged_parts,
        answer=answer,
        citations=citations,
        llm_calls=llm_calls,
        cost_usd=0.0,
        notes=notes,
    )


def _decompose(question: str, *, cap: int, llm: Decomposer,
               notes: list[str]) -> tuple[list[str], bool]:
    """调一次分解，返回 (子问题列表, 这次调用是否成功返回)。

    失败**不抛异常**：降级为"直接检索一次"（调用方看到空列表就会走那条路），
    并把原因写进 `notes`。返回第二个值是为了让"调用成功但结果不可用"和
    "调用本身失败"在计数上区分开——前者确实花了一次 LLM，后者不该被当成已发生。
    """
    try:
        raw = llm.decompose(question, cap)
    except Exception as e:  # LLM 适配器什么都可能抛（超时/解析/鉴权），一律降级
        notes.append(f"分解失败，降级为直接检索一次：{type(e).__name__}: {e}")
        return [], False
    subs = _normalize_sub_questions(raw, cap, notes)
    if len(subs) < 2:
        notes.append("分解结果不足 2 个子问题，按未分解处理（该次分解调用已发生，仍计入 llm_calls）")
        return [], True
    return subs, True


def _normalize_sub_questions(raw: object, cap: int, notes: list[str]) -> list[str]:
    """清洗模型给的子问题：去空、按序去重、截断到 `cap`。

    去重不是洁癖：重复的子问题就是重复的一次检索，白花延迟与配额。
    类型检查是防呆：有些实现只返回一个字符串，把它当可迭代会"逐字分解"出
    几十个子问题——这种错法在日志里看起来还挺像正常运行。
    """
    if isinstance(raw, str):
        notes.append("decompose 返回了字符串而不是列表，已按单个子问题处理（否则会被逐字符拆开）")
        items: Sequence[object] = [raw]
    elif isinstance(raw, Sequence):
        items = raw
    else:
        notes.append(f"decompose 返回了 {type(raw).__name__}，无法当作子问题列表，视为未分解")
        return []

    out: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for item in items:
        sub = str(item).strip()
        if not sub or sub in seen:
            skipped += 1
            continue
        seen.add(sub)
        out.append(sub)
    if skipped:
        notes.append(f"分解结果里有 {skipped} 个空/重复子问题，已去掉：重复子问题等于重复检索")
    if len(out) > cap:
        notes.append(f"子问题数 {len(out)} 超过上限 {cap}，已截断：每个子问题都要花一次检索")
        out = out[:cap]
    return out


def _answer_sub(question: str, *, search: Callable[[str], object], notes: list[str]) -> SubAnswer:
    """检索一个子问题，得到**局部编号**的证据文本与引用。

    子问题的"答案"就是它检索到的证据片段本身（带 `[n]` 标记），不再发一次生成调用：
    合成整段回答是 `compose` 的职责，每个子问题各生成一次会让 llm_calls 变成 N+1，
    而 self-ask 的收益来自**检索更准**，不是多写几段话。

    这里刻意**不**递归调用 `self_ask`：只做一层，理由见模块文档（深度会让成本无界、
    收益递减）。
    """
    hits = _search(question, search, notes)
    text, citations, hit_ids = _evidence_and_citations(hits, notes)
    return SubAnswer(question=question, answer=text, citations=citations, hit_ids=hit_ids)


def _merge_citations(parts: Sequence[SubAnswer],
                     notes: list[str]) -> tuple[list[SubAnswer], list[dict]]:
    """合并各子答案的引用：按出现顺序从 1 重编、按 chunk_id 去重，**并改写正文标记**。

    为什么必须同时改正文（最容易出错的一步）：每个子答案的引用是局部编号
    （各自从 [1] 开始）。只重编引用列表而不动正文，`[1]` 就会指向别的块——
    肉眼很难发现，但引用核对会成片失败。所以去重与重编都走**同一份**
    (局部 -> 全局) 映射，引用列表和正文标记一起落地。
    """
    global_of: dict[str, int] = {}
    citations: list[dict] = []
    merged: list[SubAnswer] = []
    stripped = 0

    for part in parts:
        remap: dict[int, int] = {}
        for cit in part.citations:
            cid = str(cit.get("chunk_id", ""))
            if not cid:
                # 没有 chunk_id 既定位不到块、也没法去重，收了只会得到打不开的引用
                continue
            try:
                local = int(cit.get("index", 0))
            except (TypeError, ValueError):
                continue
            gi = global_of.get(cid)
            if gi is None:
                gi = len(citations) + 1
                global_of[cid] = gi
                citations.append({
                    "index": gi,
                    "chunk_id": cid,
                    "title": str(cit.get("title", "")),
                    "breadcrumb": str(cit.get("breadcrumb", "")),
                    "source_url": str(cit.get("source_url", "")),
                })
            remap[local] = gi

        text, dropped = _rewrite_markers(part.answer, remap)
        stripped += dropped
        # 该子答案自己的引用换成全局编号后的副本（拷贝一份，避免各子答案共享同一个
        # dict 后互相污染）
        part_citations = [dict(citations[gi - 1]) for gi in sorted(set(remap.values()))]
        merged.append(SubAnswer(
            question=part.question,
            answer=text,
            citations=part_citations,
            hit_ids=[str(c["chunk_id"]) for c in part_citations],
        ))

    if stripped:
        notes.append(f"各子答案里有 {stripped} 处引用标记找不到对应引用，已剥除")
    return merged, citations


def _rewrite_markers(text: str, remap: Mapping[int, int]) -> tuple[str, int]:
    """按 `remap` 改写正文里的 `[n]`，返回 (新文本, 被剥除的标记数)。

    映射里没有的编号被**剥掉**而不是留着：留着等于给用户一个打不开的引用，
    比没有引用更有误导性（与 `generate.validate_citations` 同一策略）。
    """
    dropped = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal dropped
        target = remap.get(int(match.group(1)))
        if target is None:
            dropped += 1
            return ""
        return f"[{target}]"

    return _MARKER_RE.sub(_replace, text), dropped


def _compose(question: str, parts: Sequence[SubAnswer], llm: Decomposer,
             notes: list[str]) -> tuple[str, bool]:
    """调一次合成；失败时退回子答案原文，绝不把异常抛给调用方。"""
    try:
        return llm.compose(question, parts), True
    except Exception as e:  # 同 _decompose：适配器的任何异常都只降级，不往上抛
        notes.append(f"合成失败，改用子答案证据原文兜底：{type(e).__name__}: {e}")
        return _fallback_answer(parts), False


def _fallback_answer(parts: Sequence[SubAnswer]) -> str:
    """兜底回答：把已经（全局重编过的）证据原文拼起来。

    比返回空串有用得多——引用编号是对的，用户至少能自己读到来源；
    一个空回答在链路上游看起来和"没有检索到"无法区分。
    """
    return "\n\n".join(p.answer.strip() for p in parts if p.answer.strip())


# --------------------------------------------------------------------------- 检索适配
def _search(query: str, search: Callable[[str], object], notes: list[str]) -> list[object]:
    """跑一次检索，统一成"可 duck typing 的命中列表"。

    单次检索失败只让这一路没有证据，不能拖垮整个回答：多子问题的价值就在于
    证据可以来自不同检索，一路挂了就丢掉全部已经很不划算。
    """
    try:
        result = search(query)
    except Exception as e:  # 检索后端（网络/索引损坏）的异常不可枚举，一律只降级
        notes.append(f"检索失败（{query[:40]}）：{type(e).__name__}: {e}；"
                     "该子问题没有证据，其余子问题不受影响")
        return []
    hits = _hits_of(result)
    if not hits:
        notes.append(f"检索没有命中（{query[:40]}）：该子问题按无证据继续，不中断整体回答")
    return hits


def _hits_of(result: object) -> list[object]:
    """取结果对象里的 `.results`，同时容忍 `{"results": [...]}` 这种 dict 形状。

    不 `import HybridRetriever`（见模块文档），所以这里只认形状不认类型；
    测试和别处塞进来的假结果常常是 dict，多认一种形状比让调用方去凑类型便宜。
    """
    raw = getattr(result, "results", None)
    if raw is None and isinstance(result, Mapping):
        raw = result.get("results")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return list(raw)


def _evidence_and_citations(
    hits: Sequence[object], notes: list[str]
) -> tuple[str, list[dict], list[str]]:
    """把命中整理成 `[n] 标题 › 面包屑 / 正文` 的证据文本 + **局部编号**的引用。

    编号与文本在**同一个循环**里生成：分开写两次遍历，一旦某条命中因缺 chunk_id
    被跳过，正文编号与引用编号就会错位一格——这是引用错位最隐蔽的来源。
    """
    blocks: list[str] = []
    citations: list[dict] = []
    index_of: dict[str, int] = {}
    missing = 0

    for hit in hits:
        cid = _pick(hit, "chunk_id")
        if not cid:
            missing += 1
            continue
        idx = index_of.get(cid)
        if idx is None:
            idx = len(citations) + 1
            index_of[cid] = idx
            citations.append({
                "index": idx,
                "chunk_id": cid,
                "title": _pick(hit, "title"),
                "breadcrumb": _pick(hit, "breadcrumb"),
                "source_url": _pick(hit, "source_url"),
            })
        title = _pick(hit, "title")
        breadcrumb = _pick(hit, "breadcrumb")
        label = f"{title} › {breadcrumb}" if breadcrumb else title
        body = _pick(hit, "body").strip()
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS].rstrip() + " …"
        blocks.append(f"[{idx}] {label}\n{body}")

    if missing:
        notes.append(f"有 {missing} 条检索结果缺 chunk_id，已跳过："
                     "引用必须能定位到块，缺 id 也无法去重")
    hit_ids = [str(c["chunk_id"]) for c in citations]
    return "\n\n".join(blocks), citations, hit_ids


def _pick(obj: object, name: str) -> str:
    """按名字取值：先属性，再 dict 键（假结果常常是 dict）。取不到给空串。"""
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, Mapping):
        value = obj.get(name)
    return "" if value is None else str(value)
