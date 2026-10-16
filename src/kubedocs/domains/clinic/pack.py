"""clinic 域包：把领域资产与 agent 内核装配成可用的问诊/挂号/紧急 agent。

**本文件是集成点，不是又一层抽象。**内核只认 `ClinicDeps`（一组可注入的回调），
域提供知识与业务，这里把两边接起来：

    症状抽取 ← symptom_dict + polarity（含三维归因与来源判定）
    候选打分 ← hypothesis.score（规则权重 + 来源折扣）
    选题/建议 ← hypothesis.next_question / suggest_hypothesis（红线保护内置）
    急诊红线 ← redline.RiskGuard（确定性，LLM 不可绕过）
    挂号 ← BookingService → AppointmentProvider（当前是进程内 Mock；接真实 HIS 时换实现）
    回答 ← **注入的 `answerer`**；未注入时用"最佳候选 + 就医提醒"的保守兜底

**来源判定为什么在这里做**：`hypothesis` 的权重表区分"用户自发说的"与"系统建议后
确认的"，但"这条症状是不是系统建议的"只有集成层知道——它拿着上一轮的
`pending_suggestion`。所以来源归因在 `extract_evidence` 里完成，再把带来源的
证据交给打分器。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from kubedocs.agent import hypothesis
from kubedocs.agent.kernel import DEFAULT_MAX_TURNS, ClinicDeps
from kubedocs.agent.state import EvidenceSource, SymptomEvidence
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.data.symptom_dict import match_symptom_in_text
from kubedocs.domains.clinic.polarity import is_negated
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError
from kubedocs.domains.clinic.redline import (
    RiskGuard,
    emergency_advice,
    emergency_menu,
    wants_emergency_care,
)

# 口语症状最多展开成多少个语料症状键。一个词（如"咳嗽"）在语料里能对上几十个更细的症状名，
# 全靠它会把这次发言变成一次全表扫描式的投票。60 覆盖住了实测最宽的那几个词。
MAX_KEYS_PER_SYMPTOM = 60

# 逐字命中的关联键最多取几个（患者一句话里很少出现超过两三个规范症状名）。
MAX_VERBATIM_KEYS = 20

# 逐字扫描忽略的 1 字键：太容易在无关句子里撞上，噪声大于信息。
MIN_VERBATIM_KEY_CHARS = 2

# 挂号意图词表：**必须列变体**。「挂号」不是「帮我挂个号」的子串（中间插了"个"），
# 朴素子串匹配在中文里会直接失效。
BOOK_WORDS = ("挂号", "挂个号", "预约", "取号", "约号", "挂专家号")

# 确认词：用户对"确认挂号吗？"的**同意**表达。
CONFIRM_WORDS = ("确认", "确定", "是的", "对的", "好的", "可以", "行", "嗯", "要", "麻烦", "谢谢")

# `polarity.NEGATION_WORDS` 收录的是「无/没/不/未」一族，**不含祈使否定「别」**。
# 它被症状抽取与红线**共用**，往里加裸「别」会让「特别疼」「别的症状」「告别」
# 统统被误判成否定。所以只针对挂号这一组窄词表补一条祈使否定，并排除常见复合词。
_BIE_COMPOUNDS = (
    "别的",
    "别人",
    "特别",
    "分别",
    "区别",
    "告别",
    "别扭",
    "别致",
    "别提",
    "个别",
    "辨别",
)

# 「A 不 A」疑问式（能不能 / 可不可以 / 要不要 / 是不是）不是否定：
# 直接把它当普通否定，「能不能取号」会被判成"不要取号"。
# 这里是**词表内的局部归一化**，不动 `polarity`——那张表是症状抽取与红线共用的。
_ANOT_A = re.compile(r"(能|可|是|要|行|会|该|愿|好)不\1")


def _negated(text: str, word: str) -> bool:
    """`word` 在 `text` 里是否被否定（含祈使否定「别」与 A 不 A 疑问式的排除）。"""
    probe = _ANOT_A.sub(lambda m: m.group(1), text)
    if is_negated(probe, word):
        return True
    pos = probe.find(word)
    while pos != -1:
        idx = probe.rfind("别", 0, pos)
        if idx != -1:
            # 两个方向的 2 字窗口都要查：「特别」的「别」在后面，「别的」的「别」在前面。
            after = probe[idx : idx + 2]
            before = probe[max(0, idx - 1) : idx + 1]
            if after not in _BIE_COMPOUNDS and before not in _BIE_COMPOUNDS:
                return True
        pos = probe.find(word, pos + 1)
    return False


def wants_booking(text: str) -> bool:
    """用户是否**要求**挂号。

    ## 必须判否定

    `"挂号" in "我不想挂号"` 是 `True`，但那是"不要挂号"。关键词命中只证明
    "提到了挂号"，**不证明"想挂号"**——而挂号会占用一个真实号源，
    把"用户否认"读成"用户要求"会直接扣掉一个号。

    所以每个命中的词都要过一次否定判定：复用红线同一套 `polarity`
    （两套中文否定逻辑必然有一天对不上）+ 一条祈使否定补充（见 `_negated`）。能覆盖：

        我不想挂号 / 不要预约 / 不用取号 / 先别帮我挂号 / 别挂号   -> False

    ## 已知边界

    "已经预约过了，不用再约" 里的否定词在短语**之后**且跨了分句，按 `polarity`
    的作用域语义不构成否定 ⇒ 仍判为 True。这类"已完成"语义需要另一层
    （时间/完成体）判定，本项目不做。代价有限：即使误判为挂号意图，
    图路径也会先走 `confirm` 确认（见 `agent/kernel.py` 的 `node_confirm`），
    不会直接占号。
    """
    return any(w in text and not _negated(text, w) for w in BOOK_WORDS)


def confirm_booking(text: str) -> bool:
    """用户对"确认挂号吗？"的回答是否表示**同意**。

    同样走否定判定："不用了" / "先不挂" / "别确认" 必须是"不同意"。
    再次明确说出挂号意图（"确认挂号"）也算同意。
    """
    if wants_booking(text):
        return True
    return any(w in text and not _negated(text, w) for w in CONFIRM_WORDS)


@lru_cache(maxsize=1)
def link_vocabulary() -> tuple[str, ...]:
    """**当前语料**的症状关联表键，按长度降序。**进程内只读一次语料**。

    只在调用方拿不到 `links` 时用（例如只做症状抽取、不做打分的脚本）。
    装配好的路径一律把 `links` 传给 `match_symptoms`——见那里的说明：
    词表必须与打分用的是同一张表，否则抽出来的键在打分表里查不到。

    语料缺失时返回空元组，抽取退化为"只用口语词典"（不抛异常：没有语料是部署状态）。
    测试要换一套卡片语料时记得 `link_vocabulary.cache_clear()`。
    """
    try:
        from kubedocs.domains.clinic.domain import ClinicPack

        keys = tuple(str(k) for k in ClinicPack().triage_links())
    except Exception:  # noqa: BLE001 - 缺语料/格式坏都只是"这一层没有词表"
        return ()
    return tuple(sorted(keys, key=len, reverse=True))


def _vocabulary(links: Mapping[str, Sequence[str]] | None) -> tuple[str, ...]:
    """要拿来匹配的词表：**调用方给的那张关联表**优先，否则退回当前语料的表。

    这一点是刻意的：抽取与打分必须用同一张表。测试会注入一张自己的 `links`
    （键是口语词），生产用的是语料派生出来的表——如果词表写死在模块里，
    注入的那张表就永远对不上，而那正是"桥修好了、打分还是 0"的形态。
    """
    if links is None:
        return link_vocabulary()
    return tuple(sorted((str(k) for k in links), key=len, reverse=True))


def _corpus_keys_for(symptom: str, vocab: Sequence[str]) -> list[str]:
    """口语/标准症状 -> **词表里可用**的症状键。

    三档，顺序就是优先级：

    1. 这个词**本身就是**词表的键 -> 用它自己；
    2. 再补上**包含它**的键（语料的症状名比口语细：咳嗽 -> 持续性咳嗽、夜间咳嗽…）。
       按长度升序取前 `MAX_KEYS_PER_SYMPTOM` 个——越短越接近这个词本身，
       "一咳嗽胸口就疼"这种带其他症状的复合名往后排；
    3. 一个都没有 -> 返回空列表，调用方退回口语词本身（见 `match_symptoms`）。

    第 1 档与第 2 档**都要**，不是二选一：实测"瘙痒"本身在关联表里只挂着 5 个科室
    （急诊/感染/肛肠/血液/风湿免疫，全都 1 票并列），只取它自己的话前 3 名是随机的；
    把"局部瘙痒""冬季皮肤瘙痒"这些一起算进来，皮肤科才会以明显优势浮上来。
    """
    if not vocab:
        return []
    out = [k for k in vocab if k == symptom]
    out += [k for k in reversed(vocab) if len(k) > len(symptom) and symptom in k]
    return out[:MAX_KEYS_PER_SYMPTOM]


def match_symptoms(text: str, links: Mapping[str, Sequence[str]] | None = None) -> list[tuple[str, str]]:
    """患者原话 -> `[(症状键, 命中原词), ...]`，症状键**取自语料词表**。

    ## 为什么必须有这一层桥

    口语词典（`symptom_dict`）说的是"患者会怎么讲"，关联表的键来自知识卡片语料，
    粒度比口语**细**：患者说"咳嗽"，语料里是「持续性咳嗽」「夜间咳嗽」「一咳嗽胸口就疼」。
    两边直接做等值连接，实测 5 条分诊 gold 里只有 **1 个**症状键能在关联表里找到——
    分诊分数全空、准确率 0，而这不是"语料不行"，是**两套词汇之间缺一层映射**。

    两条路一起走，因为患者有时用口语、有时用规范说法：

    1. **逐字扫**词表里的键（长键优先）：患者直接说"反酸""烧心""耳痛"时命中，
       这几个词在口语词典里根本没有条目；
    2. **口语词典命中 -> 展开**成词表里包含它的键（见 `_corpus_keys_for`）。

    展开不出来（缺语料，或这个词在语料里没有任何关联）时**退回口语词本身**：
    打分器查不到关联会跳过，行为与加这层桥之前一致——不会因为语料缺失把整条路堵死。
    """
    hits: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(key: str, matched: str) -> None:
        if key and key not in seen:
            seen.add(key)
            hits.append((key, matched))

    vocab = _vocabulary(links)
    verbatim = 0
    for key in vocab:  # 已按长度降序：长键先占，避免被它的子串抢先
        if verbatim >= MAX_VERBATIM_KEYS:
            break
        if len(key) >= MIN_VERBATIM_KEY_CHARS and key in text:
            add(key, key)
            verbatim += 1

    for canonical, matched in match_symptom_in_text(text):
        for key in _corpus_keys_for(canonical, vocab) or [canonical]:
            add(key, matched)
    return hits


def extract_evidence(
    text: str,
    turn: int,
    *,
    suggested: Sequence[str] = (),
    links: Mapping[str, Sequence[str]] | None = None,
) -> list[SymptomEvidence]:
    """从用户这句话里抽症状证据（**键与 `links` 同一套词表**），并判定**来源**。

    来源决定权重（见 `hypothesis.WEIGHTS`）：系统建议后确认的降权，
    因为引导式提问会引入引导性偏倚——用户倾向于顺着说"是"。

    `links` 是**这张证据将要拿去打分的那张关联表**。抽出来的键必须能在它里面查到，
    否则打分器只会静静跳过（那正是分诊全 0 的形态）。不传时用当前语料的表，
    只适合"不打算打分"的调用方。
    """
    from kubedocs.agent.hypothesis import weight_of

    out: list[SymptomEvidence] = []
    for symptom_key, matched in match_symptoms(text, links):
        hinted = matched in suggested or symptom_key in suggested
        if is_negated(text, matched):
            source = EvidenceSource.NEGATED_AFTER_SUGGESTION if hinted else EvidenceSource.NEGATED
            present = False
        else:
            source = EvidenceSource.CONFIRMED if hinted else EvidenceSource.VOLUNTEERED
            present = True
        out.append(
            SymptomEvidence(
                symptom=symptom_key, present=present, source=source, turn=turn, weight=weight_of(source)
            )
        )
    return out


def invert_links(links: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """症状 -> 候选 反转为 候选 -> 症状（`next_question` 需要后者）。"""
    out: dict[str, list[str]] = {}
    for symptom, cands in links.items():
        for c in cands:
            out.setdefault(c, []).append(symptom)
    return {k: sorted(set(v)) for k, v in out.items()}


def build_links_from_cards(path: Path) -> dict[str, list[str]]:
    """从知识卡片语料构建 症状 -> 候选（疾病 + 科室）关联表。

    语料缺失时返回空表而不是报错：规则轨（红线/词典）不依赖它，
    **没有受限语料时三条流程仍要能跑通**。
    """
    import json

    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        cards = json.load(fh)
    links: dict[str, list[str]] = {}
    for card in cards:
        if card.get("kind") != "symptom":
            continue
        target = list(card.get("diseases", [])) + list(card.get("departments", []))
        if target:
            links[str(card["key"])] = target
    return links


@dataclass
class ClinicWiring:
    """装配好的一整套依赖，便于测试与自省。"""

    deps: ClinicDeps
    guard: RiskGuard = field(default_factory=RiskGuard)
    service: BookingService | None = None
    links: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def build_clinic_deps(
    provider: AppointmentProvider,
    *,
    links: Mapping[str, Sequence[str]] | None = None,
    answerer: Callable[[str, Any], dict[str, Any]] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    margin: float = 1.0,
    tie_breaker: Callable[[str, Sequence[str]], str] | None = None,
) -> ClinicWiring:
    """把域资产与 agent 模块装配成 `ClinicDeps`。

    `answerer` 是**可选注入**的检索回答器（需要检索器与 LLM）；
    不注入时 `_answer` 走"最佳候选 + 就医提醒"的保守兜底——
    **宁可信息少，也不要没答案**。
    """
    guard = RiskGuard()
    service = BookingService(provider)
    link_map: dict[str, list[str]] = {k: list(v) for k, v in (links or {}).items()}
    symptom_of = invert_links(link_map)
    notes: list[str] = []
    if not link_map:
        notes.append("没有症状-候选关联表（语料缺失）：候选打分与追问选择会退化，但红线/挂号不受影响")

    def _observe(text: str, state: Any) -> list[SymptomEvidence]:
        suggested = tuple(state.get("pending_suggestion", {}).get("symptoms", ()) or ())
        # 把**本装配用的那张关联表**传进去：抽出来的症状键必须能在 `_score` 的表里查到，
        # 否则打分器只会静静跳过（分诊全 0 的形态）。
        return extract_evidence(text, int(state.get("turn", 0)) + 1, suggested=suggested, links=link_map)

    def _score(evidence: Sequence[SymptomEvidence]) -> dict[str, float]:
        return hypothesis.score(evidence, link_map, tie_breaker=tie_breaker)

    def _stop(scores: dict[str, float], turn: int) -> tuple[bool, str]:
        return hypothesis.should_stop(scores, turn, max_turns=max_turns, margin=margin)

    def _next_question(scores: dict[str, float], state: Any) -> str | None:
        asked = tuple(state.get("asked", ()) or ())
        known = tuple(state.get("evidence", ()) or ())
        # 先试"押注领先候选"的试探性假设，再退回信息增益最大的一般提问。
        # 两条路径内部都做了红线过滤（红线症状一律不主动建议）。
        return hypothesis.suggest_hypothesis(scores, symptom_of, asked=asked) or hypothesis.next_question(
            scores, symptom_of, asked=asked, known=known
        )

    def _answer(question: str, state: Any) -> dict[str, Any]:
        if answerer is not None:
            return answerer(question, state)
        ranked = hypothesis.rank(dict(state.get("candidates", {})), top=3)
        if not ranked:
            return {
                "text": "目前还不足以给出方向建议。如果有明显不适，建议直接就医；"
                "出现胸痛、呼吸困难、意识不清等请立即拨打 120。",
                "citations": [],
            }
        best = hypothesis.explain(dict(state.get("candidates", {})), list(state.get("evidence", ()) or ()))
        top = "、".join(f"{name}（{score:.1f}）" for name, score in ranked)
        return {
            "text": f"根据您描述的情况，可能相关的方向：{top}。"
            f"以上仅为导诊参考，不能替代医生诊断；如症状加重请及时就医。",
            "citations": [],
            "score_detail": best,
        }

    def _book(state: Any) -> dict[str, Any]:
        """从候选里挑一个**真的可挂号科室**去挂号。

        候选集里疾病名与科室名是混在一起的（关联表两个方向都有），
        直接取 Top1 可能得到"支气管炎"这种疾病名，`resolve_department` 会失败。
        所以按分数降序找第一个出现在号源科室表里的候选；一个都没有就交给
        `BookingService` 报错（它会列出可选科室），让 agent 有机会问用户。
        """
        ranked = hypothesis.rank(dict(state.get("candidates", {})))
        known = {d["name"] for d in service.departments()}
        department = next((name for name, _ in ranked if name in known), "")
        try:
            return service.book(
                department or (ranked[0][0] if ranked else ""), str(state.get("patient_id", "anon"))
            )
        except BookingError as e:
            # 业务错误转成人话，不让 traceback 冒到用户面前
            return {"error": str(e), "department": department, "candidates": [n for n, _ in ranked[:5]]}

    deps = ClinicDeps(
        observe=_observe,
        score=_score,
        should_stop=_stop,
        next_question=_next_question,
        check_red_line=guard.check,
        answer=_answer,
        book=_book,
        wants_booking=wants_booking,
        confirm_booking=confirm_booking,
        wants_emergency_care=wants_emergency_care,
        emergency_advice=emergency_advice,
        emergency_menu=emergency_menu,
    )
    return ClinicWiring(deps=deps, guard=guard, service=service, links=link_map, notes=notes)


def clinic_cards_path() -> Path:
    """clinic 语料（知识卡片）的位置。缺失是正常情况（受限语料不入库），调用方据此降级。"""
    from kubedocs.config import repo_root

    return repo_root() / "data" / "clinic" / "cards.json"
