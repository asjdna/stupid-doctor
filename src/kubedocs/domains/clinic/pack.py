"""clinic 域包：把领域资产与 agent 内核装配成可用的问诊/挂号/紧急 agent。

**本文件是集成点，不是又一层抽象。**内核只认 `ClinicDeps`（一组可注入的回调），
域提供知识与业务，这里把两边接起来：

    症状抽取 ← symptom_dict + polarity（含三维归因与来源判定）
    候选打分 ← hypothesis.score（规则权重 + 来源折扣）
    选题/建议 ← hypothesis.next_question / suggest_hypothesis（红线保护内置）
    急诊红线 ← redline.RiskGuard（确定性，LLM 不可绕过）
    挂号 ← BookingService → AppointmentProvider（SQLite / Mock 可换）
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


def extract_evidence(text: str, turn: int, *, suggested: Sequence[str] = ()) -> list[SymptomEvidence]:
    """从用户这句话里抽症状证据，并判定**来源**。

    来源决定权重（见 `hypothesis.WEIGHTS`）：系统建议后确认的降权，
    因为引导式提问会引入建议性偏倚——用户倾向于顺着说"是"。
    """
    from kubedocs.agent.hypothesis import weight_of

    out: list[SymptomEvidence] = []
    for canonical, matched in match_symptom_in_text(text):
        hinted = matched in suggested or canonical in suggested
        if is_negated(text, matched):
            source = EvidenceSource.NEGATED_AFTER_SUGGESTION if hinted else EvidenceSource.NEGATED
            present = False
        else:
            source = EvidenceSource.CONFIRMED if hinted else EvidenceSource.VOLUNTEERED
            present = True
        out.append(
            SymptomEvidence(
                symptom=canonical, present=present, source=source, turn=turn, weight=weight_of(source)
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
    **宁可信息少，也不要没答案**。注意 `agent/selfask.py` 不在这条链上：
    当前没有任何生产路径注入它。
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
        return extract_evidence(text, int(state.get("turn", 0)) + 1, suggested=suggested)

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
