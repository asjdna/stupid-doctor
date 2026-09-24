"""clinic 域包：把领域资产与 agent 内核装配成可用的问诊/挂号/紧急 agent。

## 这个文件是"集成点"，不是"又一层抽象"

内核只认 `ClinicDeps`（七个可注入的回调），域提供知识与业务。本文件把两边接起来：

    症状抽取 ← symptom_dict + polarity（含三维归因与来源判定）
    候选打分 ← hypothesis.score（规则权重 + 来源折扣）
    选题/建议 ← hypothesis.next_question / suggest_hypothesis（红线保护内置）
    急诊红线 ← redline.RiskGuard（确定性，LLM 不可绕过）
    挂号 ← BookingService → AppointmentProvider（SQLite / Mock 可换）
    回答 ← selfask.self_ask（分解检索合成，带引用重编号）

## 来源判定为什么要在这里做

`hypothesis` 的权重表区分"用户自发说的"与"系统建议后确认的"，但"这条症状是不是
系统建议的"只有集成层知道——它拿着上一轮的 `pending_suggestion`。所以来源归因在
`extract_evidence` 里完成，再把带来源的证据交给打分器。
"""
from __future__ import annotations

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

# 挂号意图词表：**必须列变体**。
# 实测：「挂号」不是「帮我挂个号」的子串（中间插了"个"）。朴素子串匹配在中文里直接失效。
BOOK_WORDS = ("挂号", "挂个号", "预约", "取号", "约号", "挂专家号")


def wants_booking(text: str) -> bool:
    return any(w in text for w in BOOK_WORDS)


def extract_evidence(text: str, turn: int, *,
                     suggested: Sequence[str] = ()) -> list[SymptomEvidence]:
    """从用户这句话里抽症状证据，并判定**来源**。

    来源决定权重（见 `hypothesis.WEIGHTS`）：系统建议后确认的降权，
    因为引导式提问会引入建议性偏倚——用户倾向于顺着说"是"。
    """
    from kubedocs.agent.hypothesis import weight_of

    out: list[SymptomEvidence] = []
    for canonical, matched in match_symptom_in_text(text):
        hinted = matched in suggested or canonical in suggested
        if is_negated(text, matched):
            source = (EvidenceSource.NEGATED_AFTER_SUGGESTION if hinted
                      else EvidenceSource.NEGATED)
            present = False
        else:
            source = EvidenceSource.CONFIRMED if hinted else EvidenceSource.VOLUNTEERED
            present = True
        out.append(SymptomEvidence(symptom=canonical, present=present, source=source,
                                   turn=turn, weight=weight_of(source)))
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
    **没有受限语料时三条流程仍要能跑通**（见 docs/adr/0003 第五节）。
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


def build_clinic_deps(provider: AppointmentProvider, *,
                      links: Mapping[str, Sequence[str]] | None = None,
                      answerer: Callable[[str, Any], dict[str, Any]] | None = None,
                      max_turns: int = DEFAULT_MAX_TURNS,
                      margin: float = 1.0,
                      tie_breaker: Callable[[str, Sequence[str]], str] | None = None,
                      ) -> ClinicWiring:
    """把域资产与 agent 模块装配成 `ClinicDeps`。

    `answerer` 缺省时回答走 self-ask（需要检索器与 LLM，由 P3 注入）；
    这里给一个"只回当前最佳候选 + 就医提醒"的保守兜底——**宁可信息少，也不要没答案**。
    """
    guard = RiskGuard()
    service = BookingService(provider)
    link_map: dict[str, list[str]] = {k: list(v) for k, v in (links or {}).items()}
    symptom_of = invert_links(link_map)
    notes: list[str] = []
    if not link_map:
        notes.append("没有症状-候选关联表（语料缺失）：候选打分与追问选择会退化，"
                     "但红线/挂号不受影响")

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
        return (hypothesis.suggest_hypothesis(scores, symptom_of, asked=asked)
                or hypothesis.next_question(scores, symptom_of, asked=asked, known=known))

    def _answer(question: str, state: Any) -> dict[str, Any]:
        if answerer is not None:
            return answerer(question, state)
        ranked = hypothesis.rank(dict(state.get("candidates", {})), top=3)
        if not ranked:
            return {"text": "目前还不足以给出方向建议。如果有明显不适，建议直接就医；"
                            "出现胸痛、呼吸困难、意识不清等请立即拨打 120。",
                    "citations": []}
        best = hypothesis.explain(dict(state.get("candidates", {})),
                                  list(state.get("evidence", ()) or ()))
        top = "、".join(f"{name}（{score:.1f}）" for name, score in ranked)
        return {"text": f"根据您描述的情况，可能相关的方向：{top}。"
                        f"以上仅为导诊参考，不能替代医生诊断；如症状加重请及时就医。",
                "citations": [], "score_detail": best}

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
            return service.book(department or (ranked[0][0] if ranked else ""),
                                str(state.get("patient_id", "anon")))
        except BookingError as e:
            # 业务错误转成人话，不让 traceback 冒到用户面前
            return {"error": str(e), "department": department,
                    "candidates": [n for n, _ in ranked[:5]]}

    deps = ClinicDeps(
        observe=_observe, score=_score, should_stop=_stop,
        next_question=_next_question, check_red_line=guard.check,
        answer=_answer, book=_book, wants_booking=wants_booking,
        wants_emergency_care=wants_emergency_care,
        emergency_advice=emergency_advice, emergency_menu=emergency_menu,
    )
    return ClinicWiring(deps=deps, guard=guard, service=service,
                        links=link_map, notes=notes)


def clinic_cards_path() -> Path:
    """clinic 语料（知识卡片）的位置。缺失是正常情况（受限语料不入库），调用方据此降级。"""
    from kubedocs.config import repo_root

    return repo_root() / "data" / "clinic" / "cards.json"
