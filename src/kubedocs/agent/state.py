"""会话状态：图上流动的唯一数据结构。

## 为什么状态里必须带"证据来源"

医疗问诊里，系统**主动抛出试探性假设**（"我猜您可能是胸口发闷、像压着东西？"）能显著
提高症状采集效率，但会引入**建议性偏倚**：用户倾向于答"是"，即使本来没有。
于是采集到的症状可能是被问题"问出来"的，而不是患者自己想说的。

若不区分来源，这两者会等量齐观，分诊被假症状带偏。所以每条证据都记 `source`，
打分时按来源降权（权重表见 `hypothesis.py`），病历里也标注出来。

## 为什么否定证据也要记

鉴别诊断里"排除"与"命中"同等重要：用户否认某个候选的关键症状，是该候选的强反证。
只累加阳性症状的系统无法收敛。但口语里的否认可能是"没说准"（患者对症状学不熟），
所以否定证据**降权**计入，而不是满权。

## reducer 约定

`evidence` 用 `operator.add` 累加：一轮问答可能同时产生"新症状"与"否认旧症状"两类证据，
后写的节点不该覆盖先写的。其余字段默认覆盖语义（`total=False` 允许部分更新）。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, TypedDict


class EvidenceSource(str, Enum):
    """一条症状证据的来源。**这决定了它的可信权重**。"""

    VOLUNTEERED = "volunteered"
    """用户自己主动说出来的——最可信。"""

    CONFIRMED = "confirmed"
    """系统抛出试探性假设后，用户确认的——有建议性偏倚，降权。"""

    NEGATED = "negated"
    """用户自己否认的（"没有胸痛"）。"""

    NEGATED_AFTER_SUGGESTION = "negated_after_suggestion"
    """系统建议后用户否认的——比自发否认更弱（可能是被问烦了随口否）。"""

    RULE = "rule"
    """确定性规则判定（例如急诊红线命中）。不走候选打分。"""


@dataclass(frozen=True)
class SymptomEvidence:
    """一条症状证据。`weight` 由 `hypothesis.weight_of()` 按来源算出，不在这里硬编码。"""

    symptom: str
    present: bool
    source: EvidenceSource
    turn: int
    weight: float = 0.0
    note: str = ""

    @property
    def is_positive(self) -> bool:
        return self.present

    def describe(self) -> str:
        sign = "+" if self.present else "-"
        return f"{sign}{self.symptom}[{self.source.value} w={self.weight:.2f}]"


@dataclass
class RedLine:
    """急诊红线命中（确定性规则，LLM 无权绕过）。"""

    word: str
    status: str
    experiencer: str
    temporality: str
    advice_given: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "status": self.status,
            "experiencer": self.experiencer,
            "temporality": self.temporality,
            "advice_given": self.advice_given,
        }


@dataclass
class CitedAnswer:
    """一条带引用的回答（self-ask 的子答案与最终答案都用它）。"""

    text: str
    citations: list[dict] = field(default_factory=list)
    sub_question: str = ""


class ClinicState(TypedDict, total=False):
    """clinic 域在图上流动的状态。

    `total=False`：节点只更新自己关心的字段，其余保持不动（LangGraph 的默认覆盖语义）。
    """

    # ---- 输入 ----
    question: str
    patient_id: str
    turn: int
    resume_text: str
    """`interrupt()` 挂起后，用户回答被 `Command(resume=...)` 送回图里，
    内核把它放在这里供 `observe` 节点消费。不用 messages 是因为本项目不做
    通用对话，只需要"这一轮用户说了什么"。"""

    # ---- 证据与推理 ----
    evidence: Annotated[list[SymptomEvidence], operator.add]
    candidates: dict[str, float]  # 候选（疾病/科室） -> 分数
    asked: Annotated[list[str], operator.add]  # 已问过的问题，避免重复追问
    last_question: str
    pending_suggestion: dict[str, Any]  # 已抛出但未获回答的试探性假设

    # ---- 结果 ----
    red_line: dict[str, Any] | None
    triage: dict[str, Any]  # {departments, diseases, reason}
    answer: dict[str, Any]  # CitedAnswer 的序列化形式
    ticket: dict[str, Any]  # 挂号结果
    booking_confirmed: bool
    """用户是否**明确确认**了挂号。副作用只有在它为 True 时才允许发生。"""
    status: str  # asking / emergency / booking / done / refused
    stop_reason: str

    # ---- 可观测 ----
    diagnostics: Annotated[list[str], operator.add]
    llm_calls: int
    cost_usd: float


def empty_state(question: str, *, patient_id: str = "anon") -> ClinicState:
    """新会话的初始状态。显式给全字段，避免下游到处写 `.get(k, default)`。"""
    return ClinicState(
        question=question,
        patient_id=patient_id,
        turn=0,
        evidence=[],
        candidates={},
        asked=[],
        last_question="",
        pending_suggestion={},
        red_line=None,
        triage={},
        answer={},
        ticket={},
        booking_confirmed=False,
        status="asking",
        stop_reason="",
        diagnostics=[],
        llm_calls=0,
        cost_usd=0.0,
    )
