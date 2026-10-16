"""会话状态：clinic 域在 LangGraph 图上流动的唯一数据结构。

每条症状证据都带 `source`（`EvidenceSource`）：系统抛出的试探性假设会引入建议性偏倚，
用户答"是"未必是自发症状，所以打分按来源降权（权重由 `hypothesis.weight_of` 计算）。
否认同样要记——鉴别诊断里"排除"与"命中"同等重要，但口语否认可能是没说准，故降权计入。

reducer 约定：`evidence` / `asked` / `diagnostics` 用 `operator.add` 累加（一轮问答可能
同时产生新症状与否认旧症状，后写的节点不该覆盖先写的）；其余字段是覆盖语义，
`total=False` 允许节点只更新自己关心的字段。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
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

    def describe(self) -> str:
        sign = "+" if self.present else "-"
        return f"{sign}{self.symptom}[{self.source.value} w={self.weight:.2f}]"


class ClinicState(TypedDict, total=False):
    """clinic 域在图上流动的状态。

    `total=False`：节点只更新自己关心的字段，其余保持不动（LangGraph 的默认覆盖语义）。
    """

    # ---- 输入 ----
    question: str
    patient_id: str
    turn: int
    resume_text: str
    """`interrupt()` 挂起后用户的回答，由 `Command(resume=...)` 送回图里供 `observe` 消费。
    只承载"这一轮用户说了什么"，不需要通用对话历史，所以不复用 messages。"""

    # ---- 证据与推理 ----
    evidence: Annotated[list[SymptomEvidence], operator.add]
    candidates: dict[str, float]  # 候选（疾病/科室） -> 分数
    asked: Annotated[list[str], operator.add]  # 已问过的问题，避免重复追问
    last_question: str
    pending_suggestion: dict[str, Any]  # 已抛出但未获回答的试探性假设

    # ---- 结果 ----
    red_line: dict[str, Any] | None
    triage: dict[str, Any]  # {departments, diseases, reason}
    answer: dict[str, Any]  # 作答节点的输出（正文 + 引用）
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
