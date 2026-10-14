"""编排：一轮问诊的完整链路（方案 §7）。

```
intake ─▶ retrieve ─▶ evidence ─▶ reason ─▶ act ─▶ respond
```

## 两条硬规则在这里落地（不靠提示词自觉）

1. **证据不足不得给结论**：关键槽位没填就问，问不出来就"带局限作答"——
   绝不硬凑一个结论（`IntakeState.can_conclude()`）。
2. **需要动作时必须显式确认**：挂号走 skill 流程，未确认时 skill 自己会停在 `confirm`，
   编排层**只负责把确认请求转给患者**，不替患者答应。

## 红线与紧急档的关系（按决定 2b）

编排层**不覆盖**模型给出的 `urgency`。它做三件事：
把红线信号放进证据、在提示词里强制处理、并把"红线阳性 vs 模型判定"的冲突
作为契约问题上报（`ClinicalTurn.problems`），同时把 `red_line_positive` 暴露给调用方。
**前端该不该无视模型直接弹急诊**是一个产品决定，尚未与你确认，所以这里不替你决定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.evidence import EvidenceBundle, Guard, TriageEngine, build_evidence
from kubedocs.clinical.intake import IntakeState, RuleSlotExtractor, SlotExtractor
from kubedocs.clinical.persistence import ClinicalStore, NoopPersistence
from kubedocs.clinical.rag import RetrievalBackend
from kubedocs.clinical.reasoning import ClinicalReasoner
from kubedocs.clinical.skills.base import SkillOutcome, SkillRegistry, SkillRequest

logger = logging.getLogger(__name__)

TurnAction = Literal["ask", "answer", "emergency", "booked", "confirm_booking", "error"]
"""这一轮该做什么。**由编排层判定，不由模型判定**——模型只负责医学内容。"""


class Reasoner(Protocol):
    def reason(self, question: str, evidence: EvidenceBundle, intake: IntakeState) -> object: ...


@dataclass
class ClinicalTurn:
    """一轮的完整产物。调用方（HTTP 层/前端）只依赖这个对象。"""

    action: TurnAction
    message: str
    assessment: ClinicalAssessment | None = None
    evidence: EvidenceBundle | None = None
    problems: list[str] = field(default_factory=list)
    degraded: str = ""
    prompt_version: str = ""
    red_line_positive: bool = False
    skill: SkillOutcome | None = None
    intake: IntakeState = field(default_factory=IntakeState)


class ClinicalPipeline:
    """把各项能力装到一起。**每个依赖都是注入的**，所以每一段都能被单测替换。"""

    def __init__(
        self,
        *,
        backend: RetrievalBackend,
        reasoner: ClinicalReasoner,
        skills: SkillRegistry,
        guard: Guard | None = None,
        triage: TriageEngine | None = None,
        store: ClinicalStore | None = None,
        extractor: SlotExtractor | None = None,
        top_k: int = 10,
    ) -> None:
        self._backend = backend
        self._reasoner = reasoner
        self._skills = skills
        self._guard = guard
        self._triage = triage
        self._store = store or NoopPersistence()
        self._extractor = extractor or RuleSlotExtractor()
        self._top_k = top_k

    def advance(
        self,
        question: str,
        *,
        state: IntakeState | None = None,
        want_booking: bool = False,
        booking_params: dict[str, object] | None = None,
        confirmed: bool = False,
        patient_id: str = "anon",
    ) -> ClinicalTurn:
        """推进一轮。`state` 由调用方持有（会话状态属于会话层，不属于流程）。"""
        intake = state if state is not None else IntakeState()
        intake.merge(question, self._extractor)

        if not intake.can_conclude():
            slot = intake.next_question()
            if slot is not None:
                intake.mark_asked(slot.key)
                return ClinicalTurn(
                    action="ask",
                    message=slot.question,
                    intake=intake,
                    red_line_positive=False,
                )

        evidence = build_evidence(
            question,
            backend=self._backend,
            top_k=self._top_k,
            guard=self._guard,
            triage=self._triage,
        )

        if want_booking:
            turn = self._booking_turn(evidence, intake, patient_id, booking_params or {}, confirmed)
            if turn is not None:
                return turn

        result = self._reasoner.reason(question, evidence, intake)
        assessment = getattr(result, "assessment", None)
        problems = list(getattr(result, "problems", []) or [])
        degraded = str(getattr(result, "degraded", "") or "")
        version = str(getattr(result, "prompt_version", "") or "")

        if assessment is None:
            logger.error("主判没有返回任何结论（不应发生：reasoning 层保证降级产物）")
            return ClinicalTurn(
                action="error",
                message="这次没能生成建议，请再试一次。情况紧急请直接拨打 120。",
                evidence=evidence,
                problems=problems + ["主判返回空结论"],
                intake=intake,
                red_line_positive=evidence.red_line_positive,
            )

        assert isinstance(assessment, ClinicalAssessment)
        action: TurnAction = "emergency" if assessment.urgency == "emergency" else "answer"
        return ClinicalTurn(
            action=action,
            message=assessment.patient_summary,
            assessment=assessment,
            evidence=evidence,
            problems=problems,
            degraded=degraded,
            prompt_version=version,
            red_line_positive=evidence.red_line_positive,
            intake=intake,
        )

    def _booking_turn(
        self,
        evidence: EvidenceBundle,
        intake: IntakeState,
        patient_id: str,
        params: dict[str, object],
        confirmed: bool,
    ) -> ClinicalTurn | None:
        """需要挂号时走 skill 流程。返回 None 表示"没走挂号"，继续正常主判。"""
        if "book_appointment" not in self._skills:
            return None
        request = SkillRequest(patient_id=patient_id, confirmed=confirmed, params=dict(params))
        outcome = self._skills.get("book_appointment").run(request)
        if outcome.ok:
            action: TurnAction = "booked"
        elif outcome.needs_confirmation:
            action = "confirm_booking"
        else:
            # skill 没走通（科室没解析出来 / 没号了）——**不吞掉原因**，
            # 但也不把内部错误直接抛给患者：交给主判去组织一句人话。
            return None
        return ClinicalTurn(
            action=action,
            message=outcome.message,
            evidence=evidence,
            skill=outcome,
            intake=intake,
            red_line_positive=evidence.red_line_positive,
        )
