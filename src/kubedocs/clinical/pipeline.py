"""编排：一轮问诊的完整链路。

**工具循环路径（有 `tool_loop` 时，当前主路径）**：模型自己决定调哪些工具取证，
代码只提供工具清单与执行器 —— 见 `clinical/orchestration.py`。
**固定路径（没有 `tool_loop` 时）**：

```
intake ─▶ retrieve ─▶ evidence ─▶ reason ─▶ act ─▶ respond
```

两条硬规则在这里落地，不靠提示词自觉：

1. **证据不足不得给结论**：关键槽位没填就问，问不出来就"带局限作答"，
   绝不硬凑结论（`IntakeState.can_conclude()`）。**但有工具循环时不拦**：
   追问闸门抢在模型之前会拦住急症（教科书级卒中会被回成一句追问）。
2. **需要动作时必须显式确认**：挂号走 skill 流程，未确认时 skill 自己会停在 `confirm`，
   编排层只负责把确认请求转给患者，不替患者答应。

## 急诊：判断权在模型，编排层只做"自相矛盾"的检查

编排层**不覆盖**模型给出的 `urgency`。`RiskGuard` 的关键词命中**不进提示词、
不影响 `urgency`**，只用于两件事：主判不可用时给一句保守提示，以及
"命中而模型未判急重症"时留一条不一致痕迹（`ClinicalTurn.problems`）。
契约层守住唯一的硬约束：**判出 `red_flags` 却把 `urgency` 写成非 emergency ⇒ 记违规**。

> ⚠️ 这与**图路径**（`/chat`、`/patient`）相反：那条路上 `RiskGuard` 是
> "LLM 不可绕过的强制边"。两条路径故意并存，取舍相反，别混着看。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.evidence import EvidenceBundle, Guard, TriageEngine, build_evidence
from kubedocs.clinical.intake import IntakeState, RuleSlotExtractor, SlotExtractor
from kubedocs.clinical.persistence import ClinicalStore, NoopPersistence
from kubedocs.clinical.rag import RetrievalBackend
from kubedocs.clinical.reasoning import ClinicalReasoner, _extract_json
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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """模型这一轮**自己决定调用**了哪些工具（名称/成败/结果摘要）。

    单独一个字段而不是塞进 `problems`：`problems` 是**契约问题**（依据缺失、引用不存在…），
    而工具调用是**正常行为**。混在一起会让"这一轮到底有没有契约违规"读不出来。
    """
    intake: IntakeState = field(default_factory=IntakeState)


class ClinicalPipeline:
    """把各项能力装到一起。**每个依赖都是注入的**，所以每一段都能被单测替换。

    ## 两条路径，先后有序（架构决定：底层约束是大模型）

    **工具循环路径（有 `tool_loop` 时）**：模型自己决定调用哪些工具取证
    （检索/分诊/…），代码只提供工具清单与执行器。这是现在的主路径——
    "底层约束为大模型"要落到实处，就必须把"用哪些模块"也交给它。

    **固定路径（没有 `tool_loop` 时）**：代码先检索、再分诊，然后把证据交给主判。
    保留它有两个用处：单测可以只测主判；以及工具循环装配失败时还有一条能跑的链。
    """

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
        tool_loop: Any = None,
        top_k: int = 10,
    ) -> None:
        self._backend = backend
        self._reasoner = reasoner
        self._skills = skills
        self._guard = guard
        self._triage = triage
        self._store = store or NoopPersistence()
        self._extractor = extractor or RuleSlotExtractor()
        self._tool_loop = tool_loop
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
        history: list[dict[str, Any]] | None = None,
    ) -> ClinicalTurn:
        """推进一轮。`state` 与 `history` 由调用方持有（会话状态属于会话层，不属于流程）。

        `history` 是之前的对话（`[{"role","content"}, ...]`）。产品要求"科室/用药
        作为回答之后的可选扩展项、等用户来问"，而**追问要答得一致就必须有上文**——
        否则模型会把判断重跑一遍，同一个病例可能问出另一个科室。
        """
        intake = state if state is not None else IntakeState()
        intake.merge(question, self._extractor)

        # **追问闸门只在没有工具循环时才拦**（闸门只会看槽位填满没有，不会看急症）。
        # 抢在模型之前会把急症拦成一句追问：教科书级卒中被回成"这个情况多久了？"。
        # 有工具循环时追问交给模型：提示词要求"信息不足就追问"。
        if self._tool_loop is None and not intake.can_conclude():
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

        # **工具循环路径**：证据由模型自己取证得来（`build_evidence` 的那份只留作
        # 红线/分诊的**降级提示**，不进模型上下文——见 evidence.render 的说明）。
        if self._tool_loop is not None:
            return self._tool_loop_turn(question, intake, evidence, patient_id, history)

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

    def _tool_loop_turn(
        self,
        question: str,
        intake: IntakeState,
        fallback_evidence: EvidenceBundle,
        patient_id: str,
        history: list[dict[str, Any]] | None = None,
    ) -> ClinicalTurn:
        """让模型自己取证，再把它的最终答复按契约解析。

        `fallback_evidence` 是**代码预取的那一份**，这里只用它的两个字段：
        关键词命中（仅用于"模型是否漏了急症"的留痕与降级提示）
        和分诊结果（降级时当科室兜底）。**它的知识条目不进模型上下文**——
        否则"模型自己决定要不要检索"就名存实亡了。
        """
        from kubedocs.clinical.contracts import ClinicalAssessment, enforce
        from kubedocs.clinical.prompts import prompt_version, system_prompt

        loop = self._tool_loop
        try:
            result = loop.run(system=system_prompt(), question=question, intake=intake, history=history)
        except Exception as e:  # noqa: BLE001 - 工具循环失败要退回到降级产物，不崩对话
            logger.warning("工具循环失败，走降级：%s: %s", type(e).__name__, e)
            degraded_evidence = fallback_evidence.model_copy(
                update={"degraded": [*fallback_evidence.degraded, f"工具循环失败（{type(e).__name__}）"]}
            )
            return ClinicalTurn(
                action="error",
                message=self._reasoner._fallback(degraded_evidence, intake).patient_summary,  # noqa: SLF001
                evidence=degraded_evidence,
                problems=[f"工具循环失败（{type(e).__name__}）"],
                degraded=f"工具循环失败（{type(e).__name__}: {e}），已按现有证据给出有限结论",
                intake=intake,
                red_line_positive=fallback_evidence.red_line_positive,
            )

        evidence = result.evidence.model_copy(
            update={"red_line": fallback_evidence.red_line, "triage": fallback_evidence.triage}
        )
        raw = result.final_text
        try:
            payload = json.loads(_extract_json(raw))
            assessment = ClinicalAssessment.model_validate(payload)
        except Exception as e:  # noqa: BLE001 - 解析失败同样降级，不把裸文本给患者
            logger.warning("工具循环输出不合契约，走降级：%s: %s", type(e).__name__, e)
            assessments = self._reasoner._fallback(evidence, intake)  # noqa: SLF001
            return ClinicalTurn(
                action="error",
                message=assessments.patient_summary,
                assessment=assessments,
                evidence=evidence,
                problems=[f"模型输出不符合契约（{type(e).__name__}），已降级"],
                degraded="模型输出不可解析，已按已有证据给出有限结论",
                prompt_version=prompt_version(),
                red_line_positive=evidence.red_line_positive,
                intake=intake,
            )

        # **真的执行契约**：没有依据的检查/用药建议在这里被丢掉，
        # 而不是只记一笔账（见 `contracts.enforce` 的说明）。
        enforced = enforce(assessment, evidence.ids())
        assessment = enforced.assessment
        problems = enforced.problems
        if enforced.dropped:
            logger.warning("按 R3 丢弃无依据建议：%s", "、".join(enforced.dropped))
        # 追问轮也走这条路（追问由模型的 `followup_questions` 决定，见 `advance` 的说明）。
        # 但**急重症优先**：判出急症时不给"先追问"的动作，必须先给急救指引。
        if assessment.followup_questions and assessment.urgency != "emergency":
            action: TurnAction = "ask"
        else:
            action = "emergency" if assessment.urgency == "emergency" else "answer"
        return ClinicalTurn(
            action=action,
            message=assessment.patient_summary,
            assessment=assessment,
            evidence=evidence,
            problems=problems,
            tool_calls=[{"name": c.name, "ok": c.ok, "summary": c.summary} for c in result.calls],
            prompt_version=prompt_version(),
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
