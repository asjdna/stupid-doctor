"""编排：一轮问诊的完整链路。

**只有一条路径——工具循环**：模型自己决定调哪些工具取证，代码只提供工具清单与执行器，
见 `clinical/orchestration.py`。

```
intake ─▶ evidence（仅作降级兜底）─▶ 工具循环（模型自己取证）─▶ 契约化应答 ─▶ 落库
```

**落库在最后一步，而且落库失败不影响应答**：宁可少一条记录，也不能因为落库失败让患者
拿不到答复（写入异常在这里被记成 warning，见 `_persist`）。

两条硬规则在这里落地，不靠提示词自觉：

1. **证据不足不得给结论**：由模型自己追问（`followup_questions` ⇒ `action=ask`）。
   编排层**刻意不设追问闸门**：闸门只看槽位填满没有、不看急症，抢在模型之前会把急症
   拦成一句追问（教科书级卒中会被回成"这个情况多久了？"），提示词 R5 已要求它追问。
2. **急重症优先**：判出 `urgency=emergency` 时不给"先追问"的动作，必须先给急救指引。

## 急诊：判断权在模型，编排层只做"自相矛盾"的检查

编排层**不覆盖**模型给出的 `urgency`。`RiskGuard` 的关键词命中**不进提示词、
不影响 `urgency`**，只用于两件事：主判不可用时给一句保守提示，以及
"命中而模型未判急重症"时留一条不一致痕迹（`ClinicalTurn.problems`）。
契约层守住唯一的硬约束：**判出 `red_flags` 却把 `urgency` 写成非 emergency ⇒ 记违规**。

> ⚠️ 这与**图路径**（`/chat`）相反：那条路上 `RiskGuard` 是
> "LLM 不可绕过的强制边"。两条路径故意并存，取舍相反，别混着看。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.evidence import EvidenceBundle, Guard, TriageEngine, build_evidence
from kubedocs.clinical.intake import IntakeState, RuleSlotExtractor, SlotExtractor
from kubedocs.clinical.persistence import (
    ClinicalStore,
    EncounterRecord,
    NoopPersistence,
    PatientRecord,
)
from kubedocs.clinical.rag import RetrievalBackend
from kubedocs.clinical.reasoning import ClinicalReasoner, _extract_json

logger = logging.getLogger(__name__)

TurnAction = Literal["ask", "answer", "emergency", "error"]
"""这一轮该做什么。**由编排层从模型结论推导**——模型只负责医学内容。"""


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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """模型这一轮**自己决定调用**了哪些工具（名称/成败/结果摘要）。

    单独一个字段而不是塞进 `problems`：`problems` 是**契约问题**（依据缺失、引用不存在…），
    而工具调用是**正常行为**。混在一起会让"这一轮到底有没有契约违规"读不出来。
    """
    intake: IntakeState = field(default_factory=IntakeState)

    encounter_id: str = ""
    """这一轮落库的**就诊记录主键**（`uuid4().hex`）。空串 = 没落库（测试接缝/直接调 `_tool_loop_turn`）。

    HTTP 层把它回给前端：前端拿它去 `GET /consult/encounters` 对账"这一轮到底存下来了没有"。
    """


class ClinicalPipeline:
    """把各项能力装到一起。**每个依赖都是注入的**，所以每一段都能被单测替换。

    `tool_loop` 是**必需**的：一轮问诊里"用哪些工具取证"由模型决定
    （架构决定：底层约束为大模型），代码只提供工具清单与执行器。

    `backend` / `guard` / `triage` 仍然要传：它们喂的是**代码预取的降级兜底证据**
    （工具循环失败时的有限结论、红线留痕、科室兜底）。这份证据的知识条目
    **不进模型上下文**，否则"模型自己决定要不要检索"就名存实亡了。

    `store` 是就诊记录的落点（默认 `NoopPersistence` = 不落盘）。真实部署由 HTTP 层注入
    `SqlitePersistence`；这里注入而不是模块内 new，是为了单测能塞一个会抛异常的假 store
    去验证"落库失败不打断问诊"。
    """

    def __init__(
        self,
        *,
        backend: RetrievalBackend,
        reasoner: ClinicalReasoner,
        tool_loop: Any,
        guard: Guard | None = None,
        triage: TriageEngine | None = None,
        store: ClinicalStore | None = None,
        extractor: SlotExtractor | None = None,
        top_k: int = 10,
    ) -> None:
        self._backend = backend
        self._reasoner = reasoner
        self._guard = guard
        self._triage = triage
        self._store: ClinicalStore = store or NoopPersistence()
        self._extractor = extractor or RuleSlotExtractor()
        self._tool_loop = tool_loop
        self._top_k = top_k

    def advance(
        self,
        question: str,
        *,
        state: IntakeState | None = None,
        history: list[dict[str, Any]] | None = None,
        patient_id: str = "anon",
    ) -> ClinicalTurn:
        """推进一轮。`state` 与 `history` 由调用方持有（会话状态属于会话层，不属于流程）。

        `history` 是之前的对话（`[{"role","content"}, ...]`）。产品要求"科室/用药
        作为回答之后的可选扩展项、等用户来问"，而**追问要答得一致就必须有上文**——
        否则模型会把判断重跑一遍，同一个病例可能问出另一个科室。

        `patient_id` 决定这一轮**记在谁名下**（默认 `anon` = 匿名）。它是记录的归属，
        **不是认证**：谁都能声明自己是谁，认证由部署方在网关做（见 SECURITY.md）。
        它在这里的真实读者就是落库那一步（`_persist`），以及读回路径的按人查询。
        """
        intake = state if state is not None else IntakeState()
        intake.merge(question, self._extractor)

        # 这份证据只作**降级兜底**：红线/分诊进 `ClinicalTurn`，知识条目不进模型上下文。
        evidence = build_evidence(
            question,
            backend=self._backend,
            top_k=self._top_k,
            guard=self._guard,
            triage=self._triage,
        )
        turn = self._tool_loop_turn(question, intake, evidence, history)
        # 三条返回路径（正常/工具循环失败/输出不合契约）都从这里出去，所以主键在这里统一生成。
        turn.encounter_id = uuid4().hex
        self._persist(question, history, patient_id, turn)
        return turn

    def _persist(
        self,
        question: str,
        history: list[dict[str, Any]] | None,
        patient_id: str,
        turn: ClinicalTurn,
    ) -> None:
        """把这一轮写进就诊记录（患者 / 就诊 / 消息）。

        **取舍：宁可少一条记录，也不能因为落库失败让患者拿不到答复。** 写入异常在这里
        被 `logger.warning` 吞掉，本轮结果照常返回——磁盘满、库被锁、权限不对都是
        可预期的运维故障，而"已经算出来的答复"不该因为记不下来就丢掉。代价是这种轮次
        不会出现在 `GET /consult/encounters` 里，所以**失败必须留痕**（warning 里有异常类型）。

        顺序有讲究：**消息先写，再落就诊行**。`save_encounter` 只在"这次就诊还没有任何
        消息"时才用 `transcript` 播种消息表（见 persistence 的说明），反过来调就会把
        这段对话上文按轮重复写进消息表——N 轮变 N² 条。

        `institution` 用默认空串：多机构不在本次范围内，列先留着（见 persistence 模块文档）。
        """
        try:
            self._store.upsert_patient(PatientRecord(patient_id=patient_id))
            self._store.append_message(encounter_id=turn.encounter_id, role="user", text=question)
            if turn.message:
                self._store.append_message(
                    encounter_id=turn.encounter_id, role="assistant", text=turn.message
                )
            self._store.save_encounter(
                EncounterRecord(
                    encounter_id=turn.encounter_id,
                    patient_id=patient_id,
                    chief_complaint=question,
                    transcript=self._transcript(history, question, turn.message),
                    assessment=turn.assessment,
                )
            )
        except Exception as e:  # noqa: BLE001 - 落库失败不许影响这一轮的答复
            logger.warning("就诊记录落库失败，本轮结果照常返回：%s: %s", type(e).__name__, e)

    @staticmethod
    def _transcript(
        history: list[dict[str, Any]] | None, question: str, message: str
    ) -> list[dict[str, str]]:
        """这一轮用到的对话上文 + 本轮问答（存成普通 dict，不存 pydantic 对象）。"""
        rows: list[dict[str, str]] = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in (history or [])
        ]
        rows.append({"role": "user", "content": question})
        if message:
            rows.append({"role": "assistant", "content": message})
        return rows

    def _tool_loop_turn(
        self,
        question: str,
        intake: IntakeState,
        fallback_evidence: EvidenceBundle,
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
        # 追问由模型的 `followup_questions` 决定（编排层不设追问闸门）。
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
