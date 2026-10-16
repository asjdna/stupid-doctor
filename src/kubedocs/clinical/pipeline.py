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

编排层**不覆盖**模型给出的 `urgency`，也不给红线词任何短路分支。`RiskGuard` 的命中以
**线索**形式进参考标准面板（关键词 + 断言 / 经验者 / 时间性三维取值，见
`clinical/reference.py`），随系统提示交给模型——它**不替模型下结论**，
`ClinicalTurn.urgency` 始终是模型自己给的值。命中与"模型判了什么"之间的差异不写进正文，
只作为可审计读数留下：`red_line_positive` 与"命中而模型未判急重症"时的一条不一致痕迹
（`ClinicalTurn.problems`）。
契约层守住唯一的硬约束：**判出 `red_flags` 却把 `urgency` 写成非 emergency ⇒ 记违规**。

> `/chat`（`agent/clinic_react.py`）递的是**同一份面板**、同一套线索，同样没有短路分支：
> 两条路径的判断者都是模型，只有输出形状与校验手段不同。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
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
from kubedocs.clinical.reference import FactSet, build_facts
from kubedocs.clinical.safety import SafetyAudit, SafetyOutcome, apply_gate
from kubedocs.clinical.safety import append_audit as append_safety_audit
from kubedocs.clinical.verify import facts_only_answer, strip_ungrounded, verify
from kubedocs.llm_status import NO_LLM_MESSAGE

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

    reference: dict[str, Any] | None = None
    """这一轮交给模型的**参考标准面板**（结构化形状，见 `clinical/reference.py`）。

    它与 `contract_violations` 是**必须成对留档**的：只有同时留下"给它看了什么"和
    "它哪里越界了"，事后才能回答"这次是依据不够，还是模型没守规矩"。
    """

    contract_violations: list[dict[str, str]] = field(default_factory=list)
    """输出约束层拦下的东西（`clinical/verify.py`）。

    **必须回给调用方**，不能只写日志：它是"模型是否在编造"的唯一可观测读数，
    而这个读数决定了这套"判断交给大模型"的架构能不能被信任。
    """

    safety: dict[str, Any] | None = None
    """**确定性安全门**（第三层）这一轮触发了哪些不变量（见 `clinical/safety.py`）。

    与 `contract_violations` 的分工：`contract_violations` 是"模型哪里越界了"的**逐条明细**，
    `safety` 是"程序因此执行了哪几条底线"的**可对账读数**（不变量 id + 处置动作）。
    两者都进审计：前者在响应里给人看，后者另有一行写入 `safety_audit.jsonl`。
    """

    llm_calls: int = 0
    """这一轮**真正调了几次模型**（工具循环自己数的）。

    它进 `GET /metrics` 的花费统计与评测报告：只有知道"一次问诊几次调用"，
    才能解释花费与延迟的变化——只看总额看不出是"来了更多人"还是"每人多问了几轮"。
    """

    steps: int = 0
    """这一轮的循环步数（含工具调用轮）。异常高的步数通常意味着提示词或工具描述让模型绕圈了。"""


class ClinicalPipeline:
    """把各项能力装到一起。**每个依赖都是注入的**，所以每一段都能被单测替换。

    `tool_loop` 是**必需**的：一轮问诊里"用哪些工具取证"由模型决定
    （架构决定：底层约束为大模型），代码只提供工具清单与执行器。

    `reasoner`（主判）在本管线里**不再被调用**：以前它承担"工具循环失败/输出不可解析时
    按现有证据给个有限结论"，那种结论没有经过判断者，已经整体取消（现在只回
    `LLM_FAILED_MESSAGE`）。它保留是因为**评测与对照实验仍然直接用它**
    （`scripts/eval_medical_cot.py` 等走的是"不带工具的一次主判"那条路），
    这也是"主判 vs 工具循环"这两条判断路线的对照臂。

    `backend` / `guard` / `triage` 仍然要传，但**角色变了**：它们不再是"降级兜底"，
    而是**证据提供者**——它们的输出经 `reference_builder` 装成参考标准面板交给模型
    （`clinical/reference.py`），由模型判断。`build_evidence` 预取的那份证据里
    红线信号与分诊候选**会进面板**，知识条目仍只作降级兜底（模型要自己调工具检索）。

    `reference_builder` 由**域层**注入（`domains/clinic/reference.clinic_reference`）：
    面板要用的症状词表、科室表、号源事实都是域知识，管线这一层不该知道。
    不注入时用通用面板（只有红线信号 + 检索依据 + 链路问题）。

    `has_llm` 是"这个部署到底有没有大模型"。它为假时**整条管线不再产出任何判断**，
    只回一句"未连接大模型"——没有判断者就没有结论，这也是"不留降级层"的落点。

    `store` 是就诊记录的落点（默认 `NoopPersistence` = 不落库）。真实部署由 HTTP 层注入
    `PgPersistence`；这里注入而不是模块内 new，是为了单测能塞一个会抛异常的假 store
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
        reference_builder: Any = None,
        has_llm: Any = None,
        index_root: str | Path | None = None,
    ) -> None:
        self._backend = backend
        self._reasoner = reasoner
        self._guard = guard
        self._triage = triage
        self._store: ClinicalStore = store or NoopPersistence()
        self._extractor = extractor or RuleSlotExtractor()
        self._tool_loop = tool_loop
        self._top_k = top_k
        self._reference_builder = reference_builder
        # 默认"有"：单测里的打桩管线不需要每次都声明它，而"默认无 LLM"会让所有
        # 现有用例静默退化成"未连接大模型"，那是最难查的一种假绿。
        self._has_llm = has_llm or (lambda: True)
        # 安全审计日志的落点。默认 None = **不写文件**（单测与嵌入式用法不该往磁盘上留东西），
        # 真实部署由 HTTP 层注入 `settings.index_root`（见 api/consult.py）。
        self._audit_root_override: Path | None = Path(index_root) if index_root else None

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
        if not self._has_llm():
            # **没有判断者就没有结论。** 这里不检索、不装配面板、不做任何兜底：
            # 面板是"给判断者的依据"，没有判断者时它没有读者；而任何兜底回答都等于
            # 用一个没经过判断的内容冒充结论。所以只回一句话，并把这一轮照常留档。
            turn = ClinicalTurn(action="error", message=NO_LLM_MESSAGE, degraded="未连接大模型")
            turn.intake = state if state is not None else IntakeState()
            turn.encounter_id = uuid4().hex
            self._persist(question, history, patient_id, turn)
            return turn

        intake = state if state is not None else IntakeState()
        intake.merge(question, self._extractor)

        # 这份证据是**线索来源**：红线信号与分诊候选会进参考标准面板（模型据此判断），
        # 知识条目仍只作降级兜底——模型要自己调工具检索，否则"自己决定要不要检索"就作废了。
        evidence = build_evidence(
            question,
            backend=self._backend,
            top_k=self._top_k,
            guard=self._guard,
            triage=self._triage,
        )
        turn = self._tool_loop_turn(question, intake, evidence, history, patient_id=patient_id)
        # 三条返回路径（正常/工具循环失败/输出不合契约）都从这里出去，所以主键在这里统一生成。
        turn.encounter_id = uuid4().hex
        self._persist(question, history, patient_id, turn)
        self._audit_safety(turn)
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

    def _audit_safety(self, turn: ClinicalTurn) -> None:
        """把这一轮触发的安全不变量写进治理审计日志（`safety_audit.jsonl`）。

        写在落库之后、返回之前：审计要能拿 `trace_id`（= `encounter_id`）与就诊记录对上，
        而失败**不影响这一轮应答**（与落库同一条纪律，见 `clinical/safety.py`）。
        没有配审计落点（`index_root=None`，单测与嵌入式用法）时**什么都不写**。
        """
        root = self._audit_root_override
        if root is None or not turn.safety:
            return
        fired = turn.safety.get("fired") or []
        if not fired:
            return
        records = [
            SafetyAudit(
                invariant=str(row.get("invariant", "")),
                action=str(row.get("action", "record_only")),  # type: ignore[arg-type]
                detail=str(row.get("detail", "")),
                trace_id=turn.encounter_id,
                domain="clinic",
            )
            for row in fired
        ]
        append_safety_audit(root, SafetyOutcome(message=turn.message, fired=records))

    def _tool_loop_turn(
        self,
        question: str,
        intake: IntakeState,
        fallback_evidence: EvidenceBundle,
        history: list[dict[str, Any]] | None = None,
        *,
        patient_id: str = "anon",
    ) -> ClinicalTurn:
        """让模型自己取证、按契约解析，**再拿事实集校验它的输出**。

        `fallback_evidence` 是代码预取的那一份，这里只用两个字段：红线信号与分诊候选
        （它们进参考标准面板，由模型判断）。**它的知识条目不进模型上下文**——
        否则"模型自己决定要不要检索"就名存实亡了。

        `patient_id` 经 `context=` 注入工具执行体：查"我的挂号"、建待确认挂号单都要知道
        这是谁，而这个值是**服务端**给的，不进工具 schema，模型伪造不了。

        ## 校验与处置（三级）

        模型跑完循环之后，`clinical/verify.py` 拿**同一份事实集**校验它的输出：

        1. 有 `block` 级违规 → **带违规原因重生成一次**（多花一次循环，但比给患者一句
           编造的话便宜得多）；
        2. 重生成仍违规 → **不展示模型那段散文**，改由服务端拼一段只含事实的回答
           （`facts_only_answer`）；
        3. 无论哪种，违规都进 `contract_violations` 随响应回去——它是"模型有没有在编造"
           的唯一可观测读数，只写日志等于没人看得见。

        ## 模型调用失败 / 输出不可解析

        两者都**不再产出任何临床内容**（以前会"按现有证据给出有限结论"）。那种结论没有
        经过判断者，把它当结论交付等于用没人负责的内容冒充医学判断。现在只回
        `LLM_FAILED_MESSAGE`，并把 `assessment` 留空——**没有判断者就没有结论**。
        """
        from kubedocs.clinical.prompts import prompt_version, system_prompt
        from kubedocs.llm_status import LLM_FAILED_MESSAGE

        loop = self._tool_loop
        # **面板要在循环之前递给模型**：不递的话，"规则资产作为线索、由模型判断"这条架构
        # 在主路径上根本没生效——模型只会看到 system.md 与它自己检索到的东西，
        # 红线三维、症状归一、科室候选、号源事实一条都看不到（实测就是这样）。
        #
        # 但递进去的必须是**线索版**：`build_evidence` 预取的知识条目**不进提示词**，
        # 否则"模型自己决定要不要检索"就作废了（这是本管线的既有取舍）。
        clue_evidence = fallback_evidence.model_copy(update={"items": []})
        clue_facts = self._build_facts(question, clue_evidence, patient_id=patient_id)
        system = f"{system_prompt()}\n\n---\n\n{clue_facts.render(defer_retrieval=True)}"
        try:
            result = loop.run(
                system=system,
                question=question,
                intake=intake,
                history=history,
                context={"patient_id": patient_id},
            )
        except Exception as e:  # noqa: BLE001 - 模型不可用是外部故障，不该让整轮 traceback
            logger.warning("工具循环失败：%s: %s", type(e).__name__, e)
            return ClinicalTurn(
                action="error",
                message=LLM_FAILED_MESSAGE,
                evidence=fallback_evidence,
                problems=[f"工具循环失败（{type(e).__name__}）"],
                degraded=f"工具循环失败（{type(e).__name__}: {e}）",
                intake=intake,
                red_line_positive=fallback_evidence.red_line_positive,
            )

        evidence = result.evidence.model_copy(
            update={"red_line": fallback_evidence.red_line, "triage": fallback_evidence.triage}
        )
        facts = self._build_facts(question, evidence, patient_id=patient_id)
        assessment = self._parse(result.final_text)
        if assessment is None:
            # **一次格式修复重试**：工具循环跑完之后模型偶尔会吐一段散文而不是 JSON
            # （实测约每 30 轮出现 1~2 次，且不可预测）。直接回"模型调用失败"对患者是白等一场，
            # 而"请只输出 JSON"这句话几乎总能修好它——代价是一次调用，收益是一次可用回答。
            # 修不好仍然只回失败提示：**不产出任何兜底结论**。
            logger.warning("工具循环输出不合契约，重试一次：%s", result.final_text[:200])
            repair = self._retry_with_feedback(
                loop,
                question,
                intake,
                history,
                patient_id,
                "你上一次的输出不是契约要求的 JSON（可能夹了说明文字、代码围栏或缺字段）。"
                "请**只输出那一个 JSON 对象**：不要解释、不要围栏、不要多余字段；"
                "字段与取值范围按系统提示里的输出格式。",
                clue_facts,
            )
            if repair is not None:
                repaired = self._parse(repair.final_text)
                if repaired is not None:
                    result, assessment, evidence = (
                        repair,
                        repaired,
                        repair.evidence.model_copy(
                            update={
                                "red_line": fallback_evidence.red_line,
                                "triage": fallback_evidence.triage,
                            }
                        ),
                    )
                    facts = self._build_facts(question, evidence, patient_id=patient_id)

        if assessment is None:
            logger.warning("工具循环输出仍然不合契约：%s", result.final_text[:200])
            return ClinicalTurn(
                action="error",
                message=LLM_FAILED_MESSAGE,
                evidence=evidence,
                problems=["模型输出不符合契约（解析失败，已重试一次）"],
                degraded="模型输出不可解析",
                prompt_version=prompt_version(),
                red_line_positive=evidence.red_line_positive,
                intake=intake,
                reference=facts.to_payload(),
                tool_calls=[
                    {"name": c.name, "ok": c.ok, "summary": c.summary, "result": c.result}
                    for c in result.calls
                ],
                llm_calls=int(getattr(result, "llm_calls", 0) or 0),
                steps=int(getattr(result, "rounds", 0) or 0),
            )

        # **先校验、再执行契约**，顺序不能反：`enforce` 会把没有依据的检查/用药建议**删掉**，
        # 若先跑它，"检索为空却给了医学内容"这条检查就看不出任何异常（字段已被清空），
        # 而**散文里的那句话照样会送到患者眼前**。校验要看到的是模型**原始**输出。
        verification = verify(assessment, assessment.patient_summary, facts)
        # 历次尝试的违规**全部留档**：第一轮确实越界了，不能因为第二轮回正就当没发生
        # ——"模型偶尔会编"这个事实正是这套架构最需要被观测到的东西。
        recorded = list(verification.violations)
        assessment, problems, dropped = self._enforce(assessment, evidence)

        if not verification.ok():
            logger.warning("输出未通过事实校验，重生成一次：%s", verification.kinds())
            retry = self._retry_with_feedback(
                loop, question, intake, history, patient_id, verification.feedback(), clue_facts
            )
            if retry is not None:
                retry_assessment = self._parse(retry.final_text)
                if retry_assessment is not None:
                    # **证据可能变了**（它可能又检索了一次），所以事实集与校验都要重做，
                    # 不能拿第一轮的事实去校验第二轮的输出。
                    result = retry
                    evidence = retry.evidence.model_copy(
                        update={"red_line": fallback_evidence.red_line, "triage": fallback_evidence.triage}
                    )
                    facts = self._build_facts(question, evidence, patient_id=patient_id)
                    verification = verify(retry_assessment, retry_assessment.patient_summary, facts)
                    recorded.extend(verification.violations)
                    assessment, problems, dropped = self._enforce(retry_assessment, evidence)

        blocked = not verification.ok()
        if blocked:
            # 第②级：模型那段散文不上屏，改由服务端拼事实。**患者永远看不到被拦下的那句话。**
            message = facts_only_answer(
                facts, reason="、".join(sorted({v.kind for v in verification.blocking()}))
            )
            degraded = "回答未通过事实校验，已改为只陈述事实"
            logger.warning("回答未通过事实校验，已替换为事实回答：%s", verification.kinds())
        else:
            message = assessment.patient_summary
            degraded = ""

        # 结构化条目按 `strip` 级违规清理（R3 那一半复用 `contracts.enforce`，不重写一遍）。
        assessment = strip_ungrounded(assessment, verification, facts)
        if dropped:
            logger.warning("按 R3 丢弃无依据建议：%s", "、".join(dropped))

        # **第三层：确定性安全门**（`clinical/safety.py`）。放在最后一步，因为它看的是
        # "患者最终会看到的那段话"：前三层（语义理解 / 主动问诊 / 契约校验）都跑完之后，
        # 由它执行那几条谁也不能例外的底线——判了急症就必须有急救动作（缺了就补），
        # 未排除的危险信号没有下文时必须由服务端补一句追问。
        safety = apply_gate(
            assessment,
            message,
            violated_kinds=sorted({v.kind for v in recorded}),
            trace_id="",
            domain="clinic",
        )
        message = safety.message
        if safety.fired:
            logger.info("确定性安全门触发：%s", safety.ids)

        # 追问由模型的 `followup_questions` 决定（编排层不设追问闸门）。
        # 但**急重症优先**：判出急症时不给"先追问"的动作，必须先给急救指引。
        escalated = "emergency_judgment_requires_directive" in safety.ids
        followup = list(assessment.followup_questions)
        if safety.followup and safety.followup not in followup:
            # 问诊层没给出追问、而安全门判定"还没排除"时，补一句确定性的追问。
            followup.append(safety.followup)
            if not blocked:
                message = f"{message}\n\n{safety.followup}" if message else safety.followup

        if blocked:
            action: TurnAction = "error"
        elif escalated:
            # 安全门按模型自己给出的信号（critical / likely）补了急救动作 ⇒ 这一轮就是急症轮，
            # 不能因为 `urgency` 被写成 urgent 就把动作降成"常规回答"（那会让前端不弹横幅、
            # 患者也看不到"立刻去医院"）。
            action = "emergency"
        elif followup and assessment.urgency != "emergency":
            action = "ask"
        else:
            action = "emergency" if assessment.urgency == "emergency" else "answer"
        return ClinicalTurn(
            action=action,
            message=message,
            assessment=assessment,
            evidence=evidence,
            problems=problems,
            tool_calls=[
                {"name": c.name, "ok": c.ok, "summary": c.summary, "result": c.result} for c in result.calls
            ],
            prompt_version=prompt_version(),
            red_line_positive=evidence.red_line_positive,
            intake=intake,
            reference=facts.to_payload(),
            contract_violations=[
                {"kind": v.kind, "severity": v.severity, "detail": v.detail} for v in recorded
            ],
            safety=safety.to_payload(),
            degraded=degraded,
            llm_calls=int(getattr(result, "llm_calls", 0) or 0),
            steps=int(getattr(result, "rounds", 0) or 0),
        )

    def _build_facts(self, question: str, evidence: EvidenceBundle, *, patient_id: str) -> FactSet:
        """装配参考标准面板。**域层注入的建造器优先**——症状词表、科室表、号源事实都是域知识。

        不注入时给一个通用面板（红线信号 + 检索依据 + 链路问题）：它仍然能让模型看到
        "有没有检索到资料""红线三维怎么说"，只是没有域特有的线索。
        """
        if self._reference_builder is not None:
            # 建造器是注入的（类型是 Any），显式标注返回值，别把 Any 漏进本方法的声明返回类型。
            built: FactSet = self._reference_builder(question, evidence, patient_id)
            return built
        return build_facts(evidence=evidence)

    @staticmethod
    def _parse(raw: str) -> Any:
        """把模型输出解析成契约对象；解析不了返回 None（调用方必须处理，不许兜底编内容）。"""
        from kubedocs.clinical.contracts import ClinicalAssessment

        try:
            return ClinicalAssessment.model_validate(json.loads(_extract_json(raw)))
        except Exception as e:  # noqa: BLE001 - 模型输出是外部输入，任何畸形都只是"没解析出来"
            logger.warning("模型输出解析失败：%s: %s", type(e).__name__, e)
            return None

    @staticmethod
    def _enforce(assessment: Any, evidence: EvidenceBundle) -> tuple[Any, list[str], list[str]]:
        """执行契约（R3：没有依据的检查/用药建议丢掉）。返回 (结论, 问题, 被丢的东西)。"""
        from kubedocs.clinical.contracts import enforce

        enforced = enforce(assessment, evidence.ids())
        return enforced.assessment, enforced.problems, enforced.dropped

    def _retry_with_feedback(
        self,
        loop: Any,
        question: str,
        intake: IntakeState,
        history: list[dict[str, Any]] | None,
        patient_id: str,
        feedback: str,
        clue_facts: FactSet | None = None,
    ) -> Any:
        """带违规原因重生成一次。**再失败就返回 None**，由调用方换成事实回答。

        重生成时**同样要把线索版面板带上**：只带回灌的违规原因、把面板摘掉，
        等于换了一个信息更少的环境去要求它做对——那是一次必然更容易失败的重试。
        """
        from kubedocs.clinical.prompts import system_prompt

        system = system_prompt()
        if clue_facts is not None:
            system = f"{system}\n\n---\n\n{clue_facts.render(defer_retrieval=True)}"
        try:
            return loop.run(
                system=f"{system}\n\n---\n\n{feedback}",
                question=question,
                intake=intake,
                history=history,
                context={"patient_id": patient_id},
            )
        except Exception as e:  # noqa: BLE001 - 重生成失败不再兜底，交给事实回答
            logger.warning("重生成失败：%s: %s", type(e).__name__, e)
            return None
