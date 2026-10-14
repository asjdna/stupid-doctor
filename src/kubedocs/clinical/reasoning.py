"""大模型主判：证据 → 契约化输出（方案 §2、§7 的 `reason` 节点）。

## 三条职责，界限要清楚

1. **装配**：把系统提示词 + 证据 + 已采集信息 + 患者原话交给模型；
2. **校验**：把模型返回的 JSON 解析成 `ClinicalAssessment`，并用 `validate_against()`
   检查"依据是否存在、检查/用药有没有出处"；
3. **降级**：模型不可用或输出不可解析时，**不许凭空生成**，而是退回到
   "只用证据能支撑的东西"（见 `_fallback`）。

## 降级时的一个例外，必须写明白

按决定 2b，紧急档由模型判定。但**模型不可用时没有"模型判定"这回事**——
此时若红线为阳性，`_fallback` 会直接给 `emergency`。理由：规则是确定的、可复现的，
在没有模型的情况下把确定的安全信号丢掉，等于让"模型挂了"变成"急症漏判"。
这不是把红线偷偷改回闸门，而是"决策者缺席时，证据本身成为结论"。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from kubedocs.clinical.contracts import (
    ClinicalAssessment,
    DepartmentAdvice,
    validate_against,
)
from kubedocs.clinical.evidence import EvidenceBundle
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.prompts import build_user_message, prompt_version, system_prompt

logger = logging.getLogger(__name__)

EMERGENCY_FALLBACK_TEXT = (
    "你描述的情况可能比较紧急。请立即拨打 120，或马上前往最近的急诊科。"
    "在等待或前往途中如症状加重，请让身边的人协助。"
)


class LLMClient(Protocol):
    """最小 LLM 接口。与仓库既有的客户端形状一致（`complete(system, user) -> str`）。"""

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class ReasoningResult:
    """一次主判的结果。`problems` 非空表示契约有问题，但**不一定是失败**：

    调用方可以选择"带伤展示"（例如缺了一条检查建议的依据，就把那条丢掉继续），
    而不是把整轮问诊丢掉。
    """

    assessment: ClinicalAssessment | None
    raw: str = ""
    problems: list[str] = field(default_factory=list)
    degraded: str = ""
    prompt_version: str = ""


def _extract_json(text: str) -> str:
    """从模型输出里抠出 JSON。容忍 ```json 围栏与前后废话——这是现实，不是纵容。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(line for line in lines if not line.strip().startswith("```"))
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]


class ClinicalReasoner:
    """把证据交给模型，换回一个**校验过的**契约对象。"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def reason(self, question: str, evidence: EvidenceBundle, intake: IntakeState) -> ReasoningResult:
        version = prompt_version()
        user = build_user_message(question, evidence, intake)
        try:
            raw = self._client.complete(system_prompt(), user)
        except Exception as e:  # noqa: BLE001 - LLM 是外部服务：任何失败都走降级，不崩对话
            logger.warning("主判调用失败，走降级：%s: %s", type(e).__name__, e)
            return ReasoningResult(
                assessment=self._fallback(evidence, intake),
                problems=[],
                degraded=f"模型调用失败（{type(e).__name__}: {e}），已按证据给出有限结论",
                prompt_version=version,
            )

        try:
            payload = json.loads(_extract_json(raw))
            assessment = ClinicalAssessment.model_validate(payload)
        except Exception as e:  # noqa: BLE001 - 解析失败也要降级，不能把原始文本塞给患者
            logger.warning("主判输出无法解析，走降级：%s: %s", type(e).__name__, e)
            return ReasoningResult(
                assessment=self._fallback(evidence, intake),
                raw=raw,
                problems=[f"模型输出不符合契约（{type(e).__name__}），已降级"],
                degraded="模型输出不可解析，已按证据给出有限结论",
                prompt_version=version,
            )

        problems = validate_against(assessment, evidence.ids())
        if evidence.red_line_positive and assessment.urgency != "emergency":
            # 按决定 2b，模型有权判定；但**与证据冲突必须留痕**，不许静默。
            problems.append("红线阳性但模型未给 emergency 档（按既定设计由模型决定，此处仅记录冲突）")
        return ReasoningResult(assessment=assessment, raw=raw, problems=problems, prompt_version=version)

    def _fallback(self, evidence: EvidenceBundle, intake: IntakeState) -> ClinicalAssessment:
        """降级产物：只用证据能支撑的内容，**不引入任何模型知识**。"""
        if evidence.red_line_positive:
            assert evidence.red_line is not None  # 由 red_line_positive 保证
            return ClinicalAssessment(
                urgency="emergency",
                departments=list(evidence.triage),
                patient_summary=f"「{evidence.red_line.word}」需要立刻处理。{EMERGENCY_FALLBACK_TEXT}",
            )
        departments = list(evidence.triage) or [
            DepartmentAdvice(name="内科", confidence=0.0, reason="模型不可用，先按内科门诊分诊")
        ]
        if evidence.items:
            summary = (
                "这次没能生成完整建议，下面是检索到的相关资料，建议带上去医院问医生。"
                "如果你希望我继续判断，请再描述一下症状。"
            )
        else:
            summary = (
                "这次没能生成完整建议，也没有检索到相关资料。"
                "请补充描述症状（哪里不舒服、多久了），或直接到医院分诊台咨询。"
            )
        return ClinicalAssessment(
            urgency="routine",
            departments=departments,
            patient_summary=summary,
            followup_questions=[s.question for s in intake.missing()[:2]],
        )
