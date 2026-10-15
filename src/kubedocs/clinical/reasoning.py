"""大模型主判：证据 → 契约化输出。

三条职责，界限要清楚：

1. **装配**：把系统提示词 + 证据 + 已采集信息 + 患者原话交给模型；
2. **校验**：把模型返回的 JSON 解析成 `ClinicalAssessment`，并检查依据是否存在、
   检查/用药有没有出处；
3. **降级**：模型不可用或输出不可解析时**不许凭空生成**，只退回到
   "证据能支撑的东西"（见 `_fallback`）。

急重症由模型按**病症 + 严重程度**判断并输出到 `red_flags`（见 `contracts.RedFlag`），
所以关键词表（`domains.clinic.redline.RiskGuard`）**退出了判断链**：
它不进提示词、不影响 `urgency`、不触发急诊横幅。它只剩一个用途——模型不可用时
给一句保守的对话提示。注意那只是**提示**：`degraded` 非空，前端据此不弹急诊横幅。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

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
    """最小 LLM 接口：**返回文本**。

    仓库既有客户端（`generate.LiteLLMClient`）的 `complete()` 返回的是 `LLMResult`
    **对象**而不是 `str`，所以要经 `LiteLLMAdapter` 才能接进来。
    """

    def complete(self, system: str, user: str) -> str: ...


class BudgetedClient:
    """给 LLM 客户端套上**同一本日预算账**（与 CLI `ask` / `/ask` / `/chat` 共用）。

    必须有这一层：`ClinicalReasoner` 直接调模型，少了它这就是一条**能无限花钱的
    产品路径**——新增产品路径却绕过预算闸，等于把已有的花费上限作废。

    用 `reserve()` 而不是 `check()`：预留把"这次在途调用"按保守金额算进本次判断，
    否则 N 个并发会各自通过 `check()` 然后各自 `record()`，上限被冲过 N 倍。

    计费口径与 `Answerer` 完全一致：`recorder` 拿到的是 `LLMResult`，
    所以能用它的 `cost_priced` 区分"真免费"与"算不出价"。
    """

    def __init__(self, client: object, budget: Any) -> None:
        self._client = client
        self._budget = budget

    def complete(self, system: str, user: str) -> object:
        with self._budget.reserve():
            result = self._client.complete(system, user)  # type: ignore[attr-defined]
        self._budget.record(
            float(getattr(result, "cost_usd", 0.0) or 0.0),
            priced=bool(getattr(result, "cost_priced", False)),
        )
        return result

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> object:
        """工具循环里的每一次调用**都要过同一本日预算账**。

        工具循环一次问诊会调模型 1~N 次（N 由模型决定）。如果只有最后那次计费，
        前面每轮检索取证都是**免费**的——那等于给了一条"模型可以自己决定花多少钱"
        的路径，与预算闸的目的相反。
        """
        with self._budget.reserve():
            result = self._client.complete_with_tools(messages, tools)  # type: ignore[attr-defined]
        self._budget.record(
            float(getattr(result, "cost_usd", 0.0) or 0.0),
            priced=bool(getattr(result, "cost_priced", False)),
        )
        return result


class LiteLLMAdapter:
    """把仓库既有的 LLM 客户端适配成本层要求的 `complete(...) -> str`。

    真实客户端返回的是 `LLMResult` 对象而不是 `str`。若在 reasoner 里对返回值做
    `getattr(result, "text", ...)` 兜底，这个差异就会继续藏在业务代码里，
    而"接错了客户端"永远表现为"模型输出不合契约"——整条主判会**安静地走降级**：
    表面上"有回答"，实际上一次模型判断都没发生。所以差异收在这个**能被单测钉住**
    的适配器里，且类型不对时**大声抛**。

    预算闸不在这里做：它需要 `LLMResult` 的成本字段，属于"客户端包装"的职责，
    见 `BudgetedClient`。两个适配器各管一件事，可以单独替换。
    """

    def __init__(self, client: object) -> None:
        self._client = client

    def complete(self, system: str, user: str) -> str:
        result = self._client.complete(system, user)  # type: ignore[attr-defined]
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise TypeError(
                f"{type(result).__name__} 没有可用的 .text 文本，无法适配成 LLMClient。"
                "要么传入本层期望的客户端，要么为该客户端写一个适配器。"
            )
        return text


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
        from kubedocs.clinical.contracts import enforce

        enforced = enforce(assessment, evidence.ids())
        assessment, problems = enforced.assessment, enforced.problems
        if enforced.dropped:
            logger.warning("按 R3 丢弃无依据建议：%s", "、".join(enforced.dropped))
        if evidence.red_line_positive and not assessment.red_flags:
            # 关键词表命中了，但模型既没判出急重症、也没把它写进 red_flags。
            # **这只是"记一笔"，不是推翻模型**：判断权在模型手上，
            # 程序只负责把这种不一致留痕，好让人能回头查"是不是漏了一个真急症"。
            problems.append(
                f"关键词表命中「{evidence.red_line.word if evidence.red_line else '?'}」，"
                "但模型未判出任何急重症（按既定设计由模型决定，此处仅记录不一致）"
            )
        return ReasoningResult(assessment=assessment, raw=raw, problems=problems, prompt_version=version)

    def _fallback(self, evidence: EvidenceBundle, intake: IntakeState) -> ClinicalAssessment:
        """降级产物：只用证据能支撑的内容，**不引入任何模型知识**。

        主判不可用时**没有"模型判断"这回事**，所以这里不退化成沉默：
        若关键词表命中，就给一句保守的"请立即就医"提示（并把命中的词说出来，
        让用户明白系统在担心什么）。**这不冒充模型判断**——`degraded` 非空，
        前端据此不弹急诊横幅（见 `api/consult.py`），提示只出现在对话正文里。
        **没有命中时也不假装没事**：统一提示"没能给出建议，请线下就医"，
        绝不说"你应该没事"。
        """
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
