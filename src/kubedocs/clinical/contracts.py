"""产品输出契约：模型输出、前端渲染、评测断言**三方共用的唯一形状**。

模型返回的是自由文本，而产品要交付四件**结构化**东西（科室建议 / 可能病症排序 /
检查建议 / 用药建议）。没有统一形状，前端就无法断言"到底给没给检查建议"，
评测也写不出来。

两条硬约束写进类型里，而不是靠提示词让模型自觉：

1. **不给剂量**：`MedicationAdvice` **没有**剂量字段。要加剂量就得先改这个类型，
   而那是一次需要显式决策的改动。
2. **建议必须有依据**：检查与用药建议的 `evidence_ids` 不允许为空——只用检索到的
   资料，检索不到就不给。校验在 `validate_against()` 里做：
   模型可能不守规矩，代码一定会守。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Urgency = Literal["emergency", "urgent", "routine"]
"""紧急度三档。`emergency` 必须触发紧急处理指引（见提示词的硬性条款）。"""

Likelihood = Literal["high", "medium", "low"]
"""可能病症的**粗**可能性档位。

刻意不用小数：给患者一个"0.73 的可能性"是伪精度，没有任何数据支持这个数。
三档是模型能稳定给出、也是患者能用的粒度。
"""

Severity = Literal["critical", "severe"]
"""急重症的严重程度：`critical` 有即刻生命危险；`severe` 需尽快急诊处理。"""


class _Strict(BaseModel):
    """契约模型统一禁止多余字段：模型多吐一个字段就应当被发现，而不是被静默忽略。"""

    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    """可点击来源。引用必须能点开——这是"回答有依据"唯一可验证的形式。"""

    index: int
    title: str
    source_url: str
    chunk_id: str = ""


class DepartmentAdvice(_Strict):
    """科室建议。`confidence` 是 0~1，允许模型使用，但**只用于排序**，不作为结论强度。"""

    name: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class DifferentialItem(_Strict):
    """可能病症排序里的一项。**不是诊断**，措辞与责任边界见提示词。"""

    condition: str = Field(min_length=1)
    likelihood: Likelihood
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ExamAdvice(_Strict):
    """检查建议。`evidence_ids` 必须非空（依据约束由 `validate_against` 强制）。"""

    name: str = Field(min_length=1)
    why: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class MedicationAdvice(_Strict):
    """用药建议。**只有类别与理由，没有剂量**（见模块文档第 1 条）。"""

    name: str = Field(min_length=1)
    drug_class: str = ""
    why: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class RedFlag(_Strict):
    """**急重症判断**：一个具体病症 + 它的严重程度 + 判断依据（`clinical.redline` 的替代）。

    判断由大模型按**病症 + 严重程度**给出，并**要求它把依据写出来**：
    没有 `condition` 和 `why`，这条判断就无法审计、无法复现，也无法被质疑。
    关键词表仍留在代码里，但**不参与任何判断**——它的命中只作为线索进参考标准面板
    （见 `clinical/reference.py`），判断者始终是模型。
    """

    condition: str = Field(min_length=1)
    """可能的急重症（如"急性冠脉综合征""脑卒中"）。"""

    severity: Severity
    """`critical` = 有生命危险需立即抢救；`severe` = 需尽快急诊处理但暂无即刻生命危险。"""

    why: str = Field(min_length=1)
    """为什么判为急重症：病症特征 + 严重程度（如"胸痛伴大汗、放射至左肩"）。**必填**。"""

    evidence_ids: list[str] = Field(default_factory=list)
    """可选：来自知识卡片的依据。**允许为空**——急重症常可由患者原话直接识别，
    强求卡片依据会逼模型去凑 id，反而害了"不许编依据"这条约束。"""


class ClinicalAssessment(_Strict):
    """一次问诊的**最终产物**。模型必须按这个形状回，解析失败就走降级路径（见 reasoning）。"""

    urgency: Urgency
    red_flags: list[RedFlag] = Field(default_factory=list)
    """**模型自己判出的急重症**。非空 ⇒ `urgency` 必须是 `emergency`（由契约检查兜住）。"""

    departments: list[DepartmentAdvice] = Field(default_factory=list)
    differentials: list[DifferentialItem] = Field(default_factory=list)
    exams: list[ExamAdvice] = Field(default_factory=list)
    medications: list[MedicationAdvice] = Field(default_factory=list)
    patient_summary: str = ""
    """给患者看的白话总结。空的会被 `validate_against` 记为问题。"""

    followup_questions: list[str] = Field(default_factory=list)
    """还缺的信息。非空表示这一轮**不该给结论**（由 intake 决定是否追问，见 pipeline）。"""

    citations: list[Citation] = Field(default_factory=list)


def validate_against(assessment: ClinicalAssessment, evidence_ids: set[str]) -> list[str]:
    """返回契约问题列表（空列表 = 通过）。**返回而不抛**：调用方要能"带伤降级"。

    检查六件事：
    1. 每条检查/用药建议都必须有非空 `evidence_ids`；
    2. 引用的证据 id 必须真的在证据包里（不许编 id）；
    3. `patient_summary` 不能为空（患者要看的就这一句）；
    4. **`red_flags` 非空时 `urgency` 必须是 `emergency`**——这是"红线"的落点：
       判断权在模型手上，但"判出急重症却给常规建议"这种自相矛盾必须被程序拦下；
    5. `emergency` 档必须留下可执行的紧急指引（`patient_summary` 里要能看到 120 / 急诊）；
    6. 要有科室建议。
    """
    problems: list[str] = []
    for label, items in (("检查建议", assessment.exams), ("用药建议", assessment.medications)):
        for item in items:
            if not item.evidence_ids:
                problems.append(f"{label}「{item.name}」没有给出依据（evidence_ids 为空）")
                continue
            unknown = [e for e in item.evidence_ids if e not in evidence_ids]
            if unknown:
                problems.append(f"{label}「{item.name}」引用了不存在的证据 id：{unknown}")

    for flag in assessment.red_flags:
        unknown = [e for e in flag.evidence_ids if e not in evidence_ids]
        if unknown:
            problems.append(f"急重症判断「{flag.condition}」引用了不存在的证据 id：{unknown}")

    if not assessment.patient_summary.strip():
        problems.append("patient_summary 为空：患者没有任何可读的结论")

    if assessment.red_flags and assessment.urgency != "emergency":
        names = "、".join(f.condition for f in assessment.red_flags)
        problems.append(
            f"判出了急重症（{names}）却把 urgency 定为 {assessment.urgency}：两者矛盾，必须定为 emergency"
        )

    if assessment.urgency == "emergency":
        text = assessment.patient_summary
        if "120" not in text and "急诊" not in text:
            problems.append("urgency=emergency 但 patient_summary 里没有可执行的紧急指引（120 / 急诊）")

    if not assessment.departments:
        problems.append("没有任何科室建议")

    return problems


@dataclass
class Enforcement:
    """契约执行的产物：清洗后的结论 + 问题清单 + 被丢掉的东西。"""

    assessment: ClinicalAssessment
    problems: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    """被丢掉的条目名（检查/用药）。调用方要把它记进日志——**丢东西不能是静默的**。"""


def enforce(assessment: ClinicalAssessment, evidence_ids: set[str]) -> Enforcement:
    """**真的执行契约**：把没有依据的检查/用药建议**丢掉**，而不是只记一笔账。

    `validate_against()` 只返回问题字符串；本函数是那个真正执行丢弃的调用点。
    少了它，"检索不到就不给"会从硬约束退化成事后记账——患者照样会收到
    一条没有任何依据的用药建议，而问题只留在响应 JSON 的 `problems` 里。

    - **丢**：`exams` / `medications` 里没有 `evidence_ids`、或引用了不存在 id 的条目。
      它们正是硬约束要拦的东西，系统没有资格给出。
    - **保留**：`differentials`（是"可能方向"，不承诺具体处置）、`departments`、
      `red_flags`、`patient_summary`。急重症判断与科室建议不因这条被削弱。
    - `red_flags` 的 id 引用错了**只记不丢**：急重症判断宁可留着让人看见，
      也不该因为一个编号写错就消失（漏报急诊比多报危险得多）。
    """
    problems: list[str] = []
    dropped: list[str] = []
    exams: list[ExamAdvice] = []
    medications: list[MedicationAdvice] = []

    for item in assessment.exams:
        if not item.evidence_ids:
            dropped.append(f"检查「{item.name}」（没有依据）")
            continue
        unknown = [e for e in item.evidence_ids if e not in evidence_ids]
        if unknown:
            dropped.append(f"检查「{item.name}」（引用了不存在的依据 {unknown}）")
            continue
        exams.append(item)

    for med in assessment.medications:
        if not med.evidence_ids:
            dropped.append(f"用药「{med.name}」（没有依据）")
            continue
        unknown = [e for e in med.evidence_ids if e not in evidence_ids]
        if unknown:
            dropped.append(f"用药「{med.name}」（引用了不存在的依据 {unknown}）")
            continue
        medications.append(med)

    if dropped:
        problems.append(
            "已按 R3 丢弃无依据的检查/用药建议："
            + "、".join(dropped)
            + "（检索不到就不给，不用通用医学知识补足）"
        )

    # 其余检查照旧（急重症/自相矛盾/空结论/科室缺失）——它们**不该**导致丢内容，
    # 而是让调用方与运维看到"这一轮有问题"。
    #
    # 只按**前缀**排除检查/用药那两类，不能用子串匹配：红线的违规消息里也含
    # "引用了不存在的证据 id"，用子串会把「急重症判断引用错 id」一起滤掉。
    # 红线的问题必须留下——急重症引错编号也要让人看见。
    rest = [
        p for p in validate_against(assessment, evidence_ids) if not p.startswith(("检查建议", "用药建议"))
    ]
    problems.extend(rest)

    cleaned = assessment.model_copy(update={"exams": exams, "medications": medications})
    return Enforcement(assessment=cleaned, problems=problems, dropped=dropped)
