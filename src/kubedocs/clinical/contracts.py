"""产品输出契约：模型输出、前端渲染、评测断言**三方共用的唯一形状**。

## 为什么必须先有这一层

模型返回的是自由文本，而产品要交付四件**结构化**东西（科室建议 / 可能病症排序 /
检查建议 / 用药建议）。若让前端去解析自然语言，"到底给没给检查建议"就无法断言，
评测也写不出来——于是"功能做没做"只能靠人肉看，这正是本项目要避免的。

## 两条硬约束写进类型里，而不是写在提示词里靠模型自觉

1. **不给剂量**：`MedicationAdvice` **没有**剂量字段。要加剂量就得先改这个类型，
   而那是一次需要显式决策的改动（边界见 `docs/design/0001-医疗辅助诊断产品方案.md`）。
2. **建议必须有依据**：检查与用药建议的 `evidence_ids` 不允许为空——按"只用 RAG 里
   有的资料，检索不到就不给"的约定。校验在 `validate_against()` 里做，
   **不在提示词里拜托模型守规矩**：模型可能不守，代码一定会守。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Urgency = Literal["emergency", "urgent", "routine"]
"""紧急度三档。`emergency` 必须触发紧急处理指引（见提示词的硬性条款）。"""

Likelihood = Literal["high", "medium", "low"]
"""可能病症的**粗**可能性档位。

刻意不用小数：给患者一个"0.73 的可能性"是伪精度——没有任何数据支持这个数。
三档是模型能稳定给出的粒度，也是患者能用的信息。
"""


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


class ClinicalAssessment(_Strict):
    """一次问诊的**最终产物**。模型必须按这个形状回，解析失败就走降级路径（见 reasoning）。"""

    urgency: Urgency
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

    检查四件事：
    1. 每条检查/用药建议都必须有非空 `evidence_ids`；
    2. 引用的证据 id 必须真的在证据包里（不许编 id）；
    3. `patient_summary` 不能为空（患者要看的就这一句）；
    4. `emergency` 档必须留下可执行的紧急指引（`patient_summary` 里要能看到急救动作）。
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

    if not assessment.patient_summary.strip():
        problems.append("patient_summary 为空：患者没有任何可读的结论")

    if assessment.urgency == "emergency":
        text = assessment.patient_summary
        if "120" not in text and "急诊" not in text:
            problems.append("urgency=emergency 但 patient_summary 里没有可执行的紧急指引（120 / 急诊）")

    if not assessment.departments:
        problems.append("没有任何科室建议")

    return problems
