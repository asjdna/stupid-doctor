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

DangerLikelihood = Literal["ruled_out", "unlikely", "possible", "likely", "very_likely"]
"""**危险信号的不确定性档位**（这是"不确定性量化"的载体）。

刻意用有序档位而不是小数：患者端不需要伪精度，而程序端需要一个可比较的序，
来判断"这条信号排除了没有、要不要为它追问"。

- `ruled_out`：已有明确依据排除（`ruled_out_by` 必须写清依据）；
- `unlikely`：可能性低，但**没有**说清排除依据的，程序按"未排除"处理；
- `possible` / `likely` / `very_likely`：**未排除** ⇒ 触发"主动问诊"层（见 `safety.py`）。
"""

CONFIDENCE_FLOOR = 0.6
"""自报置信度低于它就算"模型自己也没底"。

这个数只用来触发一条**行为约束**（低置信 ⇒ 必须说清还缺什么信息），不参与打分、
不当作概率、也不进闸门。它的可信度由校准评测（`evaluation/calibration.py`）单独衡量——
自报置信与实际判对率对不上的话，校准报表会把它暴露出来。
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


class RedFlag(_Strict):
    """**危险信号**：一个具体病症方向 + 严重程度 + 判断依据 + 不确定性档位。

    "红线/危险信号"的识别由大模型负责，而且是**多角度**的：同一个病可以从症状性质看出来，
    也可以从症状组合、时间过程（突发还是渐进）、严重度（有没有功能受限）、高危人群
    （婴幼儿 / 孕产妇 / 老人 / 免疫抑制）、情境（外伤 / 中毒 / 用药与过敏）看出来。
    `angle` 就是用来交代"这条是从哪个角度看出来的"——**同一个病只从一个角度认定，
    是漏报最常见的成因**。

    关键词表（`domains.clinic.redline`）不在这里下判断：它的命中只作为线索进参考标准面板，
    与其他线索一起交给模型。
    """

    condition: str = Field(min_length=1)
    """可能的急重症（如"急性冠脉综合征""脑卒中"）。"""

    severity: Severity
    """`critical` = 有生命危险需立即抢救；`severe` = 需尽快急诊处理但暂无即刻生命危险。"""

    why: str = Field(min_length=1)
    """为什么判为急重症：病症特征 + 严重程度（如"胸痛伴大汗、放射至左肩"）。**必填**。"""

    angle: str = Field(default="", min_length=0)
    """从哪个角度得出的（症状性质 / 组合共现 / 时间过程 / 严重度 / 高危人群 / 情境诱因…）。

    允许为空只为兼容旧输出；**多角度扫描是提示词的硬性要求**，缺角度会被记一条留档问题。
    """

    likelihood: DangerLikelihood = "possible"
    """这条信号被排除到什么程度。**默认 `possible`（未排除）**：模型漏填时按"还没排除"处理，
    不让一个缺省值把安全问题悄悄抹掉。"""

    discriminator: str = ""
    """**能把它分成"要急救 / 不用急救"的那一个关键问题**（主动问诊层用它）。

    例：胸痛 → "有没有伴随大汗、呼吸困难，或疼痛向左肩/后背放射？"。
    服务端在需要追问时优先用它；为空则由 `safety.py` 按病症生成一句通用追问。
    """

    ruled_out_by: str = ""
    """判为 `ruled_out` 时必须写清依据（哪句话、哪个阴性表现排除的）。

    没有依据的"已排除"不算排除：程序会把它当未排除，继续要求追问。
    """

    evidence_ids: list[str] = Field(default_factory=list)
    """可选：来自知识卡片的依据。**允许为空**——急重症常可由患者原话直接识别，
    强求卡片依据会逼模型去凑 id，反而害了"不许编依据"这条约束。"""


class ClinicalAssessment(_Strict):
    """一次问诊的**最终产物**。模型必须按这个形状回，解析失败就走降级路径（见 reasoning）。"""

    urgency: Urgency
    red_flags: list[RedFlag] = Field(default_factory=list)
    """**模型自己判出的危险信号**（多角度）。非空 ⇒ `urgency` 必须是 `emergency`（契约层兜住）。"""

    departments: list[DepartmentAdvice] = Field(default_factory=list)
    differentials: list[DifferentialItem] = Field(default_factory=list)
    exams: list[ExamAdvice] = Field(default_factory=list)
    medications: list[MedicationAdvice] = Field(default_factory=list)
    patient_summary: str = ""
    """给患者看的白话总结。空的会被 `validate_against` 记为问题。"""

    followup_questions: list[str] = Field(default_factory=list)
    """还缺的信息（**要问患者的话**）。非空表示这一轮**不该给结论**（见 pipeline）。"""

    information_gaps: list[str] = Field(default_factory=list)
    """**还缺什么信息**（自然语言，与 `followup_questions` 分工：这里说"缺什么"，
    那里说"怎么问"）。

    与置信度绑成一条链：自报置信度低于 `CONFIDENCE_FLOOR` 时**必须**写清缺什么，
    否则程序判定为"低置信却不说缺什么"——那种回答既没把握又没交代，患者无从判断。
    """

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """模型对**本轮结论**的自报置信度（0~1）。

    它不是概率，也不进任何阈值：只用两处——① 低于 `CONFIDENCE_FLOOR` 时触发"必须交代
    信息缺口"这条行为约束；② 进校准报表（自报置信 vs 实际判对率），看它到底可不可信。

    **缺省是 `None` 而不是某个中间值**：模型没给就说没给（`validate_against` 会记一条），
    而不是替它填一个看起来像"中等把握"的数——那会让"没量化"看起来像"量化过了"。
    """

    citations: list[Citation] = Field(default_factory=list)


def unresolved_signals(assessment: ClinicalAssessment) -> list[RedFlag]:
    """**还没被排除**的危险信号。

    `ruled_out` 与"`unlikely` 且写清排除依据"都算排除；其余（含 `unlikely` 却没写依据）
    一律按未排除处理——**缺省值不能替模型把安全问题抹掉**。
    """
    return [
        f
        for f in assessment.red_flags
        if not (f.likelihood == "ruled_out" or (f.likelihood == "unlikely" and f.ruled_out_by.strip()))
    ]


def requires_emergency(signal: RedFlag) -> bool:
    """这条信号**必须**让整轮判成 `emergency`。

    两种情形，都是"往重的一侧"：

    - `likelihood` 是 `likely` / `very_likely`：模型自己就认为很可能，那动作只能是立刻就医；
    - `severity` 是 `critical`：有即刻生命危险的方向，**哪怕只是 `possible`**——
      拿不准的即刻危险必须先按急救处理，这不是可以让患者"再观察一下"的情形。

    反过来，`severity=severe` 且 `likelihood=possible` 的信号**允许**不判急症——
    那正是"主动问诊"层要接住的情形：先进去问清能区分的那一个问题，而不是硬给结论。
    """
    return signal.likelihood in ("likely", "very_likely") or signal.severity == "critical"


def validate_against(assessment: ClinicalAssessment, evidence_ids: set[str]) -> list[str]:
    """返回契约问题列表（空列表 = 通过）。**返回而不抛**：调用方要能"带伤降级"。

    检查这些事（前六条是契约，后三条是三层防护里能被机械核对的那部分）：

    1. 每条检查/用药建议都必须有非空 `evidence_ids`；
    2. 引用的证据 id 必须真的在证据包里（不许编 id）；
    3. `patient_summary` 不能为空（患者要看的就这一句）；
    4. 危险信号必须交代判断角度（`angle`）——多角度扫描是 L1 的硬性要求；
    5. **该判急症的信号必须判成急症**：`requires_emergency()` 为真的信号
       （`likely`/`very_likely`，或 `critical` 严重度）配非 `emergency` 的 `urgency`
       是自相矛盾，必须拦下；
    6. `emergency` 档必须留下可执行的紧急指引（`patient_summary` 里要能看到 120 / 急诊）；
    7. 要有科室建议；
    8. **未排除的危险信号必须有下文**（L2 主动问诊）：要么这一轮就是在追问
       （`followup_questions` 非空），要么判成 `emergency` 当场给动作。既不追问又不给动作，
       等于把系统自己都还在担心的东西放着不管；
    9. **不确定性要交代**（不确定性量化）：没给 `confidence` 记一条；低于
       `CONFIDENCE_FLOOR` 时必须写清 `information_gaps`——没把握又不说缺什么，
       患者无从判断这句话能不能信。
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
            problems.append(f"危险信号「{flag.condition}」引用了不存在的证据 id：{unknown}")
        if not flag.angle.strip():
            problems.append(
                f"危险信号「{flag.condition}」没有交代判断角度（angle 为空）：多角度扫描是硬性要求"
            )

    if not assessment.patient_summary.strip():
        problems.append("patient_summary 为空：患者没有任何可读的结论")

    must_be_emergency = [f for f in assessment.red_flags if requires_emergency(f)]
    if must_be_emergency and assessment.urgency != "emergency":
        names = "、".join(f.condition for f in must_be_emergency)
        problems.append(
            f"危险信号（{names}）的可能性/严重度要求立刻处理，却把 urgency 定为 "
            f"{assessment.urgency}：两者矛盾，必须定为 emergency"
        )

    if assessment.urgency == "emergency":
        text = assessment.patient_summary
        if "120" not in text and "急诊" not in text:
            problems.append("urgency=emergency 但 patient_summary 里没有可执行的紧急指引（120 / 急诊）")

    if not assessment.departments:
        problems.append("没有任何科室建议")

    pending = unresolved_signals(assessment)
    if pending and assessment.urgency != "emergency" and not assessment.followup_questions:
        names = "、".join(f.condition for f in pending)
        problems.append(
            f"危险信号（{names}）既没有被排除、也没有追问：这一轮必须问清能区分"
            "「要急救 / 不用急救」的那一个问题"
        )

    if assessment.confidence is None:
        problems.append("没有给出 confidence：不确定性没有量化，读者无法判断这句话能不能信")
    elif assessment.confidence < CONFIDENCE_FLOOR and not assessment.information_gaps:
        problems.append(
            f"自报置信度 {assessment.confidence:.2f} 低于 {CONFIDENCE_FLOOR}，"
            "却没写 information_gaps：低置信必须交代还缺什么信息"
        )

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
