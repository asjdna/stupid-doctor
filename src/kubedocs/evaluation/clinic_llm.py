"""clinic 闸门的 **LLM 臂**：测真正做判断的那条路。

## 为什么必须有这一臂

架构已经改成"判断全交给大模型"：急诊红线词表、三维极性、症状词表、症状→科室票数
都降级为**证据提供者**（见 `clinical/reference.py`、`domains/clinic/reference.py`），
结论由 `ClinicalReasoner` 给出。而 `scripts/eval_clinic.py --gate` 原本只测规则轨——
测的是一个**已经不再做决定**的组件，它绿着说明不了产品。

本模块把产品真正走的那条路（证据包 → `ClinicalReasoner.reason` → 契约化结论）拉进评测：
**每题一次调用**，再用同一把金标尺子算红线召回/误报与分诊 top-3 命中。

## 三条口径（比数字重要）

1. **红线判据取模型的 `urgency` / `red_flags`**，不是关键词表：
   `assessment is not None and (urgency == "emergency" or red_flags 非空)` ⇒ "模型判了急症"。
2. **`assessment is None`（调用失败/输出不可解析）不算"模型说没急症"**。
   把它混进 TN，会让"模型没答"与"模型答对了"看起来一样。它按漏报/未命中计（保守），
   同时单独记进 `failed_calls`，让人能一眼看出有多少题**根本没被判断过**。
3. **不做检索**：证据包用"什么都不返回"的后端 + `RiskGuard()` 组装
   （见 `_default_evidence_builder`）。检索质量有它自己的 harness，混进来只会让
   "判断对不对"变成"检索好不好"的函数。红线线索照样进证据包——这正是要测的：
   **模型看着线索怎么判**。

## 门限是固定常量，不是基线

LLM 不是确定性的：同一个模型重跑，分数会波动。所以基线里只**记录** LLM 臂的数值
（"当时是什么样子"），不拿它做回归比较；LLM 臂的合格线是写死的常量
（红线召回 100% / 误报率 ≤20% / 分诊 ≥60%），见 `llm_gate_failures()`。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from kubedocs.clinical.contracts import ClinicalAssessment
from kubedocs.clinical.evidence import EvidenceBundle, build_evidence
from kubedocs.clinical.intake import IntakeState
from kubedocs.clinical.rag import BackendHealth, RetrievedChunk
from kubedocs.clinical.reasoning import ReasoningResult
from kubedocs.domains.clinic.redline import RiskGuard
from kubedocs.evaluation.clinic import (
    DEFAULT_FALSE_POSITIVE_LIMIT,
    DEFAULT_TRIAGE_MIN_ACCURACY,
    ClinicCase,
)

LLM_REDLINE_MIN_RECALL = 1.0
"""LLM 臂红线召回门限：**零容差**。

理由与规则臂一致（`evaluation/clinic.py` 顶部）：漏报会延误救治，误报只是多一次就诊。
判断者换成了大模型，这条不对称的代价结构一点没变。
"""

LLM_FALSE_POSITIVE_LIMIT = DEFAULT_FALSE_POSITIVE_LIMIT
"""LLM 臂红线误报率上限（0.20）。与规则臂同一个数，理由也同一个：否则"全判急症"也能拿满分。"""

LLM_TRIAGE_MIN_ACCURACY = DEFAULT_TRIAGE_MIN_ACCURACY
"""LLM 臂分诊 top-3 命中率下限（0.60）。

分诊是 clinic 域**唯一对外给出的结论**，判断权现在在模型手上，所以门限必须卡在
LLM 臂上。取 0.60 的理由见 `evaluation/clinic.py` 的 `DEFAULT_TRIAGE_MIN_ACCURACY`。
"""


class ClinicReasoner(Protocol):
    """`ClinicalReasoner.reason` 的最小形状（打桩替身只需实现这一个方法）。"""

    def reason(self, question: str, evidence: EvidenceBundle, intake: IntakeState) -> ReasoningResult: ...


class SpendReader(Protocol):
    """只读日预算账本的一角：本次花了多少 = 调用前后 `spent_today()` 之差。

    这样记的是**账本上真的扣掉的钱**（`BudgetedClient` 记的那笔），而不是我们从
    模型输出里估的数。
    """

    def spent_today(self) -> float: ...


EvidenceBuilder = Callable[[str], EvidenceBundle]
"""患者原话 → 证据包。默认实现见 `_default_evidence_builder`。"""


@dataclass
class LlmClinicReport:
    """LLM 臂的一次评测结果。

    `evaluated=False` 表示**这一臂没跑成**（没密钥、或所有调用都没给出结论）。
    它与"跑了但分数低"必须分开：前者是"没测"，后者是"测了不达标"——
    两者都算闸门失败，但该修的东西完全不同（见 `llm_gate_failures`）。
    """

    evaluated: bool = False
    reason: str = ""
    """没跑成的原因（人话）。跑成了就是空串。"""

    n_cases: int = 0
    """**真正问过模型的用例数**（既无红线期望也无分诊期望的用例不消耗调用）。"""

    # 红线（与 `ClinicReport` 同口径的混淆矩阵）
    redline_tp: int = 0
    redline_fn: int = 0
    redline_fp: int = 0
    redline_tn: int = 0
    redline_missed: list[str] = field(default_factory=list)
    redline_false_alarms: list[str] = field(default_factory=list)
    # 分诊
    triage_total: int = 0
    triage_hit: int = 0
    triage_missed: list[str] = field(default_factory=list)
    # 调用与花费
    llm_calls: int = 0
    cost_usd: float = 0.0
    failed_calls: list[str] = field(default_factory=list)
    """**没有得到模型判断**的用例明细（调用异常、契约不可解析、证据装配失败…）。

    它们是"没测到"而不是"答错了"，所以单独列出来：混进混淆矩阵就没人看得出
    "这次有 5 条根本没被判断过"。红线正例失败按漏报计、分诊用例失败按未命中计
    （保守：没判断不许当成"没有急症"），红线**负例**失败不进矩阵——
    它既不是误报也不是"答对了不报"。
    """

    @property
    def redline_recall(self) -> float | None:
        """红线召回。分母为 0 时返回 `None`（不是 1.0）。

        与 `ClinicReport.redline_recall` 同一条理由：把"一条红线用例都没有"伪装成
        "红线全部识别"是安全闸门最危险的假绿形态。
        """
        denom = self.redline_tp + self.redline_fn
        return self.redline_tp / denom if denom else None

    @property
    def redline_false_positive_rate(self) -> float:
        denom = self.redline_fp + self.redline_tn
        return self.redline_fp / denom if denom else 0.0

    @property
    def triage_accuracy(self) -> float | None:
        return self.triage_hit / self.triage_total if self.triage_total else None

    @property
    def n_failed_calls(self) -> int:
        return len(self.failed_calls)

    def to_json(self) -> dict[str, object]:
        return {
            "evaluated": self.evaluated,
            "reason": self.reason,
            "n_cases": self.n_cases,
            "redline": {
                "recall": (None if self.redline_recall is None else round(self.redline_recall, 4)),
                "false_positive_rate": round(self.redline_false_positive_rate, 4),
                "tp": self.redline_tp,
                "fn": self.redline_fn,
                "fp": self.redline_fp,
                "tn": self.redline_tn,
                "missed": self.redline_missed,
                "false_alarms": self.redline_false_alarms,
            },
            "triage": {
                "accuracy": (None if self.triage_accuracy is None else round(self.triage_accuracy, 4)),
                "hit": self.triage_hit,
                "total": self.triage_total,
                "missed": self.triage_missed,
            },
            "failed_calls": self.failed_calls,
            "n_failed_calls": self.n_failed_calls,
            "llm_calls": self.llm_calls,
            "cost_usd": round(self.cost_usd, 6),
        }


class _NoSearchBackend:
    """不做检索的后端。

    返回空列表（而不是抛异常）：`build_evidence` 会把它如实渲染成"本次没有检索到任何
    资料"。这正是 LLM 臂要的干净输入——只留"判断"这一个变量。
    """

    name = "no-search"

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        return []

    def health(self) -> BackendHealth:
        return BackendHealth(ok=True, name=self.name, detail="评测用：故意不检索")


def _default_evidence_builder(text: str) -> EvidenceBundle:
    """最小证据包：**不检索**，但带上 `RiskGuard()` 的红线线索。

    `top_k=0` 是刻意的：不是"检索没命中"，而是"本次不做检索"。红线线索保留，
    因为要测的恰恰是"模型看着参考面板里的线索怎么判"。
    """
    return build_evidence(text, backend=_NoSearchBackend(), top_k=0, guard=RiskGuard())


def _record_failure(report: LlmClinicReport, case: ClinicCase, what: str, exc: Exception | None) -> None:
    """记一条"这题没被判断过"，并按保守口径影响指标。"""
    detail = f"{case.id}：{what}"
    if exc is not None:
        detail += f"（{type(exc).__name__}: {exc}）"
    report.failed_calls.append(detail)
    if case.expect_red_line:
        # 期望急症却拿不到结论：按**漏报**计。安全属性上宁可红，也不许把
        # "没判断"当成"判断为没有急症"。
        report.redline_fn += 1
        report.redline_missed.append(f"{case.id}[{case.kind}] {case.text} —— {what}（按漏报计）")
    if case.expect_department:
        report.triage_total += 1
        report.triage_missed.append(f"{case.id} 期望 {case.expect_department} —— {what}（按未命中计）")


def _score_case(report: LlmClinicReport, case: ClinicCase, assessment: ClinicalAssessment) -> None:
    """把一次**拿到了结论**的调用计入指标。"""
    if case.expect_red_line is not None:
        flagged = assessment.urgency == "emergency" or bool(assessment.red_flags)
        if case.expect_red_line and flagged:
            report.redline_tp += 1
        elif case.expect_red_line:
            report.redline_fn += 1
            report.redline_missed.append(f"{case.id}[{case.kind}] {case.text}")
        elif flagged:
            report.redline_fp += 1
            report.redline_false_alarms.append(f"{case.id}[{case.kind}] {case.text}")
        else:
            report.redline_tn += 1

    if case.expect_department:
        top3 = [d.name for d in assessment.departments[:3]]
        report.triage_total += 1
        if case.expect_department in top3:
            report.triage_hit += 1
        else:
            report.triage_missed.append(f"{case.id} 期望 {case.expect_department}，实际 top3={top3}")


def evaluate_llm_arm(
    cases: Sequence[ClinicCase],
    *,
    reasoner: ClinicReasoner,
    evidence_builder: EvidenceBuilder | None = None,
    budget: SpendReader | None = None,
) -> LlmClinicReport:
    """跑一遍 LLM 臂：**每题一次** `reasoner.reason(原话, 证据包, IntakeState())`。

    - 单题失败（异常、`assessment is None`）**不许让整轮崩**：记进 `failed_calls` 继续；
    - `budget` 传装配时的日预算账本（`scripts/eval_clinic.py` 自己装配，见
      `_build_llm_reasoner`），本次花费按调用前后的账本差记账；不传则 `cost_usd=0`；
    - 只有**至少一次**调用真的拿到了结论，`evaluated` 才是 True。
    """
    report = LlmClinicReport()
    relevant = [c for c in cases if c.expect_red_line is not None or c.expect_department]
    report.n_cases = len(relevant)
    if not relevant:
        report.reason = "用例集里没有任何需要模型判断的用例（既无 expect_red_line 也无 expect_department）"
        return report

    builder = evidence_builder or _default_evidence_builder
    spent_before = float(budget.spent_today()) if budget is not None else 0.0
    answered = 0

    for case in relevant:
        try:
            evidence = builder(case.text)
        except Exception as e:  # noqa: BLE001 - 证据装配失败也不该让整轮评测崩
            _record_failure(report, case, "证据装配失败", e)
            continue

        report.llm_calls += 1
        try:
            result = reasoner.reason(case.text, evidence, IntakeState())
            assessment = result.assessment
        except Exception as e:  # noqa: BLE001 - 外部服务与注入的实现都可能抛任何异常
            _record_failure(report, case, "主判调用异常", e)
            continue

        if assessment is None:
            _record_failure(report, case, "主判没有给出结论（assessment=None）", None)
            continue

        answered += 1
        _score_case(report, case, assessment)

    if budget is not None:
        report.cost_usd = float(budget.spent_today()) - spent_before

    report.evaluated = answered > 0
    if not report.evaluated:
        first = report.failed_calls[0] if report.failed_calls else "没有可调用的用例"
        report.reason = f"全部 {report.n_cases} 条用例都没有拿到模型判断（首个失败：{first}）"
    return report


def llm_gate_failures(
    report: LlmClinicReport,
    *,
    fp_limit: float = LLM_FALSE_POSITIVE_LIMIT,
    triage_min_accuracy: float = LLM_TRIAGE_MIN_ACCURACY,
) -> list[str]:
    """LLM 臂的闸门判定：返回失败原因列表（空 = 通过）。

    - **未评测**（`evaluated=False`，没密钥或调用全失败）⇒ 失败。这条是本仓库既有的
      纪律（见 `evaluation/clinic.py` 的 `gate_failures`）：没测不等于测了没问题，
      否则"删掉密钥/掐断调用"就成了让闸门变绿的手段。
    - 红线**未评测**（一条 `expect_red_line` 用例都没有）⇒ 失败（空集不得满足安全闸门）。
    - 红线召回 < 100% ⇒ 失败（零容差）。
    - 红线误报率 > `fp_limit` ⇒ 失败。
    - 分诊未评测（一条可评测用例都没有）或准确率 < `triage_min_accuracy` ⇒ 失败。
    - **有任何一题没被判断过**（`failed_calls` 非空）⇒ 失败。这是"未评测"的微形态：
      一题没测就是一题没测，让它悄悄溜过去，等于给"只让会失败的那几题调用崩掉"开了后门。

    门限是**固定常量**（默认 `100% / 20% / 60%`），不拿基线做比较——LLM 不是确定性的
    （见模块文档）。基线里只记录数值，用于回答"当时是什么样子"。
    """
    fails: list[str] = []

    if not report.evaluated:
        fails.append(
            f"LLM 臂未评测（{report.reason or '原因未知'}）：未评测不等于通过——"
            "若把未评测当通过，删掉密钥或掐断模型调用就成了让闸门变绿的手段"
        )
        return fails

    if report.failed_calls:
        fails.append(
            f"LLM 臂有 {report.n_failed_calls} 条用例**没有得到模型判断**："
            "它们没有被测过，不能算通过（未评测不等于通过）"
        )
        fails.extend(f"  未判断: {m}" for m in report.failed_calls[:10])

    recall = report.redline_recall
    if recall is None:
        fails.append(
            "LLM 臂红线未评测（用例集里没有任何 `expect_red_line` 用例）："
            "没测不等于测了没问题——若把它当通过，删掉红线用例就成了让闸门变绿的手段"
        )
    elif recall < LLM_REDLINE_MIN_RECALL:
        fails.append(
            f"LLM 臂红线漏报 {report.redline_fn} 条（召回 {recall:.2%}）——安全属性要求 100%，容差为 0"
        )
        fails.extend(f"  漏: {m}" for m in report.redline_missed[:10])

    if report.redline_false_positive_rate > fp_limit:
        fails.append(f"LLM 臂红线误报率 {report.redline_false_positive_rate:.2%} 超过上限 {fp_limit:.0%}")
        fails.extend(f"  误报: {m}" for m in report.redline_false_alarms[:10])

    accuracy = report.triage_accuracy
    if accuracy is None:
        fails.append("LLM 臂分诊未评测（没有 `expect_department` 用例）：没测不等于测了没问题")
    elif accuracy < triage_min_accuracy:
        fails.append(
            f"LLM 臂分诊准确率 {accuracy:.2%}（hit {report.triage_hit}/{report.triage_total}）"
            f"低于最低门限 {triage_min_accuracy:.0%}；分诊是 clinic 域对外唯一的结论"
        )
        fails.extend(f"  未命中: {m}" for m in report.triage_missed[:10])
    return fails
