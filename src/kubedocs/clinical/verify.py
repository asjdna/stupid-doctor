"""输出约束层：**拿事实集校验模型的每一处输出**，越界就拦下。

## 为什么必须有这一层

把判断权交给大模型之后，风险从"规则太笨"换成了"模型编造"。最典型的形态是**动作幻觉**：
服务端根本没挂上号，模型却写"已为您挂上，凭据号 XXXXXXXX"。这类输出对患者是
**不可挽回的误导**——他会以为号已经到手，不去复诊、不查状态。

所以「参考标准面板」（`reference.py`）与「校验器」（本模块）是一对：
面板把事实喂进去，校验器拿**同一份事实**验出来。

## 一条关键设计：不向模型要"自我声明"

校验所需的信息**全部由服务端从已有产物推导**（`ClinicalAssessment` 的字段 + 给患者那段散文），
不新增任何"请声明你有没有编造"的字段。理由很直接：**自我声明正是模型可以撒谎的地方**，
而"它引用了哪些编号""它给了哪些科室""它有没有给出医学内容"这些服务端自己就能算。
`contracts.py` 里那句"模型可能不守规矩，代码一定会守"就是这条。

## 三种严重度

| 级别 | 含义 | 处置 |
|---|---|---|
| `block` | 内容不得让患者看到 | 重生成一次；仍违规则整段换成服务端事实回答 |
| `strip` | 内容里**某一条**不合法 | 丢掉那一条（科室/建议），其余照常给患者 |
| `warn` | 需要留痕，但不该拦 | 追加一句提示 + 记进 `contract_violations` |

`block` 与 `strip` 的区别是刻意的：一条编造的科室不该让整段回答消失，而"已挂号"这种
动作幻觉必须让整段回答消失——前者是信息不全，后者是误导行为。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from kubedocs.clinical.contracts import ClinicalAssessment, enforce
from kubedocs.clinical.reference import FactSet

Severity = Literal["block", "strip", "warn"]

NO_EVIDENCE_ANSWER = (
    "没有检索到相关资料，我不能就这个问题作答。"
    "如果症状持续或加重请及时就医；出现胸痛、呼吸困难、意识不清等请立即拨打 120。"
)
"""检索为空时的**唯一**允许内容。

尾句是**安全例外**：建议就医、拨打 120 属于安全动作，不是事实断言，所以检索为空时
照样要给。否则"检索不到就说不知道"会把该报警的急症一起拦下来——那是拿安全换整洁。
真正被禁的是"你可能得的是 X 病"这类**事实断言**。
"""

_BOOKED_CLAIM_RE = re.compile(
    r"(?:已经?)(?:为您|帮你|给你)?\s*(?:挂上|挂号|取号|预约)|挂号成功|预约成功|取号成功|号已挂上"
)
"""**取号成功**的完成态说法。

两个坑都踩过：① `已` 后面要跟挂号动作词（只匹配一个"已"会把"已经三天了"误伤）；
② 必须把 `已经` 也写进去——"**已经**为您挂上了"里 `已` 后面是 `经`，只写 `已|已为您`
会整条漏掉，而那正是这句话最常见的说法。"""

_CANCELLED_CLAIM_RE = re.compile(r"已经?(?:为您|帮你)?(?:退号|取消(?:挂号|预约))|退号成功|已退号")
"""**退号成功**的完成态说法。

与取号**分开匹配**是必需的：两类动作要求的事实不是同一条。合在一起就会出现
"服务端只退过号，模型说已挂上"被放行的漏洞（一张 `cancelled` 的单据被当成
"确实有过挂号动作"）。"""

_TICKET_LABEL_RE = re.compile(r"凭据号\s*[：:是为]*\s*([A-Za-z0-9]{4,16})")
"""带标签的凭据号：`凭据号 SNXXXXXX`。

只认**明确标注成凭据号**的串，不去猜裸的 8 位大写串——后者会把正常英文单词也当成号，
误拦一段本来正确的回答。"""

_MONEY_RE = re.compile(r"(\d+)\s*元")


@dataclass(frozen=True)
class Violation:
    """一条违规。`detail` 会被回灌给模型（重生成提示）也会进响应 payload。"""

    kind: str
    detail: str
    severity: Severity = "block"

    def render(self) -> str:
        return f"[{self.severity}] {self.kind}：{self.detail}"


@dataclass
class Verification:
    """校验结果：违规清单 + 可直接回灌给模型的重生成提示。"""

    violations: list[Violation] = field(default_factory=list)

    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "block"]

    def stripping(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "strip"]

    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warn"]

    def ok(self) -> bool:
        """没有 block 就算过关；strip/warn 由调用方按自己的方式处理。"""
        return not self.blocking()

    def kinds(self) -> list[str]:
        return sorted({v.kind for v in self.violations})

    def feedback(self) -> str:
        """回灌给模型的重生成提示：**只陈述事实与违规，不重述答案**。"""
        lines = [
            "你上一轮的输出与系统事实冲突，已被拦下。请**只依据下面的事实**重写，"
            "凡事实里没有的内容，一律不要写；确实无法回答就明说没有查到相关资料。",
            "",
        ]
        lines.extend(f"- {v.detail}" for v in self.violations if v.severity == "block")
        return "\n".join(lines)


def _cited_evidence_ids(assessment: ClinicalAssessment) -> list[str]:
    """模型这次回答里引用到的全部依据编号（检查/用药/急重症判断）。

    三个列表**分开遍历**，不用 `(*a, *b, *c)`：三元展开会让 mypy 把它们推成共同基类
    `_Strict`，`.evidence_ids` 就成了"未定义属性"（类型上说不通，运行时其实是对的）。
    """
    ids: list[str] = []
    for item in assessment.exams:
        ids.extend(item.evidence_ids)
    for med in assessment.medications:
        ids.extend(med.evidence_ids)
    for flag in assessment.red_flags:
        ids.extend(flag.evidence_ids)
    return ids


def _medical_content(assessment: ClinicalAssessment) -> bool:
    """这次回答里有没有**医学实质内容**。有 ⇒ 必须有检索依据（急症安全动作除外）。"""
    return bool(assessment.exams or assessment.medications or assessment.differentials)


def _prose_violations(prose: str, facts: FactSet, *, emergency: bool) -> list[Violation]:
    """**只做能被事实锚定**的那些检查，结构化契约与散文两条路共用。

    这里的三类是同一类风险：模型宣称了一件服务端没做过的事，或写了一个服务端没给过的值。
    它们**不需要**结构化契约就能验——依据、金额、凭据号都是事实集里现存的东西，
    而"说没说过某个动作"在散文里就能认出来。
    """
    out: list[Violation] = []

    # **动作幻觉**：声称完成了挂号/退号，而事实集里没有**对应**的那个动作。
    # 这是本模块存在的首要理由（服务端没挂上、模型宣布挂上）。
    confirmed = [b for b in facts.completed_bookings() if b.kind == "confirmed"]
    cancelled = [b for b in facts.completed_bookings() if b.kind == "cancelled"]
    booked_claim = _BOOKED_CLAIM_RE.search(prose)
    cancelled_claim = _CANCELLED_CLAIM_RE.search(prose)
    if booked_claim and not confirmed:
        out.append(
            Violation(
                kind="fabricated_booking",
                detail=(
                    f"你的回答里出现了「{booked_claim.group(0)}」这类完成态说法，但服务端本轮"
                    f"**没有**任何已确认的取号动作（号源动作只能由服务端端点执行）。"
                    f"请改为「这张单子需要患者确认后才生效」这类与实际状态一致的表述。"
                ),
                severity="block",
            )
        )
    if cancelled_claim and not cancelled:
        out.append(
            Violation(
                kind="fabricated_cancellation",
                detail=(
                    f"你的回答里出现了「{cancelled_claim.group(0)}」这类说法，但服务端本轮"
                    f"**没有**任何已退号的记录。请按实际状态说明。"
                ),
                severity="block",
            )
        )

    # 凭据号幻觉：提到了一个服务端没发过的凭据号。
    real_tickets = facts.ticket_numbers()
    for match in _TICKET_LABEL_RE.finditer(prose):
        token = match.group(1)
        if token not in real_tickets:
            out.append(
                Violation(
                    kind="unknown_ticket",
                    detail=(
                        f"你写的凭据号 {token!r} 不存在"
                        + (
                            f"；服务端本轮的凭据号只有 {sorted(real_tickets)}"
                            if real_tickets
                            else "（本轮没有任何凭据号）"
                        )
                    ),
                    severity="block",
                )
            )

    # 金额幻觉：说了服务端没给过的挂号费。
    known_numbers = facts.known_numbers()
    for match in _MONEY_RE.finditer(prose):
        amount = int(match.group(1))
        if amount not in known_numbers and amount not in (0, 120):
            out.append(
                Violation(
                    kind="unknown_amount",
                    detail=(
                        f"你提到了 {amount} 元，但服务端没有给出过这个金额"
                        + (
                            f"（已知：{sorted(known_numbers)}）"
                            if known_numbers
                            else "（本轮没有任何费用事实）"
                        )
                    ),
                    severity="warn",
                )
            )

    # 红线线索未被解释（warn）：三维门控说这条可作为本人当前主诉，而模型没判急症，
    # 也没在回答里提过这个词。**不拦**——判断权在模型，但"系统担心过这件事"要留痕。
    if facts.red_line_is_reportable() and not emergency:
        for word in facts.red_line_words():
            if word and word not in prose:
                out.append(
                    Violation(
                        kind="red_line_unexplained",
                        detail=(
                            f"关键词表命中「{word}」（三维门控认为可作为患者当前主诉），"
                            f"但你没判急症、也没在总结里解释为什么不需要紧急处理"
                        ),
                        severity="warn",
                    )
                )
    return out


def verify_prose(prose: str, facts: FactSet, *, emergency: bool | None = None) -> Verification:
    """**散文路径**（`/chat` 的工具循环）的输出校验。

    与 `verify()` 的差别是**能验什么**：`/chat` 的输出是一段自由文本，没有
    `ClinicalAssessment` 那样的结构化契约，所以这里只做**能被事实锚定**的检查——
    动作幻觉、凭据号、金额、红线未解释。

    **不假装能验的**（诚实地留在这里，而不是用关键词近似一下充数）：

    - *编造依据编号*：散文里没有 `evidence_ids` 字段，认不出它引用了哪个编号；
    - *科室未接地*：散文里没有科室字段，从文本里"找科室名"会误伤（"不是心内科的问题"）；
    - *无依据却给医学内容*：判断"这段话算不算医学内容"没有可靠的机械判据，
      靠关键词表会把"这不是药"也判成用药建议。所以这条**只在结构化路径**上强制
      （`/consult` 的 `medical_without_evidence`），不在散文上假装有。

    `emergency` 不给时按"正文里有没有可执行的紧急指引"推定：那是**展示层判断**
    （决定要不要弹横幅），不是医学判断——判断已经写在正文里了。
    """
    if emergency is None:
        emergency = "120" in prose or "急诊" in prose
    return Verification(violations=_prose_violations(prose, facts, emergency=emergency))


def verify(
    assessment: ClinicalAssessment,
    prose: str,
    facts: FactSet,
    *,
    emergency: bool | None = None,
) -> Verification:
    """拿事实集校验**结构化契约路径**（`/consult`）的输出。**返回而不抛**：调用方要能"带伤降级"。

    `emergency` 不给时按 `assessment.urgency == "emergency"` 推定；显式传入是为了让
    调用方能按"横幅是否真的弹了"来判定（降级轮次里 `urgency` 可能不可信）。
    """
    if emergency is None:
        emergency = assessment.urgency == "emergency"
    out: list[Violation] = []
    known_ids = facts.evidence_ids()

    # 1) 编造依据：引用了不存在的编号。检查/用药那部分 `contracts.enforce` 已经会丢，
    #    这里仍然报出来——它的价值是**留痕**（丢东西不能是静默的）。
    unknown = sorted({e for e in _cited_evidence_ids(assessment) if e not in known_ids})
    if unknown:
        out.append(
            Violation(
                kind="fabricated_evidence",
                detail=f"引用了不存在的依据编号 {unknown}；本次可引用的编号只有 {sorted(known_ids)}",
                severity="strip",
            )
        )

    # 2) 科室未接地：建议了一个既不在科室表、也不在候选里的科室。
    #    **归一化后比较**：模型说的是"呼吸内科门诊""全科门诊"，科室表写的是"呼吸内科""全科"
    #    ——同一个科。硬比会把合法建议判成未接地并剔除（实测踩过两次）。
    allowed = facts.allowed_departments()
    if allowed:
        bad = [d.name for d in assessment.departments if facts.normalize(d.name) not in allowed]
        if bad:
            out.append(
                Violation(
                    kind="department_not_grounded",
                    detail=(f"科室建议 {bad} 不在可挂号科室/候选里；可用的只有 {sorted(allowed)}"),
                    severity="strip",
                )
            )

    # 3)~5) 散文层面的检查与 `/chat` 共用同一份实现（动作幻觉 / 凭据号 / 金额 / 红线留痕）。
    out.extend(_prose_violations(prose, facts, emergency=emergency))

    # 6) 无依据却给医学内容：检索为空时只能如实说不知道。
    #    急症安全动作例外（见 NO_EVIDENCE_ANSWER 的说明）。
    if _medical_content(assessment) and not facts.has_evidence() and not emergency:
        out.append(
            Violation(
                kind="medical_without_evidence",
                detail=(
                    "本次**没有检索到任何资料**，但你的回答里给出了病症方向/检查/用药内容。"
                    f"只允许回答：{NO_EVIDENCE_ANSWER}"
                ),
                severity="block",
            )
        )
    return Verification(violations=out)


def strip_ungrounded(
    assessment: ClinicalAssessment, verification: Verification, facts: FactSet
) -> ClinicalAssessment:
    """按 `strip` 级违规**真的删掉**不合法的条目，返回清洗后的结论。

    只删两类：编造依据的条目与未接地的科室。其它一律不动——尤其不碰 `red_flags`：
    急重症宁可留着让人看见。

    **引用那部分直接复用 `contracts.enforce`**，不在这里再写一遍。R3（"检索不到就不给"）
    只能有一个实现：两处各写一遍，迟早会出现"一处丢了、另一处没丢"的静默分歧。
    """
    kinds = verification.kinds()
    cleaned = assessment
    if "fabricated_evidence" in kinds:
        cleaned = enforce(cleaned, facts.evidence_ids()).assessment
    if "department_not_grounded" in kinds:
        allowed = facts.allowed_departments()
        # 与 `verify` 同一条口径：归一化后比较（"呼吸内科门诊" = "呼吸内科"）。
        kept = [d for d in cleaned.departments if not allowed or facts.normalize(d.name) in allowed]
        # 全被删掉时**保留原样**：一个科室建议都没有会让 `validate_against` 记一条
        # "没有任何科室建议"，反而把"科室名不接地"这件事掩盖成"模型没给科室"。
        # 宁可留着那条不合法的（患者能看到、运维能查到），也不要制造一个更含糊的失败。
        cleaned = cleaned.model_copy(update={"departments": kept or list(cleaned.departments)})
    return cleaned


def facts_only_answer(facts: FactSet, *, reason: str = "") -> str:
    """第 ② 级处置：**不展示模型那段散文**，改由服务端拼一段只含事实的回答。

    内容来源严格限定在事实集里：有没有可挂号的科室、本轮有没有已确认的挂号动作。
    拼不出任何事实时返回 `NO_EVIDENCE_ANSWER`——**宁可少说，不可编**。
    """
    parts: list[str] = []
    if facts.has_evidence():
        parts.append("系统已检索到相关资料，但这一轮的回答没有通过事实校验，因此不展示。")
    else:
        parts.append(NO_EVIDENCE_ANSWER)
    completed = facts.completed_bookings()
    if completed:
        parts.extend(b.render().lstrip("- ") for b in completed)
    else:
        parts.append("本轮没有产生任何挂号或退号动作。")
    if reason:
        parts.append(f"（拦截原因：{reason}）")
    return "\n".join(parts)
