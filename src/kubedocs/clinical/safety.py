"""第三层：**确定性安全门**——由服务端代码强制执行、模型绕不过去的那几条底线。

## 三层防护，各管一段（**不要互相替代**）

| 层 | 在哪 | 负责什么 | 不负责什么 |
|---|---|---|---|
| L1 语义理解 | 大模型（`prompts/system.md` R1） | **多角度**识别危险信号：症状性质 / 组合共现 / 时间过程 / 严重度 / 高危人群 / 情境诱因；对每条给出不确定性与依据 | 不保证不漏 —— 所以需要另外两层 |
| L2 主动问诊 | 大模型（R5）+ 契约（`contracts.validate_against`） | 没排除的危险信号必须问清"能区分要急救 / 不用急救"的那一个问题；低置信必须交代信息缺口 | 不负责安全动作的兜底 |
| L3 确定性安全门 | **本模块**（纯代码） | 三条不变量，模型说什么都不例外（见 `INVARIANTS`） | 不做医学判断、不识别新病症 |

三层的关系是**纵深**而不是备份：L1 的漏报由 L2 的追问补上（还没问清就不许下结论），
两者都失手时由 L3 兜住"至少给对动作"。任何一层被单独拿掉，另外两层都补不上它的洞。

## L3 的三条不变量

1. `no_medical_advice_without_evidence`：没有检索到资料 ⇒ 不得给方向/检查/用药建议；
2. `no_fabricated_booking_action`：不得声称服务端没执行过的挂号 / 退号 / 凭据号；
3. `emergency_judgment_requires_directive`：**判出急症就必须给可执行的急救动作**——
   这条是唯一一条"补内容"而不是"删内容"的不变量（见 `ensure_emergency_directive`）。

前两条由 `verify.py` 检测（本模块只把它们登记成不变量并按同一套口径留档），
第三条在本模块里直接执行：判定与执行分开写在两处，早就会出现"一处记得、一处漏掉"。

## 为什么是这三条

它们是**能被事实锚定**的：有没有依据、服务端有没有执行过那个动作、正文里有没有急救动作，
都可以机械判定，不需要医学知识。凡是需要医学判断的（这条信号算不算危险、这个科室对不对）
一律留在 L1/L2，交给模型并留档——把需要判断的东西写成 `if` 语句，只会得到一个
看起来安全、实际没人负责的假门。

## 审计

每次不变量被触发都写一条**结构最小化**的留档（`SafetyAudit`）：只记不变量 id、动作、
以及这一轮的 `trace_id`，**不记患者原话**。临床文本已经落在 `clinical` schema 里
（有自己的访问控制），审计日志再抄一份只会扩大泄露面。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from kubedocs.clinical.contracts import ClinicalAssessment, RedFlag, requires_emergency, unresolved_signals

logger = logging.getLogger(__name__)

SafetyAction = Literal["append_directive", "replace_answer", "strip_item", "record_only"]
"""不变量触发后的处置动作。

- `append_directive`：**补**一句可执行的安全动作（只用于急救指引那条）；
- `replace_answer`：模型那段散文不上屏，改成只含事实的回答；
- `strip_item`：丢掉不合法的某一条（检查 / 用药 / 科室）；
- `record_only`：只留档，不改患者看到的内容。
"""


@dataclass(frozen=True)
class SafetyInvariant:
    """一条确定性安全不变量。**id 是治理清单与测试覆盖的锚点**（见 `docs/GOVERNANCE.md`）。"""

    id: str
    title: str
    rationale: str
    action: SafetyAction
    """触发后的处置动作。"""

    implemented_in: tuple[str, ...]
    """实现它的文件（相对仓库根）。治理清单会核对这些路径真的存在。"""

    scope: tuple[str, ...] = ("consult",)
    """在哪条入口上强制执行。`/chat` 的散文路径拿不到结构化判断，所以急救指引那条只在
    `/consult` 上成立——这个差别必须写出来，而不是假装两条路径一样。"""


INVARIANTS: tuple[SafetyInvariant, ...] = (
    SafetyInvariant(
        id="no_medical_advice_without_evidence",
        title="没有检索到资料就不给医学内容",
        rationale=(
            "检索为空时给出方向/检查/用药建议，等于用模型自己的通用知识冒充本地资料支撑的结论，"
            "患者无法分辨哪句有依据、哪句是编的。"
        ),
        action="replace_answer",
        implemented_in=("src/kubedocs/clinical/verify.py", "src/kubedocs/clinical/pipeline.py"),
    ),
    SafetyInvariant(
        id="no_fabricated_booking_action",
        title="不得声称服务端没执行过的挂号动作",
        rationale=(
            "号源动作只能由服务端端点执行。患者一旦相信「已经挂上」，就不会再去挂号——"
            "这是不可挽回的误导，所以宁可整段回答不上屏。"
        ),
        action="replace_answer",
        implemented_in=("src/kubedocs/clinical/verify.py", "src/kubedocs/agent/clinic_react.py"),
        scope=("consult", "chat"),
    ),
    SafetyInvariant(
        id="emergency_judgment_requires_directive",
        title="判出急症必须给出可执行的急救动作",
        rationale=(
            "判了急症却没写「打 120 / 去急诊」，患者读完不知道该做什么。这条与另外两条相反："
            "它补内容而不是删内容——安全性只能往上加，不能因为模型少写一句就没了。"
        ),
        action="append_directive",
        implemented_in=("src/kubedocs/clinical/safety.py", "src/kubedocs/clinical/contracts.py"),
    ),
)
"""全部不变量。**增删一条都要同时改治理清单**（`data/governance/safety_assets.json`），
`scripts/check_governance.py` 会在 CI 里核对两边一致。"""

INVARIANTS_BY_ID: dict[str, SafetyInvariant] = {i.id: i for i in INVARIANTS}


EMERGENCY_DIRECTIVE = (
    "请立即拨打 120 或尽快前往最近的急诊科；如果出现意识不清、呼吸困难、"
    "无法缓解的胸痛或大量出血，请让身边的人立刻叫救护车。"
)
"""判出急症时的**兜底急救动作**（确定性、可执行、不含任何诊断）。

措辞刻意只有动作：它会被追加到任何一段没有急救动作的回答后面，所以它不能包含
"你可能是 X 病"这类断言——那需要依据，而这条不变量恰恰是在依据之外兜底的。
"""

_ACTIONABLE_DIRECTIVE_RE = re.compile(
    r"120|急诊|救护车|立刻就医|立即就医|马上就医|尽快就医|立刻去医院|立即去医院|马上送医"
)
"""把急救动作认出来的机械判据。

只认**可执行的去向**（叫救护车 / 去急诊 / 打 120 / 立刻就医），不认"要注意""不能大意"
这类没有动作的措辞——那种句子读完了患者仍然不知道该做什么。
"""


def has_actionable_directive(text: str) -> bool:
    """正文里有没有可执行的急救动作。"""
    return bool(_ACTIONABLE_DIRECTIVE_RE.search(text or ""))


def ensure_emergency_directive(message: str) -> tuple[str, bool]:
    """急症回答缺少急救动作时**补上**。返回 `(message, appended)`。

    幂等：已经有了就原样返回。补的内容固定（`EMERGENCY_DIRECTIVE`），
    所以这一层的输出不依赖模型这一轮说了什么——那也是"确定性"的含义。
    """
    if has_actionable_directive(message):
        return message, False
    sep = "\n\n" if message and not message.endswith("\n") else ""
    return f"{message}{sep}{EMERGENCY_DIRECTIVE}", True


def fallback_question(signals: list[RedFlag]) -> str:
    """问诊层没能给出追问时的**兜底追问**（确定性生成，不含判断）。

    优先用模型自己写的 `discriminator`（那才是它能区分"要急救 / 不用急救"的问题）；
    模型没写就按病症拼一句通用的，并说明为什么要问。
    """
    for sig in signals:
        if sig.discriminator.strip():
            return f"关于「{sig.condition}」：{sig.discriminator.strip()}"
    names = "、".join(dict.fromkeys(s.condition for s in signals if s.condition.strip()))
    if not names:
        return "请补充一下症状的细节：什么时候开始的、现在到什么程度、有没有越来越重？"
    return f"关于「{names}」：请补充说明具体表现与变化过程（什么时候开始的、有没有越来越重、还伴随什么）。"


@dataclass
class SafetyAudit:
    """一条安全门触发的留档记录（**结构最小化，不含患者原话**）。"""

    invariant: str
    action: SafetyAction
    detail: str = ""
    trace_id: str = ""
    domain: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "action": self.action,
            "detail": self.detail,
            "trace_id": self.trace_id,
            "domain": self.domain,
        }


@dataclass
class SafetyOutcome:
    """执行 L3 之后的产物：给患者的那段话 + 本轮触发的不变量 + 兜底追问。"""

    message: str
    fired: list[SafetyAudit] = field(default_factory=list)
    followup: str = ""

    @property
    def ids(self) -> list[str]:
        return sorted({f.invariant for f in self.fired})

    def to_payload(self) -> dict[str, Any]:
        """随响应回传的 `safety` 段：**看过什么、拦了什么**都要能被对账。"""
        return {
            "fired": [f.to_payload() for f in self.fired],
            "ids": self.ids,
            "followup": self.followup,
        }


_KIND_TO_INVARIANT: dict[str, str] = {
    "medical_without_evidence": "no_medical_advice_without_evidence",
    "fabricated_booking": "no_fabricated_booking_action",
    "fabricated_cancellation": "no_fabricated_booking_action",
    "unknown_ticket": "no_fabricated_booking_action",
}
"""`verify.py` 的违规类别 → 它违反的不变量。**没有映射的类别不进安全门留档**：
`warn` 级的（金额、红线未解释）不属于确定性不变量，它们只算回答质量，不该冒充安全事件。"""


def record_invariants(
    violated_kinds: list[str], *, trace_id: str = "", domain: str = "clinic"
) -> list[SafetyAudit]:
    """把 `verify.py` 检出的违规类别映射成"触发了哪几条不变量"。

    映射表里没有的类别不进安全门留档：`warn` 级的（金额、红线未解释）只是回答质量问题，
    不该冒充安全事件，否则"安全门触发次数"这个读数就没人信了。
    """
    fired: list[SafetyAudit] = []
    for kind in sorted(set(violated_kinds)):
        inv_id = _KIND_TO_INVARIANT.get(kind)
        if inv_id is None:
            continue
        fired.append(SafetyAudit(inv_id, INVARIANTS_BY_ID[inv_id].action, kind, trace_id, domain))
    return fired


def prose_gate(violated_kinds: list[str], *, trace_id: str = "", domain: str = "chat") -> SafetyOutcome:
    """散文路径（`/chat` 的工具循环）的安全门：**只留档，不改写**。

    散文路径拿不到结构化判断，所以"判出急症必须有急救动作"这条不变量在这里**不适用**
    （没有 `urgency` 可判、也没有 `patient_summary` 可补）。适用的是能被事实锚定的那两条，
    而它们已经由 `verify_prose` 拦下了——这里只把结果登记成不变量，让它进同一份审计。
    """
    return SafetyOutcome(
        message="", fired=record_invariants(violated_kinds, trace_id=trace_id, domain=domain)
    )


def apply_gate(
    assessment: ClinicalAssessment,
    message: str,
    *,
    violated_kinds: list[str],
    trace_id: str = "",
    domain: str = "clinic",
) -> SafetyOutcome:
    """执行确定性安全门，返回 `SafetyOutcome`。

    `violated_kinds` 是 `verify.py` 这一轮检测到的违规类别（前两条不变量由它检出，
    这里按同一套口径登记与留档——判定与执行写在两处，早晚会出现"一处记得一处漏掉"）。
    """
    fired = record_invariants(violated_kinds, trace_id=trace_id, domain=domain)

    out_message = message
    followup = ""
    # **急症的判据取"模型自己给出的信号"里更重的那一侧**：`urgency=emergency`，
    # 或者任何一条 `requires_emergency()` 为真的危险信号（likely/very_likely，或 critical 严重度）。
    # 只看 `urgency` 会漏掉一种真实形态：模型判出了 critical 方向却把 urgency 写成 urgent——
    # 那时契约层会记一条自相矛盾，但如果安全门也跟着它走，患者就一行急救动作都拿不到。
    emergency_floor = assessment.urgency == "emergency" or any(
        requires_emergency(f) for f in assessment.red_flags
    )
    if emergency_floor:
        out_message, appended = ensure_emergency_directive(message)
        if appended:
            fired.append(
                SafetyAudit(
                    "emergency_judgment_requires_directive",
                    "append_directive",
                    "急症判断（urgency=emergency 或 critical/likely 的危险信号）但正文没有"
                    "可执行的急救动作，已补固定指引",
                    trace_id,
                    domain,
                )
            )
    else:
        # 主动问诊层没给出追问时，由服务端补一句（确定性的，不含判断）。
        pending = unresolved_signals(assessment)
        if pending and not assessment.followup_questions:
            followup = fallback_question(pending)

    return SafetyOutcome(message=out_message, fired=fired, followup=followup)


def audit_path(index_root: Path) -> Path:
    """审计日志位置（与花费账本同目录，便于一起备份 / 轮转）。"""
    return Path(index_root) / "safety_audit.jsonl"


def append_audit(index_root: Path, outcome: SafetyOutcome, *, domain: str = "clinic") -> int:
    """把这一轮触发的不变量追加进审计日志，返回写入条数。

    写失败**不让这一轮失败**（与落库同一条纪律：宁可少一条记录，也不能让患者拿不到答复），
    但会记 warning——静默失败等于审计是假的。

    这里同时**打 Prometheus 计数**（`kubedocs_safety_invariant_fired_total`）：审计文件回答
    "发生了什么"，计数回答"最近是不是变多了"。放在同一个函数里是刻意的——两处各打一次，
    迟早会出现"记了日志没计数"或反过来的静默漂移。
    """
    if not outcome.fired:
        return 0
    stamp = datetime.now(timezone.utc).isoformat()
    path = audit_path(index_root)
    written = 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for rec in outcome.fired:
                row = {"ts": stamp, "domain": domain, **rec.to_payload()}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    except OSError as e:  # noqa: BLE001 - 磁盘问题不该影响这一轮应答
        logger.warning("安全门审计写入失败（%s）：%s", path, e)
    _count_invariants(outcome, domain=domain)
    return written


def _count_invariants(outcome: SafetyOutcome, *, domain: str) -> None:
    """打 Prometheus 计数。**观测失败不许影响业务**（prometheus 缺了也要能跑）。"""
    try:
        from kubedocs.observability import SAFETY_INVARIANT_FIRED

        for rec in outcome.fired:
            SAFETY_INVARIANT_FIRED.labels(invariant=rec.invariant, domain=domain).inc()
    except Exception as e:  # noqa: BLE001 - 指标是辅助，不该让安全门自己报错
        logger.debug("安全门计数失败（不影响业务）：%s", e)


def read_audit(index_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    """读回最近 `limit` 条审计记录（`/consult/status` 用它做治理可见性）。"""
    path = audit_path(index_root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(dict(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]
