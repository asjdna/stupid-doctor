"""参考标准面板：**规则资产给 LLM 的依据**，而不是结论。

## 这一层为什么存在

仓库里原本有一批确定性资产：急诊红线词表 + 三维门控、口语症状词表、语料症状键、
症状→科室关联票数、检索式分诊、科室补全表、号源事实。它们都**有价值**——但价值是
"提供线索"，不是"下判断"。以前它们里的一小部分直接产出结论（图路径的红线短路、
规则票数直接当科室），于是同一个问题有两个不同脑在答，而且规则那个脑看不到上下文。

现在的分工是：

| 谁 | 干什么 |
|---|---|
| 本模块 | 把上述资产装配成一段**带出处、带置信度、明说"仅供参考"**的面板 |
| 大模型 | 看着面板做**唯一**的判断（是否急症、挂哪个科、给不给建议） |
| `clinical/verify.py` | 拿同一份事实**反过来校验模型的每一处输出** |

面板与校验器用的是**同一份 `FactSet`**：说出去的依据和校验时用的依据必须是同一份事实，
否则就会出现"依据里有、校验时不算"这种最坑人的不一致。

## 三条写法上的硬规定

1. **每条都标"这不是结论"**：面板本身不能变成第二个判断者。急症线索里写的是
   "关键词表命中了某词、三维门控的结果是什么"，**不写**"这是急症"。
2. **没命中要如实写"无"**：不能因为面板空了就整段省略——省略会让模型以为"没有信号"
   与"没查"是同一件事，而它们完全不同（后者是系统故障）。
3. **业务事实要写死"本轮发生过什么"**：挂号那类不可逆动作，模型最容易凭想象宣布成功。
   面板里明确写出"本轮服务端**没有**任何已确认的挂号动作"，这是最直接的解毒剂。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kubedocs.clinical.evidence import EvidenceBundle

NO_BOOKING_FACT_NOTE = "本轮服务端**没有**任何已确认的挂号/退号动作。"
"""业务事实里的关键一句：模型不得声称发生过写操作（见 `verify.py` 的 `fabricated_booking`）。

它必须**显式出现**而不是靠"没写就是没有"来表达：模型看到空白会按对话语气补一句
"已为您挂上"，而看到这句明确的否定才有一个可对照的事实。
"""


@dataclass(frozen=True)
class SymptomSignal:
    """症状归一线索：患者原话里的词 → 口语标准词 → 语料词表的键。"""

    matched: str
    canonical: str
    corpus_keys: list[str] = field(default_factory=list)
    negated: bool = False

    def render(self) -> str:
        keys = "、".join(self.corpus_keys[:6]) or "（语料里没有对应键）"
        mark = "**已否定**" if self.negated else "未否定"
        return f"- 原话「{self.matched}」→ 标准词「{self.canonical}」→ 语料键：{keys}（{mark}）"


@dataclass(frozen=True)
class DepartmentSignal:
    """科室候选线索。`source` 说明票是哪儿来的——模型要能看出哪条更值得信。"""

    name: str
    score: float
    source: str

    def render(self) -> str:
        return f"- {self.name}（票数 {self.score:.1f}，来源：{self.source}）"


@dataclass(frozen=True)
class BookingFact:
    """号源/挂号的事实。`kind` 取 `available`（余量）/`confirmed`/`cancelled`。"""

    kind: str
    department: str = ""
    date: str = ""
    time_slot: str = ""
    fee: int = 0
    remaining: int = 0
    ticket_no: str = ""
    queue_no: int = 0
    status: str = ""

    def render(self) -> str:
        if self.kind == "available":
            return (
                f"- 号源：{self.department} {self.date} {self.time_slot} "
                f"余 {self.remaining} 个，挂号费 {self.fee} 元（服务端读数）"
            )
        verb = "已确认挂号" if self.kind == "confirmed" else "已退号"
        return (
            f"- **服务端事实：{verb}** 凭据号 {self.ticket_no}，{self.department} "
            f"{self.date} {self.time_slot}，队列号 {self.queue_no}，挂号费 {self.fee} 元，"
            f"状态 {self.status}"
        )


@dataclass
class FactSet:
    """一轮对话里**被认定为事实**的全部内容。面板与校验器共用这一份。"""

    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    symptoms: list[SymptomSignal] = field(default_factory=list)
    departments: list[DepartmentSignal] = field(default_factory=list)
    booking: list[BookingFact] = field(default_factory=list)
    known_departments: tuple[str, ...] = ()
    """可挂号的科室表。科室建议必须落在它里面（见 `verify.fabricated_department`）。"""

    normalize_department: Callable[[str], str] | None = None
    """科室名的归一化函数（**可选注入**）。

    校验必须**归一化后**比较，不能拿原始字符串硬比：模型说的是"呼吸内科门诊""全科门诊"，
    科室表写的是"呼吸内科""全科"——那是同一个科，硬比会把**合法建议判成未接地并剔除**，
    患者看到的那条建议就凭空消失了。归一化规则属于**域知识**（clinic 在
    `domains/clinic/data/departments.py`），所以这里只收一个可调用对象，自己不实现规则。
    """

    notes: list[str] = field(default_factory=list)
    """装配期的问题（某一路线索抽取失败等）。进面板，让模型知道"这次线索不全"。"""

    # ------------------------------------------------------------------ 查询
    def evidence_ids(self) -> set[str]:
        """这一轮**允许被引用**的全部依据标识。

        两种写法都算数：我们自己分配的 `K#`（提示词要求用的），以及索引里的真实
        `chunk_id`（`search_docs` 返回给模型的就是它，模型很自然会照抄）。
        只认前者会把真实命中判成编造——那是误伤，还会顺手丢掉一条本来正确的建议。
        """
        out = {i.id for i in self.evidence.items}
        out.update(i.source_id for i in self.evidence.items if i.source_id)
        return out

    def has_evidence(self) -> bool:
        """有没有**知识依据**。空 ⇒ 模型不得给出任何医学内容，只能如实说不知道。"""
        return any(i.kind == "knowledge" for i in self.evidence.items)

    def red_line(self) -> Any:
        return self.evidence.red_line

    def red_line_words(self) -> list[str]:
        red = self.red_line()
        return [str(getattr(red, "word", "") or "")] if red is not None else []

    def red_line_is_reportable(self) -> bool:
        """三分门控说这条线索能不能算"患者本人的当前主诉"。"""
        red = self.red_line()
        if red is None:
            return False
        return bool(getattr(red, "reportable", True))

    def completed_bookings(self) -> list[BookingFact]:
        return [b for b in self.booking if b.kind in ("confirmed", "cancelled")]

    def ticket_numbers(self) -> set[str]:
        return {b.ticket_no for b in self.completed_bookings() if b.ticket_no}

    def allowed_departments(self) -> set[str]:
        """科室建议的合法取值：科室表 ∪ 面板里给出的候选（**已归一化**）。"""
        out = set(self.known_departments)
        out.update(d.name for d in self.departments)
        return {self.normalize(name) for name in out}

    def normalize(self, name: str) -> str:
        """按注入的规则归一化科室名；没注入就原样返回（通用层不猜域规则）。"""
        if self.normalize_department is None:
            return str(name or "").strip()
        try:
            return str(self.normalize_department(name) or "").strip()
        except Exception:  # noqa: BLE001 - 归一化失败退回原值：宁可多报一条，不要误删
            return str(name or "").strip()

    def known_numbers(self) -> set[int]:
        """面板里出现过的数值（费用/余量/队列号）。散文里出现的数值必须落在这里。"""
        nums: set[int] = set()
        for b in self.booking:
            nums.update({int(b.fee), int(b.remaining), int(b.queue_no)})
        return {n for n in nums if n}

    # ------------------------------------------------------------------ 渲染
    def render(self) -> str:
        """渲染成提示词里的一段。**每一节都明说"这是线索，不是结论"。**"""
        lines: list[str] = [
            "【参考标准｜系统给你的线索，**不是结论**】",
            "下面是规则资产与检索给出的线索。最终判断由你做出，并对你的判断负责；",
            "线索可以与你自己的判断不一致，但不一致时请在结论里说明理由。",
            "",
            "一、急诊线索（关键词表 + 三维门控；**不构成急症结论**）",
        ]
        red = self.red_line()
        if red is None:
            lines.append("- 无（关键词表本次未命中任何词）")
        else:
            word = str(getattr(red, "word", "") or "")
            assertion = str(getattr(red, "assertion", "") or "")
            experiencer = str(getattr(red, "experiencer", "") or "")
            temporality = str(getattr(red, "temporality", "") or "")
            if not word:
                # 出现了红线词、但三维门控判定**都不作**本人当前主诉（家属/既往/否定/假设）。
                # 这一支不打印空的字段（那会渲染成「命中「」：断言= …」这种读不通的行），
                # 明细在下面逐条给。
                lines.append("- 本次有红线词出现，但三维门控判定**都不作**患者本人当前主诉（逐条见下）")
            elif self.red_line_is_reportable():
                lines.append(
                    f"- 命中「{word}」：断言={assertion} 经验者={experiencer} 时间性={temporality}"
                    f" ⇒ 三维门控认为**可作为患者当前主诉上报**，请重点评估是否急症"
                )
            else:
                lines.append(
                    f"- 命中「{word}」：断言={assertion} 经验者={experiencer} 时间性={temporality}"
                    f" ⇒ 三维门控**不认为**它可作为患者当前主诉（例如家属/既往/假设），"
                    f"请勿据此判急症，但仍要结合原话自行判断"
                )
        # 三维明细（含**不触发**的出现处）原样给出去：模型要能自己看懂
        # "系统为什么放过了这个词"，只给一个结论是不够的。
        for note in list(getattr(red, "notes", None) or [])[:6]:
            lines.append(f"  · 三维明细：{note}")
        lines.append("- 提醒：**判不判急症由你决定**；判为急症时必须在总结里给出可执行的紧急指引。")

        lines.append("")
        lines.append("二、症状归一线索（口语 → 语料词表，仅供理解原话）")
        if self.symptoms:
            lines.extend(s.render() for s in self.symptoms)
        else:
            lines.append("- 无（口语词表本次无命中）")

        lines.append("")
        lines.append("三、科室候选线索（规则票数，**只是候选**，最终由你选择）")
        if self.departments:
            lines.extend(d.render() for d in self.departments[:8])
        else:
            lines.append("- 无（规则票数为空，请仅依据知识与你的判断给科室）")
        if self.known_departments:
            lines.append(
                "- 可挂号科室（只能从这里选，或说明为什么都不合适）：" + "、".join(self.known_departments)
            )

        lines.append("")
        lines.extend(self.evidence.render().splitlines())

        lines.append("")
        lines.append("五、业务事实（号源 / 费用 / 本轮服务端实际执行过的动作）")
        completed = self.completed_bookings()
        available = [b for b in self.booking if b.kind == "available"]
        if available:
            lines.extend(b.render() for b in available[:6])
        if completed:
            lines.extend(b.render() for b in completed)
        else:
            lines.append(f"- {NO_BOOKING_FACT_NOTE}不得声称已挂号、已取号或已退号。")

        if self.notes:
            lines.append("")
            lines.append("六、线索装配问题（这些线索这次不可用）")
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines)

    def to_payload(self) -> dict[str, Any]:
        """给响应/审计用的结构化形状：**规则说了什么、原话怎么被归一**都可逐轮对账。"""
        red = self.red_line()
        return {
            "red_line": (
                None
                if red is None
                else {
                    "word": str(getattr(red, "word", "") or ""),
                    "assertion": str(getattr(red, "assertion", "") or ""),
                    "experiencer": str(getattr(red, "experiencer", "") or ""),
                    "temporality": str(getattr(red, "temporality", "") or ""),
                    "reportable": self.red_line_is_reportable(),
                }
            ),
            "symptoms": [
                {
                    "matched": s.matched,
                    "canonical": s.canonical,
                    "corpus_keys": list(s.corpus_keys),
                    "negated": s.negated,
                }
                for s in self.symptoms
            ],
            "department_candidates": [
                {"name": d.name, "score": d.score, "source": d.source} for d in self.departments
            ],
            "evidence_ids": sorted(self.evidence_ids()),
            "has_evidence": self.has_evidence(),
            "booking": [
                {
                    "kind": b.kind,
                    "department": b.department,
                    "date": b.date,
                    "time_slot": b.time_slot,
                    "fee": b.fee,
                    "remaining": b.remaining,
                    "ticket_no": b.ticket_no,
                    "queue_no": b.queue_no,
                    "status": b.status,
                }
                for b in self.booking
            ],
            "known_departments": list(self.known_departments),
            "notes": list(self.notes),
        }


def build_facts(
    *,
    evidence: EvidenceBundle | None = None,
    symptoms: list[SymptomSignal] | None = None,
    departments: list[DepartmentSignal] | None = None,
    booking: list[BookingFact] | None = None,
    known_departments: Any = (),
    normalize_department: Callable[[str], str] | None = None,
    notes: list[str] | None = None,
) -> FactSet:
    """装配事实集。各参数都可以缺——缺了面板就如实写"无"，不影响其余部分可用。"""
    return FactSet(
        evidence=evidence or EvidenceBundle(),
        symptoms=list(symptoms or []),
        departments=list(departments or []),
        booking=list(booking or []),
        known_departments=tuple(str(d) for d in (known_departments or ())),
        normalize_department=normalize_department,
        notes=list(notes or []),
    )
