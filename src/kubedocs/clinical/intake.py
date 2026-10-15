"""多轮采问：结构化槽位 + "下一问该问什么"。

"多轮采问以得到精细推断"要能被验收，就得知道**采集了哪些信息**。
自由发挥的追问无法断言、无法评测、也无法复现；槽位让"还缺什么"成为可计算的值。

**问不出来的槽位不许一直问**：规则抽取一定会漏（"心口窝堵得慌"可能匹配不到线索词），
若"缺就继续问"就会无限追问。所以每个槽位记 `asked` 次数，问满 `MAX_ASK` 次仍未填
⇒ 转入 `unfillable()` 不再问，并在给模型的上下文里写明"这项问不出来"，
让模型基于已知信息作答（并在结论里说明局限）。

抽取器可替换：`RuleSlotExtractor` 是线索词版本（可测、离线、零成本）；
换成别的 `SlotExtractor` 实现即可，`IntakeState` 不用动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

MAX_ASK = 2
"""同一个槽位最多问几次。这是防无限追问的硬闸，不是调参。"""


@dataclass(frozen=True)
class IntakeSlot:
    """一个要采集的信息项。

    `critical=True` 表示**没有它就不该给结论**（主诉）。其余槽位缺失时，
    允许"带局限作答"——因为患者可能就是不知道/不想说，卡死在这里对患者没有价值。
    """

    key: str
    question: str
    cues: tuple[str, ...]
    critical: bool = False


SLOTS: tuple[IntakeSlot, ...] = (
    IntakeSlot(
        "chief",
        "你主要是哪里不舒服？",
        ("疼", "痛", "痒", "咳", "喘", "晕", "胀", "麻", "烧", "吐", "泻", "肿", "闷", "慌"),
        critical=True,
    ),
    IntakeSlot("duration", "这个情况多久了？", ("天", "周", "月", "小时", "年", "刚才", "突然", "一直")),
    IntakeSlot(
        "associated",
        "还有别的不舒服吗？比如发烧、恶心、心慌、出汗。",
        ("发烧", "恶心", "呕吐", "心慌", "出汗", "乏力", "胸闷", "头痛"),
    ),
    IntakeSlot(
        "history",
        "以前有过类似情况吗？有没有高血压、糖尿病这类慢性病？",
        ("高血压", "糖尿病", "以前", "病史", "手术", "住院", "复发"),
    ),
    IntakeSlot("meds", "最近在吃什么药吗？", ("吃", "服用", "药", "停药", "中药", "抗生素")),
    IntakeSlot("allergy", "有没有药物过敏？", ("过敏", "不能吃", "起疹")),
)


class SlotExtractor(Protocol):
    """从患者这一轮的话里抽出某个槽位的值。抽不到返回 None。"""

    def extract(self, text: str, slot: IntakeSlot) -> str | None: ...


class RuleSlotExtractor:
    """线索词版本：命中任一 `cues` 就把**这一轮原话**当作该槽位的值。

    已知的粗糙之处：它把整句话存进去，而不是抽成结构化值；也会误触发
    （"我不发烧"命中"发烧"）。**不要**基于它的输出做统计结论。
    """

    def extract(self, text: str, slot: IntakeSlot) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        if any(cue in stripped for cue in slot.cues):
            return stripped
        return None


@dataclass
class IntakeState:
    """已采集内容 + 追问计数。**可变**：它随对话推进。"""

    filled: dict[str, str] = field(default_factory=dict)
    asked: dict[str, int] = field(default_factory=dict)

    def merge(self, text: str, extractor: SlotExtractor) -> list[str]:
        """把这一轮的话并进槽位；返回**本轮新填上**的槽位 key。

        已经填过的槽位会被覆盖（患者改口是常态，最后一次说法为准）。
        """
        filled_now: list[str] = []
        for slot in SLOTS:
            value = extractor.extract(text, slot)
            if value and self.filled.get(slot.key) != value:
                self.filled[slot.key] = value
                filled_now.append(slot.key)
        return filled_now

    def missing(self) -> list[IntakeSlot]:
        """还没填、且**还值得再问**的槽位。"""
        return [s for s in SLOTS if s.key not in self.filled and self.asked.get(s.key, 0) < MAX_ASK]

    def unfillable(self) -> list[str]:
        """问满了次数仍然空的槽位。要写进上下文，让模型知道"这项就是问不出来"。"""
        return [s.key for s in SLOTS if s.key not in self.filled and self.asked.get(s.key, 0) >= MAX_ASK]

    def next_question(self) -> IntakeSlot | None:
        """下一问。**关键槽位优先**——主诉没问到就先去问主诉。"""
        candidates = self.missing()
        if not candidates:
            return None
        critical = [s for s in candidates if s.critical]
        return (critical or candidates)[0]

    def mark_asked(self, key: str) -> None:
        self.asked[key] = self.asked.get(key, 0) + 1

    def can_conclude(self) -> bool:
        """能不能给结论：**关键槽位已填**即可。

        非关键槽位缺失不阻塞结论（会在上下文的"问不出来"里体现），
        否则只要有一项抽不出来，患者就永远拿不到答案。
        """
        return all(s.key in self.filled for s in SLOTS if s.critical)

    def render(self) -> str:
        """给模型看的"已采集情况"。空的槽位分两类写：还缺的 vs 问不出来的。"""
        lines = ["【已采集】"]
        lines.extend(f"- {s.key}：{self.filled[s.key]}" for s in SLOTS if s.key in self.filled)
        missing = [s.key for s in SLOTS if s.key not in self.filled]
        unfillable = set(self.unfillable())
        askable = [k for k in missing if k not in unfillable]
        if askable:
            lines.append("【仍缺（可以继续问）】" + "、".join(askable))
        if unfillable:
            lines.append("【问不出来（已问满次数，不要再问）】" + "、".join(sorted(unfillable)))
        return "\n".join(lines)
