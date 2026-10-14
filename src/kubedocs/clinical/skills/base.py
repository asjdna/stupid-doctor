"""Skill 框架：自描述 + 注册表 + **显式的副作用边界**。

## 与"工具(tool)"的区别，以及为什么还要 Skill

仓库里已有工具协议（`agent/tool_loop.py`）：工具是**一次函数调用**，适合"查一次号源"
这种原子动作。但挂号是**一条有状态、有先后、其中一步不许跳过的流程** ——
把它拆成零散工具交给模型自由编排，就等于把"必须显式确认"这条安全性质交给模型自觉。

Skill 补的正是这一段：**步骤是声明的，副作用是被标注的，确认是流程里的一环而不是提示词里的一句话。**

## 三条设计约束

1. **副作用必须显式标注**（`SkillStep.side_effect`）。注册表在注册时校验两条不变量：
   **副作用步骤最多一个**（不可逆点只有一处），且**不能是第一步**（先查清楚再动手，
   否则重跑不幂等）。副作用**之后**允许有只读步骤（如生成收据）——
   把"必须在最后一步"当规则会否掉合理流程，那是校验写过头了。
2. **重名直接报错**，不静默覆盖（与工具注册表同一套纪律）。
3. **未确认不执行**：`SkillOutcome.needs_confirmation` 是流程的正常出口，
   不是异常。调用方拿它去问患者，拿到答复再用 `confirmed=True` 重跑同一条 skill。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SkillStep:
    """流程里的一步。`side_effect=True` 表示这一步会改变外部世界（落库、发号、扣费）。"""

    name: str
    description: str
    side_effect: bool = False


@dataclass
class SkillRequest:
    """一次 skill 调用的输入。

    `confirmed=True` 表示**患者已明确同意**执行副作用步骤。
    它由调用方（对话层）在患者答复之后设置——skill 自己不猜"嗯大概算是同意了吧"。
    """

    patient_id: str
    confirmed: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    institution: str = ""
    """机构标识。**本轮只是占位**：多机构（方案 §1.1）尚未实现，
    留这个字段是为了让调用点先长成正确的形状，将来加隔离时不必改所有调用方。"""


@dataclass
class SkillOutcome:
    """执行结果。`stopped_at` 让调用方知道流程停在哪一步——排查时这是最有用的一个字段。"""

    ok: bool
    stopped_at: str
    message: str = ""
    needs_confirmation: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    steps_done: list[str] = field(default_factory=list)


class Skill(Protocol):
    """一个 skill。实现者声明元信息与步骤，并提供 `run`。"""

    name: str
    description: str
    steps: tuple[SkillStep, ...]

    def run(self, request: SkillRequest) -> SkillOutcome: ...


class SkillRegistry:
    """按名字索引的注册表。与 `agent.tools` 的注册表同一套纪律：重名即错。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"skill {skill.name!r} 已注册，不能重复注册")
        if not skill.steps:
            raise ValueError(f"skill {skill.name!r} 没有声明任何步骤：流程必须可被自省")
        side_effects = [s.name for s in skill.steps if s.side_effect]
        if len(side_effects) > 1:
            raise ValueError(
                f"skill {skill.name!r} 声明了多个副作用步骤 {side_effects}。"
                "副作用只允许有一处（不可逆点只有一处），否则'失败回退'会变成一件说不清的事。"
            )
        if side_effects and skill.steps[0].side_effect:
            raise ValueError(
                f"skill {skill.name!r} 的第一步就是副作用。"
                "副作用之前必须至少有一个只读步骤（先查清楚再动手），否则重跑不幂等。"
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"没有名为 {name!r} 的 skill；已注册：{sorted(self._skills)}")
        return self._skills[name]

    def describe(self) -> list[dict[str, Any]]:
        """自省输出：给 `/skills` 端点与文档生成用。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "steps": [
                    {"name": st.name, "description": st.description, "side_effect": st.side_effect}
                    for st in s.steps
                ],
            }
            for s in sorted(self._skills.values(), key=lambda x: x.name)
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)
