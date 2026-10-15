"""Skill 注册入口。

新增一个能力 = 写一个满足 `Skill` 协议的对象 + 在 `default_registry()` 里注册一行，
**不碰主流程**。这就是把能力"封装"起来的意义：加能力不该改编排代码。
"""

from __future__ import annotations

from kubedocs.clinical.skills.base import Skill, SkillOutcome, SkillRegistry, SkillRequest, SkillStep
from kubedocs.clinical.skills.booking import BookingBackend, BookingSkill

__all__ = [
    "BookingBackend",
    "BookingSkill",
    "Skill",
    "SkillOutcome",
    "SkillRegistry",
    "SkillRequest",
    "SkillStep",
    "default_registry",
]


def default_registry(*, booking_backend: BookingBackend) -> SkillRegistry:
    """装配默认 skill 集合。

    目前只有挂号。后续如果要加"查报告""改约""取消号"，
    在这里各注册一行即可；流程与确认语义由框架统一保证。
    """
    registry = SkillRegistry()
    registry.register(BookingSkill(booking_backend))
    return registry
