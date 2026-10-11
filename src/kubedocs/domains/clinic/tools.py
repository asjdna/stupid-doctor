"""clinic 域的工具：把号源能力暴露成 LLM 可调用的 `ToolSpec`。

## 为什么必须补这一层

在此之前，clinic agent 的"能力"是 `ClinicDeps` 里 8 个**硬编码回调**——
控制流完全由代码决定，LLM 只在"抽症状"和"生成回答"两个点上被调用。
那是一台**带图外壳的流水线**，不是 agent。

有了这里的 `ToolSpec`，LLM 才能通过 function calling 自己决定：
要不要先查号源、查哪个科室、什么时候该问用户而不是自己猜。

## `side_effect` 的用法

`book_appointment` 标记 `side_effect=True`——它是**不可逆的写操作**（号源减一、
生成取号单）。工具循环在执行前会要求一个确认回调；没有确认就拒绝执行并把
"需要用户确认"回灌给模型。这是安全属性，不是可选优化。
"""

from __future__ import annotations

from typing import Any

from kubedocs.agent.tools import (
    BookParams,
    GetAvailabilityParams,
    ListDepartmentsParams,
    ToolError,
    ToolSpec,
)
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.provider import BookingError

DOMAIN = "clinic"


def make_list_departments_tool(service: BookingService) -> ToolSpec:
    def _run(**kwargs: Any) -> dict[str, Any]:
        contains = str(kwargs.get("contains", "") or "").strip()
        deps = service.departments()
        if contains:
            deps = [d for d in deps if contains in str(d["name"])]
        return {"departments": deps, "n": len(deps)}

    return ToolSpec(
        name="list_departments",
        description="列出可挂号的科室（代码、名称、挂号费）。",
        params=ListDepartmentsParams,
        domain=DOMAIN,
        handler=_run,
        when_not=(
            "已经知道要挂哪个科时不要调；用户只是在描述症状、没提挂号时不要调；"
            "不确定科室名是否正确时可以调一次用于核对。"
        ),
    )


def make_availability_tool(service: BookingService) -> ToolSpec:
    def _run(**kwargs: Any) -> dict[str, Any]:
        department = str(kwargs.get("department", "") or "")
        date = str(kwargs.get("date", "") or "") or None
        try:
            avail = service.availability(department, date)
        except BookingError as e:
            raise ToolError(str(e)) from e
        return {
            "department": avail.department,
            "date": avail.date,
            "slots": avail.slots,
            "total_remaining": avail.total_remaining,
        }

    return ToolSpec(
        name="get_availability",
        description="查询某科室某天的号源余量（按时段）。",
        params=GetAvailabilityParams,
        domain=DOMAIN,
        handler=_run,
        when_not=(
            "还没确定科室时不要调（先用 list_departments 或先问清症状）；"
            "同一科室同一天不要重复调，除非距离上次查询已经隔了一轮对话。"
        ),
    )


def make_book_tool(service: BookingService) -> ToolSpec:
    def _run(**kwargs: Any) -> dict[str, Any]:
        department = str(kwargs.get("department", "") or "")
        date = str(kwargs.get("date", "") or "") or None
        slot = str(kwargs.get("time_slot", "") or "") or None
        patient_id = str(kwargs.get("patient_id", "") or "anon")
        try:
            ticket = service.book(department, patient_id, date=date, time_slot=slot)
        except BookingError as e:
            raise ToolError(str(e)) from e
        return ticket

    return ToolSpec(
        name="book_appointment",
        description="为患者取号（会真实占用一个号源，不可逆）。",
        params=BookParams,
        domain=DOMAIN,
        handler=_run,
        side_effect=True,
        when_not=(
            "**用户没有明确要求挂号时绝对不要调**；科室未确定时不要调；"
            "没查过号源余量时不要调（先 get_availability）；"
            "用户只是想了解流程、或还在描述症状时不要调。"
        ),
    )


def clinic_tools(service: BookingService) -> list[ToolSpec]:
    """clinic 域的全部工具（按注册顺序）。"""
    return [make_list_departments_tool(service), make_availability_tool(service), make_book_tool(service)]
