"""clinic 域的工具：把号源能力暴露成 LLM 可调用的 `ToolSpec`。

模型能看到**四个只读工具**：列科室、查某天号源、看整个放号窗口、查本人已挂的号，
外加一个**只提议不落库**的 `propose_booking`。

## 为什么没有"取号"工具

取号不可逆。产品要求"模型提议 → 患者点确认 → 服务端执行"，所以模型这一侧只到
**提议**为止：`propose_booking` 验证科室/日期/时段真有号，产出一张待确认单，
号源一个都不动。真正写库的是 `POST /consult/booking/confirm`，参数取自服务端存的单子。

`book_appointment` 仍然保留，但标记 `side_effect=True`——工具循环默认把它挡在外面，
只有明确知道自己在做什么的调用方（MCP 服务器加 `--allow-side-effects`）才拿得到。

同理，**退号也不给模型**：患者的挂号卡片上有退号按钮，走的是同一个 HTTP 端点。

## `patient_id` 从哪来

`my_appointments` 与 `propose_booking` 都需要知道"这是谁"。它们的参数模型里**没有**
`patient_id` 字段，值由服务端经 `context=` 注入（见 `agent/tool_loop._invoke`）——
模型没有机会伪造"我是谁"。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from kubedocs.agent.tools import (
    BookingWindowParams,
    BookParams,
    GetAvailabilityParams,
    ListDepartmentsParams,
    MyAppointmentsParams,
    ProposeBookingParams,
    ToolError,
    ToolSpec,
)
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.pending import PendingBookings
from kubedocs.domains.clinic.provider import BookingError

DOMAIN = "clinic"

MISSING_PATIENT = "缺少就诊人标识：本次会话没有拿到 patient_id，无法查/建这个人的挂号单。"

_RELATIVE_DAYS = {
    "今天": 0,
    "今日": 0,
    "明天": 1,
    "明日": 1,
    "后天": 2,
    "大后天": 3,
}
"""模型最常原样传下来的几个相对日期。

为什么必须在这一层收口：`date` 的说明写的是"YYYY-MM-DD，留空表示今天"，但患者说的是
"明天上午方便吗"，而**模型几乎必然把「明天」原样传下来**（实测如此）。号源层对日期格式
是严格校验的，于是这一次工具调用失败、整轮挂号提议不会发生——界面上只看到"模型没提议
挂号"，原因藏在工具结果里。

只认这几个词，**不做通用中文日期解析**：不假装能懂"下周三"这类表述。那种情况交给号源层
回一句人话（"date 必须是 YYYY-MM-DD"），比猜错一个日期、给患者挂错天要好得多。
"""


def _normalize_date(value: object) -> str:
    """相对日期 -> `YYYY-MM-DD`；其余原样返回（由号源层校验并给出可读错误）。"""
    text = str(value or "").strip()
    if text in _RELATIVE_DAYS:
        return (dt.date.today() + dt.timedelta(days=_RELATIVE_DAYS[text])).isoformat()
    return text


def _patient_id(kwargs: dict[str, Any]) -> str:
    """取服务端注入的就诊人标识。

    空值在这里**必须**报错而不是退回 `"anon"`：把两个不同患者记成同一个匿名标识，
    会让"我的挂号记录"里出现别人的号——一个安静的隐私事故，而不是一次显式失败。
    """
    pid = str(kwargs.get("patient_id") or "").strip()
    if not pid:
        raise ToolError(MISSING_PATIENT)
    return pid


def _fee_of(service: BookingService, department: str) -> int:
    """挂号费取自科室表（`booking.departments.fee` 的唯一出口）。

    查不到就给 0 而不是报错：费用是**展示**信息，缺了不该让挂号流程断掉。
    """
    for d in service.departments():
        if str(d.get("name")) == department:
            return int(d.get("fee") or 0)
    return 0


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
        date = _normalize_date(kwargs.get("date")) or None
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
        description="查询某科室某一天的号源余量（按时段）。",
        params=GetAvailabilityParams,
        domain=DOMAIN,
        handler=_run,
        when_not=(
            "还没确定科室时不要调（先 list_departments 或先问清症状）；"
            "用户问的是「哪天有号」而不是「某天有没有号」时改用 booking_window。"
        ),
    )


def make_booking_window_tool(service: BookingService) -> ToolSpec:
    def _run(**kwargs: Any) -> dict[str, Any]:
        department = str(kwargs.get("department", "") or "")
        days = int(kwargs.get("days", 7) or 7)
        try:
            dep = service.resolve_department(department)
            window = service.availability_window(dep, days)
        except BookingError as e:
            raise ToolError(str(e)) from e
        return {
            "department": dep,
            "days": days,
            # 只留有号的日期：模型要回答的是"哪天能挂上"，把 7 天空日子一起塞给它
            # 只会让它复述一堆"这天没号"。
            "open_days": [d for d in window if sum(int(s["remaining"]) for s in d["slots"]) > 0],
        }

    return ToolSpec(
        name="booking_window",
        description="看某科室未来几天的放号情况，返回**还有余号**的日期与时段。",
        params=BookingWindowParams,
        domain=DOMAIN,
        handler=_run,
        when_not=("用户已经指定了具体某一天时改用 get_availability；科室还没确定时不要调。"),
    )


def make_propose_booking_tool(service: BookingService, pending: PendingBookings) -> ToolSpec:
    def _run(**kwargs: Any) -> dict[str, Any]:
        pid = _patient_id(kwargs)
        department = str(kwargs.get("department", "") or "")
        date = _normalize_date(kwargs.get("date"))
        slot = str(kwargs.get("time_slot", "") or "")
        name = str(kwargs.get("patient_name", "") or "")
        try:
            dep = service.resolve_department(department)
            avail = service.availability(dep, date or None)
        except BookingError as e:
            raise ToolError(str(e)) from e
        slots = {str(s["time"]): int(s["remaining"]) for s in avail.slots}
        if slot and slot not in slots:
            raise ToolError(
                f"{dep} 在 {avail.date} 没有「{slot}」这个时段（可选：{'、'.join(slots) or '无'}）"
            )
        if not slot:
            slot = next((t for t, n in slots.items() if n > 0), "")
            if not slot:
                raise ToolError(f"{dep} 在 {avail.date} 已经没有余号了，建议换日期或科室")
        if slots[slot] <= 0:
            raise ToolError(f"{dep} {avail.date} {slot} 已无余号，请换一个时段或日期")
        proposal = pending.propose(
            patient_id=pid,
            department=dep,
            date=avail.date,
            time_slot=slot,
            fee=_fee_of(service, dep),
            patient_name=name,
            remaining=slots[slot],
        )
        return {
            "proposal": proposal.as_payload(),
            "effective_date": avail.date,
            "note": (
                "这只是一张**待确认**的挂号单，号源没有被占用。"
                "必须由患者在界面上点确认才会真正取号；"
                "请把科室、日期、时段、挂号费告诉患者并请他确认。"
            ),
        }

    return ToolSpec(
        name="propose_booking",
        description="生成一张待患者确认的挂号单（**不占号、不产生记录**）。",
        params=ProposeBookingParams,
        domain=DOMAIN,
        handler=_run,
        when_not=(
            "**用户没有明确要求挂号时绝对不要调**；科室未确定时不要调；"
            "还没查过号源余量、也不确定用户要哪一天时，先 get_availability 或 booking_window；"
            "用户只是想了解流程、或还在描述症状时不要调。"
        ),
    )


def make_my_appointments_tool(service: BookingService) -> ToolSpec:
    def _run(**kwargs: Any) -> dict[str, Any]:
        pid = _patient_id(kwargs)
        limit = int(kwargs.get("limit", 5) or 5)
        try:
            rows = service.my_appointments(pid, limit)
        except BookingError as e:
            raise ToolError(str(e)) from e
        return {
            "appointments": rows,
            "n": len(rows),
            "note": "这是患者本人的挂号记录，可以直接告诉他；退号要他在卡片上点「退号」。",
        }

    return ToolSpec(
        name="my_appointments",
        description="查询**当前就诊人本人**已有的挂号记录（含已退号的）。",
        params=MyAppointmentsParams,
        domain=DOMAIN,
        handler=_run,
        when_not=("用户没问到自己的挂号情况时不要调；不要用它去查别人（查不到，也不该查）。"),
    )


def make_book_tool(service: BookingService) -> ToolSpec:
    """**不可逆**的取号工具。工具循环默认不把它给模型，见模块文档。"""

    def _run(**kwargs: Any) -> dict[str, Any]:
        department = str(kwargs.get("department", "") or "")
        date = _normalize_date(kwargs.get("date")) or None
        slot = str(kwargs.get("time_slot", "") or "") or None
        patient_id = _patient_id(kwargs)
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


def clinic_tools(service: BookingService, pending: PendingBookings | None = None) -> list[ToolSpec]:
    """clinic 域的全部工具（按注册顺序）。

        `pending` 为
    one` 时**不注册** `propose_booking`：没有待确认单存储时，那张单子
        无处安放，注册一个必然失败的提议工具只会让模型反复重试。
    """
    tools = [
        make_list_departments_tool(service),
        make_availability_tool(service),
        make_booking_window_tool(service),
        make_my_appointments_tool(service),
    ]
    if pending is not None:
        tools.append(make_propose_booking_tool(service, pending))
    tools.append(make_book_tool(service))
    return tools
