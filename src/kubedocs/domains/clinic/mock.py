"""进程内号源替身（`MockAppointmentProvider`）——**只用于测试**。

单独留一个 Mock 而不是让测试直接用 SQLite 版，是为了三件事：**快**（不必为每次挂号
付建库建表的代价）、**可注入故障**（一行代码就能让它抛错，用于测降级路径）、
**契约一致性**（两个实现同属一个 ABC，测试里换实现不该影响断言——
如果影响了，说明有实现偏离了契约）。

它**不是**持久化的：进程退出即丢。真实使用一律走 `SqliteAppointmentProvider`。
"""

from __future__ import annotations

import secrets

from kubedocs.domains.clinic.data.departments import DEPARTMENTS, normalize_department
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError

_DEFAULT_SLOTS = ("上午", "下午")
_SLOT_CAPACITY = {"上午": 10, "下午": 6}


class MockAppointmentProvider(AppointmentProvider):
    """内存号源。可设定初始余号、可设定某科室直接报错。"""

    name = "mock"

    def __init__(
        self, *, capacity: dict[str, int] | None = None, failing_departments: tuple[str, ...] = ()
    ) -> None:
        caps = dict(_SLOT_CAPACITY)
        if capacity:
            caps.update(capacity)
        self._capacity = caps
        # (科室, 日期, 时段) -> 余号
        self._remaining: dict[tuple[str, str, str], int] = {}
        self._appointments: list[dict] = []
        self._failing = tuple(failing_departments)

    # ------------------------------------------------------------------ 内部
    def _key(self, dep: str, day: str, slot: str) -> tuple[str, str, str]:
        return (dep, day, slot)

    def _seed(self, dep: str, day: str) -> None:
        for slot in _DEFAULT_SLOTS:
            self._remaining.setdefault(self._key(dep, day, slot), self._capacity.get(slot, 8))

    # ------------------------------------------------------------------ 契约
    def list_departments(self) -> list[dict]:
        return [
            {"code": f"D{i:02d}", "name": d, "fee": 25 + (i % 5) * 5}
            for i, d in enumerate(DEPARTMENTS, start=1)
        ]

    def get_availability(self, department: str, date: str | None = None) -> list[dict]:
        dep = normalize_department(department)
        day = date or "2026-01-01"
        self._check_failing(dep)
        self._seed(dep, day)
        return [{"time": s, "remaining": self._remaining[self._key(dep, day, s)]} for s in _DEFAULT_SLOTS]

    def book(self, department: str, patient_id: str, patient_name: str = "", **kwargs: object) -> dict:
        dep = normalize_department(department)
        self._check_failing(dep)
        day = str(kwargs.get("date") or "2026-01-01")
        slot = str(kwargs.get("time_slot") or "")
        if slot not in _DEFAULT_SLOTS:
            raise BookingError(f"时段只能是 {'/'.join(_DEFAULT_SLOTS)}，收到 {slot!r}")
        self._seed(dep, day)
        key = self._key(dep, day, slot)
        if self._remaining[key] <= 0:
            raise BookingError(f"{dep}{day}{slot}的号已经挂完了，请换一个时段或日期")
        self._remaining[key] -= 1
        queue_no = 1 + sum(1 for a in self._appointments if a["department"] == dep and a["date"] == day)
        ticket = {
            "ticket_no": "M" + secrets.token_hex(3).upper(),
            "department": dep,
            "date": day,
            "time": slot,
            "queue_no": queue_no,
            "fee": 30,
            "patient_id": patient_id,
            "patient_name": patient_name,
        }
        self._appointments.append(ticket)
        return ticket

    def list_appointments(self, patient_id: str, limit: int = 20) -> list[dict]:
        mine = [a for a in self._appointments if a["patient_id"] == patient_id]
        return list(reversed(mine))[:limit]

    # ------------------------------------------------------------------ 测试钩子
    def _check_failing(self, dep: str) -> None:
        if dep in self._failing:
            raise BookingError(f"号源系统暂时不可用（{dep}）")

    def set_remaining(self, department: str, slot: str, n: int, date: str = "2026-01-01") -> None:
        """直接设定余号，用于构造"只剩 1 个号"这类边界场景。"""
        self._remaining[self._key(normalize_department(department), date, slot)] = n
