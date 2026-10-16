"""进程内号源实现（`MockAppointmentProvider`）——**离线回退与测试用的号源**。

默认号源是 `PgAppointmentProvider`（`pg_provider.py`，Postgres 的 `booking` schema）；
数据库不可达时由 `domains/clinic/assembly.py` 降级到这一个。号源放在进程内存里：
快、无 IO、可注入故障（`failing_departments`），代价是**进程退出即丢**，
也没有并发/持久化保证（那两条是对真实实现的要求，见 `AppointmentProvider.book`）。

接真实号源（HIS）时，实现 `AppointmentProvider` 的同一份契约再换装配点即可，
上层（`BookingService` 与 agent 工具）不用改。
"""

from __future__ import annotations

import secrets

from kubedocs.domains.clinic.data.departments import DEPARTMENTS, department_fee, normalize_department
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError

_DEFAULT_SLOTS = ("上午", "下午")
_SLOT_CAPACITY = {"上午": 10, "下午": 6}
_STATUS_PAID = "paid"
_STATUS_CANCELLED = "cancelled"


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
        """可挂号科室与**挂号费**。

        费用取自 `data/departments.py` 的 `department_fee`——与 Postgres 号源同一个真相源。
        这里曾经是 `25 + (i % 5) * 5` 算出来的，而 `book()` 又写死 30：于是"报价"与
        "实收"能在同一次挂号里对不上，界面上的费用与挂号后的费用差 5 块钱。
        """
        return [
            {"code": f"D{i:02d}", "name": d, "fee": department_fee(d)}
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
            "fee": department_fee(dep),
            "patient_id": patient_id,
            "patient_name": patient_name,
            # 与 Postgres 号源同一个状态位取值（假支付：确认即视为已付），
            # 否则界面上的"退号"按钮与状态显示会随后端不同而不同。
            "status": _STATUS_PAID,
        }
        self._appointments.append(ticket)
        return ticket

    def list_appointments(self, patient_id: str, limit: int = 20) -> list[dict]:
        """某患者的挂号记录（新到旧）。**退掉的号仍然在列表里**（那是历史）。"""
        mine = [a for a in self._appointments if a["patient_id"] == patient_id]
        return list(reversed(mine))[:limit]

    def cancel(self, ticket_no: str, patient_id: str) -> dict:
        """退号：标记作废并立即回补余量。规则与 Postgres 号源一致。

        四条与真实号源对齐的约束：只有本人能退（否则凭据号就是一张人人可用的作废票）、
        重复退号报错（连点两次不该凭空多出一个号）、退号**不判停诊**、余量不越过容量。

        这里没有并发保护（内存结构、单进程），这正是它是**回退实现**而不是生产实现的原因
        ——`PgAppointmentProvider` 用的是带条件的 UPDATE，那才是并发下不超卖的做法。
        """
        ticket = str(ticket_no or "").strip()
        if not ticket:
            raise BookingError("退号必须带凭据号")
        pid = str(patient_id or "").strip()
        if not pid:
            raise BookingError("退号必须带 patient_id（只有本人能退自己的号）")
        for row in self._appointments:
            if row["ticket_no"] != ticket:
                continue
            if row["patient_id"] != pid:
                raise BookingError(f"凭据号 {ticket} 不是你的挂号记录，不能退别人的号")
            if row.get("status") == _STATUS_CANCELLED:
                raise BookingError(f"凭据号 {ticket} 已经退过了，不能重复退号")
            row["status"] = _STATUS_CANCELLED
            key = self._key(row["department"], row["date"], row["time"])
            self._seed(row["department"], row["date"])
            self._remaining[key] = min(self._remaining[key] + 1, self._capacity.get(row["time"], 8))
            return {
                "ticket_no": ticket,
                "department": row["department"],
                "date": row["date"],
                "time": row["time"],
                "status": _STATUS_CANCELLED,
                "remaining": self._remaining[key],
            }
        raise BookingError(f"没有找到凭据号 {ticket} 的挂号记录")

    # ------------------------------------------------------------------ 测试钩子
    def _check_failing(self, dep: str) -> None:
        if dep in self._failing:
            raise BookingError(f"号源系统暂时不可用（{dep}）")

    def set_remaining(self, department: str, slot: str, n: int, date: str = "2026-01-01") -> None:
        """直接设定余号，用于构造"只剩 1 个号"这类边界场景。"""
        self._remaining[self._key(normalize_department(department), date, slot)] = n
