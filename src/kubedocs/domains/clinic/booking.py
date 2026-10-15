"""挂号服务（`BookingService`）：把"分诊结果"接到"号源系统"上。

组合两件事：

1. `AppointmentProvider` —— 号源从哪来（可换 SQLite / Mock / 真实 HIS）
2. 科室归一化 —— agent 说出的科室名要能对上号源系统的科室名

**这一层不做自然语言理解**：它只接受结构化参数（科室、日期、时段）。
把"用户这句话想挂哪个科"交给 agent 的工具调用去决定，是刻意的分层——
让 NLU 与号源事务各自可测。
"""

from __future__ import annotations

from dataclasses import dataclass

from kubedocs.domains.clinic.data.departments import normalize_department
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError


@dataclass(frozen=True)
class Availability:
    department: str
    date: str
    slots: list[dict]

    @property
    def total_remaining(self) -> int:
        return sum(int(s.get("remaining", 0)) for s in self.slots)

    def describe(self) -> str:
        if not self.slots:
            return f"{self.department} {self.date} 暂无可约时段"
        parts = [f"{s['time']}（余 {s['remaining']}）" for s in self.slots]
        return f"{self.department} {self.date}：" + "、".join(parts)


class BookingService:
    """号源的业务层。所有方法都会把 provider 的业务错误原样上抛（`BookingError`）。"""

    def __init__(self, provider: AppointmentProvider) -> None:
        self.provider = provider

    # ------------------------------------------------------------------ 科室
    def departments(self) -> list[dict]:
        return self.provider.list_departments()

    def resolve_department(self, name: str) -> str:
        """把口语科室名归一化到号源系统的科室名；识别不出就报错，**不猜**。"""
        dep = normalize_department(name or "")
        if not dep:
            raise BookingError(f"没能识别出科室名：{name!r}")
        known = {d["name"] for d in self.provider.list_departments()}
        if dep not in known:
            raise BookingError(f"{dep}不在可挂号科室里（可选：{'、'.join(sorted(known))}）")
        return dep

    # ------------------------------------------------------------------ 号源
    def availability(self, department: str, date: str | None = None) -> Availability:
        dep = self.resolve_department(department)
        slots = self.provider.get_availability(dep, date)
        day = date or slots and self._guess_date(slots) or ""
        return Availability(department=dep, date=str(day), slots=slots)

    @staticmethod
    def _guess_date(_slots: list[dict]) -> str:
        from datetime import date as _date

        return _date.today().isoformat()

    def first_available_slot(self, department: str, date: str | None = None) -> str | None:
        """该科室当天的第一个有余号时段；都没有则 None（agent 据此改推荐日期）。"""
        avail = self.availability(department, date)
        for s in avail.slots:
            if int(s.get("remaining", 0)) > 0:
                return str(s["time"])
        return None

    # ------------------------------------------------------------------ 取号
    def book(
        self,
        department: str,
        patient_id: str,
        *,
        date: str | None = None,
        time_slot: str | None = None,
        patient_name: str = "",
    ) -> dict:
        """取号。时段缺省时自动挑第一个有余号的时段（少一轮往返）。"""
        dep = self.resolve_department(department)
        slot = time_slot or self.first_available_slot(dep, date)
        if not slot:
            raise BookingError(f"{dep}在 {date or '今天'} 已经没有余号了，建议换日期或科室")
        return self.provider.book(dep, patient_id, patient_name, date=date, time_slot=slot)

    def my_appointments(self, patient_id: str, limit: int = 20) -> list[dict]:
        return self.provider.list_appointments(patient_id, limit)

    def cancel(self, ticket_no: str, patient_id: str) -> None:
        cancel = getattr(self.provider, "cancel", None)
        if cancel is None:
            raise BookingError("当前号源实现不支持取消")
        cancel(ticket_no, patient_id)


def describe_ticket(ticket: dict) -> str:
    """把取号结果渲染成一句人话（agent 回复用）。"""
    return (
        f"挂号成功。取号单号 {ticket['ticket_no']}，"
        f"{ticket['department']} {ticket['date']} {ticket['time']}，"
        f"排队号 {ticket['queue_no']}，挂号费 {ticket.get('fee', 0)} 元。"
    )
