"""挂号服务（`BookingService`）：把"分诊结果"接到"号源系统"上。

组合两件事：

1. `AppointmentProvider` —— 号源从哪来（当前是进程内 Mock，接真实 HIS 时换实现）
2. 科室归一化 —— agent 说出的科室名要能对上号源系统的科室名

**这一层不做自然语言理解**：它只接受结构化参数（科室、日期、时段）。
把"用户这句话想挂哪个科"交给 agent 的工具调用去决定，是刻意的分层——
让 NLU 与号源事务各自可测。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date

from kubedocs.domains.clinic.data.departments import normalize_department
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError


def _display_date(date: str | None, slots: list[dict]) -> str:
    """`Availability.date` 的三级兜底：显式传入 > 时段自带的 `date` > 今天。

    `date` 是**展示字段**（工具层直接回给模型/前端），空串会被渲染成"没有日期"，所以
    "provider 没给时段"这种正常情况也必须回一个具体日期。`date=None` 时号源的实际口径
    就是"今天"（见 `PgAppointmentProvider.get_availability`），兜底取今天才和取到的号一致。

    时段自带的 `date` 排在"今天"之前：由号源解析出的日期比调用方的假设更可信。
    """
    if date:
        return str(date)
    for slot in slots:
        if slot.get("date"):
            return str(slot["date"])
    return _date.today().isoformat()


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
        return Availability(department=dep, date=_display_date(date, slots), slots=slots)

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

    def availability_window(self, department: str, days: int = 7) -> list[dict]:
        """整个放号窗口的余量（`[{date, slots}]`），给"哪天还有号"用。

        与 `availability` 分开是刻意的：`availability` 的契约是"某一天"，而"哪天有号"
        要跨天看；用两次单日查询去拼会多两轮数据库往返，也会让上层自己写日期循环。
        号源实现不支持时报一句人话——模型据此改用单日查询，而不是收到 AttributeError。
        """
        dep = self.resolve_department(department)
        window = getattr(self.provider, "availability_window", None)
        if window is None:
            raise BookingError("当前号源实现不支持查询放号窗口，请改用 get_availability 逐天查询")
        try:
            # 显式标注返回值：`window` 是 `getattr` 取出来的（类型是 Any），
            # 直接 return 会把 Any 漏进本方法的声明返回类型。
            rows: list[dict] = window(dep, days)
            return rows
        except NotImplementedError as e:
            # ABC 给可选能力留了默认实现（抛 NotImplementedError），所以"没覆盖"这件事
            # 只能靠异常认出来——`getattr(...) is None` 对 ABC 的子类永远不成立。
            raise BookingError("当前号源实现不支持查询放号窗口，请改用 get_availability 逐天查询") from e

    def cancel(self, ticket_no: str, patient_id: str) -> dict:
        """退号，返回号源给出的取消结果（`{ticket_no, department, date, time, status, remaining}`）。

        **回 dict 而不是 None**：退号是给前端看的操作（"已退号、余量已释放"），而释放后的
        余量只有号源实现知道。

        "不支持取消"有两种形态，都要接住：provider 没有这个方法（鸭子类型），或它继承了
        ABC 的默认实现而抛 `NotImplementedError`（`MockAppointmentProvider` 就是后者）。
        **只判 `getattr(...) is None` 是不够的**——可选能力一旦在 ABC 上有了默认实现，
        `getattr` 永远拿得到东西。

        权限在号源层（只有同一 `patient_id` 能退自己的号），这一层不做也不该做鉴权：
        它拿不到"当前登录用户"，硬判会把唯一真相源拆成两半。
        """
        cancel = getattr(self.provider, "cancel", None)
        if cancel is None:
            raise BookingError("当前号源实现不支持取消")
        try:
            result = cancel(ticket_no, patient_id)
        except NotImplementedError as e:
            raise BookingError("当前号源实现不支持取消") from e
        # 兼容"只做事、不返回内容"的实现（旧契约的返回是 None）
        return dict(result) if result is not None else {}
