"""号源系统对接契约（`AppointmentProvider`）。

这层存在的唯一理由：**把"挂号"和"号源从哪来"分开**。仓库里当前只有一个实现：

| 实现 | 用途 | 特点 |
|---|---|---|
| `MockAppointmentProvider` | 生产与测试 | 进程内字典，快、无 IO、可注入故障，**重启即丢** |

接真实 HIS 时只需再实现这个 ABC，`BookingService` 与 agent 工具层都不用改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BookingError(Exception):
    """挂号失败：科室不存在 / 号源已满 / 日期非法 / 重复挂号等**可预期**的业务错误。

    刻意与编程错误区分开：这类错误应当被 agent 转成一句人话回复，
    而不是抛 traceback 给用户。
    """


class AppointmentProvider(ABC):
    """号源/预约取号系统的统一接口。"""

    name = "base"

    @abstractmethod
    def list_departments(self) -> list[dict]:
        """可挂号科室：[{code, name, fee}, ...]。"""

    @abstractmethod
    def get_availability(self, department: str, date: str | None = None) -> list[dict]:
        """某科室的号源时段：[{time, remaining}, ...]。"""

    @abstractmethod
    def book(self, department: str, patient_id: str, patient_name: str = "", **kwargs: object) -> dict:
        """取号，返回 {ticket_no, department, date, time, queue_no, fee}。

        实现必须保证**并发下不超卖**——这是号源系统唯一不能妥协的性质。
        """

    def list_appointments(self, patient_id: str, limit: int = 20) -> list[dict]:
        """某患者的挂号记录（新到旧）。默认空实现，实现方按需覆盖。"""
        return []
