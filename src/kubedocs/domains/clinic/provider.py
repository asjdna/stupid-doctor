"""号源系统对接契约（`AppointmentProvider`）。

这层存在的唯一理由：**把"挂号"和"号源从哪来"分开**。仓库里有两个实现：

| 实现 | 用途 | 特点 |
|---|---|---|
| `PgAppointmentProvider`（`pg_provider.py`） | **默认**：`KUBEDOCS_STORE=pgvector` 时的号源 | Postgres `booking` schema，跨重启、跨副本；占号是一句条件更新，**并发不超卖** |
| `MockAppointmentProvider`（`mock.py`） | 离线/测试，以及数据库不可达时的降级 | 进程内字典，快、无 IO、**重启即丢** |

接真实 HIS 时只需再实现这个 ABC，`BookingService` 与 agent 工具层都不用改。

## 契约面

必答的三个方法（`list_departments` / `get_availability` / `book`）由 `BookingService`
和 agent 工具直接调用。另外三个是**可选能力**，有默认实现，实现方按需覆盖——
`BookingService` 用 `getattr` 探测，缺了就报一句人话而不是 AttributeError：

- `list_appointments` / `cancel`：患者自助查号与退号（`/consult` 的卡片按钮走这里）
- `availability_window` / `generate_schedules` / `close_slot` / `adjust_capacity`：
  排班与号量管理（CLI `kubedocs schedule` 走这里）

**号源只到科室**（科室 × 日期 × 上午/下午），没有医生维度——这是既定的产品决定。
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
        """某患者的挂号记录（新到旧）。默认空列表，实现方按需覆盖。"""
        return []

    def cancel(self, ticket_no: str, patient_id: str) -> dict:
        """退号。默认不支持——`BookingService.cancel` 会把它转成一句人话。"""
        raise NotImplementedError

    def availability_window(self, department: str, days: int = 7) -> list[dict]:
        """某科室整个放号窗口的余量：`[{date, slots: [...]}]`。默认不支持。"""
        raise NotImplementedError

    def generate_schedules(self, days: int = 7, per_period: int = 10, start: object = None) -> int:
        """生成排班，返回**本次新增**行数（幂等）。默认不支持。"""
        raise NotImplementedError
