"""持久化号源实现（`PgAppointmentProvider`）：Postgres 落库、按科室排班、并发不超卖。

号源**自建**：我们自己排班、自己占号。**号源只到科室**（科室 × 日期 × 上午/下午），
不做医生维度。本模块只做**后端数据层与业务服务**——写操作只由服务端执行，模型永远拿不到
写工具；工具面 / HTTP / 前端是下一块，不在这里。

## schema 与表

schema 固定 `booking`，与 pgvector 存知识卡片向量的 `clinic` schema **完全无关**
（两套 schema 只是恰好在同一个库实例里）。

| 表 | 关键列 | 约束 |
|---|---|---|
| `booking.departments` | `code` PK、`name` UNIQUE、`fee` | 挂号费是 `data/departments.py` 的快照（占号时按这张表取 fee） |
| `booking.schedules` | `(department, date, period)` UNIQUE、`capacity`、`remaining`、`closed` | `remaining >= 0`、`remaining <= capacity` |
| `booking.appointments` | `ticket_no` UNIQUE、`idempotency_key` UNIQUE | `(department, date, period, queue_no)` UNIQUE、`status ∈ {pending_payment, paid, cancelled}` |

`department` 列存的是**科室名**而不是 code：`AppointmentProvider` 契约（`list_departments`
/ `get_availability` / `book`）与 `BookingService.resolve_department` 全都以**名字**为键
（`normalize_department` 的规范值），存名字才能让"归一化后的名字"直接当外键，不必在每一层
再翻译一次 code。`code` 只用于展示与对外暴露。

## 排班与放号窗口

默认规则：**每科室每天上午/下午各 10 个号，放号窗口 = 今天起 7 天**，且**可手工调整**
（`adjust_capacity` 改量、`close_slot` 停诊）。生成排班**幂等**：`INSERT ... ON CONFLICT
(department, date, period) DO NOTHING`，**绝不覆盖已有 `remaining`**——否则重跑一次就会把
别人已经占掉的号重置回去（超卖的标准成因）。

`close_slot` 只影响**新预约**：已挂出的号不作废、退号照常回补余量；`remaining` 也不动，
所以开诊后余量数字仍然是对的。退号规则最简：**随时可退、立即释放余量、不改签、
不限重复预约**。

## 挂号费与假支付

挂号费沿用**科室表**：真相源是 `data/departments.py` 的 `department_fee(name)`
（呼吸内科 35、其余 `DEFAULT_FEE` 30），`ensure_schema()` 把它种进
`booking.departments.fee`，占号时按表取——所以运营改价改表就对，不必重启进程。
假支付：保留 `pending_payment` / `paid` 状态位与 `expires_at` 字段，但当前"**确认即视为
已付**"，所以 `book()` 直接落 `status = 'paid'`、`expires_at = NULL`；`pending_payment`
是给后续"先占号再付、超时回收"留的状态位，**本模块没有超时回收器**（那需要一个后台
清扫任务，属于另一块）。

## 并发与一致性

见 `book()` 的 docstring：占号是**一个事务里的一句条件更新**（`UPDATE ... WHERE remaining > 0
AND NOT closed RETURNING remaining`），**不是**"先 SELECT 再 UPDATE"。

## 连接

用 `psycopg` + **连接池**（`psycopg_pool`），与 `store/pgvector_store.py` 同一套写法：
DSN / 池参数 / `close()` / 上下文管理器的形状都照抄，池名不同。差别只有一处：pgvector 侧
schema 初始化**必须**走直连（它的 `configure` 回调要 `register_vector`，那要求
`vector` 类型已存在，用池去建扩展会自己锁死自己），而这里没有任何扩展或类型注册，
所以 `ensure_schema()` 直接走池。

`ensure_schema()` 里的表名一律通过 `sql.Identifier` 注入（见 `_q`），值一律 `%s` 参数绑定。
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import secrets
from typing import Any

import psycopg
from psycopg import sql

from kubedocs.config import Settings
from kubedocs.domains.clinic.data.departments import DEPARTMENTS, department_fee, normalize_department
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError

# schema 名固定 booking（可用构造参数覆盖，测试用随机后缀的独立 schema）。
BOOKING_SCHEMA = "booking"

# 时段只有一个来源：排班表的 `period` 列。值域写死成中文，与 `MockAppointmentProvider`
# 的 `time_slot` 完全一致（agent 工具层传下来的就是这两个词）。
PERIODS = ("上午", "下午")

# 默认排班规则：每科室每天上午/下午各 10 个号、放号窗口今天起 7 天。
DEFAULT_PERIOD_CAPACITY = 10
DEFAULT_WINDOW_DAYS = 7

# 挂号费不在这里：`data/departments.py` 是唯一真相源（`department_fee(name)`），
# 库里的 `booking.departments.fee` 只是它的快照。

# 状态位。当前只写 PAID（假支付：确认即视为已付），PENDING 留给后续的"先占号再付"。
STATUS_PENDING_PAYMENT = "pending_payment"
STATUS_PAID = "paid"
STATUS_CANCELLED = "cancelled"
STATUSES = (STATUS_PENDING_PAYMENT, STATUS_PAID, STATUS_CANCELLED)

# 凭据号：8 位大写字母数字，**去掉 I/O/0/1**——号是要口头报给患者、也要印在单子上的，
# "到底是 I 还是一、是 O 还是零"是现场最常见的错。32 个符号 × 8 位 = 40 bit，
# 配合 `ticket_no` 上的 UNIQUE 约束与撞号重试，碰撞不可能成为可见故障。
_TICKET_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_TICKET_LEN = 8
_TICKET_ATTEMPTS = 3

# 与 `store/pgvector_store.py` 的同名规则一致：schema 名进不了参数绑定，只能白名单校验。
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = logging.getLogger(__name__)


def _ddl() -> tuple[str, ...]:
    """建表语句（逐条执行，幂等）。

    写成函数而不是模块常量，只是为了把 `PERIODS` / `STATUSES` 拼进 CHECK 里，避免
    Python 侧的值域与库里的 CHECK 各写一份、哪天改一处忘另一处。
    """
    periods = ", ".join(f"'{p}'" for p in PERIODS)
    statuses = ", ".join(f"'{s}'" for s in STATUSES)
    return (
        "CREATE TABLE IF NOT EXISTS {departments} ("
        "  code text PRIMARY KEY,"
        "  name text NOT NULL,"
        "  fee integer NOT NULL CHECK (fee >= 0),"
        "  CONSTRAINT uq_booking_departments_name UNIQUE (name))",
        # `date` 是 Postgres 的保留字之一（不是保留关键字，但作列名可读性最好时仍不必加引号）
        "CREATE TABLE IF NOT EXISTS {schedules} ("
        "  id bigserial PRIMARY KEY,"
        "  department text NOT NULL REFERENCES {departments}(name) ON DELETE CASCADE,"
        "  date date NOT NULL,"
        f"  period text NOT NULL CHECK (period IN ({periods})),"
        "  capacity integer NOT NULL CHECK (capacity >= 0),"
        "  remaining integer NOT NULL CHECK (remaining >= 0),"
        "  closed boolean NOT NULL DEFAULT false,"
        "  CONSTRAINT uq_booking_schedules_slot UNIQUE (department, date, period),"
        "  CONSTRAINT ck_booking_schedules_remaining_le_capacity CHECK (remaining <= capacity))",
        "CREATE INDEX IF NOT EXISTS idx_booking_schedules_date ON {schedules} (date)",
        "CREATE TABLE IF NOT EXISTS {appointments} ("
        "  id bigserial PRIMARY KEY,"
        "  ticket_no text NOT NULL,"
        "  department text NOT NULL,"
        "  date date NOT NULL,"
        f"  period text NOT NULL CHECK (period IN ({periods})),"
        "  patient_id text NOT NULL,"
        "  patient_name text NOT NULL DEFAULT '',"
        "  queue_no integer NOT NULL,"
        "  fee integer NOT NULL,"
        f"  status text NOT NULL CHECK (status IN ({statuses})),"
        "  idempotency_key text,"
        "  created_at timestamptz NOT NULL DEFAULT now(),"
        "  updated_at timestamptz NOT NULL DEFAULT now(),"
        "  expires_at timestamptz,"
        "  CONSTRAINT uq_booking_appointments_ticket_no UNIQUE (ticket_no),"
        # 队列号按「科室 + 日期 + **时段**」唯一：上午 1..N、下午也从 1 起（真实医院的叫号
        # 口径）。**必须把 period 写进约束和取号 SQL 两处**：只改一处的话，跨时段并发占号
        # 时两个事务各自算 MAX 会拿到同一个号（实测重号），而约束只会在写入时才炸。
        "  CONSTRAINT uq_booking_appointments_queue UNIQUE (department, date, period, queue_no),"
        # 幂等键可空，且**只有非空时**参与唯一性（Postgres 的 UNIQUE 索引里 NULL 互不冲突），
        # 因此不传幂等键的调用完全不受影响。
        "  CONSTRAINT uq_booking_appointments_idempotency_key UNIQUE (idempotency_key))",
        "CREATE INDEX IF NOT EXISTS idx_booking_appointments_patient "
        "ON {appointments} (patient_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_booking_appointments_slot ON {appointments} (department, date)",
    )


# 取单据时统一按这个列序（`_row_to_ticket` 与它的解包一一对应）。
# 列名交给 `sql.Identifier` 引用而不是手写拼接：既走上驱动提供的引用规则，也让"列清单"
# 只有一份，SELECT 与 INSERT ... RETURNING 共用。
_APPOINTMENT_COLUMNS = (
    "id",
    "ticket_no",
    "department",
    "date",
    "period",
    "patient_id",
    "patient_name",
    "queue_no",
    "fee",
    "status",
    "created_at",
    "updated_at",
    "expires_at",
)
_COLS = sql.SQL(", ").join(sql.Identifier(c) for c in _APPOINTMENT_COLUMNS)

# INSERT 的列序**不等于**取回的列序：`id` 由序列生成、`idempotency_key` 的位置也不同，
# 所以两者必须分开声明。VALUES 占位符的次序与这里逐字对应（末三个是 now()/now()/NULL）。
_APPOINTMENT_INSERT_COLUMNS = (
    "ticket_no",
    "department",
    "date",
    "period",
    "patient_id",
    "patient_name",
    "queue_no",
    "fee",
    "status",
    "idempotency_key",
    "created_at",
    "updated_at",
    "expires_at",
)
_INSERT_COLS = sql.SQL(", ").join(sql.Identifier(c) for c in _APPOINTMENT_INSERT_COLUMNS)


def probe(dsn: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    """探测数据库是否可用（只 `SELECT 1`，不建任何东西）。

    存在的理由与 `store/pgvector_store.py:probe` 相同：测试要能区分"库连不上（环境问题，
    应 skip）"与"代码错了（应失败）"。**差别是这里不要求 pgvector 扩展**——号源表只用
    Postgres 核心功能，一个没装 pgvector 的实例照样能挂号。
    """
    try:
        with psycopg.connect(dsn, connect_timeout=int(max(1, timeout))) as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - 探测就是要吞掉一切异常并转成原因字符串
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def _new_ticket_no() -> str:
    """生成一个凭据号（8 位、无易混字符）。"""
    return "".join(secrets.choice(_TICKET_ALPHABET) for _ in range(_TICKET_LEN))


def _as_date(value: object, *, field: str = "date") -> dt.date | None:
    """把 `str` / `date` / `datetime` 统一成 `date`；空值返回 None（由调用方决定默认值）。"""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as e:
        raise BookingError(f"{field} 必须是 YYYY-MM-DD，收到 {value!r}") from e


def _as_period(value: object) -> str:
    period = str(value or "").strip()
    if period not in PERIODS:
        raise BookingError(f"时段只能是 {'/'.join(PERIODS)}，收到 {value!r}")
    return period


def _as_idempotency_key(value: object) -> str | None:
    key = str(value or "").strip()
    return key or None


def _row_to_ticket(row: tuple[Any, ...]) -> dict:
    """库行 → 取号单 dict。

    `time` 这个键名是**契约**（`AppointmentProvider.book` 与 `MockAppointmentProvider`
    都叫它 `time`），值就是「上午/下午」；库里叫 `period`，不要在这里改名。
    """
    (
        _id,
        ticket_no,
        department,
        day,
        period,
        patient_id,
        patient_name,
        queue_no,
        fee,
        status,
        created_at,
        _updated_at,
        expires_at,
    ) = row
    return {
        "ticket_no": str(ticket_no),
        "department": str(department),
        "date": day.isoformat(),
        "time": str(period),
        "queue_no": int(queue_no),
        "fee": int(fee),
        "patient_id": str(patient_id),
        "patient_name": str(patient_name or ""),
        "status": str(status),
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat() if expires_at else None,
    }


class PgAppointmentProvider(AppointmentProvider):
    """Postgres 号源：`booking` schema 落库、按科室排班、并发不超卖。"""

    name = "postgres"

    def __init__(
        self,
        dsn: str | None = None,
        *,
        schema: str = BOOKING_SCHEMA,
        pool_min: int = 1,
        pool_max: int = 10,
    ) -> None:
        """`dsn` 缺省取 `Settings().database_url`（即 `KUBEDOCS_DATABASE_URL`）。

        `schema` 非法时抛 `ValueError` 而**不是** `BookingError`：后者是"可以讲给患者听的
        业务错误"，而一个拼错的 schema 名是部署事故，被 agent 吞成一句人话才是真危险。
        """
        if not _IDENT_RE.match(schema):
            raise ValueError(f"非法的 schema 名 {schema!r}：只允许 [A-Za-z_][A-Za-z0-9_]*")
        self.dsn = dsn or Settings().database_url
        self.schema = schema
        self._tbl = {
            name: sql.Identifier(schema, name) for name in ("departments", "schedules", "appointments")
        }
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover - 取决于环境
            raise RuntimeError("Postgres 号源需要 psycopg-pool：pip install 'psycopg[pool]'") from e
        self._pool = ConnectionPool(
            conninfo=self.dsn,
            min_size=pool_min,
            max_size=pool_max,
            open=False,
            # 数据库不可达时不该让调用方挂到默认的 30s
            timeout=10.0,
            name="kubedocs-booking",
        )
        self._pool.open(wait=False)
        self.ensure_schema()

    # ------------------------------------------------------------------ 基础设施
    def _q(self, template: str, **extra: object) -> sql.Composed:
        """把 `{departments}` / `{schedules}` / `{appointments}` 换成 schema 限定的标识符。

        与 `PgVectorStore._q` 同一套思路：表名不是用户输入，但把引用规则交给驱动
        （`sql.Identifier`）比口头保证"我拼的字符串安全"更可靠。
        """
        return sql.SQL(template).format(**self._tbl, **extra)

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> PgAppointmentProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ schema
    def ensure_schema(self) -> None:
        """建 schema / 表 / 索引 + 种科室（**幂等**，可反复调用）。

        科室用 `INSERT ... ON CONFLICT DO NOTHING`（不带冲突目标）：`code` 主键与 `name`
        唯一键任意一个撞上都跳过。**故意不 DO UPDATE**——`fee` 是运营可能要手工调的数据
        （例如某科室涨价），每次开进程都覆盖回初始值会把手工改动抹掉。要改费用直接改表。
        """
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
            for stmt in _ddl():
                conn.execute(self._q(stmt))
            with conn.cursor() as cur:
                cur.executemany(
                    self._q(
                        "INSERT INTO {departments}(code, name, fee) VALUES(%s, %s, %s) ON CONFLICT DO NOTHING"
                    ),
                    [
                        (f"D{i:02d}", name, department_fee(name))
                        for i, name in enumerate(DEPARTMENTS, start=1)
                    ],
                )

    # ------------------------------------------------------------------ 排班
    def generate_schedules(
        self,
        days: int = DEFAULT_WINDOW_DAYS,
        per_period: int = DEFAULT_PERIOD_CAPACITY,
        start: object = None,
    ) -> int:
        """生成放号窗口内的排班（**幂等**），返回**本次新增**的行数。

        默认：`days=7`、`per_period=10`、`start=今天` → 每科室 × 7 天 × 上午/下午。
        `start` 收 `YYYY-MM-DD` 字符串或 `date`（**故意标成 `object`**：ABC 的可选能力
        就是这么声明的，收窄参数类型会让子类不兼容父类，mypy 会直接报错）。

        **绝不能覆盖已有 `remaining`**：`ON CONFLICT (department, date, period) DO NOTHING`
        意味着"已有排班一律不动"——重跑只补缺失的行。若改成 `DO UPDATE SET remaining =
        EXCLUDED.remaining`，第二次跑就会把已经占掉的号重置成满额，等于**凭空多卖一遍**
        （这是超卖最经典的成因，比并发 bug 更常见）。

        科室清单取**表里现有的行**而不是 `DEPARTMENTS` 常量：往科室表里手工加了一个科室，
        重整排班时它就该一起排上。

        返回值是新增行数（窗口已排满时第二次跑返回 0），便于上层记日志/判断"是否真生成"。
        """
        if days <= 0:
            raise BookingError(f"排班天数必须是正整数，收到 {days!r}")
        if per_period < 0:
            raise BookingError(f"每时段号量不能是负数，收到 {per_period!r}")
        first = _as_date(start, field="start") or self._today()
        window = [first + dt.timedelta(days=i) for i in range(days)]
        if not self._department_names():
            raise BookingError("号源库里还没有科室，请先跑 ensure_schema() 初始化科室表")
        with self._pool.connection() as conn, conn.transaction():
            # 一句 SQL 生成整个窗口：`departments × 日期数组 × 时段数组` 的笛卡尔积，
            # 冲突即跳过。相比在 Python 里拼 N 行再 executemany，这里能拿到**准确的**
            # rowcount（= 本次真正新增的行数），幂等性也就有了可观测的输出。
            cur = conn.execute(
                self._q(
                    "INSERT INTO {schedules}(department, date, period, capacity, remaining) "
                    "SELECT d.name, w.day, p.period, %s, %s "
                    "FROM {departments} d "
                    "CROSS JOIN unnest(%s::date[]) AS w(day) "
                    "CROSS JOIN unnest(%s::text[]) AS p(period) "
                    "ON CONFLICT (department, date, period) DO NOTHING"
                ),
                (per_period, per_period, window, list(PERIODS)),
            )
            return int(cur.rowcount)

    def close_slot(
        self, department: str, date: str | dt.date | None = None, period: str = "", *, closed: bool = True
    ) -> None:
        """停诊 / 开诊（**只影响新预约**）。

        `closed = true` 后 `book()` 的占号条件更新会因 `NOT closed` 失败，但：
        已挂出的号**不作废**、`remaining` **不动**、退号照常回补余量。所以开诊后余量数字
        仍然是准的，也不需要"停诊时清空余量"这类补偿逻辑。

        `closed=False` 是开诊；它只翻标志位，**不会**把 `remaining` 重置回 `capacity`
        （那会把停诊期间退掉的号一起复活，等于多卖）。
        """
        dep, day, per = self._slot_args(department, date, period)
        with self._pool.connection() as conn, conn.transaction():
            written = conn.execute(
                self._q(
                    "UPDATE {schedules} SET closed = %s WHERE department = %s AND date = %s AND period = %s"
                ),
                (bool(closed), dep, day, per),
            ).rowcount
        if not written:
            action = "停诊" if closed else "开诊"
            raise BookingError(f"{dep} {day.isoformat()} {per}没有排班，无法{action}（请先生成排班）")

    def adjust_capacity(self, department: str, date: str | dt.date, period: str, capacity: int) -> dict:
        """手工改号量，返回 `{capacity, remaining, booked}`（改后）。

        语义：把 `capacity` 设成新值，`remaining` 按**已挂出多少号**平移——
        `remaining = max(0, 新号量 - 已挂出)`。调小到低于已挂出数时余量归零
        （`CHECK (remaining >= 0)` 兜底），**不会**自动取消已有挂号；此时
        `capacity < 已挂出`，退号仍能逐张回补（`cancel` 的条件是 `remaining < capacity`），
        所以不会出现"退不掉的号"。

        这里的"已挂出"**必须去数单据**（`status <> 'cancelled'`），不能拿
        `capacity - remaining` 当它：容量调小到低于已挂出数时余量被夹到 0，那个差值就退化
        成容量本身（信息已丢），再调大就会凭空多放号——一个"已挂 3 个、容量被调到 1"的时段
        调回 10 会算出 9 个余量，**多卖 2 个**。

        `FOR UPDATE` 是必需的：这句不是 `book()` 那种"判定即扣减"的单语句写法，只能显式把
        "读 → 算 → 写"整段与并发占号/退号串行化。去掉它就是一个可复现的超卖窗口。
        """
        dep, day, per = self._slot_args(department, date, period)
        new_capacity = int(capacity)
        if new_capacity < 0:
            raise BookingError(f"号量不能是负数，收到 {capacity!r}")
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                self._q(
                    "SELECT capacity FROM {schedules} "
                    "WHERE department = %s AND date = %s AND period = %s FOR UPDATE"
                ),
                (dep, day, per),
            ).fetchone()
            if row is None:
                raise BookingError(f"{dep} {day.isoformat()} {per}没有排班，无法改号量")
            old_capacity = int(row[0])
            booked = int(
                conn.execute(
                    self._q(
                        "SELECT COUNT(*) FROM {appointments} "
                        "WHERE department = %s AND date = %s AND period = %s AND status <> %s"
                    ),
                    (dep, day, per, STATUS_CANCELLED),
                ).fetchone()[0]
            )
            new_remaining = max(0, new_capacity - booked)
            conn.execute(
                self._q(
                    "UPDATE {schedules} SET capacity = %s, remaining = %s "
                    "WHERE department = %s AND date = %s AND period = %s"
                ),
                (new_capacity, new_remaining, dep, day, per),
            )
            if booked > new_capacity:
                logger.warning(
                    "把 %s %s %s 的号量从 %s 调到 %s 时已挂出 %s 个号：余量归零，"
                    "超出的 %s 个号只能靠已有挂号逐个退掉，不会自动取消",
                    dep,
                    day.isoformat(),
                    per,
                    old_capacity,
                    new_capacity,
                    booked,
                    booked - new_capacity,
                )
            return {"capacity": new_capacity, "remaining": new_remaining, "booked": booked}

    def availability_window(self, department: str, days: int = DEFAULT_WINDOW_DAYS) -> list[dict]:
        """某科室整个放号窗口的余量（**非契约方法**，给上层做"换哪个日期"用）。

        返回 `[{date, slots: [...]}]`，日期升序。`get_availability` 是"某一天"的契约，
        而"哪天还有号"需要跨天看一眼——把两次查询留给上层反而会多两轮往返。
        """
        dep = self._require_department(department)
        if days <= 0:
            raise BookingError(f"窗口天数必须是正整数，收到 {days!r}")
        first = self._today()
        last = first + dt.timedelta(days=days - 1)
        with self._pool.connection() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT date, period, capacity, remaining, closed FROM {schedules} "
                    "WHERE department = %s AND date BETWEEN %s AND %s"
                ),
                (dep, first, last),
            ).fetchall()
        by_day: dict[dt.date, list[dict]] = {}
        for day, period, capacity, remaining, closed in rows:
            by_day.setdefault(day, []).append(_slot_dict(day, period, capacity, remaining, closed))
        return [{"date": str(day), "slots": _sort_slots(slots)} for day, slots in sorted(by_day.items())]

    # ------------------------------------------------------------------ 契约：科室
    def list_departments(self) -> list[dict]:
        """可挂号科室：`[{code, name, fee}, ...]`（按 code 升序，与 Mock 的 D01… 一致）。"""
        with self._pool.connection() as conn:
            rows = conn.execute(self._q("SELECT code, name, fee FROM {departments} ORDER BY code")).fetchall()
        return [{"code": str(code), "name": str(name), "fee": int(fee)} for code, name, fee in rows]

    # ------------------------------------------------------------------ 契约：号源
    def get_availability(self, department: str, date: str | dt.date | None = None) -> list[dict]:
        """某科室某天的号源：`[{time, remaining}, ...]`（另有 `capacity` / `closed` / `date`）。

        - `date` 缺省 = **今天**。缺省**不会**自动跳到"第一个有余号的日期"：契约是"某一天"，
          而"换日期"是产品上要问用户的决定，静默改期会让上层显示的日期与实际取的号不一致。
        - `remaining` 是**可约余量**：停诊时段一律报 0（`closed=True` 另标）。
          这样 `BookingService.first_available_slot` 就不会挑中一个已停诊的时段——
          它只看 `remaining > 0`，若这里如实回"停诊但还剩 8 个号"，占号会以一个
          上层无法预判的失败告终。
        - 没有排班时返回 `[]`（不是报错）：`Availability` 本来就允许空列表。
        """
        dep = self._require_department(department)
        day = _as_date(date) or self._today()
        with self._pool.connection() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT period, capacity, remaining, closed FROM {schedules} "
                    "WHERE department = %s AND date = %s"
                ),
                (dep, day),
            ).fetchall()
        return _sort_slots(
            [
                _slot_dict(day, period, capacity, remaining, closed)
                for period, capacity, remaining, closed in rows
            ]
        )

    # ------------------------------------------------------------------ 契约：取号
    def book(self, department: str, patient_id: str, patient_name: str = "", **kwargs: object) -> dict:
        """取号，返回 `{ticket_no, department, date, time, queue_no, fee, ...}`。

        `kwargs`：`date`（缺省今天）、`time_slot`（或等价的 `period`，必须是「上午/下午」）、
        `idempotency_key`（可选）。

        ## 并发与一致性（这是号源系统唯一不能妥协的性质）

        占号是**一个事务里的一句条件更新**：

        ```sql
        UPDATE booking.schedules SET remaining = remaining - 1
         WHERE department = %s AND date = %s AND period = %s
           AND remaining > 0 AND NOT closed
        RETURNING remaining
        ```

        - **判定与扣减是同一句话，所以不可能超卖。** READ COMMITTED 下，并发事务会在
          该行的行锁上排队；先拿到锁的事务提交后，后来者用**最新版本**重新求值 WHERE
          （EvalPlanQual），于是最后一个号只可能被一个事务拿走。没有返回行 = 没抢到。
        - **不要"先 SELECT 再 UPDATE"**：两者之间是敞开的窗口，两个事务都读到
          `remaining = 1`、都认为还能占，最后一个号被卖两次——这正是要避免的。
        - 没抢到时**必须区分原因**（满 / 停诊 / 没排班），所以再只读一次该行给出准确的话术；
          笼统报"失败"会让 agent 无法判断该建议换时段还是换日期。
        - 队列号 `SELECT COALESCE(MAX(queue_no), 0) + 1 ... WHERE department = %s AND date = %s
          AND period = %s` 在**同一事务内**取：此时该 schedule 行已被上面的 UPDATE 锁住，
          同科室同日同时段的其他占号事务都在等这把锁，所以不会重号。口径是
          **科室 + 日期 + 时段**（上午 1..N、下午也从 1 起），并有
          `UNIQUE(department, date, period, queue_no)` 兜底——注意上下午落在**不同的
          schedule 行**上、互不阻塞，若把 `period` 从取号 SQL（或约束）里漏掉，跨时段并发
          就会算出同一个号（实测 20 线程上下午各 10：每轮 9~10 对重号）。
        - `ticket_no` / `idempotency_key` 撞唯一键时整个事务回滚——**扣减一起回滚**，
          所以重试（换一个 `ticket_no` 再来）不会多扣号。`idempotency_key` 命中时直接返回
          原单据、**不再扣号**，这就是幂等键的全部意义。

        业务失败抛 `BookingError`；**库连不上/池超时这类基础设施故障不包装成 `BookingError`**
        ——那会让"号源系统挂了"伪装成一句可讲给患者的业务话术，上层应该看见它。
        """
        dep = self._require_department(department)
        pid = str(patient_id or "").strip()
        if not pid:
            raise BookingError("取号必须带 patient_id（退号也只认同一个 patient_id）")
        day = _as_date(kwargs.get("date")) or self._today()
        period = _as_period(kwargs.get("time_slot") or kwargs.get("period"))
        idem = _as_idempotency_key(kwargs.get("idempotency_key"))
        last: Exception | None = None
        for _ in range(_TICKET_ATTEMPTS):
            try:
                return self._book_once(dep, day, period, pid, str(patient_name or ""), _new_ticket_no(), idem)
            except psycopg.errors.UniqueViolation as e:
                last = e
                # 幂等键撞了：说明**另一个事务**刚刚用同一个键插入了单据（本事务开头的
                # SELECT 那时还没看见它）。返回那张单据就是幂等键该有的语义——绝不能
                # 把同一个请求变成两张单、扣两个号。
                #
                # 判断依据不是错误信息里的约束名，而是"这个键现在到底有没有行"：
                # 只按约束名分派会很脆（驱动/服务端版本差异都可能让它空掉），而键存在与否
                # 是事实。键不存在 → 撞的是 ticket_no，换一个号重试即可。
                if idem is not None:
                    existing = self._by_idempotency_key(idem)
                    if existing is not None:
                        return existing
        raise BookingError(f"取号失败：连续 {_TICKET_ATTEMPTS} 次生成的凭据号都撞号了，请重试") from last

    def _book_once(
        self,
        dep: str,
        day: dt.date,
        period: str,
        patient_id: str,
        patient_name: str,
        ticket_no: str,
        idem: str | None,
    ) -> dict:
        """一次完整的占号尝试（一个事务）。重试只发生在 `ticket_no` 撞号时。"""
        with self._pool.connection() as conn, conn.transaction():
            if idem is not None:
                found = conn.execute(
                    self._q("SELECT {cols} FROM {appointments} WHERE idempotency_key = %s", cols=_COLS),
                    (idem,),
                ).fetchone()
                if found is not None:
                    return _row_to_ticket(found)
            fee_row = conn.execute(
                self._q("SELECT fee FROM {departments} WHERE name = %s"), (dep,)
            ).fetchone()
            if fee_row is None:
                # `_require_department` 已经查过一次；这里只是不让"科室在事务中途被删"
                # 变成一句费解的 NOT NULL / FK 报错。
                raise BookingError(f"{dep}不在可挂号科室里")
            taken = conn.execute(
                self._q(
                    "UPDATE {schedules} SET remaining = remaining - 1 "
                    "WHERE department = %s AND date = %s AND period = %s "
                    "  AND remaining > 0 AND NOT closed "
                    "RETURNING remaining"
                ),
                (dep, day, period),
            ).fetchone()
            if taken is None:
                raise BookingError(self._why_unavailable(conn, dep, day, period))
            queue_no = int(
                conn.execute(
                    self._q(
                        "SELECT COALESCE(MAX(queue_no), 0) + 1 FROM {appointments} "
                        "WHERE department = %s AND date = %s AND period = %s"
                    ),
                    (dep, day, period),
                ).fetchone()[0]
            )
            row = conn.execute(
                self._q(
                    "INSERT INTO {appointments}({ins}) "
                    "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now(), NULL) "
                    "RETURNING {cols}",
                    ins=_INSERT_COLS,
                    cols=_COLS,
                ),
                (
                    ticket_no,
                    dep,
                    day,
                    period,
                    patient_id,
                    patient_name,
                    queue_no,
                    int(fee_row[0]),
                    STATUS_PAID,
                    idem,
                ),
            ).fetchone()
            return _row_to_ticket(row)

    def _by_idempotency_key(self, idem: str) -> dict | None:
        """用独立事务回读幂等键对应的单据（前一个事务已回滚，不能在同一事务里读）。"""
        with self._pool.connection() as conn:
            row = conn.execute(
                self._q("SELECT {cols} FROM {appointments} WHERE idempotency_key = %s", cols=_COLS),
                (idem,),
            ).fetchone()
        return _row_to_ticket(row) if row is not None else None

    def _why_unavailable(self, conn: Any, dep: str, day: dt.date, period: str) -> str:
        """条件更新没抢到时，用一次只读查询把原因说清楚（在**同一事务内**，行状态未变）。"""
        row = conn.execute(
            self._q(
                "SELECT remaining, closed FROM {schedules} "
                "WHERE department = %s AND date = %s AND period = %s"
            ),
            (dep, day, period),
        ).fetchone()
        if row is None:
            return f"{dep} 在 {day.isoformat()} 没有{period}的排班（不在放号窗口内），请换一个日期"
        remaining, closed = int(row[0]), bool(row[1])
        if closed:
            return f"{dep} {day.isoformat()} {period}已停诊，请换一个时段或日期"
        if remaining <= 0:
            return f"{dep} {day.isoformat()} {period}的号源已满，请换一个时段或日期"
        return f"{dep} {day.isoformat()} {period}暂时无法取号，请稍后重试"

    # ------------------------------------------------------------------ 退号与记录
    def cancel(self, ticket_no: str, patient_id: str) -> dict:
        """退号：`status='cancelled'` + `remaining + 1`（同一个事务）。

        规则（已锁定的产品决定里最简单的那套）：**随时可退、立即释放余量、不改签、
        不限重复预约**。四条约束：

        - **只有同一 `patient_id` 能退**：不是本人一律拒绝（否则凭据号就是一张人人可用的
          作废票，号源会被恶意刷掉）。
        - **重复退号报错**：状态已是 `cancelled` 直接抛 `BookingError`，不再回补余量
          （否则连点两次退号就能凭空多出一个号——比超卖更隐蔽的"多卖"）。
        - 释放用条件更新 `remaining = remaining + 1 WHERE remaining < capacity`：并发退号
          在同一行上排队，不会把余量顶过 `capacity`；**不判 `closed`**——停诊只影响新预约，
          退号必须照常能退。容量被手工调小到已挂出数以下时回补条件不成立，此时**整笔回滚并
          报错**（宁可让用户找人工，也不留下一张"已退但没还号"的单子）。
        - 凭据号不存在 → 报错。
        """
        ticket = str(ticket_no or "").strip()
        if not ticket:
            raise BookingError("退号必须带凭据号")
        pid = str(patient_id or "").strip()
        if not pid:
            raise BookingError("退号必须带 patient_id（只有本人能退自己的号）")
        with self._pool.connection() as conn, conn.transaction():
            # `FOR UPDATE` 让"查状态—改状态"这段对同一张单据串行化：两个并发的退号请求
            # 只有一个能走到 UPDATE，另一个必然看见 `cancelled`（重复退号报错）。
            row = conn.execute(
                self._q(
                    "SELECT department, date, period, patient_id, status FROM {appointments} "
                    "WHERE ticket_no = %s FOR UPDATE"
                ),
                (ticket,),
            ).fetchone()
            if row is None:
                raise BookingError(f"没有找到凭据号 {ticket} 的挂号记录")
            dep, day, period, owner, status = row
            owner = str(owner)
            if owner != pid:
                raise BookingError(f"凭据号 {ticket} 不是你的挂号记录，不能退别人的号")
            if str(status) == STATUS_CANCELLED:
                raise BookingError(f"凭据号 {ticket} 已经退过了，不能重复退号")
            conn.execute(
                self._q("UPDATE {appointments} SET status = %s, updated_at = now() WHERE ticket_no = %s"),
                (STATUS_CANCELLED, ticket),
            )
            released = conn.execute(
                self._q(
                    "UPDATE {schedules} SET remaining = remaining + 1 "
                    "WHERE department = %s AND date = %s AND period = %s "
                    "  AND remaining < capacity "
                    "RETURNING remaining"
                ),
                (dep, day, period),
            ).fetchone()
            if released is None:
                raise BookingError(
                    f"凭据号 {ticket} 对应的排班已变更（{dep} {day.isoformat()} {period}），"
                    "余量无法回补，请联系人工处理"
                )
            return {
                "ticket_no": ticket,
                "department": str(dep),
                "date": day.isoformat(),
                "time": str(period),
                "status": STATUS_CANCELLED,
                "remaining": int(released[0]),
            }

    def list_appointments(self, patient_id: str, limit: int = 20) -> list[dict]:
        """某患者的挂号记录（新到旧）。只按 `patient_id` 过滤——别人的一条也看不见。

        退掉的号**仍然在列表里**（`status='cancelled'`）：那是历史，删掉就没法回答
        "我上周是不是退过一个号"。
        """
        pid = str(patient_id or "").strip()
        if not pid:
            raise BookingError("查挂号记录必须带 patient_id")
        with self._pool.connection() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT {cols} FROM {appointments} WHERE patient_id = %s "
                    "ORDER BY created_at DESC, id DESC LIMIT %s",
                    cols=_COLS,
                ),
                (pid, max(0, int(limit))),
            ).fetchall()
        return [_row_to_ticket(r) for r in rows]

    # ------------------------------------------------------------------ 内部工具
    @staticmethod
    def _today() -> dt.date:
        """ "今天"的唯一入口。抽出来是为了让"为什么这里用本地日期"只有一处解释：
        放号窗口是**营业日**概念，跟着服务器本地时区走，不用 UTC。"""
        return dt.date.today()

    def _department_names(self) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(self._q("SELECT name FROM {departments} ORDER BY code")).fetchall()
        return [str(r[0]) for r in rows]

    def _require_department(self, name: str) -> str:
        """归一化科室名并确认它在科室表里；不在就报错（**不猜**）。"""
        dep = normalize_department(name or "")
        if not dep:
            raise BookingError(f"没能识别出科室名：{name!r}")
        with self._pool.connection() as conn:
            if (
                conn.execute(self._q("SELECT 1 FROM {departments} WHERE name = %s"), (dep,)).fetchone()
                is None
            ):
                known = "、".join(
                    sorted(
                        str(r[0]) for r in conn.execute(self._q("SELECT name FROM {departments}")).fetchall()
                    )
                )
                raise BookingError(f"{dep}不在可挂号科室里（可选：{known}）")
        return dep

    def _slot_args(
        self, department: str, date: str | dt.date | None, period: object
    ) -> tuple[str, dt.date, str]:
        """停诊/改号量共用的参数解析：科室必须存在、日期缺省今天、时段必须合法。"""
        return (
            self._require_department(department),
            _as_date(date) or self._today(),
            _as_period(period),
        )


def _slot_dict(day: dt.date, period: object, capacity: object, remaining: object, closed: object) -> dict:
    """一个时段的对外形状。

    `remaining` 报**可约余量**：停诊时段一律 0（`closed=True` 另标），这样
    "有余量就一定能占上"这条不变式在上层成立。
    """
    is_closed = bool(closed)
    return {
        "time": str(period),
        "remaining": 0 if is_closed else int(remaining),
        "capacity": int(capacity),
        "closed": is_closed,
        "date": day.isoformat(),
    }


def _sort_slots(slots: list[dict]) -> list[dict]:
    """按「上午 → 下午」排序。

    不能靠 `ORDER BY period` 的排序规则（中文排序依赖 collation，换个库就可能变），
    所以按 `PERIODS` 的次序显式定序。
    """
    order = {p: i for i, p in enumerate(PERIODS)}
    return sorted(slots, key=lambda s: order.get(str(s["time"]), len(PERIODS)))


def from_settings(settings: Settings, *, schema: str = BOOKING_SCHEMA) -> PgAppointmentProvider:
    """按配置构造（装配点与 `store/pgvector_store.from_settings` 保持同一形状）。"""
    return PgAppointmentProvider(
        settings.database_url,
        schema=schema,
        pool_min=settings.db_pool_min,
        pool_max=settings.db_pool_max,
    )
