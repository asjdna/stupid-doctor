"""SQLite 号源库：可持久化的真实号源实现（`SqliteAppointmentProvider`）。

## 完整性交给数据库，而不是应用层自觉

号源系统唯一不能妥协的性质是**并发下不超卖**。把这件事放在 Python 里做
（先 SELECT 余号、再在内存里减一、再 UPDATE）在 FastAPI 线程池下必然出错：
两个请求会同时读到"还剩 1"，然后各写各的。

这里用**原子条件递减**：

    UPDATE schedules SET remaining = remaining - 1
     WHERE department=? AND date=? AND time_slot=? AND remaining > 0

`cursor.rowcount == 1` 才算拿到号。这条 UPDATE 在 SQLite 的写事务里是原子的，
`remaining > 0` 的判断与递减发生在同一条语句内，中间没有窗口。

再加三层数据库级兜底（任何绕过本类的写入路径也守得住）：

- `CHECK(remaining >= 0 AND remaining <= total)` —— 余号不可能为负
- `UNIQUE(department, date, time_slot)` —— 不可能出现重复号源行
- `UNIQUE(department, date, queue_no)` —— 排队号在同科室同天不重复

## 排队号怎么取

`MAX(queue_no) + 1`（同科室同天）在写事务内计算。`UNIQUE(department, date, queue_no)`
保证即使算错也会被数据库拒绝，而不是发出两个相同的排队号。

> 表结构移植自 clinic 域（原 medical-agent 项目）的 `db/schema_booking.py`，
> 保留其全部约束与理由。改动有两处：① 患者表在本域内自带（原项目挂在 users 上）；
> ② 触发器的"父行必须存在"兜底保留。
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

from kubedocs.domains.clinic.data.departments import DEPARTMENTS, normalize_department
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError

# 科室 -> (代码, 挂号费)。数据来自领域字典，保证与分诊输出同一套科室名。
_DEFAULT_SLOTS = ("上午", "下午")
_SLOT_CAPACITY = {"上午": 10, "下午": 6}

SCHEDULES_DDL = """
CREATE TABLE IF NOT EXISTS schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    department  TEXT    NOT NULL CHECK (length(trim(department)) > 0),
    date        TEXT    NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    time_slot   TEXT    NOT NULL CHECK (length(trim(time_slot)) > 0),
    total       INTEGER NOT NULL CHECK (total >= 0),
    remaining   INTEGER NOT NULL CHECK (remaining >= 0 AND remaining <= total),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    CONSTRAINT uq_schedule_slot UNIQUE (department, date, time_slot)
)
"""

# department/date/time_slot 在 appointments 里是**下单时的快照**（刻意冗余）：
# 号源行日后被改时段或改总量时，历史挂号仍应显示当时实际情况。
APPOINTMENTS_DDL = """
CREATE TABLE IF NOT EXISTS appointments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     TEXT    NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    schedule_id    INTEGER NOT NULL REFERENCES schedules(id) ON DELETE RESTRICT,
    department     TEXT    NOT NULL CHECK (length(trim(department)) > 0),
    date           TEXT    NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    time_slot      TEXT    NOT NULL CHECK (length(trim(time_slot)) > 0),
    ticket_no      TEXT    NOT NULL UNIQUE,
    queue_no       INTEGER NOT NULL CHECK (queue_no > 0),
    status         TEXT    NOT NULL DEFAULT 'booked' CHECK (status IN ('booked','cancelled')),
    created_via_ai INTEGER NOT NULL DEFAULT 1 CHECK (created_via_ai IN (0,1)),
    created_at     TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    CONSTRAINT uq_appointment_queue UNIQUE (department, date, queue_no)
)
"""

PATIENTS_DDL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_schedules_date      ON schedules(date)",
    "CREATE INDEX IF NOT EXISTS idx_schedules_dept_date ON schedules(department, date)",
    "CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_appointments_sched   ON appointments(schedule_id)",
    "CREATE INDEX IF NOT EXISTS idx_appointments_status  ON appointments(status)",
)

# 外键的「连接无关」兜底：PRAGMA foreign_keys 是**连接级**设置、默认关闭，
# 用 sqlite3.exe / DBeaver 直接连库时外键就变回注释——同一份数据完整性却取决于"谁连的"。
# 触发器是库内对象，任何连接都会跑，且报错信息比 FOREIGN KEY constraint failed 清楚。
_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS trg_appointments_patient_fk_insert
    BEFORE INSERT ON appointments
    WHEN NOT EXISTS (SELECT 1 FROM patients WHERE patient_id = NEW.patient_id)
    BEGIN SELECT RAISE(ABORT, '挂号引用了不存在的患者'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_appointments_schedule_fk_insert
    BEFORE INSERT ON appointments
    WHEN NOT EXISTS (SELECT 1 FROM schedules WHERE id = NEW.schedule_id)
    BEGIN SELECT RAISE(ABORT, '挂号引用了不存在的号源'); END
    """,
)


def _department_meta(name: str) -> tuple[str, int]:
    """科室 -> (代码, 挂号费)。未登记的科室也允许挂号（代码留空、费用 0）。"""
    code = ""
    fee = 0
    for i, dep in enumerate(DEPARTMENTS, start=1):
        if dep == name:
            code = f"D{i:02d}"
            fee = 25 + (i % 5) * 5
            break
    return code, fee


class SqliteAppointmentProvider(AppointmentProvider):
    """落到 SQLite 的号源实现。`db_path` 指向 clinic 域自己的库文件。"""

    name = "sqlite"

    def __init__(self, db_path: str | Path, *, days_ahead: int = 7) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.days_ahead = days_ahead
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        for ddl in (PATIENTS_DDL, SCHEDULES_DDL, APPOINTMENTS_DDL, *_INDEXES, *_TRIGGERS):
            self._con.execute(ddl)

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __enter__(self) -> SqliteAppointmentProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ 患者
    def ensure_patient(self, patient_id: str, name: str = "") -> None:
        """登记患者（幂等）。挂号前调用；否则触发器会拒绝。"""
        with self._lock:
            self._con.execute(
                "INSERT INTO patients(patient_id, name) VALUES(?, ?) "
                "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name "
                "WHERE excluded.name <> ''",
                (patient_id, name),
            )

    # ------------------------------------------------------------------ 科室/号源
    def list_departments(self) -> list[dict]:
        out = []
        for dep in DEPARTMENTS:
            code, fee = _department_meta(dep)
            out.append({"code": code, "name": dep, "fee": fee})
        return out

    def _seed(self, department: str, date: str) -> None:
        """按需生成某科室某天的号源（幂等：UNIQUE 约束保证不会重复）。"""
        for slot in _DEFAULT_SLOTS:
            cap = _SLOT_CAPACITY.get(slot, 8)
            self._con.execute(
                "INSERT INTO schedules(department, date, time_slot, total, remaining) "
                "VALUES(?,?,?,?,?) ON CONFLICT(department, date, time_slot) DO NOTHING",
                (department, date, slot, cap, cap),
            )

    def get_availability(self, department: str, date: str | None = None) -> list[dict]:
        dep = normalize_department(department)
        day = date or _date.today().isoformat()
        self._validate_date(day)
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._seed(dep, day)
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            rows = self._con.execute(
                "SELECT time_slot, remaining FROM schedules WHERE department=? AND date=? ORDER BY id",
                (dep, day),
            ).fetchall()
        return [{"time": r["time_slot"], "remaining": r["remaining"]} for r in rows]

    def upcoming_dates(self, days: int | None = None) -> list[str]:
        n = days or self.days_ahead
        today = _date.today()
        return [(today + timedelta(days=i)).isoformat() for i in range(n)]

    # ------------------------------------------------------------------ 取号
    @staticmethod
    def _validate_date(day: str) -> None:
        try:
            _date.fromisoformat(day)
        except ValueError as e:
            raise BookingError(f"日期格式不对：{day!r}（应为 YYYY-MM-DD）") from e

    def book(self, department: str, patient_id: str, patient_name: str = "", **kwargs: object) -> dict:
        dep = normalize_department(department)
        if not dep:
            raise BookingError("没能识别出要挂哪个科室")
        day = str(kwargs.get("date") or _date.today().isoformat())
        self._validate_date(day)
        slot = str(kwargs.get("time_slot") or "")
        if slot not in _DEFAULT_SLOTS:
            raise BookingError(f"时段只能是 {'/'.join(_DEFAULT_SLOTS)}，收到 {slot!r}")

        self.ensure_patient(patient_id, patient_name)
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._seed(dep, day)
                # 原子条件递减：判断"还有余号"与递减在同一条语句内完成，无并发窗口
                cur = self._con.execute(
                    "UPDATE schedules SET remaining = remaining - 1, "
                    "updated_at = datetime('now','localtime') "
                    "WHERE department=? AND date=? AND time_slot=? AND remaining > 0",
                    (dep, day, slot),
                )
                if cur.rowcount != 1:
                    raise BookingError(f"{dep}{day}{slot}的号已经挂完了，请换一个时段或日期")
                row = self._con.execute(
                    "SELECT id FROM schedules WHERE department=? AND date=? AND time_slot=?", (dep, day, slot)
                ).fetchone()
                queue_no = int(
                    self._con.execute(
                        "SELECT COALESCE(MAX(queue_no), 0) + 1 FROM appointments "
                        "WHERE department=? AND date=?",
                        (dep, day),
                    ).fetchone()[0]
                )
                ticket_no = self._new_ticket_no()
                self._con.execute(
                    "INSERT INTO appointments(patient_id, schedule_id, department, date,"
                    " time_slot, ticket_no, queue_no, status, created_via_ai)"
                    " VALUES(?,?,?,?,?,?,?,'booked',?)",
                    (
                        patient_id,
                        row["id"],
                        dep,
                        day,
                        slot,
                        ticket_no,
                        queue_no,
                        1 if kwargs.get("via_ai", True) else 0,
                    ),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        code, fee = _department_meta(dep)
        return {
            "ticket_no": ticket_no,
            "department": dep,
            "department_code": code,
            "date": day,
            "time": slot,
            "queue_no": queue_no,
            "fee": fee,
        }

    @staticmethod
    def _new_ticket_no() -> str:
        # 取号单号是患者到窗口凭的凭证，必须唯一；UNIQUE(ticket_no) 是最终保证
        return "T" + _date.today().strftime("%y%m%d") + secrets.token_hex(3).upper()

    # ------------------------------------------------------------------ 查询
    def list_appointments(self, patient_id: str, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._con.execute(
                "SELECT ticket_no, department, date, time_slot, queue_no, status, created_at"
                " FROM appointments WHERE patient_id=? ORDER BY id DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def cancel(self, ticket_no: str, patient_id: str) -> None:
        """取消挂号并**把号还回号源**（同事务内完成，不会丢号也不会多号）。"""
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                row = self._con.execute(
                    "SELECT schedule_id, status FROM appointments WHERE ticket_no=? AND patient_id=?",
                    (ticket_no, patient_id),
                ).fetchone()
                if row is None:
                    raise BookingError(f"没找到取号单 {ticket_no}")
                if row["status"] == "cancelled":
                    raise BookingError("这个号已经取消过了")
                self._con.execute(
                    "UPDATE appointments SET status='cancelled' WHERE ticket_no=?", (ticket_no,)
                )
                self._con.execute(
                    "UPDATE schedules SET remaining = MIN(remaining + 1, total), "
                    "updated_at = datetime('now','localtime') WHERE id=?",
                    (row["schedule_id"],),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
