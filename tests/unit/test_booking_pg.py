"""Postgres 号源后端的专项测试：**Postgres 不可达时整文件 skip，并写明探测到的原因**。

前置条件：一个可连的 Postgres（DSN 取自 `Settings().database_url`，即 `KUBEDOCS_DATABASE_URL`）。
连不上就 skip；每个用例用独立的 `test_booking_*` schema，结束即 `DROP SCHEMA CASCADE`，
既不污染本机正在用的 `kubedocs` 库里的 `clinic` schema（那是 pgvector 的知识卡片向量），
也不让用例互相看见对方的数据。

守的契约（对应产品已锁定的决定）：

- **自建号源只到科室**：`科室 × 日期 × 上午/下午`，没有医生维度；
- **排班幂等**：`generate_schedules` 跑两次行数不变、`remaining` 不变；先占号再跑，那个号
  的余量**不被重置**（重置 = 凭空多卖）；
- **并发不超卖**：20 个线程抢 5 个号 → 恰好 5 个成功、`remaining = 0`、没有负数；
- **队列号按「科室 + 日期 + 时段」**：上午 1..N、下午也从 1 起，唯一约束钉在同一组列上，
  上下午**并发**占号也不重号（这是真实医院的叫号口径）；
- **占号**扣减 + 队列号递增 + 满号报错；**退号**释放余量、改状态、重复退报错、退别人的报错；
- **停诊只影响新预约**（已挂出的号不作废，退号照常回补）；
- **假支付**：`book` 直接落 `paid` / `expires_at = NULL`，`pending_payment` 是保留状态位；
- **幂等键**命中时返回原单据、不重复扣号——单线程与**两线程同键同刻**都只出一张单；
- `Availability.date` 永远不是空串（显式传入 > 时段自带 > 今天）；
- 表上真有 `CHECK (remaining >= 0)` / `CHECK (remaining <= capacity)` 这两道数据库级兜底。
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
import pytest

from kubedocs.config import Settings
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.data.departments import DEFAULT_FEE, DEPARTMENTS, department_fee
from kubedocs.domains.clinic.mock import MockAppointmentProvider
from kubedocs.domains.clinic.pg_provider import PERIODS, PgAppointmentProvider, probe
from kubedocs.domains.clinic.provider import AppointmentProvider, BookingError

_DSN = Settings().database_url
_PG_OK, _PG_WHY = probe(_DSN, timeout=3.0)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _PG_OK,
        reason=f"Postgres 不可达，本文件未验证：{_PG_WHY}（DSN={_DSN}）",
    ),
]

# 凭据号必须"可读"：6~8 位大写字母数字（实现用 8 位、去掉 I/O/0/1）。
_TICKET_RE = re.compile(r"^[A-Z0-9]{6,8}$")

_TODAY = dt.date.today()


# --------------------------------------------------------------------------- 夹具
@pytest.fixture
def schema() -> str:
    """每个用例一个独立 schema：名字带随机后缀，避免并发/残留互相影响。"""
    return f"test_booking_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def make(schema: str) -> Iterator[Callable[..., PgAppointmentProvider]]:
    """构造 provider 的工厂（同一用例可以起多个，例如并发用例要更大的池）。

    清理放在夹具里而不是用例里：`DROP SCHEMA CASCADE` 必须发生在**池关闭之后**
    （池里可能还有连接），否则会留下一个带活动连接的 schema。
    """
    made: list[PgAppointmentProvider] = []

    def _make(**kwargs: Any) -> PgAppointmentProvider:
        provider = PgAppointmentProvider(_DSN, schema=schema, **kwargs)
        made.append(provider)
        return provider

    try:
        yield _make
    finally:
        for provider in made:
            provider.close()
        with psycopg.connect(_DSN, connect_timeout=5) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture
def pg(make: Callable[..., PgAppointmentProvider]) -> PgAppointmentProvider:
    """默认 provider：已建表、已种科室，**还没有排班**（用例自己决定放号窗口）。"""
    return make()


# --------------------------------------------------------------------------- SQL 小工具
def _q(schema: str) -> str:
    """schema 的带引号写法（schema 名由夹具生成，只含 [a-z0-9_]）。"""
    return f'"{schema}"'


def _rows(schema: str, query: str, params: tuple = ()) -> list[tuple]:
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        return conn.execute(query.format(s=_q(schema)), params).fetchall()


def _scalar(schema: str, query: str, params: tuple = ()) -> Any:
    return _rows(schema, query, params)[0][0]


def _exec(schema: str, query: str, params: tuple = ()) -> int:
    with psycopg.connect(_DSN, connect_timeout=5) as conn:
        return int(conn.execute(query.format(s=_q(schema)), params).rowcount)


def _remaining(schema: str, dep: str, day: dt.date, period: str) -> int:
    return int(
        _scalar(
            schema,
            "SELECT remaining FROM {s}.schedules WHERE department = %s AND date = %s AND period = %s",
            (dep, day, period),
        )
    )


def _schedule_count(schema: str) -> int:
    return int(_scalar(schema, "SELECT COUNT(*) FROM {s}.schedules"))


def _appointment_count(schema: str, status: str | None = None) -> int:
    if status is None:
        return int(_scalar(schema, "SELECT COUNT(*) FROM {s}.appointments"))
    return int(_scalar(schema, "SELECT COUNT(*) FROM {s}.appointments WHERE status = %s", (status,)))


# --------------------------------------------------------------------------- 建表
def test_ensure_schema_is_idempotent_and_seeds_departments(pg: PgAppointmentProvider, schema: str) -> None:
    """建 schema/表/索引可反复调用；科室与费用按 `data/departments.py` 种进去。

    索引/约束用系统目录核对而不是"看代码觉得对"：`ON CONFLICT (department, date, period)`
    依赖那条复合唯一约束，缺了它 `generate_schedules` 的幂等就无从谈起（而且会直接报错）。
    """
    pg.ensure_schema()
    pg.ensure_schema()

    tables = {r[0] for r in _rows(schema, "SELECT tablename FROM pg_tables WHERE schemaname = %s", (schema,))}
    assert {"departments", "schedules", "appointments"} <= tables
    indexes = {
        r[0] for r in _rows(schema, "SELECT indexname FROM pg_indexes WHERE schemaname = %s", (schema,))
    }
    assert {
        "uq_booking_departments_name",
        "uq_booking_schedules_slot",
        "uq_booking_appointments_ticket_no",
        "uq_booking_appointments_idempotency_key",
        "uq_booking_appointments_queue",
        "idx_booking_appointments_patient",
        "idx_booking_appointments_slot",
        "idx_booking_schedules_date",
    } <= indexes
    constraints = {
        (r[0], r[1])
        for r in _rows(
            schema,
            "SELECT conname, contype FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace WHERE n.nspname = %s",
            (schema,),
        )
    }
    kinds = dict(constraints)
    assert kinds.get("ck_booking_schedules_remaining_le_capacity") == "c"
    assert any(kind == "f" for kind in kinds.values()), "schedules.department 应该有外键"
    # CHECK 必须真的落到库里（应用层之外的最后一道兜底）
    checks = " ".join(
        str(r[0])
        for r in _rows(
            schema,
            "SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace "
            "WHERE n.nspname = %s AND contype = 'c'",
            (schema,),
        )
    )
    assert "remaining >= 0" in checks
    assert "remaining <= capacity" in checks

    seeded = dict(_rows(schema, "SELECT name, fee FROM {s}.departments ORDER BY code"))
    assert list(seeded) == DEPARTMENTS, "科室清单与顺序都取自 data/departments.py"
    # 费用来自 `data/departments.py`（唯一真相源），逐科室对齐
    assert seeded == {name: department_fee(name) for name in DEPARTMENTS}
    assert seeded["呼吸内科"] == 35
    assert seeded["心内科"] == DEFAULT_FEE
    assert department_fee("心血管内科") == DEFAULT_FEE, "别名也要能查到定价"
    assert pg.list_departments()[0] == {"code": "D01", "name": DEPARTMENTS[0], "fee": DEFAULT_FEE}

    # 科室是 `ON CONFLICT DO NOTHING`：**不覆盖**已有行，所以运营手工改过的费用不会被
    # 每次启动重置回初始值。
    _exec(schema, "UPDATE {s}.departments SET fee = 99 WHERE name = %s", ("心内科",))
    pg.ensure_schema()
    assert _scalar(schema, "SELECT fee FROM {s}.departments WHERE name = %s", ("心内科",)) == 99

    # 非法 schema 名是部署事故，不是"能讲给患者听的业务错误"：必须是 ValueError。
    with pytest.raises(ValueError, match="非法的 schema"):
        PgAppointmentProvider(_DSN, schema="public; DROP TABLE x")


def test_database_rejects_negative_and_over_capacity_remaining(
    pg: PgAppointmentProvider, schema: str
) -> None:
    """`remaining` 的上下界由**数据库**兜底，不只靠应用层自律。"""
    pg.generate_schedules(days=1)
    bad_updates = (
        "UPDATE {s}.schedules SET remaining = -1 WHERE department = %s AND date = %s",
        "UPDATE {s}.schedules SET remaining = capacity + 1 WHERE department = %s AND date = %s",
    )
    for stmt in bad_updates:
        with pytest.raises(psycopg.errors.CheckViolation):
            _exec(schema, stmt, ("心内科", _TODAY))


# --------------------------------------------------------------------------- 排班
def test_generate_schedules_is_idempotent_and_never_resets_remaining(
    pg: PgAppointmentProvider, schema: str
) -> None:
    """生成排班幂等：跑两次行数不变、`remaining` 不变；**先占号再生成，那个号不被重置**。

    这是"幂等"的实际含义：`ON CONFLICT ... DO NOTHING` 只补缺失的行。若写成
    `DO UPDATE SET remaining = EXCLUDED.remaining`，重跑一次就把已占的号变回满额
    ——凭空多卖一遍，而且**任何并发测试都抓不到**。
    """
    per_dept = len(DEPARTMENTS) * len(PERIODS)
    assert pg.generate_schedules() == per_dept * 7
    assert pg.generate_schedules() == 0, "第二次跑不该新增任何行"
    assert _schedule_count(schema) == per_dept * 7

    window = [r[0] for r in _rows(schema, "SELECT DISTINCT date FROM {s}.schedules ORDER BY date")]
    assert window == [_TODAY + dt.timedelta(days=i) for i in range(7)], "放号窗口 = 今天起 7 天"
    assert set(_rows(schema, "SELECT DISTINCT capacity FROM {s}.schedules")) == {(10,)}
    assert set(_rows(schema, "SELECT DISTINCT remaining FROM {s}.schedules")) == {(10,)}

    ticket = pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9

    assert pg.generate_schedules() == 0
    assert _schedule_count(schema) == per_dept * 7
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9, "重跑排班把已占的号重置了"

    # 自定义窗口与号量
    assert pg.generate_schedules(days=3, per_period=4, start=_TODAY + dt.timedelta(days=7)) == (per_dept * 3)
    assert _remaining(schema, "心内科", _TODAY + dt.timedelta(days=7), "下午") == 4
    with pytest.raises(BookingError):
        pg.generate_schedules(days=0)
    with pytest.raises(BookingError):
        pg.generate_schedules(per_period=-1)
    with pytest.raises(BookingError, match="YYYY-MM-DD"):
        pg.generate_schedules(start="2026/01/01")
    assert ticket["date"] == _TODAY.isoformat()


def test_get_availability_defaults_to_today_and_hides_closed_capacity(
    pg: PgAppointmentProvider,
) -> None:
    """`get_availability` 缺省今天，按时段升序；**停诊时段的可约余量报 0**。

    报 0 的理由：`BookingService.first_available_slot` 只看 `remaining > 0`，若停诊时段
    如实回"还剩 8 个号"，上层会挑中它、然后以一个自己无法预判的失败告终。
    """
    pg.generate_schedules(days=1)
    assert pg.get_availability("心内科") == [
        {"time": "上午", "remaining": 10, "capacity": 10, "closed": False, "date": _TODAY.isoformat()},
        {"time": "下午", "remaining": 10, "capacity": 10, "closed": False, "date": _TODAY.isoformat()},
    ]
    # 科室名走归一化（别名也算）
    assert pg.get_availability("心血管内科", _TODAY.isoformat())[0]["remaining"] == 10
    assert pg.get_availability("心内科", _TODAY + dt.timedelta(days=30)) == []
    with pytest.raises(BookingError, match="不在可挂号科室里"):
        pg.get_availability("不存在的科室")

    pg.close_slot("心内科", _TODAY, "上午")
    slots = pg.get_availability("心内科", _TODAY)
    assert slots[0] == {
        "time": "上午",
        "remaining": 0,
        "capacity": 10,
        "closed": True,
        "date": _TODAY.isoformat(),
    }
    assert slots[1]["remaining"] == 10

    # 整个放号窗口一眼看完（非契约方法，给上层回答"哪天还有号"用）
    pg.generate_schedules(days=3)
    window = pg.availability_window("心内科")
    assert [d["date"] for d in window] == [(_TODAY + dt.timedelta(days=i)).isoformat() for i in range(3)]
    assert [s["time"] for s in window[0]["slots"]] == ["上午", "下午"]
    assert window[0]["slots"][0]["closed"] is True
    assert window[1]["slots"][0]["remaining"] == 10


def test_close_slot_only_affects_new_bookings(pg: PgAppointmentProvider, schema: str) -> None:
    """停诊：只挡新预约。已挂出的号不作废、`remaining` 不动、退号照常回补；开诊不重置余量。"""
    pg.generate_schedules(days=1)
    first = pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9

    pg.close_slot("心内科", _TODAY, "上午")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9, "停诊不该动余量"
    with pytest.raises(BookingError, match="停诊"):
        pg.book("心内科", "p2", date=_TODAY, time_slot="上午")
    # 停诊期间退号必须照常能退（退号规则最简：随时可退）
    assert pg.cancel(first["ticket_no"], "p1")["remaining"] == 10

    pg.close_slot("心内科", _TODAY, "上午", closed=False)
    assert _remaining(schema, "心内科", _TODAY, "上午") == 10, "开诊不该把余量重置回 capacity"
    second = pg.book("心内科", "p2", date=_TODAY, time_slot="上午")
    assert second["queue_no"] == 2
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9

    with pytest.raises(BookingError, match="没有排班"):
        pg.close_slot("心内科", _TODAY + dt.timedelta(days=30), "上午")
    with pytest.raises(BookingError, match="时段只能是"):
        pg.close_slot("心内科", _TODAY, "半夜")


def test_adjust_capacity_moves_remaining_by_what_is_already_booked(
    pg: PgAppointmentProvider, schema: str
) -> None:
    """改号量：`remaining = max(0, 新号量 - 已挂出)`；调小到低于已挂出数时余量归零。"""
    pg.generate_schedules(days=1)
    for i in range(3):
        pg.book("心内科", f"p{i}", date=_TODAY, time_slot="上午")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 7

    assert pg.adjust_capacity("心内科", _TODAY, "上午", 5) == {
        "capacity": 5,
        "remaining": 2,
        "booked": 3,
    }
    assert pg.adjust_capacity("心内科", _TODAY, "上午", 1) == {
        "capacity": 1,
        "remaining": 0,
        "booked": 3,
    }
    with pytest.raises(BookingError, match="号源已满"):
        pg.book("心内科", "p9", date=_TODAY, time_slot="上午")
    # 调大回来：按"已挂出 3"平移，不是直接给满额
    assert pg.adjust_capacity("心内科", _TODAY, "上午", 10)["remaining"] == 7

    with pytest.raises(BookingError, match="不能是负数"):
        pg.adjust_capacity("心内科", _TODAY, "上午", -1)
    with pytest.raises(BookingError, match="没有排班"):
        pg.adjust_capacity("心内科", _TODAY + dt.timedelta(days=30), "上午", 3)


# --------------------------------------------------------------------------- 取号
def test_book_decrements_queue_no_and_uses_department_fee(pg: PgAppointmentProvider, schema: str) -> None:
    """占号扣减、队列号按时段递增、费用取科室表、凭据号可读且唯一，且落 `paid`（假支付）。"""
    pg.generate_schedules(days=2)
    first = pg.book("呼吸内科", "p1", "张三", date=_TODAY, time_slot="上午")
    assert _TICKET_RE.match(first["ticket_no"]), first["ticket_no"]
    assert first["queue_no"] == 1
    assert first["fee"] == 35, "呼吸内科 35，其余 30"
    assert first["status"] == "paid", "假支付：确认即视为已付"
    assert first["expires_at"] is None
    assert first["department"] == "呼吸内科"
    assert first["patient_name"] == "张三"
    assert first["date"] == _TODAY.isoformat()
    assert first["time"] == "上午"
    assert _remaining(schema, "呼吸内科", _TODAY, "上午") == 9

    second = pg.book("呼吸内科", "p2", date=_TODAY.isoformat(), time_slot="上午")
    assert second["queue_no"] == 2, "同科室同日同时段的队列号必须递增"
    assert second["ticket_no"] != first["ticket_no"]
    assert second["fee"] == 35
    assert _remaining(schema, "呼吸内科", _TODAY, "上午") == 8

    other = pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    assert other["fee"] == DEFAULT_FEE
    assert other["queue_no"] == 1, "换科室重新从 1 起"

    # 队列号口径 = 科室 + 日期 + **时段**：上午 1..N，下午也从 1 起（真实医院的叫号口径）
    afternoon_first = pg.book("呼吸内科", "p3", date=_TODAY, time_slot="下午")
    assert afternoon_first["queue_no"] == 1
    afternoon_second = pg.book("呼吸内科", "p4", date=_TODAY, time_slot="下午")
    assert afternoon_second["queue_no"] == 2

    # 也不跨日期：明天的上午还是从 1 起
    tomorrow = _TODAY + dt.timedelta(days=1)
    assert pg.book("呼吸内科", "p5", date=tomorrow, time_slot="上午")["queue_no"] == 1

    with pytest.raises(BookingError, match="时段只能是"):
        pg.book("呼吸内科", "p1", date=_TODAY, time_slot="半夜")
    with pytest.raises(BookingError, match="不在可挂号科室里"):
        pg.book("不存在的科室", "p1", date=_TODAY, time_slot="上午")
    with pytest.raises(BookingError, match="YYYY-MM-DD"):
        pg.book("呼吸内科", "p1", date="2026/01/01", time_slot="上午")
    with pytest.raises(BookingError, match="patient_id"):
        pg.book("呼吸内科", "", date=_TODAY, time_slot="上午")


def test_book_on_full_slot_reports_it_is_full(pg: PgAppointmentProvider, schema: str) -> None:
    """满号必须报"号源已满"（而不是笼统的失败）：agent 要据此建议换时段/换日期。"""
    pg.generate_schedules(days=1)
    pg.adjust_capacity("心内科", _TODAY, "上午", 1)
    pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    with pytest.raises(BookingError, match="号源已满"):
        pg.book("心内科", "p2", date=_TODAY, time_slot="上午")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 0
    with pytest.raises(BookingError, match="没有.*排班"):
        pg.book("心内科", "p2", date=_TODAY + dt.timedelta(days=30), time_slot="上午")


def test_no_oversell_under_concurrency(make: Callable[..., PgAppointmentProvider], schema: str) -> None:
    """**不超卖**：20 个线程抢 5 个号 → 恰好 5 个成功、`remaining = 0`、没有负数。

    条件更新（`UPDATE ... WHERE remaining > 0 AND NOT closed RETURNING remaining`）在
    READ COMMITTED 下靠行锁串行化、并用最新行重算 WHERE，所以最后一个号只可能被一个事务
    拿走。这个用例是那条推理的可执行版本：线程数远大于号量，且用 `Barrier` 让它们同时冲。
    """
    provider = make(pool_max=20)
    provider.generate_schedules(days=1)
    provider.adjust_capacity("心内科", _TODAY, "上午", 5)

    threads = 20
    barrier = threading.Barrier(threads)

    def _race(i: int) -> dict | BookingError:
        barrier.wait()
        try:
            return provider.book("心内科", f"p{i}", date=_TODAY, time_slot="上午")
        except BookingError as e:  # 没抢到是**预期结果**，不是测试失败
            return e

    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(_race, range(threads)))

    winners = [r for r in results if isinstance(r, dict)]
    losers = [r for r in results if isinstance(r, BookingError)]
    assert len(winners) == 5, f"应当恰好 5 个成功，实际 {len(winners)}"
    assert len(losers) == 15
    assert all("号源已满" in str(e) for e in losers), [str(e) for e in losers]

    assert _remaining(schema, "心内科", _TODAY, "上午") == 0
    assert _scalar(schema, "SELECT MIN(remaining) FROM {s}.schedules") == 0, "余量不许为负"
    assert len({t["ticket_no"] for t in winners}) == 5, "凭据号必须互不相同"
    assert sorted(t["queue_no"] for t in winners) == [1, 2, 3, 4, 5], "队列号必须互不相同"
    assert _appointment_count(schema, "paid") == 5


def test_queue_no_is_per_period_under_concurrency(
    make: Callable[..., PgAppointmentProvider], schema: str
) -> None:
    """上下午**并发**占号：两个时段各自从 1 起，组内互不重复、跨组允许重号。

    这条用例专门盯"跨时段共用一个计数器"的回归。上下午落在**不同的 schedule 行**上，
    占号事务之间不互相阻塞，所以 `MAX(queue_no)` 只有把 `period` 带进 WHERE
    （外加 `UNIQUE(department, date, period, queue_no)` 兜底）才可能不重号——
    去掉任何一处，这里立刻出现重号。
    """
    provider = make(pool_max=24)
    provider.generate_schedules(days=1)

    threads = 20
    barrier = threading.Barrier(threads)

    def _race(i: int) -> dict | BookingError:
        barrier.wait()
        period = PERIODS[i % len(PERIODS)]
        try:
            return provider.book("心内科", f"p{i}", date=_TODAY, time_slot=period)
        except BookingError as e:  # 两个时段各 10 个号、各 10 个线程：不该有人失败
            return e

    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(_race, range(threads)))

    winners = [r for r in results if isinstance(r, dict)]
    failed = [str(r) for r in results if isinstance(r, BookingError)]
    assert len(winners) == threads, f"两个时段各 10 个号，不该有人抢不到：{failed}"
    for period in PERIODS:
        got = sorted(t["queue_no"] for t in winners if t["time"] == period)
        assert got == list(range(1, threads // len(PERIODS) + 1)), f"{period} 的队列号是 {got}"

    for period in PERIODS:
        assert _remaining(schema, "心内科", _TODAY, period) == 0
    assert _appointment_count(schema, "paid") == threads


def test_idempotency_key_returns_the_original_ticket(pg: PgAppointmentProvider, schema: str) -> None:
    """幂等键命中时返回**原单据**、不再扣号：重试的请求不能变成两张单、两个号。"""
    pg.generate_schedules(days=1)
    first = pg.book("心内科", "p1", date=_TODAY, time_slot="上午", idempotency_key="req-1")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9

    again = pg.book("心内科", "p1", date=_TODAY, time_slot="上午", idempotency_key="req-1")
    assert again["ticket_no"] == first["ticket_no"]
    assert again["queue_no"] == first["queue_no"]
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9, "幂等命中不该再扣一个号"
    assert _appointment_count(schema) == 1

    # 换一个键就是一次新的请求
    third = pg.book("心内科", "p1", date=_TODAY, time_slot="上午", idempotency_key="req-2")
    assert third["ticket_no"] != first["ticket_no"]
    assert _remaining(schema, "心内科", _TODAY, "上午") == 8

    # 不传键（None / 空串）走普通路径，不去重
    a = pg.book("心内科", "p1", date=_TODAY, time_slot="上午", idempotency_key="")
    b = pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    assert a["ticket_no"] != b["ticket_no"]
    assert _appointment_count(schema) == 4


def test_idempotency_key_under_concurrency_yields_one_ticket(
    make: Callable[..., PgAppointmentProvider], schema: str
) -> None:
    """**两线程同键同刻**：只能出一张单、只扣一个号，两边拿到同一个 `ticket_no`。

    走的是"唯一约束 → 整个事务回滚（**扣号一起回滚**）→ 独立事务回读原单"这条路。
    两个关键点缺一不可：扣减与插入必须在**同一个事务**里（否则输的那次会白扣一个号），
    以及回读必须用**新连接/新事务**（旧事务已经因唯一冲突而作废，在里面读只会看到
    `current transaction is aborted`）。
    """
    provider = make(pool_max=8)
    provider.generate_schedules(days=1)
    barrier = threading.Barrier(2)

    def _go(_i: int) -> dict:
        barrier.wait()
        return provider.book("心内科", "p1", date=_TODAY, time_slot="上午", idempotency_key="same-request")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(_go, range(2)))

    assert first["ticket_no"] == second["ticket_no"], "同一个幂等键必须回到同一张单"
    assert first["queue_no"] == second["queue_no"]
    assert _appointment_count(schema) == 1, "同键并发只允许出一张单"
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9, "输的那次不该留下扣号"


# --------------------------------------------------------------------------- 退号与记录
def test_cancel_releases_remaining_and_rejects_wrong_patient_and_double_cancel(
    pg: PgAppointmentProvider, schema: str
) -> None:
    """退号：本人才能退、退完立刻回补余量、重复退报错、退完仍留在记录里。"""
    pg.generate_schedules(days=1)
    ticket = pg.book("心内科", "p1", date=_TODAY, time_slot="上午", idempotency_key="k")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9

    with pytest.raises(BookingError, match="不能退别人的号"):
        pg.cancel(ticket["ticket_no"], "p2")
    assert (
        _scalar(schema, "SELECT status FROM {s}.appointments WHERE ticket_no = %s", (ticket["ticket_no"],))
        == "paid"
    )
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9, "退别人的号不该释放余量"

    released = pg.cancel(ticket["ticket_no"], "p1")
    assert released["status"] == "cancelled"
    assert released["remaining"] == 10
    assert _remaining(schema, "心内科", _TODAY, "上午") == 10

    with pytest.raises(BookingError, match="已经退过"):
        pg.cancel(ticket["ticket_no"], "p1")
    assert _remaining(schema, "心内科", _TODAY, "上午") == 10, "重复退号会凭空多出一个号"

    with pytest.raises(BookingError, match="没有找到"):
        pg.cancel("NOSUCHNO", "p1")
    with pytest.raises(BookingError, match="patient_id"):
        pg.cancel(ticket["ticket_no"], "")

    # 退掉的号仍然是历史记录（否则回答不了"我上周是不是退过一个号"）
    mine = pg.list_appointments("p1")
    assert [a["ticket_no"] for a in mine] == [ticket["ticket_no"]]
    assert mine[0]["status"] == "cancelled"
    # 退号后重新占号：余量被复用，队列号继续递增
    rebooked = pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    assert rebooked["queue_no"] == 2
    assert _remaining(schema, "心内科", _TODAY, "上午") == 9


def test_my_appointments_only_returns_own_records(pg: PgAppointmentProvider, schema: str) -> None:
    """`my_appointments` 只返回自己的记录、新到旧、限量生效；上层 `BookingService` 能直接用。"""
    pg.generate_schedules(days=1)
    morning = pg.book("心内科", "p1", "甲", date=_TODAY, time_slot="上午")
    afternoon = pg.book("心内科", "p1", "甲", date=_TODAY, time_slot="下午")
    other = pg.book("心内科", "p2", "乙", date=_TODAY, time_slot="上午")

    assert isinstance(pg, AppointmentProvider), "必须实现现有 ABC"
    mine = pg.list_appointments("p1")
    assert [a["ticket_no"] for a in mine] == [afternoon["ticket_no"], morning["ticket_no"]], "新到旧"
    assert {a["patient_id"] for a in mine} == {"p1"}
    assert other["ticket_no"] not in {a["ticket_no"] for a in mine}
    assert [a["ticket_no"] for a in pg.list_appointments("p1", limit=1)] == [afternoon["ticket_no"]]
    assert pg.list_appointments("p2")[0]["ticket_no"] == other["ticket_no"]
    assert pg.list_appointments("从没挂过号的人") == []
    with pytest.raises(BookingError, match="patient_id"):
        pg.list_appointments("")

    # 上层服务不用改一行就能接上（同一个契约）
    service = BookingService(pg)
    assert [a["ticket_no"] for a in service.my_appointments("p1")] == [a["ticket_no"] for a in mine]
    assert service.departments()[0]["name"] == DEPARTMENTS[0]
    # 心内科：上午被 p1/p2 各占一个（8），下午被 p1 占一个（9）
    assert service.availability("心内科", _TODAY.isoformat()).total_remaining == 17
    booked = service.book("心内科", "p3", date=_TODAY.isoformat(), time_slot="上午")
    assert booked["department"] == "心内科"
    assert service.my_appointments("p3")[0]["ticket_no"] == booked["ticket_no"]
    assert set(PERIODS) == {s["time"] for s in service.availability("心内科", _TODAY.isoformat()).slots}


def test_booking_service_cancel_returns_provider_result(pg: PgAppointmentProvider, schema: str) -> None:
    """`BookingService.cancel` 把号源的取消结果**原样回给上层**（前端要展示"余量已释放"）。

    之前它返回 None，上层拿不到释放后的余量，只能再查一次号源。
    """
    pg.generate_schedules(days=1)
    ticket = pg.book("心内科", "p1", date=_TODAY, time_slot="上午")
    service = BookingService(pg)

    result = service.cancel(ticket["ticket_no"], "p1")
    assert result["ticket_no"] == ticket["ticket_no"]
    assert result["status"] == "cancelled"
    assert result["remaining"] == 10, "释放后的余量必须能直接展示"
    assert _remaining(schema, "心内科", _TODAY, "上午") == 10
    # 不支持取消的实现仍然是一句业务错误（Mock 至今没有 cancel）
    with pytest.raises(BookingError, match="不支持取消"):
        BookingService(MockAppointmentProvider()).cancel("X1", "p1")


def test_availability_date_is_never_empty(pg: PgAppointmentProvider) -> None:
    """`Availability.date` 的三级兜底：显式传入 > 时段自带 > 今天，**空 slots 也不能回 ""**。

    空串会被工具层原样回给模型（`get_availability` 的返回里有 `date`），渲染出来就是
    "没有日期"——而号源在 `date=None` 时查的就是今天，回今天才对得上查到的号。
    """
    service = BookingService(pg)
    assert pg.get_availability("心内科") == [] and service.availability("心内科").slots == []
    assert service.availability("心内科").date == _TODAY.isoformat(), "空 slots 时回今天，不是空串"

    future = (_TODAY + dt.timedelta(days=30)).isoformat()
    assert service.availability("心内科", future).date == future, "显式传入的日期优先回显"

    # 有排班时用时段自带的 date（号源解析出的日期比调用方的假设更可信）
    pg.generate_schedules(days=1)
    assert service.availability("心内科").date == _TODAY.isoformat()
