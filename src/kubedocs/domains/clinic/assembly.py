"""号源装配：`build_booking_service(settings)` —— 三个入口共用的**唯一**构造点。

`/consult`、`/chat`、MCP 服务器都要一个 `BookingService`。如果各自构造，就会出现
"问诊页占的号在对话页查不到"这类问题（两个连接池、两份号源视图）。所以构造逻辑放这里，
调用方只管拿。

## Postgres 优先，连不上降级

默认号源是 `PgAppointmentProvider`（`booking` schema）：号源必须跨重启、跨副本存在，
否则"昨天挂的号今天查不到"会直接毁掉演示。数据库不可达时退回进程内 `MockAppointmentProvider`
并留一条 warning——与就诊记录降级是同一个取舍：**宁可少一份持久化，也不能让患者打不开页面**。
降级状态由调用方在 `/consult/status` 里如实报出（`booking_backend`）。

`settings=None`（测试接缝里的假 runtime）直接用进程内号源，一个真实数据库都不碰。

## 排班在这里顺手补齐

拿到 Postgres 号源后立刻调一次 `generate_schedules()`：窗口内没有任何 `schedules` 行时，
科室是"可挂号"的，但每个时段都查不到——患者会看到"没有排班"，而不是"今天没号"。
它是幂等的（`ON CONFLICT DO NOTHING`，**绝不覆盖已有余量**），所以每次进程起来都调一次
既安全，也正是"放号窗口永远向前滚动"的实现方式。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_booking_service(settings: Any) -> Any:
    """按 settings 造一个 `BookingService`（Postgres 优先，失败降级进程内）。"""
    from kubedocs.domains.clinic.booking import BookingService
    from kubedocs.domains.clinic.mock import MockAppointmentProvider

    if settings is None:
        return BookingService(MockAppointmentProvider())
    try:
        from kubedocs.domains.clinic.pg_provider import from_settings as pg_from_settings
        from kubedocs.domains.clinic.pg_provider import probe

        ok, why = probe(settings.database_url)
        if not ok:
            raise RuntimeError(why)
        service = BookingService(pg_from_settings(settings))
        try:
            created = service.provider.generate_schedules()
            logger.info("号源排班就绪：本次新增 %s 个时段（窗口 7 天 × 上午/下午）", created)
        except Exception as e:  # noqa: BLE001 - 排班失败不该让服务起不来
            logger.warning("生成排班失败（%s: %s）：号源可能查不到时段", type(e).__name__, e)
        return service
    except Exception as e:  # noqa: BLE001 - 号源不可用不该拖垮整个服务
        logger.warning(
            "号源库（Postgres）不可用（%s：%s）：本次改用进程内号源，重启即丢",
            type(e).__name__,
            e,
        )
        return BookingService(MockAppointmentProvider())
