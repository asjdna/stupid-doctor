"""待确认挂号单（`PendingBookings`）：把"模型提议"与"用户点头"分成两步。

## 为什么需要这一层

挂号是**不可逆的写操作**。让模型直接调 `book_appointment` 就把两件事混成了一件：
模型既提议、又执行，患者只在事后看见一句"已为您挂上"。产品要求的是相反的顺序：

1. **模型只能提议**（`propose_booking`，只读）。它验证科室/日期/时段真有号，然后产出
   一张**待确认的挂号单**——号源一个都没动。
2. **患者点确认**才执行（`POST /consult/booking/confirm`）。执行的是服务端，
   参数取自服务端存的这张单子，**不是**前端或模型回传的字段。

## 为什么不把单子交给前端保管

如果确认请求里带的是 `{department, date, time_slot, fee}`，前端就能改金额、改科室，
或者重放别人的请求。所以确认请求只带 `proposal_id`，真正的参数留在服务端。
`patient_id` 在**提议和执行时都要重新核对**：单子只属于提议它的人。

## 一次性（one-shot）

`take()` 取出即作废。双击确认、网络重试、刷新页面重放，都只会有一张单子生效——
"重复提交变成两张号"这类问题在这里就被掐掉，不必指望下游的幂等键兜底。

## 进程内存储的取舍

单子**只在内存里**：重启进程后待确认的单子全部失效，患者重新说一句"帮我挂个号"即可
（提议是只读的，重来一次没有代价）。反过来，把单子落库就得配套一个过期清扫任务，
而它的唯一作用是让"重启后还能点 15 分钟前那张单子"成立——那件事对患者没有价值。
所以这里明确选内存版，并在 `/consult/status` 里如实报出待确认单数。
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_TTL_SECONDS = 900
"""单子有效期（15 分钟）。取值理由是"够患者读完并决定"，不是安全边界。"""

DEFAULT_MAX_PENDING = 500
"""进程内单子上限。超出按最早到期的先丢——它本来就快过期了。"""


@dataclass(frozen=True)
class BookingProposal:
    """一张待确认的挂号单。**字段在提议时定死**，执行时不再接受任何外部输入。"""

    proposal_id: str
    patient_id: str
    department: str
    date: str
    time_slot: str
    fee: int
    patient_name: str
    remaining: int
    """提议那一刻该时段的余量。只用于展示"还剩几个号"，执行时会重新判定。"""
    created_at: float
    expires_at: float

    def as_payload(self) -> dict[str, Any]:
        """给模型看 / 给前端渲染的形状。**不含 `patient_id`**：那是服务端的事。"""
        return {
            "proposal_id": self.proposal_id,
            "department": self.department,
            "date": self.date,
            "time_slot": self.time_slot,
            "fee": self.fee,
            "patient_name": self.patient_name,
            "remaining": self.remaining,
            "expires_at": _iso(self.expires_at),
        }


def _iso(ts: float) -> str:
    """时间戳 → 本地时间的 ISO 字符串（前端直接显示，不做二次换算）。"""
    import datetime as dt

    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


class PendingBookings:
    """进程内、线程安全、一次性消费的待确认挂号单。

    HTTP 层在线程池里跑端点（FastAPI 的同步端点），工具循环也可能在别的线程执行，
    所以进出都要加锁。
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_items: int = DEFAULT_MAX_PENDING,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds 必须为正数，收到 {ttl_seconds!r}")
        if max_items < 1:
            raise ValueError(f"max_items 必须 >= 1，收到 {max_items!r}")
        self.ttl_seconds = float(ttl_seconds)
        self.max_items = int(max_items)
        self._items: dict[str, BookingProposal] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 写
    def propose(
        self,
        *,
        patient_id: str,
        department: str,
        date: str,
        time_slot: str,
        fee: int,
        patient_name: str = "",
        remaining: int = 0,
        now: float | None = None,
    ) -> BookingProposal:
        """登记一张待确认单，返回它。`patient_id` 为空直接拒绝（无主单子无法核对归属）。"""
        pid = str(patient_id or "").strip()
        if not pid:
            raise ValueError("待确认挂号单必须属于某个 patient_id（否则确认时无法核对归属）")
        stamp = time.time() if now is None else float(now)
        proposal = BookingProposal(
            proposal_id=secrets.token_urlsafe(16),
            patient_id=pid,
            department=str(department),
            date=str(date),
            time_slot=str(time_slot),
            fee=int(fee),
            patient_name=str(patient_name or ""),
            remaining=int(remaining),
            created_at=stamp,
            expires_at=stamp + self.ttl_seconds,
        )
        with self._lock:
            self._sweep(stamp)
            self._items[proposal.proposal_id] = proposal
            # 超上限时丢最早过期的：它们本来就在队首，不必另算 LRU。
            while len(self._items) > self.max_items:
                oldest = min(self._items, key=lambda k: self._items[k].expires_at)
                self._items.pop(oldest, None)
        return proposal

    def take(self, proposal_id: str, *, patient_id: str, now: float | None = None) -> BookingProposal:
        """取出并作废一张单子。**一次性**：同 id 第二次调用一定失败。

        三类拒绝各有各的话术，因为患者能做的最优动作不同：
        没找到 / 过期 ⇒ 重新说一句"帮我挂号"；不是本人 ⇒ 换回自己的会话。
        """
        from kubedocs.domains.clinic.provider import BookingError

        stamp = time.time() if now is None else float(now)
        with self._lock:
            self._sweep(stamp)
            proposal = self._items.pop(str(proposal_id or ""), None)
        if proposal is None:
            raise BookingError("这张挂号单已失效（可能已确认过、或超过 15 分钟），请重新说一次「帮我挂号」")
        if proposal.patient_id != str(patient_id or "").strip():
            raise BookingError("这张挂号单不属于当前就诊人，不能确认")
        return proposal

    def drop(self, proposal_id: str) -> None:
        """作废一张单子（用户点了"先不挂"）。不存在就静默通过。"""
        with self._lock:
            self._items.pop(str(proposal_id or ""), None)

    # ------------------------------------------------------------------ 读
    def get(self, proposal_id: str, *, now: float | None = None) -> BookingProposal | None:
        """只读地看一眼（用于运维读数与测试）。**不会**消费掉单子。"""
        stamp = time.time() if now is None else float(now)
        with self._lock:
            self._sweep(stamp)
            return self._items.get(str(proposal_id or ""))

    def count(self, *, now: float | None = None) -> int:
        """当前未过期单数（`/consult/status` 用）。"""
        stamp = time.time() if now is None else float(now)
        with self._lock:
            self._sweep(stamp)
            return len(self._items)

    def clear(self) -> None:
        """清空（测试与"重新开始"用）。"""
        with self._lock:
            self._items.clear()

    # ------------------------------------------------------------------ 内部
    def _sweep(self, now: float) -> None:
        """丢掉已过期的单子。**调用方必须已持锁**。

        每次进出都顺手扫一遍：单子数量级很小（上限 500），扫一遍比开一个后台清理线程
        便宜得多，也少一类"清理线程挂了但没人发现"的故障。
        """
        if not self._items:
            return
        dead = [k for k, v in self._items.items() if v.expires_at <= now]
        for k in dead:
            self._items.pop(k, None)
