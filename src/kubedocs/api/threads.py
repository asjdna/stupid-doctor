"""`/chat` 的会话线程表：**属主绑定** + **有界淘汰**。

表键是客户端传来的 `thread_id`，所以服务端必须管住两件事：没有属主校验的话，猜到
别人的 `thread_id` 就能续聊别人的会话（历史里有别人的主诉、科室甚至挂号动作）；
没有上界的话，每个新 `thread_id` 都留下一份历史，而 `thread_id` 由调用方自带、
数量不限，进程内存会无界增长。

## 属主 = `patient_id`

属主就是请求体里与"人"绑定的 `patient_id`（`api/schemas.py`，默认 `"anon"`）。
它**不是认证**：默认值让所有匿名客户端共享同一个属主桶，它们之间仍只靠 `thread_id`
的随机性（32 位十六进制）隔离。它挡的是**误撞 / 复用别人的 `thread_id`**，
挡不住有意的身份伪造——"这是谁"必须由网关层解决。

## 为什么是条数上界（LRU）而不是 TTL

要防的是**内存无界增长**，所以界必须确定：`max_threads × max_messages` 条消息，
与请求速率无关。TTL 给不出这个保证——一阵突发流量可以在任何 TTL 到期前塞进任意多个
线程。用 LRU 而不是 FIFO：最近还在说话的会话才是该留下的那个。

## 并发

`/chat` 跑在线程池里（`asyncio.to_thread`），这张表会被多个线程同时读写，读写一律持
`threading.Lock`；尤其"先看属主再改历史"必须是**一次**临界区，否则两个请求可以各自
看到"还没有这个线程"然后各绑一个属主。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_THREADS = 256
"""同时保留的会话数上限（LRU 淘汰最久未用者）。

256 个会话 × 每会话最多 8 条消息 ≈ 最多 2048 条消息常驻。按单会话数 KB 估，
这是**几 MB 量级**的上界：对演示/单机部署足够宽松，又不会成为内存泄露的出海口。
"""

MAX_MESSAGES_PER_THREAD = 8
"""单会话保留的消息条数（4 轮 user/assistant，与 `api/app.py` 的历史口径一致）。"""


class ThreadOwnershipError(Exception):
    """该 `thread_id` 属于别的 `patient_id`。

    单独一个异常类型（而不是就地判 `if` 返回 False）是为了让调用方**必须**处理它：
    静默把它当成新会话、或静默接到别人的历史上，都是这张表要消灭的行为。
    """

    def __init__(self, thread_id: str) -> None:
        # 注意：**不回显属主是谁**。那会把"某个 thread_id 属于哪个 patient_id"
        # 变成一次探测就能拿到的信息，等于自己泄露会话归属。
        super().__init__(
            f"thread_id {thread_id!r} 已被另一个 patient_id 使用：不能续聊别人的会话。"
            "请换一个 thread_id，或省略 thread_id 让服务端新开一个会话。"
        )
        self.thread_id = thread_id


@dataclass
class _Thread:
    """一个会话：属主 + 最近若干条消息。"""

    owner: str
    messages: list[dict[str, Any]] = field(default_factory=list)


class ThreadStore:
    """有界 + 带属主的会话表。键是客户端给的 `thread_id`。"""

    def __init__(
        self, *, max_threads: int = MAX_THREADS, max_messages: int = MAX_MESSAGES_PER_THREAD
    ) -> None:
        if max_threads < 1:
            raise ValueError(f"max_threads 必须 >= 1（收到 {max_threads}）")
        if max_messages < 1:
            raise ValueError(f"max_messages 必须 >= 1（收到 {max_messages}）")
        self.max_threads = max_threads
        self.max_messages = max_messages
        self._threads: OrderedDict[str, _Thread] = OrderedDict()
        self._lock = threading.Lock()
        self.n_evicted = 0
        """被淘汰掉的线程数（累计）。留作可见性：它是"上界真的在起作用"的唯一证据。"""

    # ------------------------------------------------------------------ 属主
    def claim(self, thread_id: str, owner: str) -> None:
        """认领 `thread_id`；**已被别人占用则抛 `ThreadOwnershipError`**。

        首次出现即绑定属主（所以在处理这一轮请求**之前**调用它，
        才不会有"两个属主都以为自己先到"的窗口）。
        """
        with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                self._threads[thread_id] = _Thread(owner=owner)
                self._threads.move_to_end(thread_id)
                self._evict_locked()
                return
            if thread.owner != owner:
                raise ThreadOwnershipError(thread_id)
            self._threads.move_to_end(thread_id)

    # ------------------------------------------------------------------ 历史
    def history(self, thread_id: str, owner: str) -> list[dict[str, Any]]:
        """取该线程的历史（副本）。未知线程返回空表；属主不符抛异常。"""
        with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                return []
            if thread.owner != owner:
                raise ThreadOwnershipError(thread_id)
            self._threads.move_to_end(thread_id)
            return list(thread.messages)

    def append(self, thread_id: str, owner: str, entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """追加消息并保留最近 `max_messages` 条，返回保留后的历史。

        线程不存在时**按该属主新建**（`claim` 已经在端点里做过一次属主判定，
        这里是幂等的兜底）；属主不符同样抛异常。
        """
        with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                thread = _Thread(owner=owner)
                self._threads[thread_id] = thread
            elif thread.owner != owner:
                raise ThreadOwnershipError(thread_id)
            thread.messages = (thread.messages + list(entries))[-self.max_messages :]
            self._threads.move_to_end(thread_id)
            self._evict_locked()
            return list(thread.messages)

    # ------------------------------------------------------------------ 内省
    def owner_of(self, thread_id: str) -> str | None:
        """该线程的属主；不存在返回 None。**只给测试与运维看，不进任何响应**。"""
        with self._lock:
            thread = self._threads.get(thread_id)
            return thread.owner if thread is not None else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._threads)

    def _evict_locked(self) -> None:
        """超出上界就淘汰最久未用的（调用方必须已持锁）。"""
        while len(self._threads) > self.max_threads:
            self._threads.popitem(last=False)
            self.n_evicted += 1
