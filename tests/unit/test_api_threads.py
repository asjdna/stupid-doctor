"""`api/threads.ThreadStore` 的单测：属主绑定 + 有界淘汰。

这一层是 `/chat` 会话状态的唯一入口，所以两条性质都在这里钉死：

1. **属主绑定**：`thread_id` 首见即绑定属主；换属主抛 `ThreadOwnershipError`
   （不是"当成新会话"，也不是"接到别人的历史上"）。
2. **有界**：LRU 上界，且淘汰的是**最久未用**的那个（不是最早创建的那个）。

为什么不用 HTTP 层测这两条：端到端用例只能覆盖"被拒"这一种表现，
而"两个属主同时认领时只有一个赢"这类竞态必须在**同一个临界区**上验证。
`/chat` 跑在线程池里，所以并发那条不是假想问题。
"""

from __future__ import annotations

import threading

import pytest

from kubedocs.api.threads import (
    MAX_MESSAGES_PER_THREAD,
    MAX_THREADS,
    ThreadOwnershipError,
    ThreadStore,
)


class TestOwnership:
    def test_first_claim_binds_the_owner(self) -> None:
        store = ThreadStore()
        store.claim("t1", "p1")
        assert store.owner_of("t1") == "p1"
        assert store.history("t1", "p1") == []

    def test_same_owner_may_claim_again(self) -> None:
        """兼容性钉子：同一属主反复认领（每一轮请求都会认领一次）必须放行。"""
        store = ThreadStore()
        for _ in range(5):
            store.claim("t1", "p1")
        assert store.owner_of("t1") == "p1"

    def test_another_owner_is_rejected(self) -> None:
        store = ThreadStore()
        store.claim("t1", "p1")
        with pytest.raises(ThreadOwnershipError, match="t1"):
            store.claim("t1", "p2")
        # 被拒之后属主没有被改写（不是"后来者覆盖"）
        assert store.owner_of("t1") == "p1"

    def test_rejection_does_not_leak_the_owner(self) -> None:
        """拒绝理由里不能出现属主是谁——那会变成一次免费的身份探测。"""
        store = ThreadStore()
        store.claim("t1", "患者甲")
        with pytest.raises(ThreadOwnershipError) as e:
            store.claim("t1", "患者乙")
        assert "患者甲" not in str(e.value)

    def test_history_and_append_also_enforce_the_owner(self) -> None:
        """不只是 `claim`：真正读写历史的那两个入口必须各自把关。"""
        store = ThreadStore()
        store.claim("t1", "p1")
        store.append("t1", "p1", [{"role": "user", "content": "我咳嗽"}])
        with pytest.raises(ThreadOwnershipError):
            store.history("t1", "p2")
        with pytest.raises(ThreadOwnershipError):
            store.append("t1", "p2", [{"role": "user", "content": "越权"}])
        assert store.history("t1", "p1") == [{"role": "user", "content": "我咳嗽"}]

    def test_unknown_thread_has_no_history(self) -> None:
        assert ThreadStore().history("never-seen", "p1") == []

    def test_concurrent_claims_have_exactly_one_winner(self) -> None:
        """竞态：4 个属主同时认领同一个 thread_id ⇒ 只能有一个成功。

        "先看属主、再建线程"必须是**一次**临界区，否则它们可以各自看到"还没有这个线程"。
        """
        store = ThreadStore()
        barrier = threading.Barrier(4)
        winners: list[str] = []
        lock = threading.Lock()

        def worker(owner: str) -> None:
            barrier.wait()
            try:
                store.claim("t-shared", owner)
            except ThreadOwnershipError:
                return
            with lock:
                winners.append(owner)

        threads = [threading.Thread(target=worker, args=(f"p{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(winners) == 1, f"同一个 thread_id 被多个属主认领：{winners}"


class TestBounded:
    def test_thread_count_is_capped(self) -> None:
        store = ThreadStore(max_threads=3)
        for i in range(5):
            store.claim(f"t{i}", "p1")
        assert len(store) == 3
        assert store.n_evicted == 2

    def test_eviction_is_lru_not_fifo(self) -> None:
        """被淘汰的是**最久未用**的：刚用过的那个必须留下。"""
        store = ThreadStore(max_threads=2)
        store.claim("a", "p1")
        store.claim("b", "p1")
        store.history("a", "p1")  # a 变成"最近用过"
        store.claim("c", "p1")  # 该淘汰 b

        assert store.owner_of("c") == "p1"
        assert store.owner_of("a") == "p1", "刚用过的线程被淘汰了"
        assert store.owner_of("b") is None

    def test_evicted_thread_can_be_reclaimed_by_anyone(self) -> None:
        """淘汰掉的线程不再有属主，所以新的 thread_id 出现时不会误伤（也算一次释放）。"""
        store = ThreadStore(max_threads=1)
        store.claim("old", "p1")
        store.claim("new", "p2")
        assert store.owner_of("old") is None
        store.claim("old", "p2")  # 现在它归 p2，不再被拒
        assert store.owner_of("old") == "p2"

    def test_messages_per_thread_are_capped(self) -> None:
        store = ThreadStore(max_messages=4)
        for i in range(6):
            store.append("t1", "p1", [{"role": "user", "content": f"第{i}句"}])
        kept = store.history("t1", "p1")
        assert len(kept) == 4
        assert [m["content"] for m in kept] == ["第2句", "第3句", "第4句", "第5句"]

    def test_history_is_a_copy(self) -> None:
        """返回的是副本：调用方改它不该影响表里的会话（否则淘汰/属主都没意义了）。"""
        store = ThreadStore()
        store.append("t1", "p1", [{"role": "user", "content": "原文"}])
        got = store.history("t1", "p1")
        got.append({"role": "user", "content": "偷偷加的"})
        assert len(store.history("t1", "p1")) == 1

    def test_defaults_are_documented_constants(self) -> None:
        """默认上界必须是**有限的**正数：这才是"无界增长"的修复。"""
        assert MAX_THREADS > 0
        assert MAX_MESSAGES_PER_THREAD > 0
        assert ThreadStore().max_threads == MAX_THREADS
        assert ThreadStore().max_messages == MAX_MESSAGES_PER_THREAD

    @pytest.mark.parametrize("kwargs", [{"max_threads": 0}, {"max_messages": 0}])
    def test_zero_bound_is_rejected(self, kwargs: dict[str, int]) -> None:
        """0 上界等于"什么都不留"，是配置错误，必须报错而不是静默退化。"""
        with pytest.raises(ValueError):
            ThreadStore(**kwargs)
