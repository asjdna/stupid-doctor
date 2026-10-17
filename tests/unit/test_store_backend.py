"""后端选择：`KUBEDOCS_STORE` 的取值、工厂分派与协议覆盖。

`KUBEDOCS_STORE` 是**唯一**决定"索引数据放在哪、用什么算法检索"的开关，它的两种出错方式
都必须被挡住：**静默回退**（配了 `pgvector` 却因连不上而悄悄用本地 SQLite）与
**非法取值**（必须在启动时报错，不等到首次检索，也不自动纠正成默认值）。
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from kubedocs.config import Settings
from kubedocs.store import BACKENDS, Store, StoreError, from_settings
from kubedocs.store.pgvector_store import PgVectorStore, probe
from kubedocs.store.sqlite_store import SqliteStore
from kubedocs.store.sqlite_vec_store import SqliteVecStore


def _settings(tmp_path: Path, **over: object) -> Settings:
    base: dict[str, object] = {
        "index_dir": str(tmp_path / "index"),
        "domain": "k8s_docs",
        "embedding_provider": "hashing",
        "embedding_dim": 32,
        "embedding_model": "hashing-test",
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


class TestDefaultBackend:
    """默认后端：未显式配置时 `KUBEDOCS_STORE` 取 `pgvector`。"""

    def test_default_is_pgvector(self):
        # 用字段默认值而不是 Settings() 实例：实例会读到 .env，测的就不是代码默认值。
        assert Settings.model_fields["store"].default == "pgvector"

    def test_backend_names_are_stable(self):
        # 名字是配置面的一部分（KUBEDOCS_STORE 的取值），改名等于破坏兼容
        assert BACKENDS == ("sqlite-vec", "sqlite", "pgvector")


class TestUnreachableDatabaseFallback:
    """配置了 pgvector 但数据库不可达：降级为本地后端，**并且把原因带出来**。

    这是"离线也能演示"与"降级必须可见"两条要求的交点：静默降级会让
    "Postgres 恢复后看不到降级期间建的索引"变成一个没人预期过的差异。
    """

    def test_pgvector_unreachable_falls_back_to_local(self, tmp_path: Path):
        s = _settings(
            tmp_path,
            store="pgvector",
            database_url="postgresql://kubedocs:kubedocs@127.0.0.1:1/kubedocs",
        )
        store = from_settings(s)
        try:
            assert isinstance(store, SqliteVecStore), "数据库不可达时必须降级，而不是抛异常"
            reason = getattr(store, "fallback_reason", "")
            assert reason, "降级必须留下可见的原因（HTTP `/healthz` 与 CLI stats 都读它）"
            assert "pgvector" in reason and "sqlite-vec" in reason
        finally:
            store.close()

    def test_fallback_can_be_disabled_for_production_startup(self, tmp_path: Path):
        """生产启动路径要能要求"基础设施没起就响亮失败"。"""
        s = _settings(
            tmp_path,
            store="pgvector",
            database_url="postgresql://kubedocs:kubedocs@127.0.0.1:1/kubedocs",
        )
        with pytest.raises(Exception):  # noqa: B017 - 具体类型取决于驱动，这里只要求"不吞"
            from_settings(s, allow_local_fallback=False)

    def test_illegal_backend_still_raises_instead_of_falling_back(self, tmp_path: Path):
        """非法取值与"连不上"是两条不同的路径：前者必须报错，绝不降级成任何后端。"""
        from kubedocs.store.sqlite_store import StoreError as _StoreError

        with pytest.raises((_StoreError, ValidationError)):
            from_settings(_settings(tmp_path, store="chroma"))


class TestEnvWiring:
    """环境变量必须真的生效（配置面写在文档里但读不到是最常见的失效）。"""

    @pytest.mark.parametrize("value", ["sqlite-vec", "sqlite", "pgvector"])
    def test_env_var_selects_backend(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("KUBEDOCS_STORE", value)
        assert Settings().store == value

    def test_illegal_env_value_fails_at_startup(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KUBEDOCS_STORE", "chroma")
        with pytest.raises(ValidationError):
            Settings()

    def test_illegal_value_is_not_silently_corrected(self):
        with pytest.raises(ValidationError):
            Settings(store="chroma")  # type: ignore[arg-type]


class TestFactory:
    """工厂：选谁就构造谁；不认识就报错。"""

    def test_sqlite_backend_builds_legacy_store(self, tmp_path: Path):
        s = from_settings(_settings(tmp_path, store="sqlite"))
        try:
            assert isinstance(s, SqliteStore)
            assert not isinstance(s, SqliteVecStore)  # 两个后端不能是同一个类
            assert isinstance(s, Store)  # 运行期也满足协议
        finally:
            s.close()

    def test_sqlite_vec_backend_builds_vec_store(self, tmp_path: Path):
        s = from_settings(_settings(tmp_path, store="sqlite-vec"))
        try:
            assert isinstance(s, SqliteVecStore)
            assert s.verify_fts() == "ok"
            assert isinstance(s, Store)
        finally:
            s.close()

    @pytest.mark.parametrize(
        ("backend", "module"),
        [
            ("sqlite", "kubedocs.store.sqlite_store"),
            ("sqlite-vec", "kubedocs.store.sqlite_vec_store"),
            ("pgvector", "kubedocs.store.pgvector_store"),
        ],
    )
    def test_dispatch_targets_exactly_one_implementation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, module: str
    ):
        """分派到正确模块，且**不会**顺带构造别的实现（不做"依次试三种"那种猜测）。

        用哨兵替掉三个模块各自的 `from_settings`：工厂若在"后端名不认 / 构造失败"之后
        换个实现顶上，计数器就会显示"构造了不止一个"。

        ⚠ `pgvector` 这一档要求**库可达**：一条连不上的 DSN 会走"降级为本地 sqlite-vec"
        那条**另一条**路径（那是刻意设计，见 `TestUnreachableDatabaseFallback`），与本用例
        要证的"分派不做猜测"不是同一件事。库不可达就 skip（与其它 pgvector 用例同一口径）。
        """
        if backend == "pgvector":
            ok, why = probe(_settings(tmp_path).database_url, timeout=2.0)
            if not ok:
                pytest.skip(f"pgvector 分派用例需要可达的库：{why}")

        calls: list[str] = []

        def make(name: str):
            def _fake(_settings: Settings) -> object:
                calls.append(name)
                # 允许工厂往返回对象上挂 `fallback_reason`：这条用例只数调用次数，
                # 用 object() 会让"真的发生了降级"以 AttributeError 的形式炸掉，
                # 而它应该以 assert 的形式说清"构造了不止一个实现"。
                return types.SimpleNamespace()

            return _fake

        for name, mod in (
            ("sqlite", "kubedocs.store.sqlite_store"),
            ("sqlite-vec", "kubedocs.store.sqlite_vec_store"),
            ("pgvector", "kubedocs.store.pgvector_store"),
        ):
            monkeypatch.setattr(f"{mod}.from_settings", make(name))

        from_settings(_settings(tmp_path, store=backend))
        assert calls == [backend]

    def test_unknown_backend_raises_instead_of_falling_back(self, tmp_path: Path):
        """绕过 Settings 直接调工厂（例如下游代码自己拼了个配置对象）也要报错。"""

        class _Fake:
            store = "faiss"
            index_dir = str(tmp_path / "index")

        with pytest.raises(StoreError, match="未知的存储后端"):
            from_settings(_Fake)  # type: ignore[arg-type]


class TestPgVectorProbe:
    """`probe()` 是"库不可达就 skip"的唯一判据，它自己必须可靠：两条用例只靠一条连不通
    的 DSN，任何环境下都会跑，否则探测逻辑坏掉只会表现为"pgvector 用例全被 skip"（那是绿的）。
    """

    def test_unreachable_dsn_reports_reason_not_exception(self):
        ok, why = probe("postgresql://kubedocs:kubedocs@127.0.0.1:1/kubedocs", timeout=2.0)
        assert ok is False
        assert why.strip(), "skip 原因不能为空——空的 reason 等于没有信息"

    def test_malformed_dsn_reports_reason(self):
        ok, why = probe("not-a-dsn", timeout=2.0)
        assert ok is False and why.strip()


class TestProtocolCoverage:
    """协议与实现的同步：给 `Store` 加方法时，三个实现都得跟上。

    mypy 已在源码上保证，这里再测一次是因为 `Store` 是 `runtime_checkable` 的协议，
    `getattr` 风格的下游代码与测试替身都依赖同一份方法清单。
    """

    def test_every_implementation_covers_the_protocol(self):
        members = {name for name in dir(Store) if not name.startswith("_")}
        assert {"dense_search", "keyword_search", "upsert_docs", "finalize"} <= members
        for cls in (SqliteStore, SqliteVecStore, PgVectorStore):
            missing = {n for n in members if not callable(getattr(cls, n, None))}
            assert not missing, f"{cls.__name__} 缺少协议方法：{sorted(missing)}"

    def test_sqlite_stores_are_runtime_protocol_instances(self, tmp_path: Path):
        for backend, cls in (("sqlite", SqliteStore), ("sqlite-vec", SqliteVecStore)):
            s = from_settings(_settings(tmp_path, store=backend))
            try:
                assert isinstance(s, cls) and isinstance(s, Store)
            finally:
                s.close()
