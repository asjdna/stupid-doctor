"""后端选择：默认值、环境变量、非法取值必须报错、工厂分派。

## 这个文件守的是什么

`KUBEDOCS_STORE` 是**唯一**决定"索引数据放在哪、用什么算法检索"的开关。
它出错的方式有两种，都必须被测试挡住：

1. **静默回退**：配了 `pgvector` 却因为连不上而悄悄用本地 SQLite ——
   于是"我把数据灌进 Postgres 了，怎么检索不到"变成一个没有日志的谜。
   这里用"工厂分派"用例把它钉死：选谁就构造谁，不做任何替换。
2. **非法取值**：必须在**启动时**报错（`Settings` 的字面量校验），
   而不是等到第一次检索才炸，更不是自动纠正成默认值。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kubedocs.config import Settings
from kubedocs.store import BACKENDS, Store, StoreError, from_settings
from kubedocs.store.milvus_store import MilvusStore
from kubedocs.store.milvus_store import deps_available as milvus_deps_available
from kubedocs.store.pgvector_store import PgVectorStore, probe
from kubedocs.store.sqlite_store import SqliteStore
from kubedocs.store.sqlite_vec_store import SqliteVecStore


def milvus_available() -> tuple[bool, str]:
    """本机能否用 Milvus Lite（只查依赖，不打开数据目录——打开会占住 LOCK 不放）。"""
    return milvus_deps_available()


def _settings(tmp_path: Path, **over: object) -> Settings:
    base: dict[str, object] = {
        "index_dir": str(tmp_path / "index"), "domain": "k8s_docs",
        "embedding_provider": "hashing", "embedding_dim": 32,
        "embedding_model": "hashing-test",
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


class TestDefaultBackend:
    """默认值 = 新的 sqlite-vec 实现（用户已拍板的取舍，不是随手选的）。"""

    def test_default_is_sqlite_vec(self):
        # 用字段默认值而不是 Settings() 实例：实例会被 .env / 真实环境变量影响，
        # 那样测的是"我这台机器的配置"，不是"代码里的默认值"。
        assert Settings.model_fields["store"].default == "sqlite-vec"

    def test_backend_names_are_stable(self):
        # 名字是配置面的一部分（KUBEDOCS_STORE 的取值），改名等于破坏兼容
        assert BACKENDS == ("sqlite-vec", "sqlite", "pgvector", "milvus")


class TestEnvWiring:
    """环境变量必须真的生效（配置面写在文档里但读不到是最常见的失效）。"""

    @pytest.mark.parametrize("value", ["sqlite-vec", "sqlite", "pgvector", "milvus"])
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
            assert not isinstance(s, SqliteVecStore)      # 两个后端不能是同一个类
            assert isinstance(s, Store)                   # 运行期也满足协议
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

    @pytest.mark.parametrize(("backend", "module"), [
        ("sqlite", "kubedocs.store.sqlite_store"),
        ("sqlite-vec", "kubedocs.store.sqlite_vec_store"),
        ("pgvector", "kubedocs.store.pgvector_store"),
        ("milvus", "kubedocs.store.milvus_store"),
    ])
    def test_dispatch_targets_exactly_one_implementation(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
            backend: str, module: str):
        """分派到正确模块，且**不会**顺带构造别的实现（没有回退链）。

        用哨兵替掉四个模块各自的 `from_settings`：如果工厂在自己失败后回退，
        计数器就会显示"构造了不止一个"。
        """
        calls: list[str] = []

        def make(name: str):
            def _fake(_settings: Settings) -> object:
                calls.append(name)
                return object()
            return _fake

        for name, mod in (("sqlite", "kubedocs.store.sqlite_store"),
                          ("sqlite-vec", "kubedocs.store.sqlite_vec_store"),
                          ("pgvector", "kubedocs.store.pgvector_store"),
                          ("milvus", "kubedocs.store.milvus_store")):
            monkeypatch.setattr(f"{mod}.from_settings", make(name))

        from_settings(_settings(tmp_path, store=backend))
        assert calls == [backend]

    def test_milvus_backend_builds_milvus_store(self, tmp_path: Path):
        """`KUBEDOCS_STORE=milvus` 必须真的构造 MilvusStore（不是别的实现的别名）。"""
        from kubedocs.store.milvus_store import MilvusStore

        if not milvus_available()[0]:
            pytest.skip(f"Milvus Lite 不可用：{milvus_available()[1]}")
        s = from_settings(_settings(tmp_path, store="milvus"))
        try:
            assert isinstance(s, MilvusStore)
            assert isinstance(s, Store)
            assert s.verify_fts() == "ok"
        finally:
            s.close()

    def test_unknown_backend_raises_instead_of_falling_back(self, tmp_path: Path):
        """绕过 Settings 直接调工厂（例如下游代码自己拼了个配置对象）也要报错。"""
        class _Fake:
            store = "faiss"
            index_dir = str(tmp_path / "index")

        with pytest.raises(StoreError, match="未知的存储后端"):
            from_settings(_Fake)  # type: ignore[arg-type]


class TestPgVectorProbe:
    """`probe()` 是"数据库不可达就 skip"这条纪律的唯一判据，它自己必须可靠。

    这两条**不依赖数据库**（一条连不通的 DSN 就够了），所以在任何环境下都会跑：
    否则"探测逻辑坏了"只会表现为"pgvector 的用例全被 skip"，而那是绿的不是红的。
    """

    def test_unreachable_dsn_reports_reason_not_exception(self):
        ok, why = probe("postgresql://kubedocs:kubedocs@127.0.0.1:1/kubedocs", timeout=2.0)
        assert ok is False
        assert why.strip(), "skip 原因不能为空——空的 reason 等于没有信息"

    def test_malformed_dsn_reports_reason(self):
        ok, why = probe("not-a-dsn", timeout=2.0)
        assert ok is False and why.strip()


class TestProtocolCoverage:
    """协议与实现的同步：给 `Store` 加方法时，四个实现都得跟上。

    mypy 已经在 src 上保证了这一点；这里再测一次是因为
    `Store` 是 `runtime_checkable` 的协议，任何 `getattr` 风格的下游代码
    （以及测试替身）都依赖同一份方法清单。
    """

    def test_every_implementation_covers_the_protocol(self):
        members = {name for name in dir(Store) if not name.startswith("_")}
        assert {"dense_search", "keyword_search", "upsert_docs", "finalize"} <= members
        for cls in (SqliteStore, SqliteVecStore, PgVectorStore, MilvusStore):
            missing = {n for n in members if not callable(getattr(cls, n, None))}
            assert not missing, f"{cls.__name__} 缺少协议方法：{sorted(missing)}"

    def test_sqlite_stores_are_runtime_protocol_instances(self, tmp_path: Path):
        for backend, cls in (("sqlite", SqliteStore), ("sqlite-vec", SqliteVecStore)):
            s = from_settings(_settings(tmp_path, store=backend))
            try:
                assert isinstance(s, cls) and isinstance(s, Store)
            finally:
                s.close()

    def test_milvus_store_is_runtime_protocol_instance(self, tmp_path: Path):
        """Milvus Lite 可用时，`milvus` 后端也必须满足运行期协议。"""
        ok, why = milvus_available()
        if not ok:
            pytest.skip(f"Milvus Lite 不可用：{why}")
        s = from_settings(_settings(tmp_path, store="milvus"))
        try:
            assert isinstance(s, MilvusStore) and isinstance(s, Store)
        finally:
            s.close()
