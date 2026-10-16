"""存储层：同一 `Store` 协议下的可替换实现，以及后端选择。

`KUBEDOCS_STORE` 的取值与对应实现：

| 取值 | 实现 | 依赖 | 定位 |
|---|---|---|---|
| `pgvector`（默认） | `PgVectorStore` | Postgres + pgvector + psycopg 连接池 | **向量库**：HNSW 近似检索、一库承担全文 + 向量、多实例共享 |
| `sqlite-vec` | `SqliteVecStore` | `sqlite-vec` 扩展 | 离线回退：零外部进程、零镜像、零网络 |
| `sqlite` | `SqliteStore` | 标准库 + numpy | 旧实现：numpy 暴力内积，性能对照与最小依赖 |

## 两条不同的失败路径，处理方式刻意不同

- **非法取值**（拼错的字符串、不存在的后端）**直接报错，绝不静默回退**——
  回退成默认后端会让"我明明切到 pgvector 了，为什么数据还是旧的"变成一个没有任何
  日志的谜。字面量校验在 `Settings`（启动即失败），工厂这里再兜一层。
- **数据库不可达**（Postgres 没起、网络不通）默认**降级到 `sqlite-vec` 并留一条
  warning**，把原因挂在返回对象的 `fallback_reason` 属性上。理由是"演示/离线开发
  必须能跑起来"：一台没装 Docker 的机器上连 `ingest` 都起不来，等于把本地开发
  锁死在基础设施上。想让它响亮失败就传 `allow_local_fallback=False`（例如生产启动脚本）。

  判据是**探库结果**，不是"构造有没有抛异常"：库可达但 schema 与配置不匹配（向量维度
  不符、schema 版本不符）属于配置错误，**原样抛出**，绝不降级——降级会让"我明明配的是
  384 维"变成一个安静的、在别处才爆的谜。

降级是**有代价**的，所以必须可见：两个后端的索引各自存在不同位置，**降级期间建的索引
在 Postgres 恢复后是看不见的**。`fallback_reason` 会被 HTTP 层的 `/healthz`、`/stats`
与控制台页面读出来，不是只写进日志。
"""

from __future__ import annotations

import logging

from kubedocs.config import Settings
from kubedocs.store.base import ChunkHit, ChunkInput, DocInput, Store, StoreStats, Vector
from kubedocs.store.sqlite_store import (
    MissingVectorError,
    SqliteStore,
    StoreError,
    build_match_expr,
)
from kubedocs.store.sqlite_store import from_settings as sqlite_from_settings

logger = logging.getLogger(__name__)

__all__ = [
    "BACKENDS",
    "ChunkHit",
    "ChunkInput",
    "DocInput",
    "MissingVectorError",
    "SqliteStore",
    "Store",
    "StoreError",
    "StoreStats",
    "Vector",
    "build_match_expr",
    "from_settings",
    "sqlite_from_settings",
]

# 允许的后端名（与 Settings.store 的 Literal 保持一致，这里作为单一出口供测试与 CLI 引用）
BACKENDS = ("sqlite-vec", "sqlite", "pgvector")

# 数据库不可达时的降级目标。选 `sqlite-vec` 而不是 `sqlite`：前者是本地档里更快的那个
# （向量走 vec0 虚拟表），降级不该顺带把性能也降一档。
FALLBACK_BACKEND = "sqlite-vec"


def from_settings(settings: Settings, *, allow_local_fallback: bool = True) -> Store:
    """按 `settings.store` 构造存储实现。

    延迟导入每个实现：只想用 `sqlite` 时不该因为缺 sqlite-vec 扩展、
    或没装 psycopg 而连进程都起不来。

    `allow_local_fallback` 只对**数据库不可达**生效（见模块文档）：置 `False` 时
    pgvector 的失败会原样抛出，适合"基础设施没起就该响亮失败"的生产启动路径。
    """
    backend = settings.store
    if backend == "sqlite":
        from kubedocs.store.sqlite_store import from_settings as sqlite_fs

        return sqlite_fs(settings)
    if backend == "sqlite-vec":
        from kubedocs.store.sqlite_vec_store import from_settings as vec_from_settings

        return vec_from_settings(settings)
    if backend == "pgvector":
        from kubedocs.store.pgvector_store import from_settings as pg_from_settings
        from kubedocs.store.pgvector_store import probe as pg_probe

        # **先探库，再决定降不降级**。判据只能是"数据库可达吗"，不能是"构造抛没抛异常"：
        # `PgVectorStore` 把连接失败和"schema 与配置不匹配"（维度不符、schema 版本不符）
        # 都包成 `StoreError`，按异常类型分派会把后者也悄悄降级——那是配置错误，
        # 必须响亮失败，否则会以"索引里怎么什么都没有"的形式在别处爆出来。
        ok, why = pg_probe(settings.database_url)
        if not ok:
            if not allow_local_fallback:
                raise StoreError(f"Postgres 不可达（{why}）：{settings.database_url!r}")
            reason = (
                f"{backend} 不可达（{why}）⇒ 本次降级为 {FALLBACK_BACKEND}。"
                f"注意：降级期间写入的索引存在本地，Postgres 恢复后**看不到**它；"
                f"要回到生产后端，先起数据库再重跑 ingest。"
            )
            logger.warning("%s", reason)
            from kubedocs.store.sqlite_vec_store import from_settings as vec_from_settings

            store = vec_from_settings(settings)
            store.fallback_reason = reason  # type: ignore[attr-defined]
            return store
        # 库可达 ⇒ 任何失败都是配置问题，原样抛出（不降级）。
        return pg_from_settings(settings)
    # Settings 的 Literal 已经挡住非法值；这层兜底是为了"绕过 Settings 直接调用"的路径
    raise StoreError(
        f"未知的存储后端 {backend!r}（可选：{', '.join(BACKENDS)}）——"
        "不会静默回退到任何默认实现，请显式改成合法值"
    )
