"""存储层：同一 `Store` 协议下的可替换实现，以及后端选择。

| `KUBEDOCS_STORE` | 实现 | 依赖 | 定位 |
|---|---|---|---|
| `sqlite-vec`（默认） | `SqliteVecStore` | `sqlite-vec` 扩展 | 本地/CI：零外部进程、零镜像、零网络 |
| `sqlite` | `SqliteStore` | 标准库 + numpy | 旧实现，回退与性能对照 |
| `pgvector` | `PgVectorStore` | Postgres + pgvector + psycopg 连接池 | 生产：多实例写、一库承担全文+向量 |
| `milvus` | `MilvusStore` | `pymilvus` + `milvus-lite`（本地文件） | 可选：**唯一真正提供 ANN 的嵌入式后端**（HNSW） |

选型理由、HNSW 参数与实测数字见 `docs/adr/0004-向量存储后端.md` 与
`docs/adr/0005-milvus-后端.md`。

**非法取值直接报错，不静默回退**：配置写错时人的第一反应是"这服务怎么还在跑"，
而不是"我配错了"。静默回退成默认后端会让"我明明切到 pgvector 了，为什么数据
还是旧的"变成一个没有任何日志的谜。所以：字面量校验在 `Settings`（启动即失败），
工厂这里再兜一层（防止有人绕过 Settings 直接构造）。
"""

from __future__ import annotations

from kubedocs.config import Settings
from kubedocs.store.base import ChunkHit, ChunkInput, DocInput, Store, StoreStats, Vector
from kubedocs.store.sqlite_store import (
    MissingVectorError,
    SqliteStore,
    StoreError,
    build_match_expr,
)
from kubedocs.store.sqlite_store import from_settings as sqlite_from_settings

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
BACKENDS = ("sqlite-vec", "sqlite", "pgvector", "milvus")


def from_settings(settings: Settings) -> Store:
    """按 `settings.store` 构造存储实现。

    延迟导入每个实现：只想用 `sqlite` 时不该因为缺 sqlite-vec 扩展、
    或没装 psycopg 而连进程都起不来。
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

        return pg_from_settings(settings)
    if backend == "milvus":
        from kubedocs.store.milvus_store import from_settings as milvus_from_settings

        return milvus_from_settings(settings)
    # Settings 的 Literal 已经挡住非法值；这层兜底是为了"绕过 Settings 直接调用"的路径
    raise StoreError(
        f"未知的存储后端 {backend!r}（可选：{', '.join(BACKENDS)}）——"
        "不会静默回退到任何默认实现，请显式改成合法值"
    )
