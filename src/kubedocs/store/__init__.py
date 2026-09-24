"""存储层：同一 Store 协议下的可替换实现。"""
from __future__ import annotations

from kubedocs.store.base import ChunkHit, ChunkInput, DocInput, Store, StoreStats, Vector
from kubedocs.store.sqlite_store import (
    MissingVectorError,
    SqliteStore,
    StoreError,
    build_match_expr,
    from_settings,
)

__all__ = [
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
]
