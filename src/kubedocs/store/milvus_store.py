"""可选后端：SQLite（FTS5 全文） + **Milvus Lite**（本地文件里的 HNSW 向量索引）。

全文路完全复用 `SqliteStore`：同一份 FTS5 外内容表、同一段 `keyword_search`、
同一组 BM25 列权重，因此 `sqlite` / `sqlite-vec` / `milvus` 三个后端的全文结果
逐字节相同。混合检索 = FTS5（jieba 分词）+ 稠密向量 + RRF，所以后端之间的差异
只可能来自向量路。`pgvector` 则是 Postgres + 服务端 HNSW。

**向量与块行不在同一个事务里**：Milvus Lite 是另一个数据目录、另一个引擎，没有
跨引擎事务可用。本实现用显式排序 + 显式检测代替假设：

1. `_index_chunk_vectors` 在 SQLite 事务提交**之前**写 Milvus，因此 Milvus 写失败
   会抛错并回滚 SQLite，不会留下"块写了、向量没写"的中间态。
2. 反向情形（Milvus 已写、SQLite 提交失败）会留下孤儿向量，无法用事务消掉，改为
   **检测**：`_refresh_vec_ready()` 在打开时与每次 `finalize()` 核对"块表行数 vs
   向量行数"，不一致就拒绝向量检索（抛 `StoreError` 并给出修复命令），绝不静默少召回；
   `verify_vec()` 另做一次**主键集合**级核对（连"行数相同但内容错位"也能看出来），
   由 `cli stats --verify` 显式调用。
3. 已知无法检测的窗口：整批 Milvus 写成功、SQLite 提交失败、且两次写入的 chunk_id
   完全相同（改写同一篇文档）——此时行数与主键集合都相同，只有向量内容是旧的。
   处理方式是 `scripts/migrate_to_milvus.py` 重灌或 `ingest --rebuild`。

HNSW 与度量：`M = 16`、`efConstruction = 200` 是**建索引期**参数，改了必须重建索引；
查询期 `ef`（默认 128）可随时调。度量默认 **COSINE**，与 sqlite-vec 的
`distance_metric=cosine`、pgvector 的 `vector_cosine_ops` 一致；向量已归一化，
此时余弦序与内积序严格相同。Milvus Lite 不回传索引参数（`describe_index` 只有
index_type/metric_type/field_name），所以本实现把 `M` / `efConstruction` / 度量
写进 SQLite 元信息（`milvus_hnsw_*`），配置与索引不一致时报错，而不是静默沿用
一个参数不同的旧索引。

未安装 `pymilvus`/`milvus-lite`、数据目录打不开、初始化失败——一律抛 `StoreError`
并给出可执行的修复提示，**不会**退回 numpy 暴力扫描：静默降级会让配置错误无法被发现。

运维上必须知道的两点：

1. **以 `.db` 结尾的路径是一个目录，不是文件**：里面有 `LOCK`、`manifest.json`、
   `wal/`、`partitions/**/data_*.parquet`，备份/清理要按目录处理。
2. **独占锁**：同一数据目录同一时刻只允许一个进程打开。也就是说 API 服务开着时，
   CLI 的 ingest/eval 打不开同一份索引——比 SQLite 的 WAL（单写多读）更严格。
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from kubedocs.config import Settings
from kubedocs.normalize import Normalizer, build_normalizer
from kubedocs.store.base import ChunkInput, StoreError, Vector
from kubedocs.store.sqlite_store import SqliteStore

__all__ = ["BACKEND", "COLLECTION", "MilvusStore", "deps_available", "from_settings", "probe"]

BACKEND = "milvus"
# 集合名固定：milvus-lite 的一个数据目录里只有一个集合，名字带 "vec" 是为了让
# "这是向量索引、不是块表"一目了然（块表始终在 SQLite 里）。
COLLECTION = "chunks_vec"

# 每批写入/删除的行数。Milvus 单次调用的载荷会整段序列化进 RPC，太大会让单次调用
# 的内存与失败代价一起膨胀；1000 行是插入吞吐与可控性之间的折中。
_BATCH = 1000

# chunk_id 的主键最大长度（字节），格式为 `<rel_path>::<ordinal>`。
# 放宽不会多占存储，但写不下只会在插入时才报错，所以留足余量。
_MAX_ID_BYTES = 512

_MIGRATE_HINT = (
    "请先迁移/重建向量索引（不必重新编码）：`python scripts/migrate_to_milvus.py`，"
    "或用 `python -m kubedocs.cli ingest --rebuild` 全量重建"
)

_DEP_HINT = (
    "milvus 后端需要 `pymilvus` + `milvus-lite`（本地文件模式，无需 Docker/服务端）："
    '`pip install "pymilvus>=2.5" "milvus-lite>=2.4"`；'
    "也可以改用 KUBEDOCS_STORE=sqlite-vec / sqlite / pgvector"
)


def _import_milvus() -> Any:
    """延迟导入 pymilvus，缺包时抛 `StoreError` 并附可执行的安装提示。

    延迟导入的理由：只用 `sqlite` 的人不该因为没装 pymilvus 而起不来进程，
    而缺包时必须给出提示，不能让 `ImportError` 裸奔到调用方。
    """
    try:
        from pymilvus import DataType, MilvusClient
        from pymilvus.exceptions import MilvusException
    except ImportError as e:  # pragma: no cover - 取决于环境
        raise StoreError(f"缺少依赖：{e}。{_DEP_HINT}——不会静默回退到 numpy 扫描") from e
    return DataType, MilvusClient, MilvusException


def deps_available() -> tuple[bool, str]:
    """只检查依赖（pymilvus + milvus-lite）是否装好，**不碰文件系统**。

    与 `probe()` 分开：本地文件模式"打开即创建"，真正的探测一定有副作用，
    而只想判断"这一路能不能跑"的调用方不该被逼着建目录。
    """
    try:
        import milvus_lite  # noqa: F401
        import pymilvus  # noqa: F401
    except ImportError as e:
        return False, f"缺少依赖：{e}。{_DEP_HINT}"
    return True, ""


def probe(path: str | Path) -> tuple[bool, str]:
    """探测 Milvus Lite 是否真的可用（依赖齐全 + 能打开一个数据目录）。

    与 `pgvector_store.probe()` 同一契约：让调用方区分"环境不可用（skip）"与
    "代码错了（失败）"。本地文件模式的两条语义使探测必然有副作用：

    1. **打开即创建**：`path` 会被当作**数据目录**创建（milvus-lite 把以 `.db` 结尾的
       路径当目录用，里面有 LOCK / manifest / parquet 段文件），所以调用方传临时路径；
    2. **独占锁**：同一数据目录同一时刻只允许一个进程打开，所以"另一个进程正开着"
       也会探测失败——那属于环境原因，应当 skip 而不是当成代码错误。
    """
    ok, why = deps_available()
    if not ok:
        return False, why
    try:
        _, client_cls, _ = _import_milvus()
        client = client_cls(str(path))
        client.close()
    except Exception as e:  # noqa: BLE001 - 探测就是要吞掉一切异常并转成原因字符串
        return False, f"{type(e).__name__}: {e}"
    return True, ""


class MilvusStore(SqliteStore):
    """SQLite（FTS5 全文）+ Milvus Lite（HNSW 向量）的可选实现。"""

    def __init__(
        self,
        db_path: str | Path,
        vector_path: str | Path,
        *,
        dim: int,
        embedding_model: str = "",
        normalizer: Normalizer | None = None,
        domain: str = "",
        milvus_path: str | Path | None = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        metric: str = "COSINE",
        allow_backend_takeover: bool = False,
    ) -> None:
        # 块行与全文路完全交给基类（同一个 SCHEMA、同一段 FTS5 代码、同一套元信息闸门）。
        # `vector_path` 在这里只作为旧实现的侧车矩阵位置：本实现不读也不写它，
        # 只在 `dense_search_exact()` 与迁移脚本里用它拿一份精确答案。
        super().__init__(
            db_path,
            vector_path,
            dim=dim,
            embedding_model=embedding_model,
            normalizer=normalizer,
            domain=domain,
        )
        self.milvus_path = (
            Path(milvus_path) if milvus_path is not None else Path(db_path).with_name("milvus.db")
        )
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.metric = str(metric).upper()
        # **只给迁移脚本用**的开关，见 `_check_backend_meta`：默认 False，
        # 也就是"从别的后端抢过这份索引"必须显式声明，不能顺带发生。
        self._allow_takeover = bool(allow_backend_takeover)
        # 精确检索用的内存副本（懒加载）：只在 `dense_search_exact()` 里用，
        # **不在查询路径上**——向量不常驻内存正是这个后端的特点之一。
        self._exact: tuple[list[str], np.ndarray] | None = None

        self._DataType, self._MilvusClient, self._MilvusException = _import_milvus()
        self._client = self._open_client()
        self._ensure_collection()
        self._check_backend_meta()
        self._check_hnsw_meta()
        self._refresh_vec_ready()

    # ------------------------------------------------------------------ 初始化
    def _open_client(self) -> Any:
        """打开 Milvus Lite 数据目录（本地文件模式，无服务端）。

        路径必须先自己校验：milvus-lite 只接受以 `.db` 结尾的本地路径或
        unix/http/tcp 形式的远端地址，否则抛出的 `uri ... is illegal` 对用户毫无帮助。

        **一个 `.db` 结尾的路径在 milvus-lite 里是目录，不是文件**（内含 LOCK /
        manifest.json / wal / parquet 段文件），且**同一目录同一时刻只允许一个进程
        打开**（第二个进程报 `DataDirLockedError`）。所以打开失败最常见的原因——
        被别的进程占着、父目录不可写——必须写进错误信息里。
        """
        if self.milvus_path.suffix.lower() != ".db":
            raise StoreError(
                f"Milvus Lite 数据目录必须以 .db 结尾（当前 {self.milvus_path}）；"
                "milvus-lite 只认以 .db 结尾的本地数据目录或 unix/http/tcp 形式的远端地址，"
                "请把 KUBEDOCS_MILVUS_DB 改成例如 "
                f"{self.milvus_path.with_suffix('.db')}"
            )
        self.milvus_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._MilvusClient(str(self.milvus_path))
        except Exception as e:  # noqa: BLE001 - 任何初始化异常都要转成人话提示
            raise StoreError(
                f"打开 Milvus Lite 数据目录失败（{self.milvus_path}）："
                f"{type(e).__name__}: {e}；常见原因："
                "① 另一个进程（正在跑的 API / 另一个 ingest / 上一次没退出的 CLI）"
                "持有同一数据目录的**独占锁**——milvus-lite 不支持多进程共用一个目录，"
                "请先停掉它；② 目录不可写或磁盘空间不足；③ 依赖没装好。"
                "本后端不会静默回退到别的向量实现"
            ) from e

    def _ensure_collection(self) -> None:
        """幂等建集合（含 HNSW 索引），并把声明维度读回来核对。

        `create_collection` 对已存在的集合会报错（不像 SQL 的 IF NOT EXISTS），
        幂等性只能靠先 `has_collection` 判断；并发建集合时再兜一层"建失败后复查
        是否真的存在"，存在就当幂等，否则才报错。
        """
        if not self._client.has_collection(COLLECTION):
            try:
                self._create_collection()
            except self._MilvusException as e:
                if not self._client.has_collection(COLLECTION):
                    raise StoreError(f"创建 Milvus 集合 {COLLECTION} 失败：{type(e).__name__}: {e}") from e
        declared = self._declared_dim()
        if declared is not None and declared != self.dim:
            raise StoreError(
                f"Milvus 向量索引维度不符：文件里是 {declared}，当前配置 {self.dim}；"
                "换了嵌入模型必须重建向量索引（`ingest --rebuild` 或 "
                "`python scripts/migrate_to_milvus.py`）"
            )

    def _create_collection(self) -> None:
        DataType = self._DataType
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=_MAX_ID_BYTES)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        index = self._client.prepare_index_params()
        index.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type=self.metric,
            params={"M": self.hnsw_m, "efConstruction": self.hnsw_ef_construction},
        )
        self._client.create_collection(COLLECTION, schema=schema, index_params=index)

    def _declared_dim(self) -> int | None:
        """从集合 schema 里读回向量维度（读不到就返回 None，交给别的闸门兜）。"""
        try:
            info = self._client.describe_collection(COLLECTION)
        except self._MilvusException as e:  # pragma: no cover - 只可能在引擎异常时发生
            raise StoreError(f"读取 Milvus 集合 schema 失败：{type(e).__name__}: {e}") from e
        for field in info.get("fields", []):
            if field.get("name") == "embedding":
                dim = field.get("params", {}).get("dim")
                return int(dim) if dim is not None else None
        return None

    def _check_backend_meta(self) -> None:
        """同一份 SQLite 索引的向量只允许由一个后端写入。

        行数核对挡不住跨后端写入：先用 sqlite-vec 建好索引、再用 milvus 摄取几批，
        此时 `chunks_vec` 里旧向量行数与块表仍然一致，sqlite-vec 会静默返回陈旧向量。
        只有"这份索引的向量是谁写的"这个标记能看出来。

        `allow_backend_takeover` 是唯一的出口，只有真正把向量搬过去的迁移脚本才打开它。
        """
        got = self._meta().get("vector_backend", "")
        if got and got != BACKEND and not self._allow_takeover:
            raise StoreError(
                f"索引由另一个后端写入（vector_backend={got!r}，本实现是 {BACKEND!r}）；"
                "同一份索引不要在多个后端间反复切换写（块表会更新、另一个后端的向量表不会，"
                "结果是静默读到陈旧向量）。要迁移向量请跑 "
                "`python scripts/migrate_to_milvus.py`（只搬向量，不重新编码）；"
                "要换回另一个后端请先 `python scripts/migrate_to_milvus.py --detach`，"
                "再 `python -m kubedocs.cli ingest --rebuild`"
            )
        if got != BACKEND:
            self._set_meta("vector_backend", BACKEND)

    def _check_hnsw_meta(self) -> None:
        """HNSW 建索引参数必须与索引一致，不一致就报错。

        Milvus Lite 不回传索引参数（`describe_index` 只有 index_type/metric_type/
        field_name），所以用过的值只能由本实现记进索引元信息。
        `ef`（查询期）不进这道闸门：它本来就应该随时可调。
        """
        recorded = {
            "milvus_hnsw_m": str(self.hnsw_m),
            "milvus_hnsw_ef_construction": str(self.hnsw_ef_construction),
            "milvus_metric": self.metric,
        }
        meta = self._meta()
        for key, want in recorded.items():
            got = meta.get(key, "")
            if not got:
                self._set_meta(key, want)
            elif got != want:
                raise StoreError(
                    f"向量索引的建索引参数与当前配置不符：{key} 库里是 {got!r}，配置是 {want!r}。"
                    f"M / efConstruction / 度量只能重建索引才能改，{_MIGRATE_HINT}"
                )

    def _meta(self) -> dict[str, str]:
        with self._lock:
            return {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}

    def _refresh_vec_ready(self) -> None:
        """核对"块表行数 vs Milvus 向量行数"，不一致就拒绝向量检索。

        这是两个引擎之间唯一一处的**廉价**一致性检查（行数查询是毫秒级），所以放在
        打开时与每次 `finalize()`。更强的**主键集合**级核对在 `verify_vec()`。
        """
        with self._lock:
            n_chunks = int(self._con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        n_vec = self._row_count()
        self._vec_error: str | None = None
        if n_chunks != n_vec:
            self._vec_error = f"Milvus 向量索引与块表不一致：块 {n_chunks} 行，向量 {n_vec} 行。" + (
                _MIGRATE_HINT if n_vec == 0 else "请重建或重灌向量索引"
            )

    def _row_count(self) -> int:
        try:
            stats = self._client.get_collection_stats(COLLECTION)
        except self._MilvusException as e:  # pragma: no cover - 引擎异常路径
            raise StoreError(f"读取 Milvus 行数失败：{type(e).__name__}: {e}") from e
        return int(stats.get("row_count", 0))

    def close(self) -> None:
        # 顺序：先关 Milvus（它会落盘/释放句柄），再关 SQLite。
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                # 这里**显式**忽略关闭异常：Milvus Lite 的每次写入都是同步落盘的，
                # 关失败没有可执行的补救动作，而它一旦抛出去就会盖住调用方正在处理的
                # 真正异常（并且让后面的 SQLite 连接关不掉）。检索/写入路径上的任何
                # 异常仍然一律外抛。
                client.close()
        super().close()

    def __enter__(self) -> MilvusStore:
        return self

    # ------------------------------------------------------------------ 写入
    def _index_chunk_vectors(self, con: Any, chunks: list[ChunkInput], vectors: dict[str, Vector]) -> None:
        """把本批向量写进 Milvus——**在 SQLite 事务提交之前**。

        这个顺序保证 Milvus 写失败时抛出 `StoreError`、由 `_tx` 回滚 SQLite，
        不会留下"块写了、向量没写"的中间态；反过来先提交再写 Milvus，失败时会
        留下行数不一致的索引。
        """
        if not chunks:
            return
        rows = [
            {"chunk_id": c.chunk_id, "embedding": np.asarray(vectors[c.chunk_id], dtype=np.float32)}
            for c in chunks
        ]
        self._exact = None
        for i in range(0, len(rows), _BATCH):
            batch = rows[i : i + _BATCH]
            try:
                self._client.upsert(COLLECTION, batch)
            except self._MilvusException as e:
                raise StoreError(
                    f"写入 Milvus 向量失败（{len(batch)} 行，例：{batch[0]['chunk_id']}）："
                    f"{type(e).__name__}: {e}；本次 SQLite 事务已回滚，"
                    '不会留下"块写了、向量没写"的中间态'
                ) from e

    def _drop_doc_chunks(self, con: Any, doc_id: str) -> int:
        """删一篇文档的块：**先删 Milvus 向量再删块行**。

        顺序反了就再也查不到这篇文档的 chunk_id，向量会永久残留。
        """
        with self._lock:
            ids = [
                r["chunk_id"] for r in con.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
            ]
        self._delete_vectors(ids)
        return super()._drop_doc_chunks(con, doc_id)

    def _delete_vectors(self, chunk_ids: Iterable[str]) -> None:
        ids = [str(c) for c in chunk_ids]
        if not ids:
            return
        self._exact = None
        for i in range(0, len(ids), _BATCH):
            batch = ids[i : i + _BATCH]
            try:
                self._client.delete(COLLECTION, ids=batch)
            except self._MilvusException as e:
                raise StoreError(
                    f"从 Milvus 删除向量失败（{len(batch)} 个，例：{batch[0]}）："
                    f"{type(e).__name__}: {e}；块表与向量索引可能已不一致，"
                    f"请重跑 ingest 或 {_MIGRATE_HINT}"
                ) from e

    def finalize(self) -> None:
        """收尾：盖时间戳、清待写缓冲、复核"块表 vs 向量索引"。

        与 `sqlite-vec` 不同，这里不是同一个事务的收尾，所以 `finalize()` 是"这一批
        写完，两个引擎是否还对齐"的最后一道闸门：不一致就抛错，让摄取在能定位的
        边界上失败，而不是留给下一次检索去发现。
        """
        with self._lock:
            self._set_meta("indexed_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._pending.clear()
            self._dirty = False
        self._refresh_vec_ready()
        if self._vec_error:
            raise StoreError(
                self._vec_error + "（本次写入的块行已经落库，但向量侧没跟上，说明两个引擎的写入没有同时成功）"
            )

    def rebuild_from_scratch(self) -> None:
        """清空全部数据：SQLite 的表、FTS5 索引，以及 Milvus 集合（drop 后重建）。"""
        with self._tx() as con:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM documents")
        try:
            self._client.drop_collection(COLLECTION)
        except self._MilvusException as e:  # pragma: no cover - 引擎异常路径
            raise StoreError(f"清空 Milvus 集合失败：{type(e).__name__}: {e}") from e
        self._create_collection()
        self._exact = None
        self._pending.clear()
        self._dirty = False
        self._reload_index()
        # 新建的集合是按当前配置建的，"实际用到的参数"要重新记一遍
        self._set_meta("milvus_hnsw_m", str(self.hnsw_m))
        self._set_meta("milvus_hnsw_ef_construction", str(self.hnsw_ef_construction))
        self._set_meta("milvus_metric", self.metric)
        self._refresh_vec_ready()

    # ------------------------------------------------------------------ 读取
    def existing_hash_to_vector(self, hashes: set[str]) -> dict[str, Vector]:
        """按 content_hash 复用旧向量（跨篇复用，不重新编码）。

        先按 content_hash 在 SQLite 里定位 chunk_id，再按主键从 Milvus 取向量：
        Milvus 的标量过滤只认它自己的字段，而 content_hash 在块表侧。
        """
        if not hashes:
            return {}
        if self._vec_error:
            raise StoreError(self._vec_error)
        wanted = list(hashes)
        out: dict[str, Vector] = {}
        for i in range(0, len(wanted), 400):
            batch = wanted[i : i + 400]
            marks = ",".join("?" * len(batch))
            with self._lock:
                # marks 只由 '?' 组成，值一律走参数绑定，不存在拼接注入
                rows = self._con.execute(
                    f"SELECT chunk_id, content_hash FROM chunks WHERE content_hash IN ({marks})",  # noqa: S608
                    batch,
                ).fetchall()
            if not rows:
                continue
            hash_of = {str(r["chunk_id"]): str(r["content_hash"]) for r in rows}
            out.update(self._fetch_vectors(list(hash_of), hash_of))
        return out

    def _fetch_vectors(self, chunk_ids: list[str], hash_of: dict[str, str]) -> dict[str, Vector]:
        """按主键批量取向量。取不到的块（向量侧缺行）直接跳过——行数闸门会先报错。"""
        got: dict[str, Vector] = {}
        for i in range(0, len(chunk_ids), _BATCH):
            batch = chunk_ids[i : i + _BATCH]
            try:
                rows = self._client.get(COLLECTION, ids=batch, output_fields=["chunk_id", "embedding"])
            except self._MilvusException as e:
                raise StoreError(f"从 Milvus 读取向量失败：{type(e).__name__}: {e}；{_MIGRATE_HINT}") from e
            for row in rows:
                cid = str(row["chunk_id"])
                if cid in hash_of:
                    got[hash_of[cid]] = np.asarray(row["embedding"], dtype=np.float32)
        return got

    def _reload_index(self) -> None:
        """覆写基类：本实现不把向量读进内存（向量由 Milvus 管理）。

        基类在这里 `np.load` 整个 `.npy` 侧车矩阵并常驻内存，本实现省掉这一步。
        """
        with self._lock:
            self._row_index: list[str] = []
            self._old_row: dict[str, int] = {}
            self._hash_row: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)

    # ------------------------------------------------------------------ 检索
    def _as_query_vector(self, query_vector: Vector) -> np.ndarray:
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise StoreError(f"查询向量维度 {q.shape[0]} 与索引 {self.dim} 不符")
        return np.ascontiguousarray(q, dtype=np.float32)

    def dense_search(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        """向量路：Milvus 的 **HNSW 近似检索**，按相似度降序。

        **HNSW 是近似索引**：同样的输入不保证给出与精确检索相同的前 k。这一点不
        允许含糊，所以本实现提供 `dense_search_exact()` 来度量召回率。
        返回的名次从 1 开始，与另外几个后端同构。
        """
        if self._vec_error:
            raise StoreError(self._vec_error)
        if limit <= 0:
            return []
        q = self._as_query_vector(query_vector)
        try:
            res = self._client.search(
                COLLECTION,
                data=[q],
                limit=limit,
                anns_field="embedding",
                search_params={"metric_type": self.metric, "params": {"ef": int(self.hnsw_ef_search)}},
                output_fields=["chunk_id"],
            )
        except self._MilvusException as e:
            raise StoreError(
                f"Milvus 向量检索失败：{type(e).__name__}: {e}；检查数据文件（{self.milvus_path}）是否可用"
            ) from e
        hits = res[0] if res else []
        return [(str(h["chunk_id"]), i) for i, h in enumerate(hits, start=1)]

    def dense_search_exact(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        """**不用 HNSW** 的精确检索：度量近似索引召回率的基准。

        它把全量向量拉回内存算一次精确余弦序，之后每次查询是一次 BLAS 矩阵乘。
        代价必须写清楚：**这是刻意留在查询路径之外的方法**——向量不常驻内存是本
        后端的特点，不该被基准工具吃掉。
        """
        if limit <= 0:
            return []
        q = self._as_query_vector(query_vector)
        ids, mat = self._exact_index()
        if not ids:
            return []
        n = min(limit, len(ids))
        sims = mat @ q
        idx = np.argpartition(-sims, n - 1)[:n]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(ids[int(i)], rank) for rank, i in enumerate(idx, start=1)]

    def _exact_index(self) -> tuple[list[str], np.ndarray]:
        """懒加载全量 (chunk_id, 向量)：只给 `dense_search_exact()` 用。"""
        if self._exact is not None:
            return self._exact
        n = self._row_count()
        if n == 0:
            self._exact = ([], np.zeros((0, self.dim), dtype=np.float32))
            return self._exact
        ids: list[str] = []
        mat = np.zeros((n, self.dim), dtype=np.float32)
        # 向量已归一化（摄取层保证），所以精确余弦序 == 内积序，这里直接算内积
        try:
            rows = self._client.query(
                COLLECTION, filter="", limit=n + 1, output_fields=["chunk_id", "embedding"]
            )
        except self._MilvusException as e:
            raise StoreError(f"拉取全量向量用于精确对照失败：{type(e).__name__}: {e}") from e
        for row in rows:
            ids.append(str(row["chunk_id"]))
            mat[len(ids) - 1] = np.asarray(row["embedding"], dtype=np.float32)
        mat = mat[: len(ids)]
        self._exact = (ids, mat)
        return self._exact

    # ------------------------------------------------------------------ 运维
    def adopt_legacy_matrix(self) -> int:
        """把旧实现的 `.npy` 矩阵搬进 Milvus 的 HNSW 索引（**不重新编码**）。

        向量本身一个字节都没变，只是存放位置变了，所以搬运比全量重编码快得多。
        正确性由 `chunks.vector_row` ↔ 矩阵行号的对应关系保证（旧实现 `finalize()`
        写入的那个映射），越界即报错，不做任何猜测。

        幂等：先 drop 集合再整批灌，重复跑不会产生重复行。它同时把 `vector_backend`
        这道闸门切到 `milvus`——这是唯一一处"向量真的被搬过去"的地方。
        """
        path = Path(self.vector_path)
        if not path.exists():
            raise StoreError(
                f"找不到旧向量矩阵 {path}；没有可迁移的向量。"
                "若索引本来就是 milvus 建的，直接跑 ingest 即可；"
                "否则请用 `python -m kubedocs.cli ingest --rebuild` 重新编码"
            )
        mat = np.load(path)
        with self._lock:
            rows = [
                (str(r["chunk_id"]), int(r["vector_row"]))
                for r in self._con.execute("SELECT chunk_id, vector_row FROM chunks")
            ]
        if mat.ndim != 2 or mat.shape[1] != self.dim:
            raise StoreError(f"{path} 形状 {mat.shape} 与 dim={self.dim} 不符")
        if len(rows) != mat.shape[0]:
            raise StoreError(
                f"块表 {len(rows)} 行与矩阵 {mat.shape[0]} 行不符——索引本身已不一致，"
                "请用 `ingest --rebuild` 重建而不是迁移"
            )
        payload: list[dict[str, object]] = []
        for chunk_id, vr in rows:
            if vr < 0 or vr >= mat.shape[0]:
                raise StoreError(f"块 {chunk_id} 的 vector_row={vr} 越界（矩阵 {mat.shape}）")
            payload.append({"chunk_id": chunk_id, "embedding": np.asarray(mat[vr], dtype=np.float32)})
        try:
            self._client.drop_collection(COLLECTION)
        except self._MilvusException as e:  # pragma: no cover - 引擎异常路径
            raise StoreError(f"清空 Milvus 集合失败：{type(e).__name__}: {e}") from e
        self._create_collection()
        for i in range(0, len(payload), _BATCH):
            try:
                self._client.upsert(COLLECTION, payload[i : i + _BATCH])
            except self._MilvusException as e:
                raise StoreError(f"灌入 Milvus 失败（第 {i} 行起）：{type(e).__name__}: {e}") from e
        self._exact = None
        self._set_meta("vector_backend", BACKEND)
        self._set_meta("milvus_hnsw_m", str(self.hnsw_m))
        self._set_meta("milvus_hnsw_ef_construction", str(self.hnsw_ef_construction))
        self._set_meta("milvus_metric", self.metric)
        self._refresh_vec_ready()
        return len(payload)

    def detach(self) -> None:
        """把这份索引从 milvus 上摘下来：清空集合 + 撤掉后端标记。

        `vector_backend` 闸门是双向的：它挡住"静默读陈旧向量"，但也会让"我想换回
        sqlite-vec、先 `ingest --rebuild`"这一步连 store 都构造不出来。所以要换回去
        必须先打开闸门，本方法就是那个人：它只宣布"这里不再有 milvus 向量"，
        **不动块表与全文索引**；换回别的后端仍需 `ingest --rebuild` 重新产生向量。

        清空后集合**重建为空集合**（而不是留一个不存在的集合）：数据目录因此仍是一份
        自洽状态，后续操作看到的是"块有、向量没有"，会明确报错并给出修复命令，
        而不是撞上一句 `collection does not exist`。
        """
        try:
            self._client.drop_collection(COLLECTION)
        except self._MilvusException as e:  # pragma: no cover - 引擎异常路径
            raise StoreError(f"清空 Milvus 集合失败：{type(e).__name__}: {e}") from e
        self._create_collection()
        with self._lock:
            self._con.execute(
                "DELETE FROM meta WHERE key IN "
                "('vector_backend', 'milvus_hnsw_m', 'milvus_hnsw_ef_construction', "
                "'milvus_metric')"
            )
        self._exact = None
        self._refresh_vec_ready()

    def verify_vec(self) -> str:
        """向量索引自检：**行数** + **主键集合**两级核对。返回 `"ok"` 或 `"FAIL: ..."`。

        比 `_refresh_vec_ready()` 强的地方是主键集合：行数相同但内容错位（删了 A 的
        向量、又重复写了 B 的向量）只有它看得出来。代价是 Milvus Lite 逐行回传主键较慢，
        所以它是**显式**操作（`cli stats --verify`），不在每次检索或 finalize 上跑。

        诊断信息优先给出**集合差**（谁缺了、谁多了），因为它比"行数 5 vs 6"更能直接
        指向修复动作；`_vec_error` 只在集合一致时兜底报告。
        """
        try:
            milvus_ids = set(self._all_ids())
            with self._lock:
                sqlite_ids = {str(r["chunk_id"]) for r in self._con.execute("SELECT chunk_id FROM chunks")}
        except StoreError as e:
            return f"FAIL: {e}"
        if milvus_ids != sqlite_ids:
            missing = sorted(sqlite_ids - milvus_ids)[:3]
            extra = sorted(milvus_ids - sqlite_ids)[:3]
            return (
                f"FAIL: 主键集合不一致——块表 {len(sqlite_ids)} 个、向量 {len(milvus_ids)} 个；"
                f"向量侧缺 {len(sqlite_ids - milvus_ids)} 个（例：{missing}），"
                f"多 {len(milvus_ids - sqlite_ids)} 个（例：{extra}）"
            )
        if self._vec_error:
            return f"FAIL: {self._vec_error}"
        return "ok"

    def _all_ids(self) -> list[str]:
        """全量主键。Milvus Lite 逐行回传较慢，这里会扫一遍集合。"""
        n = self._row_count()
        if n == 0:
            return []
        try:
            rows = self._client.query(COLLECTION, filter="", limit=n + 1, output_fields=["chunk_id"])
        except self._MilvusException as e:
            raise StoreError(f"读取 Milvus 主键失败：{type(e).__name__}: {e}") from e
        return [str(r["chunk_id"]) for r in rows]


def from_settings(settings: Settings) -> MilvusStore:
    """按配置构造 milvus 后端（数据文件位置、HNSW 参数、度量都在这里落地）。"""
    return MilvusStore(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        normalizer=build_normalizer(settings.normalizer),
        domain=settings.domain,
        milvus_path=settings.milvus_path,
        hnsw_m=settings.milvus_hnsw_m,
        hnsw_ef_construction=settings.milvus_hnsw_ef_construction,
        hnsw_ef_search=settings.milvus_hnsw_ef_search,
        metric=settings.milvus_metric,
    )
