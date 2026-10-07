"""本地默认索引：SQLite（FTS5 全文） + sqlite-vec 向量索引。

## 这个实现换掉的到底是什么（先说实话）

它**不是**"用 ANN 索引替掉暴力扫描"。`sqlite-vec` 0.1.9 的 `vec0` 虚表
是 **C 语言实现的暴力近邻扫描**（带 SIMD、按 chunk 分块），HNSW/IVF 只在其
roadmap 上——本机实测 `EXPLAIN QUERY PLAN` 也确认是 `SCAN ... VIRTUAL TABLE`。
所以"换 ANN"这个说法在本版本上不成立，必须写明（见 ADR-0004「未验证/被证否的假设」）。

它真正换掉的是**向量数据的位置与生命周期**：

| 关注点 | 旧实现（numpy 矩阵） | 本实现（vec0 虚表） |
|---|---|---|
| 向量存放 | `.npy` 侧车文件，与 DB 不同步 | DB 内的虚表，与块行**同一事务** |
| 删除一篇文档 | 建新矩阵再原子替换（18861×384 ≈ 29MB 重写） | 按 chunk_id 删对应行，O(删除量) |
| 内存常驻 | 全量向量常驻内存 | 不常驻（由扩展管理） |
| 一致性 | "块写了、矩阵没写"是可能状态 | 不存在这个中间态 |
| 相似度语义 | 原始内积（要求向量已归一化） | 余弦距离（每行归一化后比较） |

即：在 1.9 万块这个量级，它**不是延迟上的胜利**（实测见 ADR-0004 的对照表），
而是**一致性、可增删、内存占用**上的胜利。把它设成默认是因为它删掉了
"侧车文件"这个最容易失配的东西；性能结论按实测数字写，不按期待写。

## 关键设计

1. **向量表主键就是 `chunk_id`**（`vec0(chunk_id text primary key, ...)`），
   不用 `chunks.rowid` 中转。旧实现必须绕道 rowid 并因此要处理"rowid 会被回收"
   这个坑（见 `sqlite_store.py` 文档第 2 条）；直接用 chunk_id 后这个坑根本不存在，
   删除/覆盖也只按主键定位，不受回收影响。
2. **向量与块行同事务**：`upsert_docs` 在一个 `BEGIN IMMEDIATE` 里写 `chunks`、
   `chunks_fts`、`chunks_vec`，因此不存在"块在、向量不在"的中间态，
   `finalize()` 不再需要重建任何东西（旧实现要在那里重写整个矩阵）。
3. **全文路完全继承旧实现**（同一个 `chunks_fts` 外内容表、同一个 bm25 与列权重），
   所以 `sqlite` 与 `sqlite-vec` 两个后端的 `keyword_search` 结果**逐字节相同**——
   换后端时唯一的变量就是向量路，这对定位指标变化是决定性的。
4. **不一致就报错，不静默**：块表与向量表行数不一致时，向量路直接抛
   `StoreError` 并给出可执行的修复命令；全文路仍然可用。
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from kubedocs.config import Settings
from kubedocs.normalize import Normalizer, build_normalizer
from kubedocs.store.base import ChunkInput, StoreError, Vector
from kubedocs.store.sqlite_store import SqliteStore

BACKEND = "sqlite-vec"
VEC_TABLE = "chunks_vec"
# vec0 里向量列的合法小端 float32 字节序（sqlite-vec 的 raw 格式就是小端 float32）
_BLOB_DTYPE = np.dtype("<f4")

# 从虚表 DDL 里读回声明维度：`embedding float[384] distance_metric=cosine`
_DIM_RE = re.compile(r"embedding\s+float\s*\[\s*(\d+)\s*\]")

_MIGRATE_HINT = ("请先迁移旧索引（不重新编码）："
                 "`python scripts/migrate_to_sqlite_vec.py`，"
                 "或用 `python -m kubedocs.cli ingest --rebuild` 全量重建")


def _sql(template: str, **parts: object) -> str:
    """本模块**唯一**的动态 SQL 拼装点。

    动态部分只有两种，都不可能来自用户输入：
    - 表名 `chunks_vec`（模块常量）；
    - `IN (...)` 里的 `?` 占位符个数（由 `_marks(n)` 生成）。
    值一律走 sqlite3 的参数绑定。集中到一个函数里，是为了让"哪些部分是动态的"
    只有一处需要审阅——比在十几处分别论证"我这里拼得安全"可靠得多。
    """
    return template.format(**parts)


def _marks(n: int) -> str:
    """生成 n 个 `?` 占位符（动态的只有个数，值的绑定由 sqlite3 负责）。"""
    return ",".join("?" * n)


class SqliteVecStore(SqliteStore):
    """SQLite（FTS5）+ sqlite-vec 向量索引。`Store` 协议的本地默认实现。"""

    def __init__(self, db_path: str | Path, vector_path: str | Path, *, dim: int,
                 embedding_model: str = "", normalizer: Normalizer | None = None,
                 domain: str = "", cache_kib: int = 131072) -> None:
        # `vector_path` 只是**旧实现的侧车矩阵位置**，本实现不读也不写它；
        # 仅用于"迁移旧索引"时定位那个文件，以及告诉用户它已经不需要了。
        super().__init__(db_path, vector_path, dim=dim, embedding_model=embedding_model,
                         normalizer=normalizer, domain=domain)
        # 页缓存必须按向量表大小调大。实测（18861×384，向量表 29MB，同一台机器）：
        #   默认 2MB 缓存   KNN top10 p50 66~147ms   ← 每次查询都在穿透页缓存读盘
        #   128MB 缓存      KNN top10 p50 12.0ms
        #   256MB mmap      KNN top10 p50 20.5ms     ← mmap 反而更慢，所以不开
        # 这是本实现相对基类唯一需要额外调的连接参数；调与不调差 5~12 倍，
        # 而它完全不影响正确性——正是那种"不测就永远发现不了"的退化。
        self._con.execute(f"PRAGMA cache_size=-{int(cache_kib)}")
        self._load_extension()
        self._create_vec_table()
        self._check_backend_meta()
        self._refresh_vec_ready()

    # ------------------------------------------------------------------ 初始化
    def _load_extension(self) -> None:
        """加载 sqlite-vec 扩展。

        失败必须**硬报错**而不是退回暴力扫描：静默回退会让"我明明配了 sqlite-vec"
        与"实际在跑什么"分离，而两者给出的排序可能不同，指标变化将无从解释。
        """
        try:
            import sqlite_vec
        except ImportError as e:  # pragma: no cover - 取决于环境
            raise StoreError(
                "缺少 sqlite-vec：请把 `sqlite-vec>=0.1.9` 加进依赖（或改用 "
                "KUBEDOCS_STORE=sqlite / pgvector）"
            ) from e
        with self._lock:
            self._con.enable_load_extension(True)
            try:
                sqlite_vec.load(self._con)
            except (sqlite3.Error, OSError) as e:
                raise StoreError(
                    f"sqlite-vec 扩展加载失败（{type(e).__name__}: {e}）；"
                    "本后端不可用，请改用 KUBEDOCS_STORE=sqlite 或 pgvector——"
                    "不会静默退回暴力扫描"
                ) from e
            finally:
                self._con.enable_load_extension(False)

    def _create_vec_table(self) -> None:
        """建向量表（幂等）并核对声明维度。

        `CREATE VIRTUAL TABLE IF NOT EXISTS` 遇到已存在的表**不会**校验维度，
        所以维度必须在建表后显式读回比较——否则"换了嵌入模型但忘了重建"会
        表现为插入/查询报一个难懂的扩展错误，而不是一句可执行的提示。
        """
        ddl = _sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {tbl} USING vec0("
            "chunk_id text primary key, "
            "embedding float[{dim}] distance_metric=cosine)",
            tbl=VEC_TABLE, dim=int(self.dim))
        with self._lock:
            self._con.execute(ddl)
            row = self._con.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (VEC_TABLE,)).fetchone()
        declared = row["sql"] if row is not None else ""
        m = _DIM_RE.search(declared or "")
        if m is None:  # pragma: no cover - 只可能在扩展行为变化时发生
            raise StoreError(f"无法从 {VEC_TABLE} 的 DDL 里读回向量维度：{declared!r}")
        got = int(m.group(1))
        if got != self.dim:
            raise StoreError(
                f"向量索引维度不符：库里是 {got}，当前配置 {self.dim}；"
                "换了嵌入模型必须重建索引（`ingest --rebuild`）"
            )

    def _check_backend_meta(self) -> None:
        with self._lock:
            rows = {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}
        got = rows.get("vector_backend", "")
        if got and got != BACKEND:
            raise StoreError(
                f"索引由另一个后端写入（vector_backend={got!r}，本实现是 {BACKEND!r}）；"
                "同一份索引不要在多个后端间反复切换写，请用 ingest --rebuild 重建"
            )
        if got != BACKEND:
            self._set_meta("vector_backend", BACKEND)

    def _refresh_vec_ready(self) -> None:
        """核对向量表与块表行数是否一致（不一致时向量路必须报错而不是少召回）。"""
        with self._lock:
            n_chunks = int(self._con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            n_vec = int(self._con.execute(
                _sql("SELECT COUNT(*) FROM {tbl}", tbl=VEC_TABLE)).fetchone()[0])
        self._vec_error: str | None = None
        if n_chunks != n_vec:
            self._vec_error = (
                f"向量索引与块表不一致：块 {n_chunks} 行，向量 {n_vec} 行。"
                + (_MIGRATE_HINT if n_vec == 0 else "请重建索引（ingest --rebuild）")
            )

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # ------------------------------------------------------------------ 写入
    def _index_chunk_vectors(self, con: sqlite3.Connection, chunks: list[ChunkInput],
                             vectors: dict[str, Vector]) -> None:
        """把本批向量写进 vec0 表——与块行同一事务，不存在中间态。

        为什么用 `ndarray.tobytes()` 而不是 `sqlite_vec.serialize_float32()`：
        后者用 `struct.pack` 逐个 float 走 Python 层，18861×384 次调用是纯开销；
        `.tobytes()` 是内存直拷。两者都产出 sqlite-vec 要求的**小端 float32**
        原始字节（`<f4` 显式声明字节序，不依赖平台）。
        """
        if not chunks:
            return
        rows = [
            (c.chunk_id, np.asarray(vectors[c.chunk_id], dtype=_BLOB_DTYPE).tobytes())
            for c in chunks
        ]
        con.executemany(_sql(
            "INSERT INTO {tbl}(chunk_id, embedding) VALUES(?, ?)", tbl=VEC_TABLE), rows)

    def _drop_doc_chunks(self, con: sqlite3.Connection, doc_id: str) -> int:
        """删一篇文档的块：**先删向量行再删块行**（顺序反了就找不回 chunk_id 了）。"""
        with self._lock:
            ids = [r["chunk_id"] for r in
                   con.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]
        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            con.execute(_sql(
                "DELETE FROM {tbl} WHERE chunk_id IN ({marks})",
                tbl=VEC_TABLE, marks=_marks(len(batch))), batch)
        return super()._drop_doc_chunks(con, doc_id)

    def finalize(self) -> None:
        """收尾：本实现没有"待并入的矩阵"，只需盖时间戳并复核一致性。

        旧实现在这里重建整份 `.npy`（18861×384 ≈ 29MB 重写）并回写每个块的
        `vector_row`——那是"向量与块行不同步"的代价。这里向量在写入时就已经
        和块行在同一事务里落库，所以 finalize 是真正的收尾，而不是一次重排。
        """
        with self._lock:
            self._set_meta("indexed_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._pending.clear()
            self._dirty = False
        self._refresh_vec_ready()

    def rebuild_from_scratch(self) -> None:
        with self._tx() as con:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
            con.execute(_sql("DELETE FROM {tbl}", tbl=VEC_TABLE))
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM documents")
        # 旧实现的侧车矩阵已无对应块，留着只会让"切回 sqlite 后端"读到陈旧数据
        if Path(self.vector_path).exists():
            Path(self.vector_path).unlink()
        self._pending.clear()
        self._dirty = False
        self._reload_index()
        self._refresh_vec_ready()

    # ------------------------------------------------------------------ 读取
    def existing_hash_to_vector(self, hashes: set[str]) -> dict[str, Vector]:
        """按 content_hash 复用旧向量（用于跨篇复用，不重新编码）。"""
        if not hashes or self._vec_error:
            if self._vec_error:
                raise StoreError(self._vec_error)
            return {}
        out: dict[str, Vector] = {}
        items = list(hashes)
        with self._lock:
            for i in range(0, len(items), 400):
                batch = items[i:i + 400]
                marks = ",".join("?" * len(batch))
                by_id = {
                    r["chunk_id"]: r["content_hash"] for r in self._con.execute(
                        _sql("SELECT chunk_id, content_hash FROM chunks "
                             "WHERE content_hash IN ({marks})", marks=marks),
                        batch)
                }
                for blob, content_hash in self._fetch_embeddings(by_id):
                    out[content_hash] = self._as_vector(blob)
        return out

    def _fetch_embeddings(self, chunk_ids: dict[str, str]) -> list[tuple[bytes, str]]:
        """批量取向量原始字节：返回 [(embedding_blob, content_hash)]（缺行直接跳过）。"""
        out: list[tuple[bytes, str]] = []
        ids = list(chunk_ids)
        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            rows = self._con.execute(_sql(
                "SELECT chunk_id, embedding FROM {tbl} WHERE chunk_id IN ({marks})",
                tbl=VEC_TABLE, marks=_marks(len(batch))), batch).fetchall()
            for r in rows:
                out.append((r["embedding"], chunk_ids[r["chunk_id"]]))
        return out

    def _as_vector(self, blob: bytes) -> Vector:
        return np.frombuffer(blob, dtype=_BLOB_DTYPE).astype(np.float32, copy=True)

    def dense_search(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        """向量路：vec0 的 KNN 查询，按余弦距离升序 = 余弦相似度降序。

        与旧实现的排序语义对照：旧实现算的是**原始内积**，本实现算的是**余弦**。
        向量已归一化（摄取层的 `_sanity_check` 保证）时二者严格同序——这正是
        "换后端不该改指标"的前提，也是我在 ADR-0004 里用真实评测对照验证的那一点。
        """
        if self._vec_error:
            raise StoreError(self._vec_error)
        if limit <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise StoreError(f"查询向量维度 {q.shape[0]} 与索引 {self.dim} 不符")
        blob = np.ascontiguousarray(q, dtype=_BLOB_DTYPE).tobytes()
        with self._lock:
            # 刻意**不**先 COUNT(*) 再 min(limit, n)：vec0 的 COUNT 本身就是一次全表扫描，
            # 会让每次查询的开销翻倍（实测 p50 从 12ms 涨到 31ms）。
            # 而 k 大于行数时 vec0 只返回现有的行、不报错——把上限交给它更省一次扫描。
            rows = self._con.execute(_sql(
                "SELECT chunk_id FROM {tbl} "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance", tbl=VEC_TABLE),
                (blob, limit)).fetchall()
        return [(r["chunk_id"], i) for i, r in enumerate(rows, start=1)]

    # ------------------------------------------------------------------ 运维
    def verify_vec(self) -> str:
        """向量索引自检：行数一致 + 每行向量长度等于声明维度。"""
        if self._vec_error:
            return f"FAIL: {self._vec_error}"
        with self._lock:
            bad = int(self._con.execute(
                _sql("SELECT COUNT(*) FROM {tbl} WHERE length(embedding) != ?",
                     tbl=VEC_TABLE), (self.dim * 4,)).fetchone()[0])
        if bad:
            return f"FAIL: {bad} 行向量的字节长度不等于 {self.dim}×4"
        return "ok"

    def adopt_legacy_matrix(self) -> int:
        """把旧实现的 `.npy` 矩阵搬进 `chunks_vec`（**不重新编码**）。

        为什么这条路径值得存在：全量重编码是 1.9 万块 × 本机 9.4 块/秒 ≈ 33 分钟，
        而向量本身一个字节都没变——只是存放位置变了。迁移的正确性由
        `vector_row` 与矩阵行号的对应关系保证（旧实现 `finalize` 里写入的那个映射），
        越界即报错，不做任何猜测。
        """
        path = Path(self.vector_path)
        if not path.exists():
            raise StoreError(
                f"找不到旧向量矩阵 {path}；没有可迁移的向量，请用 ingest --rebuild 重建")
        mat = np.load(path)
        with self._lock:
            rows = [(r["chunk_id"], int(r["vector_row"])) for r in
                    self._con.execute("SELECT chunk_id, vector_row FROM chunks")]
        if mat.ndim != 2 or mat.shape[1] != self.dim:
            raise StoreError(f"{path} 形状 {mat.shape} 与 dim={self.dim} 不符")
        if len(rows) != mat.shape[0]:
            raise StoreError(
                f"块表 {len(rows)} 行与矩阵 {mat.shape[0]} 行不符——索引本身已不一致，"
                "请用 ingest --rebuild 重建而不是迁移")
        payload: list[tuple[str, bytes]] = []
        for chunk_id, vr in rows:
            if vr < 0 or vr >= mat.shape[0]:
                raise StoreError(f"块 {chunk_id} 的 vector_row={vr} 越界（矩阵 {mat.shape}）")
            payload.append((chunk_id, np.asarray(mat[vr], dtype=_BLOB_DTYPE).tobytes()))
        with self._tx() as con:
            con.execute(_sql("DELETE FROM {tbl}", tbl=VEC_TABLE))
            con.executemany(_sql(
                "INSERT INTO {tbl}(chunk_id, embedding) VALUES(?, ?)", tbl=VEC_TABLE),
                payload)
        self._set_meta("vector_backend", BACKEND)
        self._refresh_vec_ready()
        return len(payload)

    def _reload_index(self) -> None:
        """覆写基类：本实现不需要把向量读进内存，只需要空的占位。

        基类在这里 `np.load` 整个 `.npy`（18861×384 ≈ 29MB 常驻内存）——
        本实现的向量由扩展管理，这一步正是要省掉的东西。
        """
        with self._lock:
            self._row_index: list[str] = []
            self._old_row: dict[str, int] = {}
            self._hash_row: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)


def from_settings(settings: Settings) -> SqliteVecStore:
    """按配置构造本地默认索引。"""
    return SqliteVecStore(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        normalizer=build_normalizer(settings.normalizer),
        domain=settings.domain,
        cache_kib=settings.sqlite_cache_kib,
    )
