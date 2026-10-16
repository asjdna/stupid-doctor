"""本地嵌入式索引：SQLite（FTS5 全文） + numpy 向量矩阵。

同一 Store 协议下有多个实现，由 `KUBEDOCS_STORE` 选择（见 `store/__init__.py`）：

| 取值 | 实现 | 定位 |
|---|---|---|
| `sqlite-vec` | `sqlite_vec_store.SqliteVecStore` | **默认**：FTS5 + sqlite-vec 向量索引 |
| `sqlite` | 本文件 | FTS5 + numpy 暴力内积 + `.npy` 侧车矩阵 |
| `pgvector` | `pgvector_store.PgVectorStore` | 生产：Postgres + pgvector HNSW |

本实现是唯一"零扩展、纯 numpy"的一个：全部由标准库 + numpy 提供，
**零外部进程、零镜像、零网络**，因此可作 sqlite-vec 扩展装不上时的回退，
以及性能对照的基线。检索与生成代码完全相同，切换后端只改配置。

边界与代价：

- 单写者：SQLite 并发写入有限，只适合单进程摄取；多实例写要走 `pgvector`；
- 全量向量常驻内存（`dim` 越大占用越高），语料再大两个数量级就该换 ANN 索引；
- 删除一篇文档要**重建整个矩阵**（`finalize` 压实 `vector_row`），
  语料频繁增删时这一步是纯开销；`sqlite-vec` 版没有它，向量随块行增删。

两个必须规避的坑：

1. **FTS5 外内容表（external content）的删除**：不能直接用普通 `DELETE`，必须先用
   FTS5 的 `'delete'` 指令摘掉索引项，再删内容行；顺序反了索引会永久不一致
   （后续查询报 malformed）。见 `_drop_doc_chunks`。
2. **向量矩阵与 rowid 的对应**：`chunks` 的 rowid 在删除后会被回收，新插入的块可能
   拿到比现存块更小的 rowid，所以不能假设"rowid 单调递增"。`finalize()` 每次按当前
   rowid 顺序重建矩阵并回写 `vector_row`。

父子切割在存储层的落法：**父块不进任何索引**（`chunks_fts` 与向量索引都不含它），
它只躺在普通表 `parents` 里，靠 `chunks.parent_id` 指过去，读命中时**批量**取回正文。
所以两层共用同一套"向量与块行同事务""覆盖按文档先删后写"的规则，没有第二套一致性模型；
父块缺失（老库、或调用方不传 `parents`）是合法退化：语境退回子块正文。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from kubedocs.config import Settings
from kubedocs.normalize import IdentityNormalizer, Normalizer, build_normalizer
from kubedocs.store.base import (
    ChunkHit,
    ChunkInput,
    DocInput,
    MissingVectorError,
    ParentInput,
    StoreError,
    StoreStats,
    Vector,
)

# 词元抽取与 MATCH 表达式在 store/textquery.py 里（各后端共用同一套词元规则，
# 只在表达式语法上分叉）。这里转出 `build_match_expr` 保持既有导入路径可用。
from kubedocs.store.textquery import build_match_expr

__all__ = [
    "MissingVectorError",
    "SqliteStore",
    "StoreError",
    "build_match_expr",
    "from_settings",
]

SCHEMA_VERSION = 2
"""布局版本。**增量的新表/新列不跟着加它**：老库补一列就能读，只是没有父块语境，
那是可用的退化状态；把版本号一加，`_load_meta_or_init` 就会拒绝打开老库、
逼所有人删库重建——正是 `_migrate_schema` 要避免的事。"""

# FTS5 外内容表：正文只存一份（在 chunks 里），FTS 只存倒排索引。
#
# 索引列是 `body_normalized` 而不是 `body`：索引形态与展示形态必须分开。
# 英文两者相同；中文必须先 jieba 预分词才能被 FTS5 正确切词（详见 normalize.py）。
# `body` 保留原文用于展示与引用，`body_normalized` 只服务于倒排索引。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    rel_path     TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    n_chunks     INTEGER NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    title          TEXT NOT NULL,
    breadcrumb     TEXT NOT NULL,
    source_url     TEXT NOT NULL,
    body           TEXT NOT NULL,
    body_normalized TEXT NOT NULL DEFAULT '',
    content_hash   TEXT NOT NULL,
    n_chars        INTEGER NOT NULL,
    vector_row     INTEGER NOT NULL,
    parent_id      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

-- 父块：一个 Markdown 小节 / 一页 PDF / 一张幻灯片，是**喂给模型**的完整语境。
-- 它刻意不进任何索引：没有 body_normalized、没有向量行——检索用子块（命中精确），
-- 生成用父块（语境完整），两者各司其职，混在一起只会让"召回"与"喂料"互相污染。
CREATE TABLE IF NOT EXISTS parents (
    parent_id    TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    title        TEXT NOT NULL,
    breadcrumb   TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    body         TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    n_chars      INTEGER NOT NULL,
    location     TEXT NOT NULL DEFAULT '',
    n_children   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_parents_doc ON parents(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body_normalized, title, breadcrumb,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

# 注意上面**没有** `idx_chunks_parent`：老库的 `chunks` 还没有这一列，建索引会先于补列
# 执行而报 `no such column`。它只出现在 `_migrate_schema` 里，补列之后建。

# BM25 列权重，顺序必须与 chunks_fts 的列声明一致：(body_normalized, title, breadcrumb)。
# 标题命中比正文命中更能说明"这一块就是讲这件事的"，面包屑次之。
_BM25_WEIGHTS = (1.0, 3.0, 1.5)


class SqliteStore:
    """SQLite + FTS5 + numpy 的本地索引实现。"""

    def __init__(
        self,
        db_path: str | Path,
        vector_path: str | Path,
        *,
        dim: int,
        embedding_model: str = "",
        normalizer: Normalizer | None = None,
        domain: str = "",
    ) -> None:
        self.db_path = Path(db_path)
        self.vector_path = Path(vector_path)
        self.dim = dim
        self.embedding_model = embedding_model
        self.domain = domain
        # 归一化器必须与建索引时用的一致，否则查询与索引处在两个"词汇空间"里，
        # 表现为"索引里明明有却搜不到"，且没有任何报错。
        self.normalizer: Normalizer = normalizer or IdentityNormalizer()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False + 全库访问加锁：FastAPI 的同步端点跑在线程池里，
        # 一个连接跨线程使用本身安全，前提是同一时刻只有一处在用（由锁保证）。
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.executescript(_SCHEMA)
        self._migrate_schema()

        self._pending: dict[str, Vector] = {}
        self._dirty = False
        self._load_meta_or_init()
        self._reload_index()

    def _migrate_schema(self) -> None:
        """把**已存在**的旧库原地补到当前列布局。

        建表语句是 `CREATE TABLE IF NOT EXISTS`：对已经存在的表它什么都不做，所以新加的列
        必须在这里显式补。老库的 `chunks` 没有 `parent_id`，补上就能读——只是没有父块语境。

        为什么必须平滑：换一次切块方式就要求所有人删库重建是不可接受的，而且"老库能读、
        只是没有父块语境"本身是完全可用的退化状态（`ChunkHit.context_text` 会退回子块正文），
        没有任何数据不一致。所以既不删库、也不加 `SCHEMA_VERSION`（那等于换一种方式拒绝老库）。
        """
        with self._lock:
            cols = {r["name"] for r in self._con.execute("PRAGMA table_info(chunks)")}
            if "parent_id" not in cols:
                # NOT NULL + DEFAULT ''：老子块"没有父块"的表示与代码里的空串语义天然一致
                self._con.execute("ALTER TABLE chunks ADD COLUMN parent_id TEXT NOT NULL DEFAULT ''")
            # 读命中时按 parent_id 批量取父块，走索引才不会退化成全表扫；
            # 它只能在补列之后建（见 `_SCHEMA` 下方的说明）。
            self._con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)")

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                yield self._con
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            else:
                self._con.execute("COMMIT")

    def _load_meta_or_init(self) -> None:
        with self._lock:
            rows = {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}
            if not rows:
                self._con.executemany(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    [
                        ("schema_version", str(SCHEMA_VERSION)),
                        ("embedding_model", self.embedding_model),
                        ("embedding_dim", str(self.dim)),
                        ("normalizer", self.normalizer.name),
                        ("domain", self.domain),
                        ("indexed_at", ""),
                    ],
                )
                return
            got = int(rows.get("schema_version", "0"))
            if got != SCHEMA_VERSION:
                raise StoreError(
                    f"索引 schema 版本不符：库里是 {got}，代码要求 {SCHEMA_VERSION}；"
                    f"请跑 `python scripts/migrate_index_layout.py` 迁移，"
                    f"或用 `ingest --rebuild` 重建"
                )
            old_dim = int(rows.get("embedding_dim", "0"))
            if old_dim != self.dim:
                # 维度不一致会让矩阵乘法直接崩，早报比晚崩好
                raise StoreError(
                    f"索引向量维度不符：库里是 {old_dim}，当前配置 {self.dim}；换了嵌入模型必须重建索引"
                )
            old_model = rows.get("embedding_model", "")
            if old_model and self.embedding_model and old_model != self.embedding_model:
                raise StoreError(
                    f"索引嵌入模型不符：库里是 {old_model!r}，当前配置 {self.embedding_model!r}；"
                    "不同模型的向量空间不可混用，请重建索引"
                )
            old_norm = rows.get("normalizer", "")
            if old_norm and old_norm != self.normalizer.name:
                # 归一化器变了 = 倒排索引里的词元变了，旧索引对新查询基本不可用；
                # 这种不一致只会表现为"搜不到"，必须显式拒绝。
                raise StoreError(
                    f"索引归一化器不符：库里是 {old_norm!r}，当前配置 "
                    f"{self.normalizer.name!r}；请用 `ingest --rebuild` 重建索引"
                )

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def _reload_index(self) -> None:
        """把矩阵与 rowid 映射读进内存。向量矩阵按 vector_row 升序排列。

        `vector_row` 的连续性必须在这里校验：`dense_search` 用矩阵行号 `i` 去查
        `self._row_index[i]`，即"矩阵第 i 行的向量属于哪个 chunk"完全由 `vector_row`
        决定。一旦 `vector_row` 出现空洞（例如调了 `delete_docs()` 之后没有
        `finalize()`），`_row_index` 会比矩阵短、标签整体前移——轻则**静默返回错误
        内容**（查询 A 的向量，返回 B 的正文），重则 `IndexError` 越界。

        这两种结果都是给出错误答案，不是服务降级，所以加载时就失败；修复办法只有
        重建索引。
        """
        with self._lock:
            rows = [int(r["vector_row"]) for r in self._con.execute("SELECT vector_row FROM chunks")]
            self._row_index = [
                r["chunk_id"] for r in self._con.execute("SELECT chunk_id FROM chunks ORDER BY vector_row")
            ]
            self._old_row = {
                r["chunk_id"]: r["vector_row"]
                for r in self._con.execute("SELECT chunk_id, vector_row FROM chunks")
            }
            self._hash_row = {
                r["content_hash"]: r["vector_row"]
                for r in self._con.execute("SELECT content_hash, vector_row FROM chunks ORDER BY vector_row")
            }
        if self.vector_path.exists():
            mat = np.load(self.vector_path)
        else:
            mat = np.zeros((0, self.dim), dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self.dim:
            raise StoreError(f"{self.vector_path} 形状 {mat.shape} 与 dim={self.dim} 不符")
        if sorted(rows) != list(range(len(rows))):
            raise StoreError(
                f"索引不一致：{len(rows)} 个块的 vector_row 不是连续的 0..{len(rows) - 1}"
                f"（实际 {min(rows) if rows else '-'}..{max(rows) if rows else '-'}）。"
                "成因通常是**删除文档后没有调用 finalize()**（纯删除的一轮 ingest 曾漏掉它）。"
                "向量与块已经错位，继续查询会返回错误内容。\n"
                "⚠️ 重建时**必须指定同一个后端**：`vector_row` 与 `chunk_vectors.npy` 是 "
                "`sqlite` 后端独有的，**`sqlite-vec` / `pgvector` 后端不维护它们**"
                "（在那些后端上这一列是 -1，本后端因此打不开）。所以「照默认重建」会原样再失败：\n"
                "    kubedocs --store sqlite ingest --rebuild      # 要读这份索引就得用 sqlite 后端重建\n"
                "    # 或者：继续用 sqlite-vec/pgvector，改回对应后端即可（那份数据本来就是它们的）"
            )
        if mat.shape[0] != len(rows):
            raise StoreError(
                f"索引不一致：向量矩阵有 {mat.shape[0]} 行，chunks 表有 {len(rows)} 行。"
                "成因通常是删除文档后没有调用 finalize()。"
                "重建时要指定同一个后端：`kubedocs --store sqlite ingest --rebuild`"
            )
        self._matrix: np.ndarray = np.ascontiguousarray(mat, dtype=np.float32)

    # ------------------------------------------------------------------ 读取
    def existing_doc_hashes(self) -> dict[str, str]:
        with self._lock:
            return {
                r["rel_path"]: r["content_hash"]
                for r in self._con.execute("SELECT rel_path, content_hash FROM documents")
            }

    def existing_hash_to_vector(self, hashes: set[str]) -> dict[str, Vector]:
        """按 content_hash 复用旧向量。只读取命中的行，不做全量拷贝。"""
        out: dict[str, Vector] = {}
        for h in hashes:
            row = self._hash_row.get(h)
            if row is None or row >= len(self._matrix):
                continue
            out[h] = self._matrix[row]
        return out

    def existing_doc_ids(self) -> set[str]:
        with self._lock:
            return {r["doc_id"] for r in self._con.execute("SELECT doc_id FROM documents")}

    def doc_texts(self, doc_ids: list[str]) -> dict[str, str]:
        """拼接指定文档的全部**子块**正文（小写化，供标注集 `must_contain` 自检）。

        刻意只拼子块、**不含父块**：这个函数回答的是"标注集里提到的内容是否真的进了索引"，
        而进索引的只有子块。把父块拼进来会掩盖"子块没进索引"这种真问题——父块与子块正文
        高度重叠，父块在、子块丢时拼接结果照样包含那个词，自检就永远绿。
        """
        if not doc_ids:
            return {}
        out: dict[str, list[str]] = {d: [] for d in doc_ids}
        with self._lock:
            for i in range(0, len(doc_ids), 400):
                batch = doc_ids[i : i + 400]
                marks = ",".join("?" * len(batch))
                # marks 只由 '?' 组成，值一律走参数绑定，不存在拼接注入
                sql = f"SELECT doc_id, body FROM chunks WHERE doc_id IN ({marks}) ORDER BY doc_id, ordinal"  # noqa: S608
                for r in self._con.execute(sql, batch):
                    out[r["doc_id"]].append(r["body"])
        return {d: "\n".join(parts).lower() for d, parts in out.items()}

    def stats(self) -> StoreStats:
        with self._lock:
            n_docs = self._con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            row = self._con.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(n_chars), 0) AS chars FROM chunks"
            ).fetchone()
            n_code = self._con.execute("SELECT COUNT(*) FROM chunks WHERE body LIKE '%```%'").fetchone()[0]
            # 父块数单独读数：它应当与 n_chunks 同量级但不是同一个数（"几段语境" vs
            # "几个检索单位"），体检时要能分别看见——父子切割没生效时这里会是 0。
            n_parents = self._con.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
            meta = {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}
        return StoreStats(
            n_docs=n_docs,
            n_chunks=row["n"],
            total_chars=row["chars"],
            n_with_code=n_code,
            embedding_model=meta.get("embedding_model", ""),
            embedding_dim=int(meta.get("embedding_dim", "0")),
            indexed_at=meta.get("indexed_at", ""),
            domain=meta.get("domain", ""),
            normalizer=meta.get("normalizer", ""),
            n_parents=n_parents,
        )

    def _parent_texts(self, parent_ids: set[str]) -> dict[str, str]:
        """按 parent_id **批量**取父块正文（缺的键就是不存在的父块）。

        一次 `IN` 查完，绝不在调用方的循环里一条条查：`get_chunks` 的入参是检索结果，
        条数随 top-k 线性增长，逐条查会变成标准的 N+1（本地库表现为"取 20 条命中要
        20 次往返"，pgvector 上则是 20 次网络往返）。分批是为了避开 SQLite 变量数上限。
        """
        ids = [p for p in parent_ids if p]
        if not ids:
            return {}
        out: dict[str, str] = {}
        with self._lock:
            for i in range(0, len(ids), 400):
                batch = ids[i : i + 400]
                marks = ",".join("?" * len(batch))
                # marks 只由 '?' 组成，值一律走参数绑定
                sql = f"SELECT parent_id, body FROM parents WHERE parent_id IN ({marks})"  # noqa: S608
                for r in self._con.execute(sql, batch):
                    out[r["parent_id"]] = r["body"]
        return out

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkHit]:
        """按 chunk_id 取回块（保序由调用方负责）。父块正文在这里**批量**补齐。"""
        if not chunk_ids:
            return {}
        rows: list[sqlite3.Row] = []
        with self._lock:
            # 分批 IN，避开 SQLite 变量数上限（默认 999）
            for i in range(0, len(chunk_ids), 400):
                batch = chunk_ids[i : i + 400]
                marks = ",".join("?" * len(batch))
                # marks 只由 '?' 组成，值一律走参数绑定
                sql = f"SELECT chunk_id, doc_id, ordinal, title, breadcrumb, source_url, body, parent_id FROM chunks WHERE chunk_id IN ({marks})"  # noqa: S608
                rows.extend(self._con.execute(sql, batch).fetchall())
        # 先取完父块再拼装：不能边遍历边走库（N+1），也不能在持锁的批次循环里嵌套查询。
        # 缺父块（老库 / `parents=None` 写入）就留空串，由 `context_text` 退回子块正文。
        texts = self._parent_texts({r["parent_id"] for r in rows if r["parent_id"]})
        out: dict[str, ChunkHit] = {}
        for r in rows:
            out[r["chunk_id"]] = ChunkHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                ordinal=r["ordinal"],
                title=r["title"],
                breadcrumb=r["breadcrumb"],
                source_url=r["source_url"],
                body=r["body"],
                parent_id=r["parent_id"],
                parent_text=texts.get(r["parent_id"], ""),
            )
        return out

    # ------------------------------------------------------------------ 检索
    def keyword_search(self, query: str, limit: int) -> list[tuple[str, int]]:
        # 查询必须先过**同一个**归一化器：中文查询要 jieba 分词后才可能命中
        # 中文索引；英文 identity 时是原样透传。
        expr = build_match_expr(self.normalizer.query_text(query))
        if not expr or limit <= 0:
            return []
        w = ", ".join(str(x) for x in _BM25_WEIGHTS)

        # 分两步取，不要在 FTS 上直接 JOIN chunks：JOIN 会让查询计划变成
        # "先连上全部命中行再排序"，命中上万行时明显变慢。两步走只需要对最终 limit
        # 行做一次 rowid 查表，名次仍由 FTS 的 bm25 顺序决定，与一步式完全一致。
        with self._lock:
            fts_sql = f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, {w}) LIMIT ?"  # noqa: S608
            ids = [r[0] for r in self._con.execute(fts_sql, (expr, limit))]
            if not ids:
                return []
            marks = ",".join("?" * len(ids))
            rows = self._con.execute(
                f"SELECT rowid, chunk_id FROM chunks WHERE rowid IN ({marks})",  # noqa: S608 - marks 全是 ?
                ids,
            ).fetchall()
        by_rowid = {r[0]: r[1] for r in rows}
        return [(by_rowid[r], i) for i, r in enumerate(ids, start=1) if r in by_rowid]

    def dense_search(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        if limit <= 0 or not self._row_index or self._matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise StoreError(f"查询向量维度 {q.shape[0]} 与索引 {self.dim} 不符")
        n = min(limit, len(self._row_index), self._matrix.shape[0])
        if n <= 0:
            return []
        sims = self._matrix @ q  # 向量已归一化，内积即余弦
        # argpartition 取 top-n 后局部排序：全量 sort 在 10 万块量级是纯浪费
        idx = np.argpartition(-sims, n - 1)[:n]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(self._row_index[int(i)], rank) for rank, i in enumerate(idx, start=1)]

    # ------------------------------------------------------------------ 写入
    def _drop_doc_chunks(self, con: sqlite3.Connection, doc_id: str) -> int:
        """删掉一篇文档的所有块。**顺序不能反**：先摘 FTS 索引项，再删内容行，
        否则索引会永久不一致（见模块文档第 1 条）。
        """
        n = con.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]
        con.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, body_normalized, title, breadcrumb) "
            "SELECT 'delete', rowid, body_normalized, title, breadcrumb FROM chunks "
            "WHERE doc_id = ?",
            (doc_id,),
        )
        con.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return int(n)

    def _drop_doc_parents(self, con: sqlite3.Connection, doc_id: str) -> None:
        """删掉一篇文档的父块。

        父块必须与子块同生共死：覆盖时不清，同一 `parent_id` 再写会撞主键；残留的旧父块
        更会让新子块取到**过期的语境**（喂错上下文比喂空上下文更坏）。删除文档时不清，
        父块表就只增不减。
        """
        con.execute("DELETE FROM parents WHERE doc_id = ?", (doc_id,))

    def delete_docs(self, rel_paths: set[str]) -> int:
        if not rel_paths:
            return 0
        removed = 0
        with self._tx() as con:
            for rel in rel_paths:
                row = con.execute("SELECT doc_id FROM documents WHERE rel_path = ?", (rel,)).fetchone()
                if row is None:
                    continue
                removed += self._drop_doc_chunks(con, row["doc_id"])
                self._drop_doc_parents(con, row["doc_id"])
                con.execute("DELETE FROM documents WHERE doc_id = ?", (row["doc_id"],))
        self._dirty = True
        return removed

    def upsert_docs(
        self,
        docs: list[DocInput],
        chunks: list[ChunkInput],
        vectors: dict[str, Vector],
        parents: list[ParentInput] | None = None,
    ) -> None:
        if not docs and not chunks and not parents:
            return
        missing = [c.chunk_id for c in chunks if c.chunk_id not in vectors]
        if missing:
            raise MissingVectorError(
                f"{len(missing)} 个块没有向量（例：{missing[:3]}）——"
                "插入无向量的块会让向量路静默漏召回，这里直接拒绝"
            )
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._tx() as con:
            for d in docs:
                self._drop_doc_chunks(con, d.doc_id)
                con.execute("DELETE FROM documents WHERE doc_id = ?", (d.doc_id,))
                n_chunks = sum(1 for c in chunks if c.doc_id == d.doc_id)
                con.execute(
                    "INSERT INTO documents(doc_id, rel_path, title, source_url, content_hash,"
                    " n_chunks, updated_at) VALUES(?,?,?,?,?,?,?)",
                    (d.doc_id, d.rel_path, d.title, d.source_url, d.content_hash, n_chunks, now),
                )
            # 覆盖的粒度是文档：父块与子块一起清掉再写。按"本次涉及的全部 doc_id"清，而不只
            # `docs` 里的——`parents` 里出现的 doc_id 同样会被覆盖。`parents=None` 时也照清：
            # 那表示"这次调用不提供父块"，留着上一篇的旧父块只会喂出过期语境。
            for doc_id in {d.doc_id for d in docs} | {p.doc_id for p in (parents or [])}:
                self._drop_doc_parents(con, doc_id)
            for c in chunks:
                cur = con.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, ordinal, title, breadcrumb, source_url,"
                    " body, body_normalized, content_hash, n_chars, vector_row, parent_id)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.ordinal,
                        c.title,
                        c.breadcrumb,
                        c.source_url,
                        c.body,
                        self.normalizer.index_text(c.body),
                        c.content_hash,
                        c.n_chars,
                        -1,
                        c.parent_id,
                    ),
                )
                rowid = cur.lastrowid
                # 外内容表要显式给 rowid，让索引项与内容行一一对应
                con.execute(
                    "INSERT INTO chunks_fts(rowid, body_normalized, title, breadcrumb) VALUES(?,?,?,?)",
                    (rowid, self.normalizer.index_text(c.body), c.title, c.breadcrumb),
                )
                self._pending[c.chunk_id] = np.asarray(vectors[c.chunk_id], dtype=np.float32)
            # 父块只进普通表：不进 FTS、不进向量索引。它与块行、向量在**同一事务**里落库，
            # 所以不存在"子块写了、语境还没写"的中间态。
            for p in parents or ():
                con.execute(
                    "INSERT INTO parents(parent_id, doc_id, ordinal, title, breadcrumb, source_url,"
                    " body, content_hash, n_chars, location, n_children)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        p.parent_id,
                        p.doc_id,
                        p.ordinal,
                        p.title,
                        p.breadcrumb,
                        p.source_url,
                        p.body,
                        p.content_hash,
                        p.n_chars,
                        p.location,
                        p.n_children,
                    ),
                )
            self._index_chunk_vectors(con, chunks, vectors)
        self._dirty = True

    def _index_chunk_vectors(
        self, con: sqlite3.Connection, chunks: list[ChunkInput], vectors: dict[str, Vector]
    ) -> None:
        """钩子：把刚写入事务的块向量落到本实现的向量结构里。

        基类（numpy 矩阵）的向量**不在这里落盘**：`chunks.rowid` 会被 SQLite 回收，
        矩阵必须等本批全部插入完成后按最终 rowid 顺序重建（见 `finalize`），
        所以基类只把向量暂存在 `_pending`。这个钩子因此是空实现。

        `SqliteVecStore` 覆写它，把向量写进 `vec0` 虚表——与块行**同一事务**提交，
        不需要"重排"这一步，也不会有"块写了、向量没写"的中间态。
        """
        return

    def finalize(self) -> None:
        """把本轮新增向量并入矩阵、压实 vector_row 并落盘。

        重排是必须的：rowid 会被回收，删掉最大 rowid 的块后新块可能拿到更小的 rowid，
        于是"按 rowid 顺序"与"按矩阵行号顺序"不再一致。这里一律以当前 rowid 顺序重建，
        映射由 vector_row 落库固化；本批未重编码的块复用旧矩阵中的向量。
        """
        with self._lock:
            rows = self._con.execute("SELECT rowid, chunk_id FROM chunks ORDER BY rowid").fetchall()
            ids = [r["chunk_id"] for r in rows]
            new_mat = np.zeros((len(ids), self.dim), dtype=np.float32)
            for i, cid in enumerate(ids):
                v = self._pending.get(cid)
                if v is None:
                    old = self._old_row.get(cid)
                    if old is None or old >= len(self._matrix):
                        raise MissingVectorError(f"块 {cid} 既无新向量也无旧向量；请用 rebuild 重建索引")
                    v = self._matrix[old]
                if v.shape[0] != self.dim:
                    raise StoreError(f"块 {cid} 向量维度 {v.shape[0]} != {self.dim}")
                new_mat[i] = v

            tmp = self.vector_path.with_suffix(".npy.tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, new_mat)
            # 先写临时文件再原子替换：中途崩了也不会留下半个矩阵
            tmp.replace(self.vector_path)

            with self._tx() as con:
                con.executemany(
                    "UPDATE chunks SET vector_row = ? WHERE chunk_id = ?",
                    [(i, cid) for i, cid in enumerate(ids)],
                )
                con.execute(
                    "INSERT INTO meta(key, value) VALUES('indexed_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"),),
                )
            self._pending.clear()
            self._dirty = False
        self._reload_index()

    def rebuild_from_scratch(self) -> None:
        """清空全部数据（向量矩阵也一并清掉）。用于换模型/换分块参数后重建。"""
        with self._tx() as con:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM documents")
            # 父块也要清：重建后子块会按新的切块方式重写，旧父块没有任何东西再引用它，
            # 留着只会让下一轮的 `stats().n_parents` 虚高、`parents` 表只增不减。
            con.execute("DELETE FROM parents")
        if self.vector_path.exists():
            self.vector_path.unlink()
        self._pending.clear()
        self._dirty = False
        self._reload_index()

    def verify_fts(self) -> str:
        """对 FTS 索引做一致性自检（外内容表最容易在这里出问题）。"""
        with self._lock:
            try:
                self._con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
            except sqlite3.DatabaseError as e:
                return f"FAIL: {e}"
        return "ok"

    def doc_summary(self, limit: int = 20) -> list[dict[str, object]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT rel_path, n_chunks, updated_at FROM documents ORDER BY rel_path LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def meta_dump(self) -> dict[str, str]:
        with self._lock:
            return {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}


def from_settings(settings: Settings) -> SqliteStore:
    """按配置构造当前域的本地索引（路径解析、有效模型名、归一化器都在这里落地）。"""
    return SqliteStore(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        normalizer=build_normalizer(settings.normalizer),
        domain=settings.domain,
    )
