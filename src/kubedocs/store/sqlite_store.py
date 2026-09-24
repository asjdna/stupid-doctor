"""本地嵌入式索引：SQLite（FTS5 全文） + numpy 向量矩阵。

## 为什么是这套组合而不是 Postgres+pgvector

先把"业务逻辑跑通"和"可一键部署"拆成两件事。前者只需要一个能落盘、
能全文检索、能做向量近邻的存储；这套组合全部由标准库 + numpy 提供，
**零外部进程、零镜像、零网络**，本机立刻可跑。

代价与边界（写清楚，免得日后误判）：
- 单写者：SQLite 并发写入有限，只适合单进程摄取；线上多实例写要走 pgvector 版；
- 全量向量常驻内存：18861×384 float32 ≈ 29MB，可接受；再大两个数量级就该换 ANN 索引；
- 无 ANN：暴力内积在 10 万块量级仍是毫秒级，超过就该上 HNSW/IVF。

架构上这不算"降级实现"，而是同一 Store 协议下的**另一种实现**——
检索与生成代码一行都不用改，切换只改配置。

## 两个容易踩的坑，这里直接规避

1. **FTS5 外内容表（external content）的删除**：外内容表不能用普通
   `DELETE`，必须先用 FTS5 的 `'delete'` 指令把索引项摘掉，再删内容行；
   且顺序反了索引会永久不一致（后续查询报 malformed）。见 `_drop_doc_chunks`。
2. **向量矩阵与 rowid 的对应**：`chunks` 的 rowid 在删除后会**被回收**，
   新插入的块可能拿到比现存块更小的 rowid。所以不能假设"rowid 单调递增"，
   `finalize()` 每次按当前 rowid 顺序**重建**矩阵并回写 `vector_row`。
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from kubedocs.config import Settings
from kubedocs.store.base import (
    ChunkHit,
    ChunkInput,
    DocInput,
    StoreStats,
    Vector,
)

SCHEMA_VERSION = 1

# FTS5 外内容表：正文只存一份（在 chunks 里），FTS 只存倒排索引。
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
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    title        TEXT NOT NULL,
    breadcrumb   TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    body         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    n_chars      INTEGER NOT NULL,
    vector_row   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body, title, breadcrumb,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

# BM25 列权重，顺序必须与 chunks_fts 的列声明一致：(body, title, breadcrumb)。
# 标题命中比正文命中更能说明"这一块就是讲这件事的"，面包屑次之。
_BM25_WEIGHTS = (1.0, 3.0, 1.5)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# 只清掉功能词。不做词形还原——那是 FTS5 的 porter 分词器负责的事。
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "not", "of", "on",
    "or", "that", "the", "their", "then", "there", "these", "this", "to", "up", "use",
    "used", "using", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
})


class StoreError(RuntimeError):
    """存储层可预期错误（配置/一致性问题），调用方应把它当用户错误处理。"""


class MissingVectorError(StoreError):
    """某块的向量既不在本次运行的结果里、也不在旧矩阵里——不能静默填 0。"""


def build_match_expr(query: str) -> str:
    """把自由文本查询改写成安全的 FTS5 MATCH 表达式。

    直接 `WHERE chunks_fts MATCH ?` 传原文是**注入面**：FTS5 的 MATCH 有自己的
    语法（`-` `*` `"` `(` `:` `AND/OR/NOT`），用户输入 `pod -limit` 会被解释成
    "含 pod 且不含 limit"，输入 `"` 直接报语法错、请求 500。

    这里只抽取字母数字下划线词元，逐个加双引号短语化，再用 OR 连接：
    - 词元集天然不含特殊字符，注入面关闭；
    - 用 OR 而非 AND 是为了**召回**（多词查询在文档里往往并不同现），
      相关性交给 BM25 排序——包含全部词元的文档排名自然会高。
    """
    tokens = [t for t in _TOKEN_RE.findall(query) if t.lower() not in _STOPWORDS]
    if not tokens:
        # 全是被停用词/特殊符号，此时"什么都不匹配"比"匹配全部"更诚实
        return ""
    # 同一个词重复出现会拉高该项权重，先去重保序
    seen: dict[str, None] = {}
    for t in tokens:
        seen.setdefault(t.lower(), None)
    return " OR ".join(f'"{t}"' for t in seen)


class SqliteStore:
    """SQLite + FTS5 + numpy 的本地索引实现。"""

    def __init__(self, db_path: str | Path, vector_path: str | Path, *, dim: int,
                 embedding_model: str = "") -> None:
        self.db_path = Path(db_path)
        self.vector_path = Path(vector_path)
        self.dim = dim
        self.embedding_model = embedding_model
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False + 全库访问加锁：FastAPI 的同步端点跑在线程池里，
        # 一个连接跨线程使用本身是安全的，前提是同一时刻只有一处在用（这里用锁保证）。
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False,
                                    isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.executescript(_SCHEMA)

        self._pending: dict[str, Vector] = {}
        self._dirty = False
        self._load_meta_or_init()
        self._reload_index()

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
                        ("indexed_at", ""),
                    ],
                )
                return
            got = int(rows.get("schema_version", "0"))
            if got != SCHEMA_VERSION:
                raise StoreError(
                    f"索引 schema 版本不符：库里是 {got}，代码要求 {SCHEMA_VERSION}；"
                    "请删除索引目录后重新 ingest"
                )
            old_dim = int(rows.get("embedding_dim", "0"))
            if old_dim != self.dim:
                # 维度不一致会让矩阵乘法直接崩，早报比晚崩好
                raise StoreError(
                    f"索引向量维度不符：库里是 {old_dim}，当前配置 {self.dim}；"
                    "换了嵌入模型必须重建索引"
                )
            old_model = rows.get("embedding_model", "")
            if old_model and self.embedding_model and old_model != self.embedding_model:
                raise StoreError(
                    f"索引嵌入模型不符：库里是 {old_model!r}，当前配置 {self.embedding_model!r}；"
                    "不同模型的向量空间不可混用，请重建索引"
                )

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def _reload_index(self) -> None:
        """把矩阵与 rowid 映射读进内存。向量矩阵按 vector_row 升序排列。"""
        with self._lock:
            self._row_index = [
                r["chunk_id"] for r in self._con.execute(
                    "SELECT chunk_id FROM chunks ORDER BY vector_row")
            ]
            self._old_row = {
                r["chunk_id"]: r["vector_row"] for r in self._con.execute(
                    "SELECT chunk_id, vector_row FROM chunks")
            }
            self._hash_row = {
                r["content_hash"]: r["vector_row"] for r in self._con.execute(
                    "SELECT content_hash, vector_row FROM chunks ORDER BY vector_row")
            }
        if self.vector_path.exists():
            mat = np.load(self.vector_path)
        else:
            mat = np.zeros((0, self.dim), dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self.dim:
            raise StoreError(f"{self.vector_path} 形状 {mat.shape} 与 dim={self.dim} 不符")
        self._matrix: np.ndarray = np.ascontiguousarray(mat, dtype=np.float32)

    # ------------------------------------------------------------------ 读取
    def existing_doc_hashes(self) -> dict[str, str]:
        with self._lock:
            return {r["rel_path"]: r["content_hash"]
                    for r in self._con.execute("SELECT rel_path, content_hash FROM documents")}

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
        """拼接指定文档的全部块正文（小写化，供标注集 `must_contain` 自检）。"""
        if not doc_ids:
            return {}
        out: dict[str, list[str]] = {d: [] for d in doc_ids}
        with self._lock:
            for i in range(0, len(doc_ids), 400):
                batch = doc_ids[i:i + 400]
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
            n_code = self._con.execute(
                "SELECT COUNT(*) FROM chunks WHERE body LIKE '%```%'"
            ).fetchone()[0]
            meta = {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}
        return StoreStats(
            n_docs=n_docs, n_chunks=row["n"], total_chars=row["chars"], n_with_code=n_code,
            embedding_model=meta.get("embedding_model", ""),
            embedding_dim=int(meta.get("embedding_dim", "0")),
            indexed_at=meta.get("indexed_at", ""),
        )

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkHit]:
        if not chunk_ids:
            return {}
        out: dict[str, ChunkHit] = {}
        with self._lock:
            # 分批 IN，避开 SQLite 变量数上限（默认 999）
            for i in range(0, len(chunk_ids), 400):
                batch = chunk_ids[i:i + 400]
                marks = ",".join("?" * len(batch))
                # marks 只由 '?' 组成，值一律走参数绑定
                sql = f"SELECT chunk_id, doc_id, ordinal, title, breadcrumb, source_url, body FROM chunks WHERE chunk_id IN ({marks})"  # noqa: S608
                for r in self._con.execute(sql, batch):
                    out[r["chunk_id"]] = ChunkHit(
                        chunk_id=r["chunk_id"], doc_id=r["doc_id"], ordinal=r["ordinal"],
                        title=r["title"], breadcrumb=r["breadcrumb"],
                        source_url=r["source_url"], body=r["body"],
                    )
        return out

    # ------------------------------------------------------------------ 检索
    def keyword_search(self, query: str, limit: int) -> list[tuple[str, int]]:
        expr = build_match_expr(query)
        if not expr or limit <= 0:
            return []
        w = ", ".join(str(x) for x in _BM25_WEIGHTS)
        # expr 已由 build_match_expr 收窄为纯词元短语，w 是代码里的常量元组
        sql = f"SELECT c.chunk_id AS chunk_id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, {w}) LIMIT ?"  # noqa: S608
        with self._lock:
            rows = self._con.execute(sql, (expr, limit)).fetchall()
        return [(r["chunk_id"], i) for i, r in enumerate(rows, start=1)]

    def dense_search(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        if limit <= 0 or not self._row_index or self._matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise StoreError(f"查询向量维度 {q.shape[0]} 与索引 {self.dim} 不符")
        n = min(limit, len(self._row_index), self._matrix.shape[0])
        if n <= 0:
            return []
        sims = self._matrix @ q            # 向量已归一化，内积即余弦
        # argpartition 取 top-n 后局部排序：全量 sort 在 10 万块量级是纯浪费
        idx = np.argpartition(-sims, n - 1)[:n]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(self._row_index[int(i)], rank) for rank, i in enumerate(idx, start=1)]

    # ------------------------------------------------------------------ 写入
    def _drop_doc_chunks(self, con: sqlite3.Connection, doc_id: str) -> int:
        """删掉一篇文档的所有块。**顺序不能反**，见模块文档第 1 条。"""
        n = con.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]
        con.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, body, title, breadcrumb) "
            "SELECT 'delete', rowid, body, title, breadcrumb FROM chunks WHERE doc_id = ?",
            (doc_id,),
        )
        con.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return int(n)

    def delete_docs(self, rel_paths: set[str]) -> int:
        if not rel_paths:
            return 0
        removed = 0
        with self._tx() as con:
            for rel in rel_paths:
                row = con.execute("SELECT doc_id FROM documents WHERE rel_path = ?",
                                  (rel,)).fetchone()
                if row is None:
                    continue
                removed += self._drop_doc_chunks(con, row["doc_id"])
                con.execute("DELETE FROM documents WHERE doc_id = ?", (row["doc_id"],))
        self._dirty = True
        return removed

    def upsert_docs(self, docs: list[DocInput], chunks: list[ChunkInput],
                    vectors: dict[str, Vector]) -> None:
        if not docs and not chunks:
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
            for c in chunks:
                cur = con.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, ordinal, title, breadcrumb, source_url,"
                    " body, content_hash, n_chars, vector_row) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (c.chunk_id, c.doc_id, c.ordinal, c.title, c.breadcrumb, c.source_url,
                     c.body, c.content_hash, c.n_chars, -1),
                )
                rowid = cur.lastrowid
                # 外内容表要显式给 rowid，让索引项与内容行一一对应
                con.execute(
                    "INSERT INTO chunks_fts(rowid, body, title, breadcrumb) VALUES(?,?,?,?)",
                    (rowid, c.body, c.title, c.breadcrumb),
                )
                self._pending[c.chunk_id] = np.asarray(vectors[c.chunk_id], dtype=np.float32)
        self._dirty = True

    def finalize(self) -> None:
        """把本轮新增向量并入矩阵、压实 vector_row 并落盘。

        重排是必须的：rowid 会被回收，删掉最大 rowid 的块后新块可能拿到更小的
        rowid，于是"按 rowid 顺序"与"按矩阵行号顺序"不再一致。这里一律以当前
        rowid 顺序重建，映射关系由 vector_row 落库固化。
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
                        raise MissingVectorError(
                            f"块 {cid} 既无新向量也无旧向量；请用 rebuild 重建索引"
                        )
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
                "SELECT rel_path, n_chunks, updated_at FROM documents "
                "ORDER BY rel_path LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def meta_dump(self) -> dict[str, str]:
        with self._lock:
            return {r["key"]: r["value"] for r in self._con.execute("SELECT key, value FROM meta")}


def from_settings(settings: Settings) -> SqliteStore:
    """按配置构造本地索引（把路径解析与"有效模型名"的细节挡在 store 之外）。"""
    return SqliteStore(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
    )


def dump_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
