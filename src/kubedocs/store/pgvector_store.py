"""生产存储实现：Postgres + pgvector（HNSW）。

## 为什么需要它（而不是"本地 SQLite 就够了"）

本地实现的三条边界，正好是它补上的三件事：

| 边界 | 本地 SQLite | 本实现 |
|---|---|---|
| 单写者 | 只适合单进程摄取 | 多实例并发写、事务与行锁 |
| 无 ANN | 1.9 万块还行，百万级不行 | **HNSW 索引**，召回/延迟可调 |
| 无多租户 | 一个域一个文件 | 一个域一个 **schema**，可加 RLS |
| 运维 | 备份=拷文件 | 标准备份/PITR/连接池/可观测 |

## 全文检索：与 FTS5 的**真实差异**（不粉饰）

1. **排序函数不同源**。FTS5 用的是真 BM25（含 IDF 与文档长度归一化）；
   Postgres 核心**没有 BM25**，这里用 `ts_rank`（TF 加权，无 IDF、无长度归一化），
   列权重通过 `setweight` + `ts_rank(weights, ...)` 对齐成 title 3.0 /
   breadcrumb 1.5 / body 1.0（与 FTS5 的 `bm25(chunks_fts, 1.0, 3.0, 1.5)` 同序尝试）。
   结论：**相同的命中集合下，排序可能不同**——尤其"常见词 vs 稀有词"的相对权重。
   要 BM25 就得装扩展（ParadeDB/pg_search），本任务不引入。
2. **词干与停用词**：`english` 配置 = Snowball 词干 + 英文停用词表；
   FTS5 的 `porter` 只做 Porter 词干、**不删停用词**。查询侧我们已自行去停用词，
   但索引侧 Postgres 会把纯停用词的行索引成空——极端查询下命中集合会有差异。
3. **分词器**：Postgres 默认解析器 vs FTS5 `unicode61`。对
   `kube-apiserver` / `v1.29` 这类标识符，两者都按非字母数字切分，但边界情况
   （下划线、点号与数字的组合）不完全一致。
4. **中文**：这里**没有** jieba。采用的折中是把**应用侧的归一化文本**
   （`normalizer.index_text()`，中文域就是 jieba 预分词结果）写进 `body_normalized`，
   再用 `to_tsvector('english', body_normalized)` 建索引。因此：
   - 中文**能**检索（分词在应用层做，与 SQLite 侧是同一个归一化器）；
   - 但 `english` 配置会对中文词元跑 Snowball 词干（对 CJK 是空操作），
     且**不会**对中文做任何切分——若把 `normalizer=identity` 用于中文语料，
     Postgres 与 FTS5 一样会把整句当成一个词元，同样搜不到。
   - 换句话说：**中文能力来自应用层归一化器，不来自 Postgres**；
     想在库内做中文分词需要 zhparser/pg_jieba 扩展，本任务未引入。
5. **`tsvector` 是生成列（`GENERATED ALWAYS AS ... STORED`）**，无法与正文脱钩——
   比 FTS5 外内容表"删错顺序就永久不一致"要结实，这是本实现相对更安全的一点。

## HNSW 参数与理由

- `m = 16`：每个节点的出边数。pgvector 默认值；1.9 万块规模下 8 会明显掉召回，
  32 以上收益递减而索引体积/建图时间线性上升。**m 只能重建索引才能改。**
- `ef_construction = 64`：建图时的候选队列。默认值。与 m 一起决定图的质量；
  本规模下建索引只需数秒到数十秒，调到 200 的收益要在更大规模上才体现。
- `ef_search = 200`（查询期 `hnsw.ef_search`）：pgvector 默认 40，本仓库原先取 100，
  2026-10-09 起提到 200。**加大的理由是精确召回**，依据是本机实测（同轮、同一份
  18861 块索引、50 条金标 ×3 轮）：向量路 top-10 与 numpy 精确检索逐位相同的查询数
  100 → 200 时从 36/50 涨到 43/50（平均集合重合 0.9420 → 0.9800），而 dense p50 只从
  4.29ms 涨到 5.79ms；再调到 400 只多 1 条（44/50、0.9840）却再贵 1.45ms，**收益递减**。
  它**不需要重建索引**，可随时按需调（`Settings.pg_hnsw_ef_search`）。
  ⚠️ 注意它治不了混合检索的质量落差——那条落差来自全文路，与 ef_search 无关
  （实测：ef=100/200/400 的 hybrid hit/recall 完全一样），见 ADR-0004 第九节。
- 距离用 `vector_cosine_ops` + `<=>`：与另外两个实现的余弦语义一致。
  向量已归一化时，余弦序与内积序严格相同。

## 连接

用 `psycopg` + **连接池**（`psycopg_pool`）：HTTP 层是多线程的，每次查询新建连接
在 100ms 量级，会把检索延迟整体抬一个数量级。池的每条连接在 `configure` 回调里
设置 `hnsw.ef_search` 并注册 pgvector 类型适配器。

**但 schema 初始化不能走池**：`configure` 里的 `register_vector` 要求 `vector` 类型
已存在，用池去 `CREATE EXTENSION` 会自己把自己锁死（实测见 `__init__` 的注释）。
所以顺序固定为「直连建 schema → 再开池」。

SQL 里的表名一律通过 `sql.Identifier` 注入（见 `_q`），值一律走 `%s` 参数绑定：
不给自己留"这句是我手拼的、应该安全"的口头承诺。
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import psycopg
from psycopg import sql

from kubedocs.config import Settings
from kubedocs.normalize import IdentityNormalizer, Normalizer, build_normalizer
from kubedocs.store.base import (
    ChunkHit,
    ChunkInput,
    DocInput,
    MissingVectorError,
    StoreError,
    StoreStats,
    Vector,
)
from kubedocs.store.textquery import build_tsquery_expr

# pgvector 侧自己的 schema 版本（与 SQLite 的 SCHEMA_VERSION 无关：两者是不同引擎）
PG_SCHEMA_VERSION = 1

# ts_rank 的权重数组顺序是 {D, C, B, A}，必须与建表时的 setweight 对应：
# body_normalized=D，breadcrumb=B，title=A（C 未使用，占位跟随 body）。
#
# ⚠️ `ts_rank` 的权重**必须落在 [0, 1] 内**，否则 Postgres 直接报
# `InvalidParameterValue: weight out of range`（`tsrank.c` 里就是这么校验的）。
# 所以这里**不能**照抄 FTS5 `bm25(chunks_fts, 1.0, 3.0, 1.5)` 那组倍数——原实现
# 写了 1.0/1.5/3.0，导致 `keyword_search` 一调用就抛错（实测），全文路从未跑通。
# 合法写法是给一组**比例相同、最大值为 1** 的权重：body 1/3、breadcrumb 1/2、title 1。
# ts_rank 对权重是线性的（整体乘一个常数只等比缩放分数、不改次序），
# 因此这与"body : breadcrumb : title = 1 : 1.5 : 3"的原意**同序**，只是分数尺度不同；
# 而 `ORDER BY ts_rank(...) DESC` 只依赖次序，故意图得以保留。
_TS_RANK_WEIGHTS = (1 / 3, 1 / 3, 1 / 2, 1.0)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TSV_EXPR = (
    "setweight(to_tsvector('english', coalesce(body_normalized, '')), 'D') || "
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(breadcrumb, '')), 'B')"
)

# 建表 DDL（语句用 `;` 分隔，逐条执行）。`embedding` 与 `tsv` 都在块行内，
# 不存在"块在、向量不在"的中间态，也不需要任何侧车文件（对比 SQLite 侧的 .npy）。
# `{documents}` / `{chunks}` 是 sql.Identifier 占位符，`{dim}`/`{m}`/`{efc}` 是整数字面量。
_DDL_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS {documents} ("
    "  doc_id text PRIMARY KEY,"
    "  rel_path text NOT NULL UNIQUE,"
    "  title text NOT NULL,"
    "  source_url text NOT NULL,"
    "  content_hash text NOT NULL,"
    "  n_chunks integer NOT NULL,"
    "  updated_at text NOT NULL)",
    "CREATE TABLE IF NOT EXISTS {chunks} ("
    "  chunk_id text PRIMARY KEY,"
    "  doc_id text NOT NULL REFERENCES {documents}(doc_id) ON DELETE CASCADE,"
    "  ordinal integer NOT NULL,"
    "  title text NOT NULL,"
    "  breadcrumb text NOT NULL,"
    "  source_url text NOT NULL,"
    "  body text NOT NULL,"
    "  body_normalized text NOT NULL DEFAULT '',"
    "  content_hash text NOT NULL,"
    "  n_chars integer NOT NULL,"
    "  embedding vector({dim}) NOT NULL,"
    f"  tsv tsvector GENERATED ALWAYS AS ({_TSV_EXPR}) STORED)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON {chunks} (doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON {chunks} (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON {chunks} USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON {chunks} "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {efc})",
)


def probe(dsn: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    """探测数据库是否可用（不做任何建表动作）。

    存在的理由：pgvector 的测试与前置检查都需要区分"数据库连不上（环境问题，
    应 skip / 给出提示）"与"代码错了（应失败）"。把两者混在一起，
    CI 要么红得莫名其妙，要么绿得莫名其妙。
    """
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover - 取决于环境
        return False, f"未安装 psycopg：{e}"
    try:
        with psycopg.connect(dsn, connect_timeout=int(max(1, timeout))) as conn:
            conn.execute("SELECT 1")
            row = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone()
    except Exception as e:  # noqa: BLE001 - 探测就是要吞掉一切异常并转成原因字符串
        return False, f"{type(e).__name__}: {e}"
    if row is None:
        return False, "Postgres 可用，但没有 pgvector 扩展（pg_available_extensions 里没有 vector）"
    return True, ""


class PgVectorStore:
    """Postgres + pgvector（HNSW）的生产实现。"""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str,
        dim: int,
        embedding_model: str = "",
        normalizer: Normalizer | None = None,
        domain: str = "",
        pool_min: int = 1,
        pool_max: int = 10,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 200,
    ) -> None:
        """`hnsw_ef_search` 的默认值必须与 `Settings.pg_hnsw_ef_search` **保持一致**：
        两处默认值不同就会出现"直接构造时是 100、走配置时是 200"的双真相，
        而这类差异只会表现为"同一份数据换个入口召回就不一样"。
        """
        if not _IDENT_RE.match(schema):
            raise StoreError(f"非法的 schema 名 {schema!r}：只允许 [A-Za-z_][A-Za-z0-9_]*")
        self.dsn = dsn
        self.schema = schema
        self.dim = dim
        self.embedding_model = embedding_model
        self.domain = domain
        # 归一化器必须与建索引时一致（与 SQLite 侧同一个理由：不一致只会表现为"搜不到"）
        self.normalizer: Normalizer = normalizer or IdentityNormalizer()
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self._dirty = False
        self._tbl = {name: sql.Identifier(schema, name) for name in ("documents", "chunks", "meta")}

        self._register_vector = _import_register_vector()
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover - 取决于环境
            raise StoreError("pgvector 后端需要 psycopg-pool：pip install 'psycopg[pool]'") from e
        # ⚠️ 顺序是**有意义**的：schema 初始化必须在连接池建立**之前**跑完。
        # 池的 `configure` 回调会对每条新连接执行 `register_vector`，而那个函数要求
        # 库里**已经存在** `vector` 类型；池一旦 open，后台就会立刻开始建连接。
        # 于是"建扩展"被排在"注册类型"之后，而注册类型又只能靠池的连接去执行——
        # 死锁。实测（compose 刚起、`\dx` 只有 plpgsql 的全新库）100% 复现：
        #   error connecting in 'kubedocs-pgvector': vector type not found in the database
        #   psycopg_pool.PoolTimeout: couldn't get a connection after 10.00 sec
        # 而 `CREATE EXTENSION` 那句**根本没被执行到**。所以 schema 这一步走直连，
        # 池只服务查询/写入流量（这也是池该干的事：一次性建表不需要 10 条连接）。
        self._ensure_schema()
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=pool_min,
            max_size=pool_max,
            open=False,
            configure=self._configure_connection,
            # 数据库不可达时不该让调用方挂到默认的 30s
            timeout=10.0,
            name="kubedocs-pgvector",
        )
        self._pool.open(wait=False)

    def _q(self, template: str, **extra: object) -> sql.Composed:
        """把 `{chunks}` / `{documents}` / `{meta}` 换成 **schema 限定的标识符**。

        表名不是用户输入（`schema` 在构造时已按 `_IDENT_RE` 校验过），但与其口头
        保证"我拼的字符串安全"，不如把引用规则交给驱动：`sql.Identifier` 正是
        psycopg 为这件事提供的 API，动态 SQL 也就不再是一个无法静态区分的模式。
        """
        return sql.SQL(template).format(**self._tbl, **extra)

    def _configure_connection(self, conn: Any) -> None:
        """池里每条新连接：设置查询期 HNSW 候选队列 + 注册 pgvector 适配器。

        `register_vector` 让 numpy 数组可以直接作为向量参数传入/取出
        （否则只能自己拼 `'[1,2,3]'::vector` 文本，读回来还要手写解析）。

        **结尾必须 `commit()`**：`SET` 与类型注册都会开一个隐式事务，而
        `psycopg_pool` 要求 `configure` 回调把连接留在 **IDLE** 状态。不提交的话
        每条新连接都会被判为
        `connection left in status INTRANS by configure function ...: discarded`
        并立刻丢弃——表现为池永远填不满、每次取连接都等到 `PoolTimeout`。
        这是实测踩到的第三个坑（前两个见 `__init__` 与 `_init_meta`）。
        """
        conn.execute(f"SET hnsw.ef_search = {int(self.hnsw_ef_search)}")
        self._register_vector(conn)
        conn.commit()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        """建扩展/schema/表/索引（幂等）。摄取与 finalize 都能自己把 schema 准备好。

        **用直连而不是连接池**：见 `__init__` 里的顺序说明——池的 `configure`
        会 `register_vector`，而那要求 `vector` 类型已存在；用池来建扩展就是死锁。
        建表是一次性动作，本来也不需要池。
        """
        try:
            with psycopg.connect(self.dsn, connect_timeout=10) as conn:
                try:
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except Exception as e:  # noqa: BLE001
                    raise StoreError(
                        f"无法启用 pgvector 扩展（{type(e).__name__}: {e}）；"
                        "需要数据库装了 pgvector 且当前角色有权限（compose 里的 "
                        "pgvector/pgvector:pg16 镜像自带该扩展）"
                    ) from e
                conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
                for stmt in _DDL_STATEMENTS:
                    conn.execute(
                        self._q(
                            stmt,
                            dim=sql.Literal(int(self.dim)),
                            m=sql.Literal(int(self.hnsw_m)),
                            efc=sql.Literal(int(self.hnsw_ef_construction)),
                        )
                    )
                conn.execute(
                    self._q("CREATE TABLE IF NOT EXISTS {meta} (key text PRIMARY KEY, value text NOT NULL)")
                )
                self._init_meta(conn)
        except StoreError:
            raise
        except Exception as e:  # noqa: BLE001
            raise StoreError(
                f"连接/初始化 Postgres 失败（{type(e).__name__}: {e}）；"
                f"检查 KUBEDOCS_DATABASE_URL 与数据库可达性（当前 {self.dsn!r}）"
            ) from e

    def _init_meta(self, conn: Any) -> None:
        """首次写入元信息；已存在则做与 SQLite 侧同样的闸门校验。

        `executemany` 在 `psycopg.Cursor` 上而**不在** `Connection` 上（实测
        psycopg 3.3.6：`Connection.executemany` 不存在），所以这里显式开游标——
        `Connection.execute` 只是"快捷开一个游标"的语法糖，没有对应的批量版本。
        """
        rows = dict(conn.execute(self._q("SELECT key, value FROM {meta}")).fetchall())
        if not rows:
            with conn.cursor() as cur:
                cur.executemany(
                    self._q("INSERT INTO {meta}(key, value) VALUES(%s, %s)"),
                    [
                        ("schema_version", str(PG_SCHEMA_VERSION)),
                        ("embedding_model", self.embedding_model),
                        ("embedding_dim", str(self.dim)),
                        ("normalizer", self.normalizer.name),
                        ("domain", self.domain),
                        ("indexed_at", ""),
                    ],
                )
            return
        got = int(rows.get("schema_version", "0"))
        if got != PG_SCHEMA_VERSION:
            raise StoreError(f"Postgres 索引 schema 版本不符：库里是 {got}，代码要求 {PG_SCHEMA_VERSION}")
        old_dim = int(rows.get("embedding_dim", "0"))
        if old_dim != self.dim:
            raise StoreError(
                f"索引向量维度不符：库里是 {old_dim}，当前配置 {self.dim}；"
                "换了嵌入模型必须重建索引（ingest --rebuild）"
            )
        old_model = rows.get("embedding_model", "")
        if old_model and self.embedding_model and old_model != self.embedding_model:
            raise StoreError(
                f"索引嵌入模型不符：库里是 {old_model!r}，当前配置 {self.embedding_model!r}；"
                "不同模型的向量空间不可混用，请重建索引"
            )
        old_norm = rows.get("normalizer", "")
        if old_norm and old_norm != self.normalizer.name:
            raise StoreError(
                f"索引归一化器不符：库里是 {old_norm!r}，当前配置 {self.normalizer.name!r}；"
                "请用 ingest --rebuild 重建索引"
            )

    def _set_meta(self, key: str, value: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                self._q(
                    "INSERT INTO {meta}(key, value) VALUES(%s, %s) "
                    "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value"
                ),
                (key, value),
            )

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ 读取
    def existing_doc_hashes(self) -> dict[str, str]:
        return {
            str(rel): str(h)
            for rel, h in self._pairs(self._q("SELECT rel_path, content_hash FROM {documents}"))
        }

    def existing_doc_ids(self) -> set[str]:
        return {cid for (cid,) in self._pairs(self._q("SELECT doc_id FROM {documents}"))}

    def _pairs(self, query: sql.Composed, params: tuple[object, ...] = ()) -> list[tuple[Any, ...]]:
        with self._pool.connection() as conn:
            return list(conn.execute(query, params).fetchall())

    def existing_hash_to_vector(self, hashes: set[str]) -> dict[str, Vector]:
        """按 content_hash 复用旧向量（跨篇复用同一模板块，不重复编码）。"""
        if not hashes:
            return {}
        rows = self._pairs(
            self._q(
                "SELECT DISTINCT ON (content_hash) content_hash, embedding "
                "FROM {chunks} WHERE content_hash = ANY(%s) ORDER BY content_hash"
            ),
            (list(hashes),),
        )
        return {str(h): self._as_vector(v) for h, v in rows}

    @staticmethod
    def _as_vector(value: object) -> Vector:
        """把驱动读回的 pgvector 值统一成 float32 的 `ndarray`。

        **不能直接 `np.asarray(value)`**：pgvector 0.5.0 的 `Vector` 不是 `ndarray`
        子类（实测 MRO 只有 `object`），`np.asarray` 会把它当标量，报
        `float() argument must be a string or a real number, not 'Vector'`。
        它自带 `to_numpy()`；对已经是 ndarray 的情况（没注册适配器时）也兼容。
        """
        to_numpy = getattr(value, "to_numpy", None)
        arr = to_numpy() if callable(to_numpy) else value
        return np.asarray(arr, dtype=np.float32)

    def doc_texts(self, doc_ids: list[str]) -> dict[str, str]:
        """拼接指定文档的全部块正文（小写化，供标注集 `must_contain` 自检）。"""
        if not doc_ids:
            return {}
        # 请求了但索引里没有的文档返回空串（与 SQLite 侧一致），而不是缺键
        out: dict[str, str] = dict.fromkeys(doc_ids, "")
        rows = self._pairs(
            self._q(
                "SELECT doc_id, string_agg(body, E'\\n' ORDER BY ordinal) "
                "FROM {chunks} WHERE doc_id = ANY(%s) GROUP BY doc_id"
            ),
            (doc_ids,),
        )
        for doc_id, text in rows:
            out[str(doc_id)] = str(text).lower()
        return out

    def stats(self) -> StoreStats:
        with self._pool.connection() as conn:
            n_docs = int(conn.execute(self._q("SELECT COUNT(*) FROM {documents}")).fetchone()[0])
            row = conn.execute(
                self._q("SELECT COUNT(*) AS n, COALESCE(SUM(n_chars), 0) AS chars FROM {chunks}")
            ).fetchone()
            n_code = int(
                conn.execute(
                    self._q("SELECT COUNT(*) FROM {chunks} WHERE body LIKE %s"), ("%```%",)
                ).fetchone()[0]
            )
            meta = dict(conn.execute(self._q("SELECT key, value FROM {meta}")).fetchall())
        return StoreStats(
            n_docs=n_docs,
            n_chunks=int(row[0]),
            total_chars=int(row[1]),
            n_with_code=n_code,
            embedding_model=str(meta.get("embedding_model", "")),
            embedding_dim=int(meta.get("embedding_dim", "0")),
            indexed_at=str(meta.get("indexed_at", "")),
            domain=str(meta.get("domain", "")),
            normalizer=str(meta.get("normalizer", "")),
        )

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkHit]:
        if not chunk_ids:
            return {}
        rows = self._pairs(
            self._q(
                "SELECT chunk_id, doc_id, ordinal, title, breadcrumb, source_url, body "
                "FROM {chunks} WHERE chunk_id = ANY(%s)"
            ),
            (chunk_ids,),
        )
        out: dict[str, ChunkHit] = {}
        for cid, doc_id, ordinal, title, breadcrumb, url, body in rows:
            out[str(cid)] = ChunkHit(
                chunk_id=str(cid),
                doc_id=str(doc_id),
                ordinal=int(ordinal),
                title=str(title),
                breadcrumb=str(breadcrumb),
                source_url=str(url),
                body=str(body),
            )
        return out

    def meta_dump(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self._pairs(self._q("SELECT key, value FROM {meta}"))}

    def doc_summary(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self._pairs(
            self._q("SELECT rel_path, n_chunks, updated_at FROM {documents} ORDER BY rel_path LIMIT %s"),
            (limit,),
        )
        return [{"rel_path": r[0], "n_chunks": r[1], "updated_at": r[2]} for r in rows]

    # ------------------------------------------------------------------ 检索
    def keyword_search(self, query: str, limit: int) -> list[tuple[str, int]]:
        """全文路：`tsv @@ to_tsquery` 走 GIN 索引，`ts_rank` 排序。

        与 FTS5 的差异见模块文档第 1~3 条（**排序函数不同源**是主要的一条：
        `ts_rank` 没有 IDF 与文档长度归一化）。
        并列分数按 chunk_id 定序：FTS5 的并列顺序由内部实现决定、不可依赖；
        这里给一个**确定性**次序，至少让"同样的输入给同样的输出"成立。
        """
        expr = build_tsquery_expr(self.normalizer.query_text(query))
        if not expr or limit <= 0:
            return []
        rows = self._pairs(
            self._q(
                "SELECT chunk_id FROM {chunks} "
                "WHERE tsv @@ to_tsquery('english', %s) "
                "ORDER BY ts_rank(%s::float4[], tsv, to_tsquery('english', %s)) DESC, "
                "chunk_id LIMIT %s"
            ),
            (expr, list(_TS_RANK_WEIGHTS), expr, limit),
        )
        return [(str(r[0]), i) for i, r in enumerate(rows, start=1)]

    def dense_search(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        """向量路：HNSW 索引 + 余弦距离升序（= 余弦相似度降序）。

        返回形态与另外两个实现一致：`list[(chunk_id, rank)]`，rank 从 1 起。
        """
        if limit <= 0:
            return []
        q = self._as_query_vector(query_vector)
        rows = self._pairs(
            self._q("SELECT chunk_id FROM {chunks} ORDER BY embedding <=> %s LIMIT %s"), (q, limit)
        )
        return [(str(r[0]), i) for i, r in enumerate(rows, start=1)]

    def dense_search_exact(self, query_vector: Vector, limit: int) -> list[tuple[str, int]]:
        """**不用索引**的精确检索：度量 HNSW 召回率的基准。

        没有它，"HNSW 够不够准"就只能靠感觉。它关掉索引扫描，强制顺序扫描全表
        算精确余弦序——1.9 万块约几十毫秒，作为离线度量完全可接受。
        """
        if limit <= 0:
            return []
        q = self._as_query_vector(query_vector)
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("SET LOCAL enable_indexscan = off")
            conn.execute("SET LOCAL enable_bitmapscan = off")
            rows = conn.execute(
                self._q("SELECT chunk_id FROM {chunks} ORDER BY embedding <=> %s LIMIT %s"), (q, limit)
            ).fetchall()
        return [(str(r[0]), i) for i, r in enumerate(rows, start=1)]

    def _as_query_vector(self, query_vector: Vector) -> np.ndarray:
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise StoreError(f"查询向量维度 {q.shape[0]} 与索引 {self.dim} 不符")
        return np.ascontiguousarray(q, dtype=np.float32)

    # ------------------------------------------------------------------ 写入
    def upsert_docs(self, docs: list[DocInput], chunks: list[ChunkInput], vectors: dict[str, Vector]) -> None:
        if not docs and not chunks:
            return
        missing = [c.chunk_id for c in chunks if c.chunk_id not in vectors]
        if missing:
            raise MissingVectorError(
                f"{len(missing)} 个块没有向量（例：{missing[:3]}）——"
                "插入无向量的块会让向量路静默漏召回，这里直接拒绝"
            )
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._pool.connection() as conn, conn.transaction():
            for d in docs:
                # 先删整篇（chunks 上有 ON DELETE CASCADE），再插——覆盖语义与 SQLite 侧一致
                conn.execute(self._q("DELETE FROM {documents} WHERE doc_id = %s"), (d.doc_id,))
                n_chunks = sum(1 for c in chunks if c.doc_id == d.doc_id)
                conn.execute(
                    self._q(
                        "INSERT INTO {documents}(doc_id, rel_path, title, source_url, "
                        "content_hash, n_chunks, updated_at) "
                        "VALUES(%s, %s, %s, %s, %s, %s, %s)"
                    ),
                    (d.doc_id, d.rel_path, d.title, d.source_url, d.content_hash, n_chunks, now),
                )
            with conn.cursor() as cur:
                cur.executemany(
                    self._q(
                        "INSERT INTO {chunks}(chunk_id, doc_id, ordinal, title, breadcrumb, "
                        "source_url, body, body_normalized, content_hash, n_chars, embedding) "
                        "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ),
                    [
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
                            np.ascontiguousarray(np.asarray(vectors[c.chunk_id], dtype=np.float32)),
                        )
                        for c in chunks
                    ],
                )
        self._dirty = True

    def delete_docs(self, rel_paths: set[str]) -> int:
        if not rel_paths:
            return 0
        with self._pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                self._q("SELECT doc_id, n_chunks FROM {documents} WHERE rel_path = ANY(%s)"),
                (list(rel_paths),),
            ).fetchall()
            if rows:
                conn.execute(
                    self._q("DELETE FROM {documents} WHERE doc_id = ANY(%s)"), ([r[0] for r in rows],)
                )
        self._dirty = True
        return int(sum(int(r[1]) for r in rows))

    def finalize(self) -> None:
        """收尾：盖时间戳；批量写入后跑一次 `ANALYZE` 让规划器知道新分布。

        与 SQLite 侧的关键差别：**向量随块行一起提交**（`embedding NOT NULL` 同一行），
        没有"待并入的矩阵"，也没有"块写了、向量没写"的中间态。
        """
        with self._pool.connection() as conn:
            conn.execute(
                self._q(
                    "INSERT INTO {meta}(key, value) VALUES('indexed_at', %s) "
                    "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value"
                ),
                (time.strftime("%Y-%m-%dT%H:%M:%S"),),
            )
            if self._dirty:
                # GIN / HNSW 索引的统计信息在批量写入后会过期，ANALYZE 让规划器选对计划
                conn.execute(self._q("ANALYZE {chunks}"))
        self._dirty = False

    # ------------------------------------------------------------------ 运维
    def rebuild_from_scratch(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(self._q("TRUNCATE TABLE {chunks}, {documents}"))
        self._set_meta("indexed_at", "")
        self._dirty = False

    def verify_fts(self) -> str:
        """全文索引自检。

        `tsv` 是生成列，**不可能**与正文漂移（这点比 FTS5 外内容表结实）；
        所以这里检查"索引本身还在不在"与"生成列有没有空值"——少了 GIN 索引时
        查询结果仍然正确、但会退化成顺序扫描，那是**静默的性能事故**。
        """
        with self._pool.connection() as conn:
            idx = conn.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = 'chunks' "
                "AND indexname = 'idx_chunks_tsv'",
                (self.schema,),
            ).fetchone()
            if idx is None:
                return "FAIL: 缺少 GIN 索引 idx_chunks_tsv（全文检索会退化为顺序扫描）"
            n_null = int(
                conn.execute(self._q("SELECT COUNT(*) FROM {chunks} WHERE tsv IS NULL")).fetchone()[0]
            )
        if n_null:
            return f"FAIL: {n_null} 行的 tsv 为空"
        return "ok"

    def verify_hnsw(self) -> str:
        """向量索引自检：HNSW 索引是否存在（缺了会静默退化为精确暴力扫描）。"""
        with self._pool.connection() as conn:
            idx = conn.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = 'chunks' "
                "AND indexname = 'idx_chunks_embedding'",
                (self.schema,),
            ).fetchone()
        return "ok" if idx is not None else "FAIL: 缺少 HNSW 索引 idx_chunks_embedding"


def _import_register_vector() -> Any:
    """延迟导入 `pgvector.psycopg.register_vector`（缺包时给出可执行的提示）。"""
    try:
        from pgvector.psycopg import register_vector
    except ImportError as e:  # pragma: no cover - 取决于环境
        raise StoreError("pgvector 后端需要 pgvector 的 Python 包（pip install pgvector）") from e
    return register_vector


def from_settings(settings: Settings) -> PgVectorStore:
    """按配置构造生产实现（schema 用域名：一个域一份索引，互不干扰）。"""
    return PgVectorStore(
        settings.database_url,
        schema=settings.domain,
        dim=settings.embedding_dim,
        embedding_model=settings.effective_embedding_model,
        normalizer=build_normalizer(settings.normalizer),
        domain=settings.domain,
        pool_min=settings.db_pool_min,
        pool_max=settings.db_pool_max,
        hnsw_m=settings.pg_hnsw_m,
        hnsw_ef_construction=settings.pg_hnsw_ef_construction,
        hnsw_ef_search=settings.pg_hnsw_ef_search,
    )
