"""生产存储实现：Postgres + pgvector（HNSW）。

相对本地 SQLite 实现补上的三件事：多实例并发写（事务与行锁）、HNSW 近似检索
（块数上百万时不必全表扫描）、一个域一个 **schema** 的多租户隔离；备份、PITR、
连接池与可观测性也直接沿用 Postgres。

## 全文检索：pg_search 的真 BM25

关键字路默认走 ParadeDB 的 `pg_search`：建 `USING paradedb`（tantivy）索引，
排序用 `pdb.score()`，它是带 IDF 与文档长度归一化的真 BM25。Postgres 核心自带的
`ts_rank` 只有 TF 加权（没有 IDF、没有文档长度归一化），**不是 BM25**，只作为降级
引擎，且降级会被明确标出（`meta.fts_engine`、`verify_fts()` 返回 `DEGRADED: ...`、
构造时一条 WARNING 日志）；`fts_engine="pg_search"` 要求真 BM25，扩展不可用则直接
抛 `StoreError`。

与默认后端 SQLite / sqlite-vec 的对照：

| | SQLite / sqlite-vec（默认后端） | pgvector（本实现） |
|---|---|---|
| 全文索引 | FTS5 外内容表 | `CREATE INDEX ... USING paradedb`（tantivy BM25） |
| 分词 | `tokenize='porter unicode61'` | `pdb.chinese_compatible('stemmer=english')` |
| 索引字段 | 三个列（body / title / breadcrumb）+ 列权重 | 一个字段：三列拼接 |
| 打分 | BM25（IDF + 长度归一化） | BM25（IDF + 长度归一化） |

- **打分实现不同**：SQLite 的 `bm25()` 与 tantivy 的 BM25 是两份独立实现
  （k1/b 默认值与 TF 饱和曲线未必一致），所以命中集合相同也不保证名次相同。
- **列权重不能照抄** FTS5 的 `bm25(chunks_fts, 1.0, 3.0, 1.5)`：FTS5 的权重乘在
  按整行长度归一化的 TF 上，tantivy 的 `pdb.boost` 乘在按字段自身长度归一化的
  BM25 上，而标题/面包屑只有几个词，"标题命中"会被放大成压倒性信号。本实现改成
  三列拼接成**单字段**：归一化域与 IDF 域各收敛成一个，是与 FTS5"整行归一化"最接近
  的对应物，同时保住"三个字段都搜得到"。要重新引入权重必须重测。
- **分词器**：英文按 Unicode 词边界切分 + Snowball 词干（对齐 FTS5 的 porter）；
  CJK 按**单字**切分，不做中文分词、不查词典。因此中文检索能力仍来自应用层归一化器
  （`normalizer=chinese` 的 jieba 预分词，索引与查询两侧都写成空格分隔的词）。没配
  归一化器时，中文查询会被切成单字并 OR（召回高、精度低），而不是一个词元都搜不到。
- **停用词**：pg_search 的 `simple` / `chinese_compatible` 不删停用词，查询侧的去停
  用词由 `store/textquery.py` 自己做，因此两侧索引里的词表并不相同。

`tsv` 生成列 + GIN 索引保留下来，作为降级引擎的索引，与 `USING paradedb` 索引并存、
互不干扰；代价是每行多存一份 tsvector。

## HNSW 参数

- `m = 16`：每个节点的出边数，pgvector 默认值。在约 1.9 万块这个规模下 8 会明显掉
  召回，32 以上收益递减而索引体积/建图时间线性上升。**改 m 只能重建索引。**
- `ef_construction = 64`：建图时的候选队列，默认值，与 m 一起决定图的质量。
- `ef_search = 200`（查询期 `hnsw.ef_search`）：pgvector 默认 40，取 200 是为了提高
  向量路与精确检索的吻合度，查询开销只小幅上升，继续加大收益递减。它**不需要重建
  索引**，可随时按需调（`Settings.pg_hnsw_ef_search`）；它只影响向量路，改善不了
  混合检索里来自全文路的质量落差。
- 距离用 `vector_cosine_ops` + `<=>`：与另外两个实现的余弦语义一致。
  向量已归一化时，余弦序与内积序严格相同。

## 父子两层

父子切割之后库里有两张事实表：`chunks`（子块，进 BM25 与 HNSW）与 `parents`（父块，
一个 Markdown 小节 / 一页 PDF / 一张幻灯片）。父块表**刻意不进任何索引**——它没有
`tsv`、没有 `embedding`，因此"父块混进检索结果"在 schema 层面就不可能发生，而不必
依赖每条查询记得加过滤条件。它只在子块命中后按 `parent_id` 被整段取回，作为交给模型
的语境（`ChunkHit.context_text`）。

`chunks.parent_id` 是 `NOT NULL DEFAULT ''`：空串 = 这份索引建在父子切割之前，读路径会
退回用子块正文当语境。所以这一列可以用 `ADD COLUMN IF NOT EXISTS` 无损升级老库，
**不需要 bump `PG_SCHEMA_VERSION`**（bump 会让老库在打开时被元信息闸门直接拒绝）。

## 连接

用 `psycopg` + **连接池**（`psycopg_pool`）：HTTP 层是多线程的，每次查询新建连接
在 100ms 量级，会把检索延迟整体抬一个数量级。池的每条连接在 `configure` 回调里
设置 `hnsw.ef_search` 并注册 pgvector 类型适配器。

**但 schema 初始化不能走池**：`configure` 里的 `register_vector` 要求 `vector` 类型
已存在，用池去 `CREATE EXTENSION` 会自己把自己锁死。所以顺序固定为「直连建 schema
→ 再开池」。

SQL 里的表名一律通过 `sql.Identifier` 注入（见 `_q`），值一律走 `%s` 参数绑定。
"""

from __future__ import annotations

import logging
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
    ParentInput,
    StoreError,
    StoreStats,
    Vector,
)
from kubedocs.store.textquery import build_tsquery_expr, query_tokens

# pgvector 侧自己的 schema 版本（与 SQLite 的 SCHEMA_VERSION 无关：两者是不同引擎）
PG_SCHEMA_VERSION = 1

# ts_rank 的权重数组顺序是 {D, C, B, A}，必须与建表时的 setweight 对应：
# body_normalized=D，breadcrumb=B，title=A（C 未使用，占位跟随 body）。
#
# ⚠️ `ts_rank` 的权重**必须落在 [0, 1] 内**，否则 Postgres 直接报
# `InvalidParameterValue: weight out of range`。所以这里不能照抄 FTS5
# `bm25(chunks_fts, 1.0, 3.0, 1.5)` 那组倍数，而要写一组**比例相同、最大值为 1** 的
# 权重：body 1/3、breadcrumb 1/2、title 1。ts_rank 对权重是线性的（整体乘一个常数
# 只等比缩放分数、不改次序），因此与"body : breadcrumb : title = 1 : 1.5 : 3"同序，
# 而 `ORDER BY ts_rank(...) DESC` 只依赖次序，故意图得以保留。
_TS_RANK_WEIGHTS = (1 / 3, 1 / 3, 1 / 2, 1.0)

# ------------------------------------------------------------------ 全文引擎
# `fts_engine` 的三个取值（构造参数，不是环境变量：它是"这个库能不能跑 BM25"的
# 事实判断，不该由部署环境随手改；`auto` 是默认，也是唯一会被 app 用到的值）。
_FTS_AUTO = "auto"
_FTS_PG_SEARCH = "pg_search"
_FTS_TS_RANK = "ts_rank"
_FTS_MODES = (_FTS_AUTO, _FTS_PG_SEARCH, _FTS_TS_RANK)

_PG_SEARCH_EXT = "pg_search"
# BM25 索引名。**一个表只能有一个 `USING paradedb` 索引**（扩展的硬约束），
# 所以这个名字同时也是"这个表有没有 BM25 索引"的唯一标识。
_BM25_INDEX = "idx_chunks_bm25"

# 三个字段共用的分词器：英文 Snowball 词干（对齐 FTS5 的 porter）+ CJK 单字切分。
# 具体行为见模块文档「全文检索」一节。
_BM25_TOKENIZER = "pdb.chinese_compatible('stemmer=english')"

# BM25 索引只建**一个**字段：`title + breadcrumb + body_normalized` 拼起来的文本。
# 索引里的字段名（`alias=`）只用于 EXPLAIN 可读性；查询侧必须用**同一个表达式**引用
# 它——PostgreSQL 的表达式索引匹配是句法的，两边要逐字一致，所以抽成常量。
# 注意括号：`::` 的优先级高于 `||`，`a || b::tok` 只会把 `b` 转型，因此这里把整个
# 拼接表达式连同外层括号一起定义，DDL 与查询两侧共用同一个字符串。
_BM25_TEXT_EXPR = "(title || ' ' || breadcrumb || ' ' || body_normalized)"
_BM25_FIELD_ALIAS = "fts_text"
# `alias=` 必须写在分词器**参数括号里面**，写到外面是语法错误。
_BM25_TOKENIZER_ALIASED = "pdb.chinese_compatible('stemmer=english', 'alias=" + _BM25_FIELD_ALIAS + "')"
_BM25_DDL = (
    "CREATE INDEX IF NOT EXISTS " + _BM25_INDEX + " ON {chunks} USING paradedb "
    "(({expr}::" + _BM25_TOKENIZER_ALIASED + "))"
)

# BM25 索引不照抄 FTS5 的列权重，而是三列拼接成单字段——理由见模块文档「全文检索」。

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TSV_EXPR = (
    "setweight(to_tsvector('english', coalesce(body_normalized, '')), 'D') || "
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(breadcrumb, '')), 'B')"
)

# 建表 DDL（语句逐条执行）。`embedding` 与 `tsv` 都在块行内，不存在"块在、向量不在"
# 的中间态，也不需要任何侧车文件（对比 SQLite 侧的 .npy）。
# `{documents}` / `{chunks}` / `{parents}` 是 sql.Identifier 占位符，
# `{dim}`/`{m}`/`{efc}` 是整数字面量。
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
    "  parent_id text NOT NULL DEFAULT '',"
    "  embedding vector({dim}) NOT NULL,"
    f"  tsv tsvector GENERATED ALWAYS AS ({_TSV_EXPR}) STORED)",
    # 父块表：**没有 tsv、没有 embedding、没有任何索引字段**——父块不参与检索，所以
    # "它会不会混进检索结果"在 schema 层面就不成立，而不是靠每条查询自觉过滤。
    #
    # 它**没有指向 documents 的外键**：父块与子块在写入路径上是一起清、一起写的，
    # 而"覆盖某篇"的删除是按 doc_id 显式点名的（见 upsert_docs / delete_docs）；
    # 加外键反而会引入"父块行必须先有 documents 行"这种隐性顺序约束。
    "CREATE TABLE IF NOT EXISTS {parents} ("
    "  parent_id text PRIMARY KEY,"
    "  doc_id text NOT NULL,"
    "  ordinal integer NOT NULL,"
    "  title text NOT NULL,"
    "  breadcrumb text NOT NULL,"
    "  source_url text NOT NULL,"
    "  body text NOT NULL,"
    "  content_hash text NOT NULL DEFAULT '',"
    "  n_chars integer NOT NULL,"
    "  location text NOT NULL DEFAULT '',"
    "  n_children integer NOT NULL DEFAULT 0)",
    # ⚠️ 老库（父子切割之前建的，CREATE TABLE 里没有这一列）必须能原地升级：
    # `ADD COLUMN IF NOT EXISTS` 在**新库与老库上都幂等**（新库里它只是 CREATE TABLE
    # 声明的重复 no-op）。**故意不 bump PG_SCHEMA_VERSION**：这一列是可加列、无损升级，
    # 而 bump 版本号会让老库在 `_init_meta` 的闸门处直接被拒绝，把"能平滑升级"变成
    # "必须丢库重建"。
    "ALTER TABLE {chunks} ADD COLUMN IF NOT EXISTS parent_id text NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON {chunks} (doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON {chunks} (content_hash)",
    # 读路径按 parent_id 批量取父块，这条索引支撑那一句 `WHERE parent_id = ANY(%s)`
    "CREATE INDEX IF NOT EXISTS idx_chunks_parent ON {chunks} (parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_parents_doc ON {parents} (doc_id)",
    # ⚠️ 这个 GIN 索引只服务**降级引擎**（ts_rank）：BM25 走下面那条
    # `USING paradedb` 索引，两者并存、互不干扰。
    "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON {chunks} USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON {chunks} "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {efc})",
)


# BM25 索引 DDL。**不在 `_DDL_STATEMENTS` 里**：它依赖 pg_search 扩展，而扩展可能
# 装不上（那时整条语句会以"unknown access method paradedb"失败，连表都建不出来），
# 所以单独一条、只在探测到扩展可用时执行。
#
# 分词器写成 `表达式::pdb.xxx(...)` 而不是索引选项：pg_search 0.26.1 的索引级
# `text_fields` 已被忽略，列级配置只剩表达式这一条路，而表达式字段**必须带
# `alias=`**，否则报 `indexed expression requires a tokenizer cast with an alias`。


def probe(dsn: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    """探测数据库是否可用（不做任何建表动作）。

    存在的理由：测试与前置检查需要区分"数据库连不上（环境问题，应 skip / 给出提示）"
    与"代码错了（应失败）"，两者混在一起会让门禁出现莫名其妙的红或绿。
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


def _pg_search_available(conn: Any) -> tuple[bool, str]:
    """探测这个库能不能用 pg_search（真 BM25），能用就顺手把扩展建出来。

    **判据不能只看 `pg_available_extensions`**：pg_search 是 preload 型扩展，扩展
    文件在、而 `shared_preload_libraries` 里没有它时，`CREATE EXTENSION` 会直接失败
    （`pg_search must be loaded via shared_preload_libraries`），所以真正的判据是
    "建一下试试"。

    `CREATE EXTENSION` 放在**嵌套事务（SAVEPOINT）**里：它失败时不该毒化外层事务，
    否则后面的建表语句会统统以 `current transaction is aborted` 失败，看起来像
    "表建不出来"，而不是真相"扩展装不上"。
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = %s", (_PG_SEARCH_EXT,)
        ).fetchone()
    except Exception as e:  # noqa: BLE001 - 探测要把任何失败都转成原因字符串
        return False, f"查 pg_available_extensions 失败（{type(e).__name__}: {e}）"
    if row is None:
        return False, f"这个 Postgres 实例里没有 {_PG_SEARCH_EXT} 扩展文件（镜像不带它）"
    try:
        with conn.transaction():  # 嵌套 → SAVEPOINT，失败只回滚这一段
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    except Exception as e:  # noqa: BLE001
        return False, (
            f"CREATE EXTENSION {_PG_SEARCH_EXT} 失败（{type(e).__name__}: {e}）；"
            "pg_search 是 preload 型扩展，需要 shared_preload_libraries=pg_search 并重启实例"
        )
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
        fts_engine: str = _FTS_AUTO,
    ) -> None:
        """`hnsw_ef_search` 的默认值必须与 `Settings.pg_hnsw_ef_search` **保持一致**：
        两处默认值不同就会出现"直接构造时是 100、走配置时是 200"的双真相，而这类差异
        只表现为"同一份数据换个入口召回就不一样"。

        `fts_engine` 决定关键字路用哪个引擎（**不是**环境变量，见常量区说明）：
        - `auto`（默认）：库里有 `pg_search` 就用真 BM25，没有就降级到 `ts_rank`，
          并在 meta / verify_fts / WARNING 日志三处标出来；app 走的就是这条。
        - `pg_search`：要求真 BM25，扩展不可用**直接报错**（生产闸门用得上）。
        - `ts_rank`：显式要旧引擎（对照旧数字用），同样带降级标记。
        """
        if not _IDENT_RE.match(schema):
            raise StoreError(f"非法的 schema 名 {schema!r}：只允许 [A-Za-z_][A-Za-z0-9_]*")
        if fts_engine not in _FTS_MODES:
            raise StoreError(f"非法的 fts_engine {fts_engine!r}：只允许 {_FTS_MODES}")
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
        self._fts_mode = fts_engine
        # 由 `_setup_fts()` 在 `_ensure_schema()` 里定下：
        # `_FTS_PG_SEARCH`（真 BM25）或 `_FTS_TS_RANK`（降级），以及降级的原因。
        self._fts_engine = _FTS_TS_RANK
        self._fts_reason = ""
        self._dirty = False
        self._tbl = {
            name: sql.Identifier(schema, name) for name in ("documents", "chunks", "parents", "meta")
        }

        self._register_vector = _import_register_vector()
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover - 取决于环境
            raise StoreError("pgvector 后端需要 psycopg-pool：pip install 'psycopg[pool]'") from e
        # ⚠️ 顺序是**有意义**的：schema 初始化必须在连接池建立**之前**跑完。池的
        # `configure` 回调会对每条新连接执行 `register_vector`，而那要求库里**已经
        # 存在** `vector` 类型；池一旦 open 就会立刻开始建连接，于是"建扩展"被排在
        # "注册类型"之后，而注册类型又只能靠池的连接去执行——死锁（表现为
        # PoolTimeout，且 `CREATE EXTENSION` 根本没被执行到）。所以 schema 这一步走
        # 直连，池只服务查询/写入流量。
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

        表名不是用户输入（`schema` 在构造时已按 `_IDENT_RE` 校验过），但与其口头保证
        "我拼的字符串安全"，不如把引用规则交给驱动：`sql.Identifier` 正是 psycopg 为
        这件事提供的 API，动态 SQL 也就不再是无法静态区分的模式。
        """
        return sql.SQL(template).format(**self._tbl, **extra)

    def _configure_connection(self, conn: Any) -> None:
        """池里每条新连接：设置查询期 HNSW 候选队列 + 注册 pgvector 适配器。

        `register_vector` 让 numpy 数组可以直接作为向量参数传入/取出（否则只能自己
        拼 `'[1,2,3]'::vector` 文本，读回来还要手写解析）。

        **结尾必须 `commit()`**：`SET` 与类型注册都会开一个隐式事务，而 `psycopg_pool`
        要求 `configure` 回调把连接留在 **IDLE** 状态；不提交的连接会被判为
        `connection left in status INTRANS by configure function ...: discarded` 并
        立刻丢弃，表现为池永远填不满、每次取连接都等到 `PoolTimeout`。
        """
        conn.execute(f"SET hnsw.ef_search = {int(self.hnsw_ef_search)}")
        self._register_vector(conn)
        conn.commit()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        """建扩展/schema/表/索引（幂等）。摄取与 finalize 都能自己把 schema 准备好。

        **用直连而不是连接池**：池的 `configure` 会 `register_vector`，而那要求
        `vector` 类型已存在；用池来建扩展就是死锁。建表是一次性动作，也不需要池。
        """
        try:
            with psycopg.connect(self.dsn, connect_timeout=10) as conn:
                try:
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except Exception as e:  # noqa: BLE001
                    raise StoreError(
                        f"无法启用 pgvector 扩展（{type(e).__name__}: {e}）；"
                        "需要数据库装了 pgvector 且当前角色有权限（docker-compose.yml 里的 "
                        "paradedb/paradedb:0.26.1-pg16 镜像自带 pgvector + pg_search）"
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
                # 全文引擎要在表建好之后才定（BM25 索引依赖 chunks 的列），
                # 也必须在这里写进 meta —— 那时连接池还没开（`_set_meta` 用池）。
                self._setup_fts(conn)
                conn.execute(
                    self._q(
                        "INSERT INTO {meta}(key, value) VALUES(%s, %s) "
                        "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value"
                    ),
                    ("fts_engine", self.fts_engine_label),
                )
        except StoreError:
            raise
        except Exception as e:  # noqa: BLE001
            raise StoreError(
                f"连接/初始化 Postgres 失败（{type(e).__name__}: {e}）；"
                f"检查 KUBEDOCS_DATABASE_URL 与数据库可达性（当前 {self.dsn!r}）"
            ) from e

    # ------------------------------------------------------------------ 全文引擎
    @property
    def fts_engine(self) -> str:
        """解析后的全文引擎：`"pg_search"`（真 BM25）或 `"ts_rank"`（降级）。"""
        return self._fts_engine

    @property
    def fts_engine_label(self) -> str:
        """写进 `meta.fts_engine` 的自述字符串——**降级必须在这里被看见**。"""
        if self._fts_engine == _FTS_PG_SEARCH:
            return f"{_FTS_PG_SEARCH}-bm25 (paradedb/pg_search, USING paradedb + pdb.score)"
        return f"{_FTS_TS_RANK}-DEGRADED (无 IDF/长度归一化，不是 BM25；原因：{self._fts_reason})"

    def _setup_fts(self, conn: Any) -> None:
        """定下全文引擎；能上 pg_search 就建 BM25 索引，不能就**标记**降级。

        `auto` 会降级但不隐瞒（meta / verify_fts / WARNING 日志三处都写着），
        `pg_search` 则宁可报错也不降级。
        """
        if self._fts_mode == _FTS_TS_RANK:
            self._fts_engine = _FTS_TS_RANK
            self._fts_reason = "调用方显式指定 fts_engine='ts_rank'"
        else:
            available, why = _pg_search_available(conn)
            if available:
                try:
                    conn.execute(self._q(_BM25_DDL, expr=sql.SQL(_BM25_TEXT_EXPR)))
                except Exception as e:  # noqa: BLE001
                    raise StoreError(
                        f"pg_search 可用，但建 BM25 索引失败（{type(e).__name__}: {e}）；"
                        "索引建不出来就不该继续——没有索引的 `|||` 查询会退化成顺序扫描，"
                        "而那是静默的性能事故"
                    ) from e
                self._fts_engine = _FTS_PG_SEARCH
                self._fts_reason = ""
            else:
                if self._fts_mode == _FTS_PG_SEARCH:
                    raise StoreError(
                        f"fts_engine='pg_search' 要求真 BM25，但扩展不可用：{why}；"
                        "请换用带 pg_search 的镜像（docker-compose.yml 已钉 "
                        "paradedb/paradedb:0.26.1-pg16），或改用 fts_engine='auto'/'ts_rank' "
                        "接受带标记的降级"
                    )
                self._fts_engine = _FTS_TS_RANK
                self._fts_reason = why
        if self._fts_engine == _FTS_TS_RANK:
            logger.warning(
                "全文路降级为 ts_rank（%s）：ts_rank 只有 TF 加权、没有 IDF 与文档长度归一化，"
                "**不是 BM25**，检索质量会低于冻结基线（实测 hybrid hit@10 0.8400 → 0.7400）。"
                "要真 BM25 请换用带 pg_search 的 Postgres（docker-compose.yml 已钉 "
                "paradedb/paradedb:0.26.1-pg16）。",
                self._fts_reason,
            )

    def _init_meta(self, conn: Any) -> None:
        """首次写入元信息；已存在则做与 SQLite 侧同样的闸门校验。

        `executemany` 在 `psycopg.Cursor` 上而**不在** `Connection` 上，所以这里显式
        开游标——`Connection.execute` 只是"快捷开一个游标"的语法糖，没有批量版本。
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
            # ⚠️ 维度是**列类型**，只能靠丢掉整个 schema 来改（索引是派生物，重建
            # 即可）：`ingest --rebuild` 走 `rebuild_from_scratch()`，而它只 `TRUNCATE`
            # 表，`embedding vector(384)` 的列定义不会因此改变，再跑一次还是同一个错。
            raise StoreError(
                f"索引向量维度不符：库里是 {old_dim}，当前配置 {self.dim}。\n"
                f"  维度是 vector 列的**类型**，`ingest --rebuild` 只 TRUNCATE、改不了它。\n"
                f"  丢掉这个 schema 再重建（索引是派生物，重建即可）：\n"
                f"    Postgres: DROP SCHEMA IF EXISTS {self.schema} CASCADE;\n"
                f"    然后重跑 ingest（sqlite 后端删掉 <index-dir>/{self.schema} 即可）"
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
        子类，`np.asarray` 会把它当标量，报
        `float() argument must be a string or a real number, not 'Vector'`。
        它自带 `to_numpy()`；值已是 ndarray 时（没注册适配器）也兼容。
        """
        to_numpy = getattr(value, "to_numpy", None)
        arr = to_numpy() if callable(to_numpy) else value
        return np.asarray(arr, dtype=np.float32)

    def doc_texts(self, doc_ids: list[str]) -> dict[str, str]:
        """拼接指定文档的全部**子块**正文（小写化，供标注集 `must_contain` 自检）。

        **父块刻意不拼进来**：这个函数的用途是回答"标注集要求的字面量确实进了被检索的
        文本吗"，而父块不进索引、检索时根本看不见它。把父块拼进来只会让一个"子块没写进
        索引"的真问题被父块里的同一句话掩盖（`must_contain` 过了，检索却永远命不中）。
        """
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
            # 父块数与块数分别读：父子切割之后它们同量级但**不是**同一个数，
            # "有几段语境"和"有几个检索单位"是两件要分开看见的事。
            n_parents = int(conn.execute(self._q("SELECT COUNT(*) FROM {parents}")).fetchone()[0])
            meta = dict(conn.execute(self._q("SELECT key, value FROM {meta}")).fetchall())
        return StoreStats(
            n_docs=n_docs,
            n_chunks=int(row[0]),
            total_chars=int(row[1]),
            n_with_code=n_code,
            n_parents=n_parents,
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
                "SELECT chunk_id, doc_id, ordinal, title, breadcrumb, source_url, body, parent_id "
                "FROM {chunks} WHERE chunk_id = ANY(%s)"
            ),
            (chunk_ids,),
        )
        # 父块**批量**取回（一句 `WHERE parent_id = ANY(%s)`）：命中列表动辄几十条，
        # 在循环里逐条查父块会把"取回上下文"变成 N+1 次往返，而这一步在检索延迟的
        # 关键路径上。缺失的 parent_id（索引建在父子切割之前、或调用方 parents=None）
        # 直接留在 `parent_text = ""`，由 `context_text` 退回子块正文。
        parent_ids = sorted({str(r[7]) for r in rows if r[7]})
        parents: dict[str, str] = {}
        if parent_ids:
            parents = {
                str(pid): str(body)
                for pid, body in self._pairs(
                    self._q("SELECT parent_id, body FROM {parents} WHERE parent_id = ANY(%s)"),
                    (parent_ids,),
                )
            }
        out: dict[str, ChunkHit] = {}
        for cid, doc_id, ordinal, title, breadcrumb, url, body, parent_id in rows:
            pid = str(parent_id or "")
            out[str(cid)] = ChunkHit(
                chunk_id=str(cid),
                doc_id=str(doc_id),
                ordinal=int(ordinal),
                title=str(title),
                breadcrumb=str(breadcrumb),
                source_url=str(url),
                body=str(body),
                parent_id=pid,
                parent_text=parents.get(pid, ""),
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
        """全文路：`pg_search` 的 BM25 索引 + `pdb.score()` 排序（降级时是 `ts_rank`）。

        两条路共用同一套查询词元（`normalizer.query_text` + `textquery.query_tokens`）：
        换引擎不改查询语义。

        并列分数按 chunk_id 定序，给一个**确定性**次序（FTS5 的并列顺序由内部实现
        决定、不可依赖）。代价是 `ORDER BY score DESC, chunk_id` 不能被 pg_search
        下推成 Top-K 扫描，计划为 `Limit → Sort → ParadeDB Base Scan`，p50 仍在
        10ms 量级。
        """
        if limit <= 0:
            return []
        text = self.normalizer.query_text(query)
        if self._fts_engine == _FTS_PG_SEARCH:
            return self._bm25_search(text, limit)
        return self._ts_rank_search(text, limit)

    def _bm25_search(self, normalized_query: str, limit: int) -> list[tuple[str, int]]:
        """真 BM25：`USING paradedb` 索引 + `pdb.score()`。

        索引只有**一个字段**（`title + breadcrumb + body_normalized` 的拼接，见
        `_BM25_TEXT_EXPR`）；为什么不是三个字段各带权重，见模块文档「全文检索」。

        `pdb.match_disjunction(...)` 是**显式的析取（OR）**，与 FTS5 侧 `"a" OR "b"`
        的召回语义一致；查询串不经过 pg_search 的查询语法解析，所以 `textquery.py` 里
        "只抽 `\\w+`、按 OR 连接"的那套注入面分析在这里依然成立（而且更强）。

        ⚠️ 两条维护约束：不要用 `field ||| %s::pdb.boost(3.0)`——psycopg 到点改用
        服务端 prepare 后，`$1` 会被推断成 `pdb.boost`，而 `|||` 没有这个重载（报
        "The right-hand side of the ||| operator must be a text or text array value"）；
        `pdb.match_disjunction(%s)` 的参数类型由函数签名（`text`）钉死。表达式字段还
        必须用同一个表达式在 WHERE 里引用（PostgreSQL 表达式索引的句法匹配），不能拿
        `alias=` 当列名用（`column "fts_text" does not exist`）。

        ⚠️ 没命中任何词元的查询（空串、纯标点）返回 0 行而不是报错——与 FTS5 一致。
        """
        tokens = query_tokens(normalized_query)
        if not tokens:
            return []
        rows = self._pairs(
            self._q(
                "SELECT chunk_id FROM {chunks} "
                "WHERE {expr} @@@ pdb.match_disjunction(%s) "
                "ORDER BY pdb.score(chunk_id) DESC, chunk_id LIMIT %s",
                expr=sql.SQL(_BM25_TEXT_EXPR),
            ),
            (" ".join(tokens), limit),
        )
        return [(str(r[0]), i) for i, r in enumerate(rows, start=1)]

    def _ts_rank_search(self, normalized_query: str, limit: int) -> list[tuple[str, int]]:
        """降级引擎：`tsv @@ to_tsquery` 走 GIN，`ts_rank` 排序（**没有 IDF**）。

        只在 `_setup_fts()` 判定了扩展不可用（或调用方显式选了它）时才会走到，
        且 `meta.fts_engine` / `verify_fts()` 都会把它标成 DEGRADED。
        """
        expr = build_tsquery_expr(normalized_query)
        if not expr:
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

        它关掉索引扫描、强制顺序扫描全表算精确余弦序——1.9 万块约几十毫秒，作为离线
        度量完全可接受。
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
    def upsert_docs(
        self,
        docs: list[DocInput],
        chunks: list[ChunkInput],
        vectors: dict[str, Vector],
        parents: list[ParentInput] | None = None,
    ) -> None:
        """写入/覆盖文档、**父块**与**子块**（父块可选）。

        `parents=None`（旧调用方、只看检索层的测试）走的是"只写子块"的老路径：子块的
        `parent_id` 仍照写，但库里没有对应父块行，读路径因此退回子块正文当语境——
        这与"这份索引建在父子切割之前"是同一种状态，不需要额外分支。

        父块与子块在**同一个事务**里清、同一个事务里写：任何"父块写了、子块没写"的
        中间态都会在崩溃后留下指向不存在父块的 `parent_id`，比没有父块更难排查。
        """
        parent_rows = list(parents or ())
        if not docs and not chunks and not parent_rows:
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
                # 父块表没有外键，级联删不到它，只能在这里显式点名同一个 doc_id；
                # 一次文档一句 DELETE，不随块数增长，批量写法不受影响。
                conn.execute(self._q("DELETE FROM {parents} WHERE doc_id = %s"), (d.doc_id,))
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
                        "source_url, body, body_normalized, content_hash, n_chars, parent_id, "
                        "embedding) "
                        "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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
                            c.parent_id,
                            np.ascontiguousarray(np.asarray(vectors[c.chunk_id], dtype=np.float32)),
                        )
                        for c in chunks
                    ],
                )
                if parent_rows:
                    # 父块同样走 executemany：条数虽少于子块，但"每块一句 SQL"在
                    # 上万父块的批次里同样是致命的往返开销。
                    cur.executemany(
                        self._q(
                            "INSERT INTO {parents}(parent_id, doc_id, ordinal, title, breadcrumb, "
                            "source_url, body, content_hash, n_chars, location, n_children) "
                            "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        ),
                        [
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
                            )
                            for p in parent_rows
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
                doc_ids = [r[0] for r in rows]
                # 父块没有外键，删除同样要显式点名 doc_id（返回语义仍是"删掉的块数"）
                conn.execute(self._q("DELETE FROM {parents} WHERE doc_id = ANY(%s)"), (doc_ids,))
                conn.execute(self._q("DELETE FROM {documents} WHERE doc_id = ANY(%s)"), (doc_ids,))
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
        # 父块必须一起 TRUNCATE：它是与子块同一批写入的事实表，漏掉它会让
        # "重建后的索引"里残留一堆孤儿父块（n_parents 虚高，且串味的旧语境仍可取回）。
        with self._pool.connection() as conn:
            conn.execute(self._q("TRUNCATE TABLE {chunks}, {parents}, {documents}"))
        self._set_meta("indexed_at", "")
        self._dirty = False

    def verify_fts(self) -> str:
        """全文索引自检：**先报引擎，再报索引**。

        返回三态（`store/base.py` 的契约只写了 ok/FAIL，这里多一个 DEGRADED）：
        - `"ok"`：BM25 引擎在位、索引在位；
        - `"DEGRADED: ..."`：跑得动，但引擎是 `ts_rank`（无 pg_search），检索质量
          **低于**基线。它不是"索引坏了"，而是"这条路的排序函数不是 BM25"；
        - `"FAIL: ..."`：索引真的缺失/漂移（缺索引时结果仍正确、但会退化成顺序扫描，
          那是**静默的性能事故**）。

        DEGRADED 不并进 `"ok"`：`cli stats --verify` 把非 `"ok"` 视为不满足前置条件，
        降级的 pgvector 后端就该在那一步被拦住。
        """
        with self._pool.connection() as conn:
            if self._fts_engine == _FTS_PG_SEARCH:
                idx = conn.execute(
                    "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = 'chunks' "
                    "AND indexname = %s",
                    (self.schema, _BM25_INDEX),
                ).fetchone()
                if idx is None:
                    return f"FAIL: 缺少 BM25 索引 {_BM25_INDEX}（关键字路会退化为顺序扫描）"
                return "ok"
            # 降级引擎：GIN 索引与生成列照旧自检，但结论**先**标成 DEGRADED
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
        return (
            f"DEGRADED: 全文引擎是 ts_rank（{self._fts_reason}）；"
            "ts_rank 无 IDF/长度归一化、不是 BM25，hybrid hit@10 会低于冻结基线。"
            "要 BM25 请换用带 pg_search 的 Postgres（docker-compose.yml 已钉 "
            "paradedb/paradedb:0.26.1-pg16）。索引本身 ok"
        )

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
