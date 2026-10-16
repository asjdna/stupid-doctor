"""就诊记录（患者 / 就诊 / 消息）的持久化：**真的落进 Postgres 的 `clinical` schema**。

`PgPersistence` 是默认实现，`NoopPersistence` 只留给"显式不落盘"的场景：测试、临时演示，
以及**数据库不可达时上层选择的降级**（`api/consult.py` 的取舍：宁可这轮不留档，也不能让
患者拿不到答复）。存储层自己**不吞异常**：失败如实抛给调用方，由持有产品取舍的那一层决定
（`ClinicalPipeline` 的选择是"记一条 warning 后继续"——见该模块的说明）。

## 为什么单独一个 schema，而不是塞进索引库

就诊记录与向量索引是两种性质的数据：

- **索引是派生物**：语料在、脚本在，随时可以重建，丢了只是重跑一次 `kubedocs ingest`；
  同一个 schema 里还混着 FTS 表、向量表与 chunk 元数据，`ingest` 会按域整批改写它们。
- **就诊记录不是派生物**：它是原始数据，重建不出来。丢了就是真的丢了。

所以它落在**自己的 schema**（默认 `clinical`，与索引 schema 即域名分开）。放在同一个
Postgres 实例里只是**部署上就近**（备份、PITR、监控一套机制即可），读写路径与生命周期
完全独立：`ingest` 只写域名 schema，不会碰 `clinical`。

## 多机构：`institution` 是**保留列**，不是租户隔离

本次不做多机构（行级安全 / 连接级 schema / 分库）。`institution` 现在只是把"归属哪个
机构"这个唯一确定得了的形状先占住：读写都只用默认空串，将来加隔离时不必改表结构与
所有调用点的签名。约定：读路径传**空串 = 不过滤**（当前全部数据都是空机构），
传非空值时按精确匹配过滤——所以拿一个真实机构名去查，会**查不到**任何东西。

## 线程安全：连接池，没有锁

HTTP 层在线程池里跑轮次（`/consult` 是同步端点），所以存储必须能被多线程并发使用。
这里用 `psycopg_pool.ConnectionPool`（与 `store/pgvector_store.py` 同一套做法）：
每条连接只属于**当时借到它的那个线程**，用完还池，因此不存在"多个线程共用一个连接对象"
这件事，也就**不需要 `RLock` 去串行化读写**。

SQLite 版的那把锁是**单连接的产物**（一个连接对象不允许并发使用；隐式事务还是连接级的，
一边读一边写会报 "cannot start a transaction within a transaction"）。换成 Postgres 后
这两条约束都不在了：事务是每连接独立的，并发写的正确性由 MVCC + 主键/行锁保证——
锁在这里只会把并发能力重新丢掉，所以删掉。

## 失败与探测

构造时（`ensure_schema()`，幂等）走**直连**建 schema/表/索引，之后再开池：建 schema 是
一次性动作，且它失败时必须让构造直接抛错（上层据此降级为 `NoopPersistence`），
不能让一个半开的池留在那里。失败统一抛 `PersistenceError`，消息里带上脱敏后的 DSN 与
排查方向（`KUBEDOCS_DATABASE_URL` / 数据库可达性）。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import quote, urlsplit, urlunsplit

import psycopg
from psycopg import sql

from kubedocs.clinical.contracts import ClinicalAssessment

logger = logging.getLogger(__name__)

DEFAULT_ENCOUNTER_LIMIT = 20
"""`list_encounters` 的默认条数（读路径是给人看的，不需要一次拉几百条）。"""

DEFAULT_SCHEMA = "clinical"
"""就诊记录所在的 schema。与索引 schema（域名）分开：一个是原始数据，一个是派生物。"""

DEFAULT_BOOKING_LIMIT = 20
"""`list_bookings` 的默认条数（与就诊记录同理，读路径是给人看的）。"""

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 建表 DDL（语句逐条执行，顺序有意义：encounters 的 FK 指向 patients）。
# 表名通过 `{patients}` 这类占位符注入 `sql.Identifier`（见 `_q`），值一律走 `%s` 绑定。
#
# 注意两处刻意的取舍：
#   · 两个 JSON 列**不写 DEFAULT '{}'**：本模板走 `str.format`，字面量花括号要写成
#     `{{}}`，维护时极易改错；空值由代码显式传参写入（见 `upsert_patient` /
#     `save_encounter`），`_assessment_or_none` 也把空串与 `{}` 一视同仁。
#   · `encounters.seq` 是**内部列**，不在对外契约里：`list_encounters` 需要"同一
#     `created_at` 时按插入顺序倒序"的稳定兜底，而 Postgres 没有 SQLite 的 `rowid`
#     （`ctid` 会因 UPDATE 变动、不能当稳定次序）。`bigserial` 只在插入时取值，
#     `ON CONFLICT DO UPDATE` 不会改它，正好等价于 `rowid`。
_DDL_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS {patients} ("
    "  patient_id text PRIMARY KEY,"
    "  institution text NOT NULL DEFAULT '',"
    "  display_name text NOT NULL DEFAULT '',"
    "  attributes_json text NOT NULL,"
    "  created_at text NOT NULL DEFAULT '',"
    "  updated_at text NOT NULL DEFAULT '')",
    "CREATE TABLE IF NOT EXISTS {encounters} ("
    "  encounter_id text PRIMARY KEY,"
    "  patient_id text NOT NULL REFERENCES {patients}(patient_id),"
    "  institution text NOT NULL DEFAULT '',"
    "  chief_complaint text NOT NULL DEFAULT '',"
    "  assessment_json text NOT NULL,"
    "  created_at text NOT NULL DEFAULT '',"
    "  seq bigserial)",
    # 刻意**不加**指向 encounters 的外键：消息是 append-only 的增量写入，管线里
    # `append_message` 先于 `save_encounter`（顺序理由见 save_encounter 的说明），
    # 加了外键就会把这条正常路径变成"先有鸡还是先有蛋"。
    # 患者上的外键仍然有东西可查：就诊不许挂在不存在的患者上。
    "CREATE TABLE IF NOT EXISTS {messages} ("
    "  id bigserial PRIMARY KEY,"
    "  encounter_id text NOT NULL,"
    "  role text NOT NULL,"
    "  text text NOT NULL DEFAULT '',"
    "  institution text NOT NULL DEFAULT '',"
    "  created_at text NOT NULL DEFAULT '')",
    # 挂号与就诊的**单向**关联：就诊记录知道"这一轮挂出了哪张号"，号源那边完全不知道
    # 就诊记录的存在。所以这张表是**投影**，不是号源的第二份真相——权威状态在
    # `booking.appointments`，这里只留一份给问诊界面读的快照。
    #
    # 不存 `patient_id`：它由 `encounter_id → encounters.patient_id` 唯一确定，
    # 存两份迟早会在某一处被改而另一处没改。也不加指向 `encounters` 的外键——
    # 与 `messages` 同理，挂号可能先于就诊行落库（见 `save_encounter` 的顺序说明），
    # 加外键会把正常路径变成"先有鸡还是先有蛋"。
    "CREATE TABLE IF NOT EXISTS {bookings} ("
    "  ticket_no text PRIMARY KEY,"
    "  encounter_id text NOT NULL DEFAULT '',"
    "  department text NOT NULL DEFAULT '',"
    "  booked_date text NOT NULL DEFAULT '',"
    "  time_slot text NOT NULL DEFAULT '',"
    "  fee integer NOT NULL DEFAULT 0,"
    "  status text NOT NULL DEFAULT '',"
    "  created_at text NOT NULL DEFAULT '',"
    "  updated_at text NOT NULL DEFAULT '')",
    # 读路径就是"某患者最近若干次就诊"：按 (patient_id, created_at) 建索引才不必全表扫。
    "CREATE INDEX IF NOT EXISTS idx_encounters_patient ON {encounters} (patient_id, created_at)",
    # 消息按就诊取回，且要保持写入顺序（bigserial id 就是顺序）。
    "CREATE INDEX IF NOT EXISTS idx_messages_encounter ON {messages} (encounter_id, id)",
    # 挂号按就诊回看（"这次问诊挂出了哪张号"），新到旧。
    "CREATE INDEX IF NOT EXISTS idx_bookings_encounter ON {bookings} (encounter_id, created_at)",
)


class PersistenceError(RuntimeError):
    """就诊记录库不可用（连不上 / 建不出 schema / 缺驱动）。

    单独一个类型而不是直接抛 `psycopg.Error`：上层要区分的是"记录库这条路不可用"
    （据此降级为 `NoopPersistence`）与真正的代码错误，而不是驱动层的具体异常谱系。
    """


def _now() -> str:
    """统一的时间戳：UTC + 秒精度 ISO 字符串。

    用字符串而不是 `timestamptz`：与既有契约（`EncounterRecord.created_at: str`）一致，
    排查时人能直接读，且同一格式下**字典序 == 时间序**，`ORDER BY created_at DESC`
    不必在 SQL 里做转换。时间来源也只有一个：写路径，不在读路径二次推断。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact_dsn(dsn: str) -> str:
    """脱敏 DSN：`/consult/status` 要说清"记录落在哪个库"，但**不能回显口令**。

    尽力而为：URL 形式走 `urlsplit` 重建 netloc；`key=value` 的 conninfo 走正则。
    解析不出来时返回一句固定文案，而不是把原串（可能带口令）漏出去。
    """
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "<无法解析的 DSN>"
    if not parts.scheme:
        return re.sub(r"(?i)(password\s*=\s*)\S+", r"\1***", dsn)
    userinfo = f"{quote(parts.username)}:***@" if parts.username is not None else ""
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:  # 端口不是数字：仍然要把口令挡掉
        port = ""
    netloc = f"{userinfo}{parts.hostname or ''}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _json_dict(raw: str) -> dict[str, str]:
    """把 `attributes_json` 读回字典。坏数据不让读路径崩，返回空字典并留一条日志。"""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        logger.warning("患者属性 JSON 不可解析，按空属性处理：%s", e)
        return {}
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def _assessment_or_none(raw: str) -> ClinicalAssessment | None:
    """把 `assessment_json` 读回契约对象。

    **单条坏数据不该让整个读路径失败**：解析不了就返回 `None` 并记日志，
    其余就诊记录照常返回（历史里可能存着旧版本契约写下的 JSON）。
    """
    if not raw or raw.strip() in ("", "{}"):
        return None
    try:
        return ClinicalAssessment.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001 - 库里可能有旧版本契约写下的行
        logger.warning("就诊记录里的结论 JSON 不可解析，该字段按缺失处理：%s: %s", type(e).__name__, e)
        return None


@dataclass
class PatientRecord:
    """患者主体。`institution` 是机构标识——**当前只是保留列**（见模块文档）。"""

    patient_id: str
    institution: str = ""
    display_name: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class EncounterRecord:
    """一次就诊（一轮问诊）。

    `transcript` 是"这一轮用到的对话上文 + 本轮问答"的快照，`created_at` 是落库时间
    （读取时由存储填上；写入时可以显式给，便于测试与补录历史）。
    """

    encounter_id: str
    patient_id: str
    institution: str = ""
    chief_complaint: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)
    assessment: ClinicalAssessment | None = None
    created_at: str = ""


def _booking_row(row: tuple[Any, ...]) -> BookingLink:
    """库行 → `BookingLink`（列序与 `list_bookings` 的 SELECT 一一对应）。"""
    return BookingLink(
        ticket_no=str(row[0]),
        encounter_id=str(row[1]),
        department=str(row[2]),
        date=str(row[3]),
        time_slot=str(row[4]),
        fee=int(row[5]),
        status=str(row[6]),
        created_at=str(row[7]),
    )


@dataclass
class BookingLink:
    """ "这次就诊挂出了哪张号"的关联行（**单向**：就诊 → 号，号源那边不知情）。

    它是号源状态的**投影**，不是权威：权威在 `booking.appointments`。这里的 `status`
    只在"从问诊界面退号"时同步更新，用来让回看路径不必跨 schema 联表。
    """

    ticket_no: str
    encounter_id: str = ""
    department: str = ""
    date: str = ""
    time_slot: str = ""
    fee: int = 0
    status: str = ""
    created_at: str = ""


class ClinicalStore(Protocol):
    """产品层需要的持久化能力。实现是 `PgPersistence`（默认）或 `NoopPersistence`。"""

    def upsert_patient(self, record: PatientRecord) -> None: ...

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None: ...

    def save_encounter(self, record: EncounterRecord) -> None: ...

    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None: ...

    def list_encounters(
        self, patient_id: str, *, institution: str = "", limit: int = DEFAULT_ENCOUNTER_LIMIT
    ) -> list[EncounterRecord]:
        """读回某患者最近若干次就诊（新 → 旧），用于"记录真的存住了"的验证与回看。"""
        ...

    def add_booking(self, record: BookingLink) -> None:
        """记下"这次就诊挂出了哪张号"（单向关联）。"""
        ...

    def set_booking_status(self, ticket_no: str, status: str) -> None:
        """更新挂号关联里的状态（退号时同步）。"""
        ...

    def list_bookings(
        self, *, encounter_id: str = "", limit: int = DEFAULT_BOOKING_LIMIT
    ) -> list[BookingLink]:
        """回看某次就诊挂出的号（新 → 旧）。"""
        ...

    def list_bookings_many(
        self, encounter_ids: Sequence[str], *, limit: int = DEFAULT_BOOKING_LIMIT
    ) -> dict[str, list[BookingLink]]:
        """批量回看（避免读路径 N+1）。"""
        ...


class PgPersistence:
    """`ClinicalStore` 的真实实现：三张表落进 Postgres 的 `clinical` schema。

    表结构见 `_DDL_STATEMENTS`；每条消息一行（append-only），就诊行存主诉与结论 JSON。
    `list_encounters` 会把结论 JSON 还原成契约对象、并用消息表重建这次就诊的对话。

    连接走 `psycopg_pool`（见模块文档「线程安全」）；schema 初始化走直连，且**在开池
    之前**完成——构造失败要立刻抛给上层，而不是留下一个半开的池。
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = DEFAULT_SCHEMA,
        pool_min: int = 1,
        pool_max: int = 10,
    ) -> None:
        self.dsn = dsn
        self.schema = schema
        self._tbl = {
            name: sql.Identifier(schema, name) for name in ("patients", "encounters", "messages", "bookings")
        }
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover - 取决于环境
            raise PersistenceError(
                "就诊记录的 Postgres 实现需要 psycopg-pool：pip install 'psycopg[pool]'"
            ) from e
        self.ensure_schema()
        # `timeout` 是**借连接**的上限，不是连接超时：数据库不可达时不该让调用方挂到
        # 驱动的默认值上（问诊每轮只写 2~3 行，10 秒拿不到连接就已经该降级了）。
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=pool_min,
            max_size=pool_max,
            open=False,
            timeout=10.0,
            name="kubedocs-clinical",
        )
        # 不在构造里等连接建好：Postgres 慢启动时 `/consult` 仍要能起来（真连不上会在
        # 第一次读写时抛错，由管线/端点按既有取舍处理）。
        self._pool.open(wait=False)

    def _q(self, template: str) -> sql.Composed:
        """把 `{patients}` / `{encounters}` / `{messages}` 换成 **schema 限定的标识符**。

        表名不是用户输入（`schema` 已按 `_IDENT_RE` 校验过），但与其口头保证"我拼的字符串
        安全"，不如把引用规则交给驱动：`sql.Identifier` 正是 psycopg 为这件事提供的 API。
        """
        return sql.SQL(template).format(**self._tbl)

    # ---------------------------------------------------------------- schema
    def ensure_schema(self) -> None:
        """建 schema / 表 / 索引（幂等）。构造时自动跑一次；重复调用是安全的。

        **走直连而不是连接池**：这是一次性动作，且它失败时构造必须直接抛错
        （上层据此降级为 `NoopPersistence`）。`DROP`/`CREATE` 都不会做——幂等靠的是
        `IF NOT EXISTS`，已有数据一律原地保留。
        """
        if not _IDENT_RE.match(self.schema):
            raise PersistenceError(f"非法的 schema 名 {self.schema!r}：只允许 [A-Za-z_][A-Za-z0-9_]*")
        try:
            with psycopg.connect(self.dsn, connect_timeout=10) as conn:
                conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
                for stmt in _DDL_STATEMENTS:
                    conn.execute(self._q(stmt))
        except Exception as e:  # noqa: BLE001 - 任何失败都要转成"记录库不可用"
            raise PersistenceError(
                f"连接/初始化 Postgres 失败（{type(e).__name__}: {e}）；"
                f"检查 KUBEDOCS_DATABASE_URL 与数据库可达性（当前 {redact_dsn(self.dsn)}）"
            ) from e

    # ---------------------------------------------------------------- 运维读数
    @property
    def location(self) -> str:
        """`/consult/status` 用的落点描述：`schema@脱敏 DSN`（口令不回显）。"""
        return f"{self.schema}@{redact_dsn(self.dsn)}"

    def counts(self) -> dict[str, int]:
        """各表的行数。给 `/consult/status` 用：运维要能看见它**真的在写**。

        不属于 `ClinicalStore` 协议：计数是运维读数，不是产品能力。
        """
        # 每条语句写死而不是循环拼表名：动态 SQL 的注入面为 0，也不与静态检查较劲。
        with self._pool.connection() as conn:
            return {
                "patients": int(conn.execute(self._q("SELECT COUNT(*) FROM {patients}")).fetchone()[0]),
                "encounters": int(conn.execute(self._q("SELECT COUNT(*) FROM {encounters}")).fetchone()[0]),
                "messages": int(conn.execute(self._q("SELECT COUNT(*) FROM {messages}")).fetchone()[0]),
                "bookings": int(conn.execute(self._q("SELECT COUNT(*) FROM {bookings}")).fetchone()[0]),
            }

    def close(self) -> None:
        """关掉连接池（进程退出、测试里模拟"重启"）。关掉后本对象不可再用。"""
        self._pool.close()

    def __enter__(self) -> PgPersistence:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- 写路径
    def upsert_patient(self, record: PatientRecord) -> None:
        """新增或**合并**患者。

        合并规则：空串 / 空字典**不覆盖**已有值。理由是调用点的形态——
        管线每一轮都会 `upsert_patient(PatientRecord(patient_id=...))` 只为把记录挂到
        这个人名下，若按"整行覆盖"实现，它会顺手把别处录的姓名与属性抹成空。
        同 id 反复写只有一行（主键 + `ON CONFLICT DO UPDATE`），`created_at` 保留首次值。

        "属性是不是空的"在 Python 侧判定后作为参数传入（而不是在 SQL 里写 `'{}'`）：
        SQL 模板走 `str.format`，字面量花括号要写两遍，容易在维护时改错。
        """
        attributes = json.dumps(record.attributes or {}, ensure_ascii=False)
        blank = attributes in ("", "{}")
        now = _now()
        with self._pool.connection() as conn:
            conn.execute(
                self._q(
                    "INSERT INTO {patients} AS p "
                    "(patient_id, institution, display_name, attributes_json, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (patient_id) DO UPDATE SET "
                    "  institution = CASE WHEN excluded.institution <> '' "
                    "                     THEN excluded.institution ELSE p.institution END, "
                    "  display_name = CASE WHEN excluded.display_name <> '' "
                    "                      THEN excluded.display_name ELSE p.display_name END, "
                    "  attributes_json = CASE WHEN %s THEN p.attributes_json "
                    "                         ELSE excluded.attributes_json END, "
                    "  updated_at = excluded.updated_at"
                ),
                (record.patient_id, record.institution, record.display_name, attributes, now, now, blank),
            )

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None:
        """**追加**一条消息（append-only：这里没有更新与删除）。

        这是对话正文的落点。一次就诊的多条消息靠自增 `id` 保持顺序，
        读回时按它排序，所以"用户说了什么、助手回了什么"不会错位。
        """
        with self._pool.connection() as conn:
            conn.execute(
                self._q(
                    "INSERT INTO {messages} (encounter_id, role, text, institution, created_at) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ),
                (encounter_id, role, text, institution, _now()),
            )

    def save_encounter(self, record: EncounterRecord) -> None:
        """落一次就诊：主诉 + 结论 JSON + 时间。

        关于 `transcript`：消息表才是对话正文的落点（`append_message` 是**增量**入口），
        所以这里只在**这次就诊还没有任何消息**时才用 `transcript` 播种消息行——
        这样单独调 `save_encounter` 的调用方也不会丢掉对话；而"先 `append_message`
        再 `save_encounter`"的正常路径（管线就是这么调的）不会把整段上文按轮重复写一遍：
        N 轮对话只会得到 N 组消息，不会得到 N² 组。

        `created_at` 只在插入时生效：重复保存同一次就诊不改变"它是什么时候发生的"。

        三步（确保患者 → 落就诊 → 播种消息）在**一个事务**里：中途失败要么全成、要么
        全不成——不会留下"就诊行在、消息不在"的半条记录。`_has_messages` 因此必须在
        同一事务里读（否则并发下两个写入者会同时判"还没有消息"而重复播种）。
        """
        created_at = record.created_at or _now()
        payload = record.assessment.model_dump_json() if record.assessment is not None else "{}"
        with self._pool.connection() as conn, conn.transaction():
            # 外键指向 patients：落就诊前先确保患者行存在（不覆盖已录的姓名/属性）。
            # `attributes_json` 在这里显式传 `'{}'`：空属性由代码写入（DDL 里没有 DEFAULT，
            # 理由见 _DDL_STATEMENTS 的说明）。
            conn.execute(
                self._q(
                    "INSERT INTO {patients} "
                    "(patient_id, institution, display_name, attributes_json, created_at, updated_at) "
                    "VALUES (%s, %s, '', %s, %s, %s) "
                    "ON CONFLICT (patient_id) DO NOTHING"
                ),
                (record.patient_id, record.institution, "{}", created_at, created_at),
            )
            conn.execute(
                self._q(
                    "INSERT INTO {encounters} "
                    "(encounter_id, patient_id, institution, chief_complaint, assessment_json, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (encounter_id) DO UPDATE SET "
                    "  patient_id = excluded.patient_id,"
                    "  institution = excluded.institution,"
                    "  chief_complaint = excluded.chief_complaint,"
                    "  assessment_json = excluded.assessment_json"
                ),
                (
                    record.encounter_id,
                    record.patient_id,
                    record.institution,
                    record.chief_complaint,
                    payload,
                    created_at,
                ),
            )
            if record.transcript and not self._has_messages(conn, record.encounter_id):
                with conn.cursor() as cur:
                    cur.executemany(
                        self._q(
                            "INSERT INTO {messages} "
                            "(encounter_id, role, text, institution, created_at) "
                            "VALUES (%s, %s, %s, %s, %s)"
                        ),
                        [
                            (
                                record.encounter_id,
                                str(m.get("role", "")),
                                str(m.get("content", "")),
                                record.institution,
                                created_at,
                            )
                            for m in record.transcript
                        ],
                    )

    # ---------------------------------------------------------------- 读路径
    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None:
        """按 id 取患者；没有就返回 `None`（"没有这个人"不是错误）。"""
        query = (
            "SELECT patient_id, institution, display_name, attributes_json "
            "FROM {patients} WHERE patient_id = %s"
        )
        params: list[Any] = [patient_id]
        if institution:
            query += " AND institution = %s"
            params.append(institution)
        with self._pool.connection() as conn:
            row = conn.execute(self._q(query), params).fetchone()
        if row is None:
            return None
        return PatientRecord(
            patient_id=str(row[0]),
            institution=str(row[1]),
            display_name=str(row[2]),
            attributes=_json_dict(str(row[3])),
        )

    def list_encounters(
        self, patient_id: str, *, institution: str = "", limit: int = DEFAULT_ENCOUNTER_LIMIT
    ) -> list[EncounterRecord]:
        """某患者最近若干次就诊，**新 → 旧**，`limit` 是上界。

        每条的 `transcript` 由消息表按写入顺序重建，`assessment` 由结论 JSON 还原
        （含 `red_flags` 等全部字段）。`created_at` 相同的行按内部 `seq` 倒序兜底，
        所以同一秒内的连续几轮也有稳定次序。
        """
        query = (
            "SELECT encounter_id, patient_id, institution, chief_complaint, created_at, assessment_json "
            "FROM {encounters} WHERE patient_id = %s"
        )
        params: list[Any] = [patient_id]
        if institution:
            query += " AND institution = %s"
            params.append(institution)
        query += " ORDER BY created_at DESC, seq DESC LIMIT %s"
        params.append(max(1, int(limit)))

        with self._pool.connection() as conn:
            rows = conn.execute(self._q(query), params).fetchall()
            ids = [str(r[0]) for r in rows]
            # 一次把消息全取回来（而不是每条就诊查一次）：limit 已是上界，N+1 不值当。
            transcripts: dict[str, list[dict[str, str]]] = {i: [] for i in ids}
            if ids:
                for m in conn.execute(
                    self._q(
                        "SELECT encounter_id, role, text FROM {messages} "
                        "WHERE encounter_id = ANY(%s) ORDER BY id"
                    ),
                    (ids,),
                ).fetchall():
                    transcripts[str(m[0])].append({"role": str(m[1]), "content": str(m[2])})

        return [
            EncounterRecord(
                encounter_id=str(r[0]),
                patient_id=str(r[1]),
                institution=str(r[2]),
                chief_complaint=str(r[3]),
                created_at=str(r[4]),
                assessment=_assessment_or_none(str(r[5])),
                transcript=transcripts[str(r[0])],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- 挂号关联
    def add_booking(self, record: BookingLink) -> None:
        """记下"这次就诊挂出了哪张号"。

        `ON CONFLICT (ticket_no) DO NOTHING`：确认挂号那条路径可能被重试（网络抖动、
        用户刷新后重放），同一张号不该在关联表里出现两行。**不 `DO UPDATE`**——号源那边
        已经判过一次，这里再覆盖只会把"哪次就诊挂的号"改掉，属于篡改历史。
        """
        now = _now()
        with self._pool.connection() as conn:
            conn.execute(
                self._q(
                    "INSERT INTO {bookings} "
                    "(ticket_no, encounter_id, department, booked_date, time_slot, fee, status,"
                    " created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (ticket_no) DO NOTHING"
                ),
                (
                    record.ticket_no,
                    record.encounter_id,
                    record.department,
                    record.date,
                    record.time_slot,
                    int(record.fee),
                    record.status,
                    record.created_at or now,
                    now,
                ),
            )

    def set_booking_status(self, ticket_no: str, status: str) -> None:
        """同步挂号关联里的状态（退号时调）。凭据号不存在就静默通过——
        退号的权威判定在号源层，这里只是把快照跟上，不该因为投影缺行而报错。"""
        with self._pool.connection() as conn:
            conn.execute(
                self._q("UPDATE {bookings} SET status = %s, updated_at = %s WHERE ticket_no = %s"),
                (str(status), _now(), str(ticket_no)),
            )

    def list_bookings(
        self, *, encounter_id: str = "", limit: int = DEFAULT_BOOKING_LIMIT
    ) -> list[BookingLink]:
        """某次就诊挂出的号（新 → 旧）。`encounter_id` 为空返回空列表——**不返回全部**：
        那等于把所有人的挂号记录倒出来，而这个方法的每个调用点都只关心一次就诊。"""
        if not encounter_id:
            return []
        return self.list_bookings_many([encounter_id], limit=limit).get(str(encounter_id), [])

    def list_bookings_many(
        self, encounter_ids: Sequence[str], *, limit: int = DEFAULT_BOOKING_LIMIT
    ) -> dict[str, list[BookingLink]]:
        """一次取回**多次就诊**各自的挂号关联，按 `encounter_id` 分组。

        存在的理由就是那条读路径的形状：`GET /consult/encounters` 一次返回 N 次就诊，
        每条都单查一次就是标准的 N+1。这里一句 `= ANY(%s)` 取完再在内存里分组。

        `limit` 是**每次就诊**的上界（不是总数）：调用方要的是"每次就诊挂出的号"，
        让某一次就诊把自己那 20 条占满而把别的挤掉，是错的分页语义。
        """
        ids = [str(i) for i in encounter_ids if str(i)]
        grouped: dict[str, list[BookingLink]] = {i: [] for i in ids}
        if not ids:
            return grouped
        with self._pool.connection() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT ticket_no, encounter_id, department, booked_date, time_slot, fee,"
                    " status, created_at FROM {bookings} WHERE encounter_id = ANY(%s) "
                    "ORDER BY created_at DESC, ticket_no"
                ),
                (ids,),
            ).fetchall()
        cap = max(1, int(limit))
        for row in rows:
            link = _booking_row(row)
            bucket = grouped.get(link.encounter_id)
            if bucket is not None and len(bucket) < cap:
                bucket.append(link)
        return grouped

    # ---------------------------------------------------------------- 内部
    def _has_messages(self, conn: Any, encounter_id: str) -> bool:
        """这次就诊有没有消息了。**必须在 `save_encounter` 的事务里调**（理由见该方法）。

        SQL 里的表名照旧走 `_q` 限定到本 schema：不依赖连接的 `search_path`。
        """
        row = conn.execute(
            self._q("SELECT 1 FROM {messages} WHERE encounter_id = %s LIMIT 1"), (encounter_id,)
        ).fetchone()
        return row is not None


class NoopPersistence:
    """**什么都不存**的实现。

    默认实现是 `PgPersistence`（由上层决定用哪个），这个类保留给两类场景：
    测试里不需要落库，以及部署方**显式**选择"这轮不留记录"（含数据库不可达时的降级）。

    它仍然存在（而不是删掉让类型变成可选）的理由：内存字典会给人"数据存住了"的错觉，
    重启才发现没有；`NoopPersistence` 从命名到行为都不含糊——**这里没有持久化**。
    """

    def upsert_patient(self, record: PatientRecord) -> None:
        del record

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None:
        del encounter_id, role, text, institution

    def save_encounter(self, record: EncounterRecord) -> None:
        del record

    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None:
        del patient_id, institution
        return None

    def list_encounters(
        self, patient_id: str, *, institution: str = "", limit: int = DEFAULT_ENCOUNTER_LIMIT
    ) -> list[EncounterRecord]:
        del patient_id, institution, limit
        return []

    def counts(self) -> dict[str, int]:
        """没有落库就没有行数（`/consult/status` 因此能如实显示全 0）。"""
        return {"patients": 0, "encounters": 0, "messages": 0, "bookings": 0}

    def add_booking(self, record: BookingLink) -> None:
        del record

    def set_booking_status(self, ticket_no: str, status: str) -> None:
        del ticket_no, status

    def list_bookings(
        self, *, encounter_id: str = "", limit: int = DEFAULT_BOOKING_LIMIT
    ) -> list[BookingLink]:
        del encounter_id, limit
        return []

    def list_bookings_many(
        self, encounter_ids: Sequence[str], *, limit: int = DEFAULT_BOOKING_LIMIT
    ) -> dict[str, list[BookingLink]]:
        del limit
        return {str(i): [] for i in encounter_ids}

    def close(self) -> None:
        """与 `PgPersistence` 同形（上层可以不区分地关掉它）。"""
