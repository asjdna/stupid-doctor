"""就诊记录（患者 / 就诊 / 消息）的持久化：**真的落进一个 SQLite 文件**。

`SqlitePersistence` 是默认实现，`NoopPersistence` 只留给"显式不落盘"的场景（测试、
临时演示）。存储层自己**不吞异常**：失败如实抛给调用方，由持有产品取舍的那一层决定
（`ClinicalPipeline` 的选择是"记一条 warning 后继续"——见该模块的说明）。

## 为什么单独一个文件，而不是塞进索引库

就诊记录与向量索引是两种性质的数据：

- **索引是派生物**：语料在、脚本在，随时可以重建，丢了只是重跑一次 `kubedocs ingest`；
  同一个库文件里还混着 FTS 表、向量表与 chunk 元数据，`ingest` 会按域整批改写它们。
- **就诊记录不是派生物**：它是原始数据，重建不出来。丢了就是真的丢了。

所以它落在**自己的文件**里（默认 `<index_dir>/clinic/records.db`）。它放在 `data/index/`
下只是**部署上就近**（数据都在 `data/` 里、备份一处即可），读写路径与生命周期完全独立：
`ingest` 只写 `kubedocs.db` / `chunk_vectors.npy`，不会碰它。

## 多机构：`institution` 是**保留列**，不是租户隔离

本次不做多机构（行级安全 / 连接级 schema / 分库）。`institution` 现在只是把"归属哪个
机构"这个唯一确定得了的形状先占住：读写都只用默认空串，将来加隔离时不必改表结构与
所有调用点的签名。约定：读路径传**空串 = 不过滤**（当前全部数据都是空机构），
传非空值时按精确匹配过滤——所以拿一个真实机构名去查，会**查不到**任何东西。

## 线程安全：一条长连接 + 一把锁

HTTP 层在线程池里跑轮次（`/consult` 是同步端点），连接因此不能绑死在创建它的线程上
（`check_same_thread=False`）。代价是**同一个连接对象不允许并发使用**，所以用一把
`threading.RLock` 把**全部读写**串行化（不只是写：SQLite 的隐式事务是连接级的，
一边读一边写会让另一侧报"cannot start a transaction within a transaction"）。

选长连接而不是"每次开一条短连接"：问诊每轮只写 2~3 行，锁竞争可以忽略；而短连接
每轮都要重开文件、重设 PRAGMA，还要在 Windows 上处理 WAL 文件句柄的释放时机。
真到了写不动的那天，换 Postgres 实现同一个 `ClinicalStore`，而不是在这里做连接池。
`busy_timeout` 另开一道保险：多进程（两个 worker）指到同一个文件时，忙等而不是立刻报错。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from kubedocs.clinical.contracts import ClinicalAssessment

logger = logging.getLogger(__name__)

DEFAULT_ENCOUNTER_LIMIT = 20
"""`list_encounters` 的默认条数（读路径是给人看的，不需要一次拉几百条）。"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT PRIMARY KEY,
    institution     TEXT NOT NULL DEFAULT '',
    display_name    TEXT NOT NULL DEFAULT '',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS encounters (
    encounter_id    TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    institution     TEXT NOT NULL DEFAULT '',
    chief_complaint TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT '',
    assessment_json TEXT NOT NULL DEFAULT '{}',
    -- 就诊必须挂在患者上：`PRAGMA foreign_keys=ON` 靠这条约束真的有东西可查。
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id TEXT NOT NULL,
    role         TEXT NOT NULL,
    text         TEXT NOT NULL DEFAULT '',
    institution  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT ''
    -- 刻意**不加**指向 encounters 的外键：消息是 append-only 的增量写入，管线里
    -- `append_message` 先于 `save_encounter`（顺序理由见 save_encounter 的说明），
    -- 加了外键就会把这条正常路径变成"先有鸡还是先有蛋"。
    -- `PRAGMA foreign_keys=ON` 仍然有东西可查：就诊不许挂在不存在的患者上。
);

-- 读路径就是"某患者最近若干次就诊"：按 (patient_id, created_at) 建索引才不必全表扫。
CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id, created_at);
-- 消息按就诊取回，且要保持写入顺序（AUTOINCREMENT id 就是顺序）。
CREATE INDEX IF NOT EXISTS idx_messages_encounter ON messages(encounter_id, id);
"""


def _now() -> str:
    """统一的时间戳：UTC + 秒精度 ISO 字符串。

    用字符串而不是 epoch 数字：直接 `sqlite3` 打开文件排查时人能读；同一格式下
    **字典序 == 时间序**，`ORDER BY created_at DESC` 直接可用，不必在 SQL 里做转换。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


class ClinicalStore(Protocol):
    """产品层需要的持久化能力。实现可以是 SQLite（默认）、Postgres，也可以是 `NoopPersistence`。"""

    def upsert_patient(self, record: PatientRecord) -> None: ...

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None: ...

    def save_encounter(self, record: EncounterRecord) -> None: ...

    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None: ...

    def list_encounters(
        self, patient_id: str, *, institution: str = "", limit: int = DEFAULT_ENCOUNTER_LIMIT
    ) -> list[EncounterRecord]:
        """读回某患者最近若干次就诊（新 → 旧），用于"记录真的存住了"的验证与回看。"""
        ...


class SqlitePersistence:
    """`ClinicalStore` 的真实实现：三张表落进一个 SQLite 文件。

    表结构见 `_SCHEMA`；每条消息一行（append-only），就诊行存主诉与结论 JSON。
    `list_encounters` 会把结论 JSON 还原成契约对象、并用消息表重建这次就诊的对话。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # check_same_thread=False：连接要跨线程用（HTTP 线程池），并发由 self._lock 挡。
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL：读不阻塞写、写不阻塞读；崩溃后也能从 WAL 恢复到最后一次提交。
            self._conn.execute("PRAGMA journal_mode=WAL")
            # 外键约束默认**关着**，必须每个连接显式打开（encounters -> patients）。
            self._conn.execute("PRAGMA foreign_keys=ON")
            # 多进程指同一文件时忙等 5 秒再报错，而不是立刻抛 locked。
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- 写路径
    def upsert_patient(self, record: PatientRecord) -> None:
        """新增或**合并**患者。

        合并规则：空串 / 空字典**不覆盖**已有值。理由是调用点的形态——
        管线每一轮都会 `upsert_patient(PatientRecord(patient_id=...))` 只为把记录挂到
        这个人名下，若按"整行覆盖"实现，它会顺手把别处录的姓名与属性抹成空。
        同 id 反复写只有一行（主键 + `ON CONFLICT DO UPDATE`）。
        """
        attributes = json.dumps(record.attributes or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO patients (patient_id, institution, display_name, attributes_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    institution = CASE WHEN excluded.institution <> '' THEN excluded.institution ELSE patients.institution END,
                    display_name = CASE WHEN excluded.display_name <> '' THEN excluded.display_name ELSE patients.display_name END,
                    attributes_json = CASE WHEN excluded.attributes_json IN ('', '{}') THEN patients.attributes_json ELSE excluded.attributes_json END,
                    updated_at = excluded.updated_at
                """,
                (record.patient_id, record.institution, record.display_name, attributes, _now()),
            )
            self._conn.commit()

    def append_message(self, *, encounter_id: str, role: str, text: str, institution: str = "") -> None:
        """**追加**一条消息（append-only：这里没有更新与删除）。

        这是对话正文的落点。一次就诊的多条消息靠自增 `id` 保持顺序，
        读回时按它排序，所以"用户说了什么、助手回了什么"不会错位。
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (encounter_id, role, text, institution, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (encounter_id, role, text, institution, _now()),
            )
            self._conn.commit()

    def save_encounter(self, record: EncounterRecord) -> None:
        """落一次就诊：主诉 + 结论 JSON + 时间。

        关于 `transcript`：消息表才是对话正文的落点（`append_message` 是**增量**入口），
        所以这里只在**这次就诊还没有任何消息**时才用 `transcript` 播种消息行——
        这样单独调 `save_encounter` 的调用方也不会丢掉对话；而"先 `append_message`
        再 `save_encounter`"的正常路径（管线就是这么调的）不会把整段上文按轮重复写一遍：
        N 轮对话只会得到 N 组消息，不会得到 N² 组。

        `created_at` 只在插入时生效：重复保存同一次就诊不改变"它是什么时候发生的"。
        """
        created_at = record.created_at or _now()
        payload = record.assessment.model_dump_json() if record.assessment is not None else "{}"
        with self._lock:
            # 外键指向 patients：落就诊前先确保患者行存在（不覆盖已录的姓名/属性）。
            self._conn.execute(
                "INSERT OR IGNORE INTO patients (patient_id, institution, display_name, attributes_json, updated_at) "
                "VALUES (?, ?, '', '{}', ?)",
                (record.patient_id, record.institution, created_at),
            )
            self._conn.execute(
                """
                INSERT INTO encounters
                    (encounter_id, patient_id, institution, chief_complaint, created_at, assessment_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(encounter_id) DO UPDATE SET
                    patient_id = excluded.patient_id,
                    institution = excluded.institution,
                    chief_complaint = excluded.chief_complaint,
                    assessment_json = excluded.assessment_json
                """,
                (
                    record.encounter_id,
                    record.patient_id,
                    record.institution,
                    record.chief_complaint,
                    created_at,
                    payload,
                ),
            )
            if record.transcript and not self._has_messages(record.encounter_id):
                self._conn.executemany(
                    "INSERT INTO messages (encounter_id, role, text, institution, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
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
            self._conn.commit()

    # ---------------------------------------------------------------- 读路径
    def get_patient(self, patient_id: str, *, institution: str = "") -> PatientRecord | None:
        """按 id 取患者；没有就返回 `None`（"没有这个人"不是错误）。"""
        sql = (
            "SELECT patient_id, institution, display_name, attributes_json FROM patients WHERE patient_id = ?"
        )
        params: list[Any] = [patient_id]
        if institution:
            sql += " AND institution = ?"
            params.append(institution)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return PatientRecord(
            patient_id=str(row["patient_id"]),
            institution=str(row["institution"]),
            display_name=str(row["display_name"]),
            attributes=_json_dict(str(row["attributes_json"])),
        )

    def list_encounters(
        self, patient_id: str, *, institution: str = "", limit: int = DEFAULT_ENCOUNTER_LIMIT
    ) -> list[EncounterRecord]:
        """某患者最近若干次就诊，**新 → 旧**，`limit` 是上界。

        每条的 `transcript` 由消息表按写入顺序重建，`assessment` 由结论 JSON 还原
        （含 `red_flags` 等全部字段）。`created_at` 相同的行按插入顺序倒序兜底，
        所以同一秒内的连续几轮也有稳定次序。
        """
        sql = (
            "SELECT encounter_id, patient_id, institution, chief_complaint, created_at, assessment_json "
            "FROM encounters WHERE patient_id = ?"
        )
        params: list[Any] = [patient_id]
        if institution:
            sql += " AND institution = ?"
            params.append(institution)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            ids = [str(r["encounter_id"]) for r in rows]
            # 一次把消息全取回来（而不是每条就诊查一次）：limit 已是上界，N+1 不值当。
            transcripts: dict[str, list[dict[str, str]]] = {i: [] for i in ids}
            if ids:
                placeholders = ",".join("?" * len(ids))
                for m in self._conn.execute(
                    # 拼进去的只有问号个数，值仍走参数绑定（`ids` 作为参数传入）
                    "SELECT encounter_id, role, text FROM messages "  # noqa: S608
                    f"WHERE encounter_id IN ({placeholders}) ORDER BY id",
                    ids,
                ):
                    transcripts[str(m["encounter_id"])].append(
                        {"role": str(m["role"]), "content": str(m["text"])}
                    )

        return [
            EncounterRecord(
                encounter_id=str(r["encounter_id"]),
                patient_id=str(r["patient_id"]),
                institution=str(r["institution"]),
                chief_complaint=str(r["chief_complaint"]),
                created_at=str(r["created_at"]),
                assessment=_assessment_or_none(str(r["assessment_json"])),
                transcript=transcripts[str(r["encounter_id"])],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- 运维
    def counts(self) -> dict[str, int]:
        """三张表的行数。给 `/consult/status` 用：运维要能看见它**真的在写**。

        不属于 `ClinicalStore` 协议：计数是运维读数，不是产品能力。
        """
        with self._lock:
            return {
                table: int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608
                for table in ("patients", "encounters", "messages")
            }

    def close(self) -> None:
        """关掉连接（进程退出、测试里模拟"重启"）。关掉后本对象不可再用。"""
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- 内部
    def _has_messages(self, encounter_id: str) -> bool:
        """这次就诊有没有消息了。调用方必须已持锁。"""
        row = self._conn.execute(
            "SELECT 1 FROM messages WHERE encounter_id = ? LIMIT 1", (encounter_id,)
        ).fetchone()
        return row is not None


class NoopPersistence:
    """**什么都不存**的实现。

    默认实现是 `SqlitePersistence`（由上层决定用哪个），这个类保留给两类场景：
    测试里不需要落盘，以及部署方**显式**选择"这轮不留记录"。

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
        """没有落盘就没有行数（`/consult/status` 因此能如实显示全 0）。"""
        return {"patients": 0, "encounters": 0, "messages": 0}
