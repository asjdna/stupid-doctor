"""索引迁移：单域平铺布局 -> 多域布局，并把 schema v1 升到 v2。

## 这次迁移**不重新编码任何向量**

v2 的唯一结构变化是：全文索引从 `chunks.body` 改为 `chunks.body_normalized`
（为中文预分词让路，见 `normalize.py`）。英文域的归一化是原样透传，
所以迁移只是：

    ALTER TABLE chunks ADD COLUMN body_normalized ...
    UPDATE chunks SET body_normalized = body          -- 原样，不涉及模型
    DROP/CREATE chunks_fts（列名改了）
    INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')   -- 从内容表重建倒排索引

全程不碰 `chunk_vectors.npy`，也就**不需要跑 34 分钟的嵌入**。
这是把"索引形态"与"展示形态"分开之后的直接好处。

用法：
    python scripts/migrate_index_layout.py --dry-run     # 只看会做什么
    python scripts/migrate_index_layout.py               # 真正执行
    python scripts/migrate_index_layout.py --verify-only # 只做一致性体检

退出码：0 成功 / 1 迁移失败 / 3 前置条件不满足。
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.config import Settings  # noqa: E402

TARGET_VERSION = 2

_V2_FTS = """
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    body_normalized, title, breadcrumb,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
)
"""


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _fts_columns(con: sqlite3.Connection, table: str) -> set[str]:
    # FTS5 虚表的列可用 table_info 读到
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _layout(src: Path, dst: Path, dry_run: bool) -> bool:
    """把平铺索引搬进域目录。返回是否发生了搬运。"""
    if not src.exists():
        return False
    moved = False
    for name in ("kubedocs.db", "chunk_vectors.npy"):
        f = src / name
        if not f.exists():
            continue
        target = dst / name
        if target.exists() and target.stat().st_size == f.stat().st_size:
            print(f"  目标已存在同名同大小文件，跳过：{target}")
            continue
        print(f"  移动 {f} -> {target}")
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(target))
        moved = True
    return moved


def _schema_version(con: sqlite3.Connection) -> int:
    row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return int(row[0]) if row else 0


def _upgrade_schema(db: Path, dry_run: bool) -> int:
    """v1 -> v2。幂等：已经是 v2 就什么都不做。"""
    con = sqlite3.connect(str(db))
    try:
        version = _schema_version(con)
        n_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"  schema_version={version}，chunks={n_chunks}")

        if version == TARGET_VERSION:
            cols = _columns(con, "chunks")
            fts_cols = _fts_columns(con, "chunks_fts")
            if "body_normalized" in cols and "body_normalized" in fts_cols:
                print("  已是 v2 且结构正确，无需迁移")
                return 0
            print("  ⚠ 版本号是 v2 但结构不符，按 v1 处理")
        if version not in (1, TARGET_VERSION):
            print(f"  ✗ 不支持的起始版本 {version}（只支持 1 -> {TARGET_VERSION}）")
            return 1
        if n_chunks == 0:
            print("  ✗ 索引为空；直接删除索引目录后重新 ingest 更省事")
            return 3

        if dry_run:
            print("  [dry-run] 将执行：ADD COLUMN body_normalized / UPDATE = body / "
                  "DROP+CREATE chunks_fts / FTS 'rebuild' / 写入 schema_version=2")
            return 0

        cols = _columns(con, "chunks")
        if "body_normalized" not in cols:
            con.execute("ALTER TABLE chunks ADD COLUMN body_normalized TEXT NOT NULL DEFAULT ''")
        con.execute("UPDATE chunks SET body_normalized = body WHERE body_normalized = ''")
        con.execute("DROP TABLE IF EXISTS chunks_fts")
        con.execute(_V2_FTS)
        # FTS5 外内容表的官方重建指令：从内容表重新生成倒排索引
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.execute("INSERT INTO meta(key, value) VALUES('normalizer', 'identity') "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value")
        con.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(TARGET_VERSION),))
        con.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(TARGET_VERSION),))
        con.commit()
        print(f"  ✓ 已升级到 v{_schema_version(con)}")
        return 0
    finally:
        con.close()


def _verify(db: Path) -> int:
    con = sqlite3.connect(str(db))
    try:
        if "body_normalized" not in _columns(con, "chunks"):
            print("  ✗ chunks 表还没有 body_normalized 列（尚未迁移；dry-run 下属正常）")
            return 3
        n_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        non_empty = con.execute(
            "SELECT COUNT(*) FROM chunks WHERE body_normalized <> ''").fetchone()[0]
        fts_n = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        print(f"  documents={n_docs} chunks={n_chunks} body_normalized 非空={non_empty} "
              f"fts 行数={fts_n}")
        try:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
            print("  FTS 一致性：ok")
        except sqlite3.DatabaseError as e:
            print(f"  ✗ FTS 一致性失败：{e}")
            return 1
        if n_chunks and non_empty != n_chunks:
            print(f"  ✗ 有 {n_chunks - non_empty} 个块没有归一化文本，倒排索引会漏")
            return 1
        if n_chunks and fts_n != n_chunks:
            print(f"  ✗ FTS 行数({fts_n}) 与 chunks({n_chunks}) 不一致")
            return 1
        return 0
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="索引迁移：多域布局 + schema v1->v2")
    p.add_argument("--domain", default="k8s_docs", help="目标域目录名")
    p.add_argument("--index-dir", default=None, help="索引根目录（默认取配置）")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)

    base = Path(args.index_dir) if args.index_dir else Settings().index_root
    dst = base / args.domain
    db = dst / "kubedocs.db"

    print(f"索引根目录：{base}")
    print(f"目标域目录：{dst}")

    if not args.verify_only:
        print("\n[1/2] 布局迁移（平铺 -> 域目录）")
        _layout(base, dst, args.dry_run)
    if not db.exists():
        # dry-run 下文件当然还没搬过去：退回源位置继续演练后续步骤，
        # 否则 dry-run 会在第一步就断掉、看不到 schema 升级会做什么。
        legacy = base / "kubedocs.db"
        if args.dry_run and legacy.exists():
            print(f"  （dry-run：尚未搬运，改用源位置 {legacy} 演练 schema 步骤）")
            db = legacy
        else:
            print(f"✗ 找不到索引库：{db}")
            return 3

    if not args.verify_only:
        print("\n[2/2] schema 升级")
        rc = _upgrade_schema(db, args.dry_run)
        if rc != 0:
            return rc

    print("\n[体检]")
    return _verify(db)


if __name__ == "__main__":
    sys.exit(main())
