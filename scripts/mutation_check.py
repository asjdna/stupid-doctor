"""变异验证：故意改坏关键逻辑，确认对应测试**变红**，然后原样还原。

一个测试如果在实现被改坏之后依然是绿的，那它就没有在守任何东西。
这里对 4 处关键逻辑各做一次"回退式变异"，跑指定的测试并记录退出码。

用法：python scripts/mutation_check.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

CASES = [
    (
        "M1 sqlite-vec 不写向量（_index_chunk_vectors 变空实现）",
        "src/kubedocs/store/sqlite_vec_store.py",
        """        rows = [
            (c.chunk_id, np.asarray(vectors[c.chunk_id], dtype=_BLOB_DTYPE).tobytes())
            for c in chunks
        ]
        con.executemany(_sql(
            "INSERT INTO {tbl}(chunk_id, embedding) VALUES(?, ?)", tbl=VEC_TABLE), rows)""",
        """        return""",
        "tests/unit/test_store_contract.py::test_dense_search_shape_and_rank_start_at_one",
    ),
    (
        "M2 删文档时不删向量行（_drop_doc_chunks 跳过向量清理）",
        "src/kubedocs/store/sqlite_vec_store.py",
        """        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            con.execute(_sql(
                "DELETE FROM {tbl} WHERE chunk_id IN ({marks})",
                tbl=VEC_TABLE, marks=_marks(len(batch))), batch)
        return super()._drop_doc_chunks(con, doc_id)""",
        """        return super()._drop_doc_chunks(con, doc_id)""",
        "tests/unit/test_store_contract.py::test_delete_docs_removes_from_both_channels",
    ),
    (
        "M3 未知后端静默回退到 sqlite（工厂不再报错）",
        "src/kubedocs/store/__init__.py",
        """    raise StoreError(
        f"未知的存储后端 {backend!r}（可选：{', '.join(BACKENDS)}）——"
        "不会静默回退到任何默认实现，请显式改成合法值"
    )""",
        """    return sqlite_from_settings(settings)  # type: ignore[arg-type]""",
        "tests/unit/test_store_backend.py::TestFactory::test_unknown_backend_raises_instead_of_falling_back",
    ),
    (
        "M4 去掉页缓存 pragma（性能退化无人守）",
        "src/kubedocs/store/sqlite_vec_store.py",
        '        self._con.execute(f"PRAGMA cache_size=-{int(cache_kib)}")',
        "        pass",
        "tests/unit/test_store_contract.py::TestSqliteVecBackend::test_page_cache_is_raised",
    ),
]


def main() -> int:
    failures = 0
    for name, rel, old, new, node in CASES:
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if old not in original:
            print(f"[跳过] {name}：待替换片段未找到")
            failures += 1
            continue
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            # 命令是本文件里写死的（解释器路径 + 固定参数），不存在外部输入参与构造；
            # S603/S607 在这个场景下是误报，逐条说明而不是全局忽略
            proc = subprocess.run([PY, "-m", "pytest", node, "-p", "no:cacheprovider"],  # noqa: S603
                                  cwd=str(ROOT), capture_output=True, text=True)
            red = proc.returncode != 0
            last = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
            print(f"[{'OK' if red else '没变红!'}] {name} -> exit {proc.returncode}  {last}")
            if not red:
                failures += 1
        finally:
            path.write_text(original, encoding="utf-8")
    print("\n变异验证汇总：", "全部按预期变红" if failures == 0 else f"{failures} 项不符合预期")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
