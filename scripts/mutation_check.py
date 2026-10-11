"""变异验证：故意改坏关键逻辑，确认对应测试**变红**，然后原样还原。

一个测试如果在实现被改坏之后依然是绿的，那它就没有在守任何东西。
这里对若干处关键逻辑各做一次"回退式变异"，跑指定的测试并记录退出码。

分组：
- M1~M4：存储层（sqlite-vec 写向量 / 删文档 / 后端工厂 / 页缓存）；
- M5：clinic 闸门里的**分诊**判据（摘掉它，"分诊 0 分也通过"的缺陷就该回来）；
- M6：工具循环**调用前**的预算检查（摘掉它，`/chat` 就又能超预算调模型）；
- M7 / M10：`/ask` 与 `/chat` 两条路径上"**算不出价按 0 记**"的缺陷；
- M8 / M9：`/chat` 会话表的**属主校验**与**LRU 淘汰**。

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
    (
        "M5 clinic 闸门摘掉分诊判据（回到「分诊 0 分也算通过」）",
        "src/kubedocs/evaluation/clinic.py",
        """    accuracy = report.triage_accuracy
    if not report.triage_evaluated or accuracy is None:
        fails.append(
            "分诊未评测（triage.evaluated=False 或没有可评测用例）："
            "没测不等于测了没问题——若把它当通过，“删掉语料”就成了让闸门变绿的手段"
        )
    elif accuracy < triage_min_accuracy:
        fails.append(
            f"分诊准确率 {accuracy:.2%}（hit {report.triage_hit}/{report.triage_total}）"
            f"低于最低门限 {triage_min_accuracy:.0%}；分诊是 clinic 域对外唯一的结论，"
            "答不出/答错科室等于给出错误指引"
        )
        fails.extend(f"  未命中: {m}" for m in report.triage_missed[:10])
    return fails""",
        """    _ = triage_min_accuracy          # 变异：分诊不进闸门
    return fails""",
        "tests/unit/test_clinic_eval.py::TestTriageIsInTheGate::test_zero_triage_accuracy_fails_the_gate",
    ),
    (
        "M6 工具循环不做调用前的预算检查（/chat 又能量超预算调模型）",
        "src/kubedocs/agent/tool_loop.py",
        """        if budget is not None:
            # 调用**之前**检查：超限时这一次模型调用根本不发生（不是"先花了再说"）。
            # 这里刻意放在 try 之外：异常必须上抛给调用方，而不是变成一条 note。
            budget.check()""",
        """        if False:                    # 变异：只记不查
            budget.check()""",
        "tests/unit/test_tool_loop.py::TestBudgetGateInTheLoop::test_over_budget_stops_calling_the_model",
    ),
    (
        "M7 /ask 路径把「算不出价」按 0 记（闸门再也拦不住）",
        "src/kubedocs/generate.py",
        """            cost, priced = 0.0, False
        return LLMResult(""",
        """            cost, priced = 0.0, True
        return LLMResult(""",
        "tests/unit/test_generate.py::TestRealClientFlagsUnknownPrice"
        "::test_price_map_miss_is_flagged_not_zeroed",
    ),
    (
        "M8 /chat 会话表不校验属主（猜到 thread_id 就能续聊别人）",
        "src/kubedocs/api/threads.py",
        """                self._threads.move_to_end(thread_id)
                self._evict_locked()
                return
            if thread.owner != owner:
                raise ThreadOwnershipError(thread_id)
            self._threads.move_to_end(thread_id)""",
        """                self._threads.move_to_end(thread_id)
                self._evict_locked()
                return
            if False:                    # 变异：属主校验摘掉
                raise ThreadOwnershipError(thread_id)
            thread.owner = owner
            self._threads.move_to_end(thread_id)""",
        "tests/unit/test_api_threads.py::TestOwnership::test_another_owner_is_rejected",
    ),
    (
        "M9 /chat 会话表不淘汰（回到无界增长）",
        "src/kubedocs/api/threads.py",
        "        while len(self._threads) > self.max_threads:",
        "        while False:                 # 变异：去掉淘汰",
        "tests/unit/test_api_threads.py::TestBounded::test_thread_count_is_capped",
    ),
    (
        "M10 /chat 的 ToolCaller 把「算不出价」按 0 报（原先那个空操作 except）",
        "src/kubedocs/agent/llm_tool_caller.py",
        """            cost, priced = 0.0, False
            self.unpriced_calls += 1""",
        """            cost, priced = 0.0, True
            self.unpriced_calls += 0""",
        "tests/unit/test_tool_integration.py::TestToolCallerReportsUnknownPrice"
        "::test_price_map_miss_is_flagged_not_zeroed",
    ),
]


def _read(path: pathlib.Path) -> str:
    """按**原样**读文本：不翻译换行（`newline=""`）。

    这个脚本会改写仓库里的真实源文件，所以换行必须逐字节保留：Windows 上
    `read_text` / `write_text` 默认会做换行翻译，写回时把 LF 变成 CRLF——
    实测把 `sqlite_vec_store.py` 的 382 行、`store/__init__.py` 的 77 行全改成 CRLF，
    `git status` 立刻显示这两个**没被这次变异碰过内容**的文件被修改了。
    本仓库要求 LF（`.gitattributes` 是 `* text=auto eol=lf`），所以读写都显式关掉翻译。
    """
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(path: pathlib.Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main() -> int:
    failures = 0
    for name, rel, old, new, node in CASES:
        path = ROOT / rel
        original = _read(path)
        if old not in original:
            print(f"[跳过] {name}：待替换片段未找到")
            failures += 1
            continue
        _write(path, original.replace(old, new, 1))
        try:
            # 命令是本文件里写死的（解释器路径 + 固定参数），不存在外部输入参与构造；
            # S603/S607 在这个场景下是误报，逐条说明而不是全局忽略。
            # ⚠️ noqa 必须挂在 `subprocess.run(` 这一行：S603 的诊断**落在调用起始行**，
            # 而 `ruff format` 会把参数拆到后续行——把 noqa 留在参数行上等于没抑制
            # （实测：格式化后 `ruff check .` 反而变红，2026 年那次全量格式化就是）。
            proc = subprocess.run(  # noqa: S603
                [PY, "-m", "pytest", node, "-p", "no:cacheprovider"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            red = proc.returncode != 0
            last = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
            print(f"[{'OK' if red else '没变红!'}] {name} -> exit {proc.returncode}  {last}")
            if not red:
                failures += 1
        finally:
            _write(path, original)
    print("\n变异验证汇总：", "全部按预期变红" if failures == 0 else f"{failures} 项不符合预期")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
