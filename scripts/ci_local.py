"""在本地按 CI 的口径把三个 job 跑一遍（`ci.yml` 的等价物）。

## 为什么需要它

`.github/workflows/ci.yml` 是这个仓库的门禁，但它**一次都没在真实 runner 上跑过**
（仓库没有 remote，见 SECURITY.md §6）。本地能替代的只有 `make check`，
而 `make check` == `lint fmt type test refs`，只等价于 **quality job 的静态+测试部分**:

- `pipeline` job（全量语料 + 哈希向量全链路自检）本地没有对应目标；
- `gate` job（真实模型 + 冻结基线闸门）也没有；
- 两者在 CI 里都是**阻塞**的，所以"本地 `make check` 绿了"并不等于"CI 会绿"。

这个脚本把三个 job 按同样的顺序、同样的环境变量在本地跑一遍，并给出逐 job 的判定。

## 与 CI 的**差异**（如实列出，不假装等价）

| 差异 | 说明 |
|---|---|
| 不重新 checkout | 用当前工作树。CI 是干净检出，本地脏树里多出来的未跟踪文件**不会**被 CI 看到 |
| 不建 Python 环境 | 用当前解释器。CI 用 `actions/setup-python@v5` 装 3.10 + `pip cache` |
| 不缓存 | CI 的 `gate` job 靠 cache 复用索引与模型；本地直接用现成的 `data/index` 与 HF 缓存 |
| Postgres 由你提供 | `quality` job 在 CI 里有 service 容器。本地需自己起（`docker compose up -d db`），否则 pgvector 那族用例会 skip —— 脚本会**把它当失败**，因为 CI 里它们必须真跑 |
| `--hf-offline` | CI 的 runner 有外网，本地按 README「环境约束」必须带（模型已缓存时联网校验会挂死 >10 分钟）。脚本默认加上 |

## 副作用隔离（最要紧的一条）

`pipeline` job 用 `hashing` 建索引，**绝不能**覆盖 `data/index/<domain>` 那份真实索引
（重建要 35 分钟）。脚本因此把它建到临时目录，跑完即删。
`gate` job 反过来**必须**用真实索引，所以它只读不写（`eval --gate` 不写索引）。

## 用法

    .venv/Scripts/python scripts/ci_local.py                 # 三个 job 全跑
    .venv/Scripts/python scripts/ci_local.py --jobs quality   # 只跑一个
    .venv/Scripts/python scripts/ci_local.py --keep-temp      # 保留 pipeline 的临时索引
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
"""仓库根（`scripts/ci_local.py` -> 上溯两层）。"""

JOBS = ("quality", "pipeline", "gate")

DEFAULT_DATABASE_URL = "postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs"
"""与 `Settings.database_url` 的默认值同形，也与 ci.yml 里 quality job 的 env 一致。"""


class JobFailed(RuntimeError):
    """某个 job 里有一条命令返回非零。"""


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path = REPO) -> None:
    """跑一条命令，输出直接透传到终端（CI 里也是这么看的）。"""
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    started = time.perf_counter()
    merged = {**os.environ, **(env or {})}
    # 命令全部由本文件构造（`sys.executable` + 字面量参数 + 自己创建的临时目录），
    # 没有任何外部输入参与拼接；`--jobs` 也已按 `JOBS` 白名单校验过。
    # S603 在这个场景下是误报，逐条说明而不是全局忽略（与 scripts/mutation_check.py 同惯例）。
    # ⚠️ noqa 必须挂在 `subprocess.run(` 这一行：S603 的诊断落在**调用起始行**，
    # 而 `ruff format` 会把参数拆到后续行——留在参数行上等于没抑制。
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(cwd),
        env=merged,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise JobFailed(f"退出码 {proc.returncode}：{printable}（耗时 {elapsed:.1f}s）")
    print(f"  ✓ 耗时 {elapsed:.1f}s", flush=True)


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _check_environment() -> None:
    """先证明"我们跑的是这个仓库的 venv"，再开始 —— 否则失败信息会很难读。"""
    try:
        import kubedocs  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 环境问题，不是逻辑分支
        raise SystemExit(
            "无法 import kubedocs。请用仓库自己的解释器运行：\n"
            "    .venv\\Scripts\\python scripts\\ci_local.py      # Windows\n"
            "    .venv/bin/python scripts/ci_local.py            # POSIX\n"
            f"（当前解释器：{sys.executable}；原因：{exc}）"
        ) from exc
    if not (REPO / "data" / "raw").is_dir():
        print(
            "  [注意] data/raw 不存在 ⇒ pipeline job 需要先在本地拉一次语料：\n"
            "         python scripts/fetch_corpus.py\n",
            flush=True,
        )


def _postgres_reachable(dsn: str) -> tuple[bool, str]:
    """quality job 依赖 Postgres：`ci.yml` 里有 service，本地要自己起。

    判据与 CI 一致地严格：**连得上还不够**，`pgvector` 那族用例真跑了才算数。
    所以这里只做"能不能连"的前置判断，是否真跑由 pytest 的 `-rs` 输出说话。
    """
    try:
        import psycopg
    except ImportError:  # pragma: no cover
        return False, 'psycopg 没装（pip install -e ".[dev]"）'
    try:
        with psycopg.connect(dsn, connect_timeout=3):
            return True, "ok"
    except Exception as exc:  # noqa: BLE001 - 前置探测：任何失败都只是"连不上"
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- job
def job_quality(*, hf_offline: bool) -> None:
    """静态检查 + 单元测试（对应 ci.yml 的 quality job）。"""
    del hf_offline  # quality job 不加载模型，与离线无关
    _banner("job: quality —— 静态检查 + 单元测试（需要真实 Postgres）")

    dsn = os.environ.get("KUBEDOCS_DATABASE_URL", DEFAULT_DATABASE_URL)
    reachable, why = _postgres_reachable(dsn)
    if reachable:
        print(f"  Postgres 可达：{dsn}")
    else:
        # CI 有 service 容器，pgvector 用例必须真跑；本地连不上时它们会 skip，
        # 而"skip"与"通过"在日志里长得几乎一样 —— 所以这里直接判失败。
        raise JobFailed(
            f"Postgres 不可达（{why}）。ci.yml 里 quality job 有 Postgres service，"
            "那族用例在 CI 里必须真跑，本地连不上就不算跑过。\n"
            "         起库：docker compose up -d db   （或设 KUBEDOCS_DATABASE_URL 指向你的库）"
        )

    _run(_py("-m", "pytest", "tests/unit", "-p", "no:cacheprovider", "-rs"))
    _run(_py("-m", "ruff", "check", "."))
    _run(_py("-m", "ruff", "format", "--check", "."))
    _run(_py("-m", "mypy", "src", "scripts"))
    _run(_py("scripts/check_doc_refs.py"))
    # 依赖声明一致性：`src/` 直接 import 的第三方包必须都写进 pyproject。
    # 实测撞过两次（`jieba` 缺声明 ⇒ 容器 `ingest --domain clinic` 挂；
    # `system.md` 没进 wheel ⇒ 容器 `/consult` 挂），
    # 两者都是"本机 editable 开发永远看不见、真装才现形"的类型。
    _run(_py("scripts/check_deps.py"))


def job_pipeline(*, hf_offline: bool) -> None:
    """全链路自检：全量语料 + 哈希向量（对应 ci.yml 的 pipeline job）。

    索引建到**临时目录**：CI 是干净检出所以无所谓，本地那份 `data/index/<domain>`
    是 35 分钟才能重建的资产，绝不能被 hashing 索引覆盖掉。
    """
    del hf_offline  # hashing 不加载模型，与离线无关
    _banner("job: pipeline —— 全量语料 + 哈希向量（不下载模型、不碰真实索引）")

    corpus = REPO / "data" / "raw"
    if not corpus.is_dir():
        raise JobFailed(f"缺少语料目录 {corpus}；先跑 python scripts/fetch_corpus.py")

    env = {"KUBEDOCS_EMBEDDING_PROVIDER": "hashing"}
    tmp = Path(tempfile.mkdtemp(prefix="kd-ci-pipeline-"))
    print(f"  临时索引目录：{tmp}（跑完即删，真实索引不动）", flush=True)
    try:
        _run(_py("-m", "kubedocs.cli", "--index-dir", str(tmp), "ingest"), env=env)
        _run(_py("-m", "kubedocs.cli", "--index-dir", str(tmp), "stats", "--verify"), env=env)
        _run(_py("scripts/check_gold.py"), env=env)
        # **故意不带 --gate**：hashing 的模型名与冻结基线（bge-small-en-v1.5）不可比，
        # 闸门会正确地拒绝比较并退 2。这里只用退出码验证链路通。
        # 另外：pyproject 的 addopts 已含 -q，**不要再叠 -q**（会吞掉 "N passed" 汇总行）。
        _run(_py("-m", "kubedocs.cli", "--index-dir", str(tmp), "eval", "--top-k", "10"), env=env)
    finally:
        if os.environ.get("KUBEDOCS_CI_KEEP_TEMP") == "1":
            print(f"  [保留] 临时索引仍在：{tmp}", flush=True)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def job_gate(*, hf_offline: bool) -> None:
    """检索质量闸门（对应 ci.yml 的 gate job）。

    CI 里这个 job 会（缓存未命中时）拉语料 + 用真实模型重建索引；本地已经有真实索引，
    所以只跑"预检 + 闸门"这两步。**只读**，不重建。
    """
    _banner("job: gate —— 真实模型 + 冻结基线（指标退化就红）")
    flag = ["--hf-offline"] if hf_offline else []
    _run(_py("scripts/check_gold.py"))
    _run(_py("-m", "kubedocs.cli", *flag, "eval", "--top-k", "10", "--gate"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--jobs",
        default=",".join(JOBS),
        help=f"要跑的 job，逗号分隔（默认全部：{','.join(JOBS)}）",
    )
    parser.add_argument(
        "--no-hf-offline",
        action="store_true",
        help="不传 --hf-offline（CI 的 runner 有外网，本地默认要传，理由见 README「环境约束」）",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留 pipeline job 的临时索引目录（排查失败时有用）",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="某个 job 失败后继续跑后面的（CI 里 pipeline/gate 有 needs: [quality]，默认与 CI 一致：不继续）",
    )
    args = parser.parse_args(argv)

    wanted = [j.strip() for j in args.jobs.split(",") if j.strip()]
    unknown = [j for j in wanted if j not in JOBS]
    if unknown:
        parser.error(f"未知 job：{unknown}；可选：{list(JOBS)}")

    if args.keep_temp:
        os.environ["KUBEDOCS_CI_KEEP_TEMP"] = "1"

    _check_environment()

    finished: list[str] = []
    failed: dict[str, str] = {}
    for name in wanted:
        runner = {"quality": job_quality, "pipeline": job_pipeline, "gate": job_gate}[name]
        try:
            runner(hf_offline=not args.no_hf_offline)
        except JobFailed as exc:
            failed[name] = str(exc)
            _banner(f"job {name} 失败")
            print(f"  {exc}", flush=True)
            # ci.yml 里 pipeline 与 gate 都声明了 `needs: [quality]`，所以 quality 挂掉时
            # 它们在 CI 里**根本不会启动**。默认照此停止，否则"本地跑出来的结论"与 CI
            # 对不上——而这个脚本的全部价值就在于忠实。想看全貌再加 --keep-going。
            if not args.keep_going:
                print("\n  按 CI 语义（needs: [quality]）停止；继续跑后面用 --keep-going", flush=True)
                break
        else:
            finished.append(name)

    _banner("汇总")
    for name in wanted:
        if name in failed:
            print(f"  ✗ {name} —— {failed[name]}", flush=True)
        elif name in finished:
            print(f"  ✓ {name}", flush=True)
        else:
            print(f"  – {name}（未运行：前一个 job 失败，CI 里也不会启动）", flush=True)

    ran = len(finished) + len(failed)
    if failed:
        print(
            f"\n已运行的 {ran} 个 job 里有 {len(failed)} 个失败。"
            "\n注意：这**不等于** GitHub runner 上的结论（差异见本文件顶部那张表）。",
            flush=True,
        )
        return 1
    print(
        f"\n{ran}/{ran} 个 job 通过。"
        "\n注意：这**不等于** GitHub runner 上的结论 —— 本脚本不重新 checkout、不建环境、不缓存，"
        "也不覆盖 CI 的矩阵与 service 容器语义。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
