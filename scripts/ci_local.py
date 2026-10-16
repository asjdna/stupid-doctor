"""跑完整套验证：quality（静态检查 + 单元测试）· pipeline（全量语料 + 哈希向量自检）·
gate（真实模型 + 冻结基线）。

`make check` 只是 quality 里的静态检查与测试；pipeline 与 gate 是它之外的两道硬闸门。
"`make check` 绿了"不等于全套通过 —— 要一个可对外引用的结论就跑这个脚本。

| job | 覆盖 | 前置 |
|---|---|---|
| quality | pytest（含需要真实 Postgres 的用例）· ruff check · ruff format --check · mypy · 文档锚点校验 · 临床治理一致性 · 依赖声明校验 | Postgres 可达 |
| pipeline | 全量语料摄取 + 索引自检 + 冻结标注集自检（哈希向量，不下载模型） | `data/raw/` 语料 |
| gate | 真实嵌入模型 + 冻结基线的检索闸门 | 真实索引 `data/index/` 与模型缓存 |

前置条件说明：

- Postgres 自己提供：`docker compose up -d db`，或设 `KUBEDOCS_DATABASE_URL` 指向你的库。
  连不上时 pgvector 那族用例会 skip，而 skip 不等于跑过，所以脚本直接判失败。
- `--hf-offline` 默认加上：模型已缓存时离线加载，避免联网校验长时间挂起。
- 用当前解释器与现成的 `data/index` / HF 缓存，不重建环境；未跟踪文件会参与本次运行。

副作用隔离：`pipeline` job 用 `hashing` 建索引，建到临时目录、跑完即删，绝不覆盖
`data/index/<domain>` 那份真实索引；`gate` job 只读真实索引，不重建。

用法：
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
"""仓库根（本文件上溯两层）。"""

JOBS = ("quality", "pipeline", "gate")

DEFAULT_DATABASE_URL = "postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs"
"""quality job 的默认连接串：与 `Settings.database_url` 的默认值一致。"""


class JobFailed(RuntimeError):
    """某个 job 里有一条命令返回非零。"""


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path = REPO) -> None:
    """跑一条命令，输出透传到终端；非零退出码抛 `JobFailed`。"""
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    started = time.perf_counter()
    merged = {**os.environ, **(env or {})}
    # 命令全部由本文件构造（`sys.executable` + 字面量参数 + 自建临时目录），无外部输入参与
    # 拼接，`--jobs` 也按 `JOBS` 白名单校验过，S603 属误报。抑制标记必须挂在
    # `subprocess.run(` 这一行：诊断落在调用起始行，`ruff format` 会把参数拆到后续行。
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
    """先确认跑的是本仓库的 venv，否则后续失败信息很难读。"""
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
    """判断 quality job 依赖的 Postgres 能不能连上（要自己起：`docker compose up -d db`）。

    只管"能不能连"：连得上不等于 pgvector 那族用例真跑了，那由 pytest 的 `-rs` 输出说话。
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
    """静态检查 + 单元测试。"""
    del hf_offline  # quality job 不加载模型，与离线无关
    _banner("job: quality —— 静态检查 + 单元测试（需要真实 Postgres）")

    dsn = os.environ.get("KUBEDOCS_DATABASE_URL", DEFAULT_DATABASE_URL)
    reachable, why = _postgres_reachable(dsn)
    if reachable:
        print(f"  Postgres 可达：{dsn}")
    else:
        # pgvector 那族用例必须真跑；连不上时它们会 skip，而 skip 与通过在日志里几乎一样。
        raise JobFailed(
            f"Postgres 不可达（{why}）。那族用例连不上库就不算跑过。\n"
            "         起库：docker compose up -d db   （或设 KUBEDOCS_DATABASE_URL 指向你的库）"
        )

    _run(_py("-m", "pytest", "tests/unit", "-p", "no:cacheprovider", "-rs"))
    _run(_py("-m", "ruff", "check", "."))
    _run(_py("-m", "ruff", "format", "--check", "."))
    _run(_py("-m", "mypy", "src", "scripts"))
    _run(_py("scripts/check_doc_refs.py"))
    # 临床治理：安全资产清单 ↔ 代码里的不变量 ↔ 测试覆盖三方必须一致。
    # **这是唯一按"内容"卡的那道门**——指标（召回/误报/分诊）只报告不卡退出码，
    # 因为拿自建小样本的分数当门限，只会逼着后来者去改门限。
    _run(_py("scripts/check_governance.py"))
    # 依赖声明一致性：`src/` 直接 import 的第三方包必须都写进 pyproject；缺声明只在
    # 干净环境（容器）里才现形，本机 editable 开发看不见。
    _run(_py("scripts/check_deps.py"))


def job_pipeline(*, hf_offline: bool) -> None:
    """全链路自检：全量语料 + 哈希向量。

    索引建到临时目录：本地那份 `data/index/<domain>` 重建要很久，不能被 hashing 覆盖。

    **必须显式钉住本地后端**：`--index-dir` 只对本地后端有意义（pgvector 的索引在
    Postgres 的 schema 里），而默认后端是 pgvector。不钉住的话这一 job 会往真实
    Postgres 里灌 hashing 的 32 维向量——既污染生产 schema，又根本没在测隔离性。
    """
    del hf_offline  # hashing 不加载模型，与离线无关
    _banner("job: pipeline —— 全量语料 + 哈希向量（不下载模型、不碰真实索引）")

    corpus = REPO / "data" / "raw"
    if not corpus.is_dir():
        raise JobFailed(f"缺少语料目录 {corpus}；先跑 python scripts/fetch_corpus.py")

    env = {"KUBEDOCS_EMBEDDING_PROVIDER": "hashing", "KUBEDOCS_STORE": "sqlite-vec"}
    tmp = Path(tempfile.mkdtemp(prefix="kd-ci-pipeline-"))
    print(f"  临时索引目录：{tmp}（跑完即删，真实索引不动）", flush=True)
    try:
        _run(_py("-m", "kubedocs.cli", "--index-dir", str(tmp), "ingest"), env=env)
        _run(_py("-m", "kubedocs.cli", "--index-dir", str(tmp), "stats", "--verify"), env=env)
        _run(_py("scripts/check_gold.py"), env=env)
        # 故意不带 --gate：hashing 的模型名与冻结基线不可比，闸门会正确地拒绝比较并退 2，
        # 这里只用退出码验证链路通。另：pyproject 的 addopts 已含 -q，不要再叠。
        _run(_py("-m", "kubedocs.cli", "--index-dir", str(tmp), "eval", "--top-k", "10"), env=env)
    finally:
        if os.environ.get("KUBEDOCS_CI_KEEP_TEMP") == "1":
            print(f"  [保留] 临时索引仍在：{tmp}", flush=True)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def job_gate(*, hf_offline: bool) -> None:
    """检索质量闸门。

    真实索引已存在，所以只跑"预检 + 闸门"两步。**只读**，不重建索引。

    **按配置的后端读索引，不钉本地后端**：基线是按**某一份具体索引**冻结的
    （`corpus_sha` 把"文档 + 每篇块数"连同嵌入模型一起钉进基线），所以闸门必须读基线
    对应的那一份。产品默认跑在 pgvector 上，闸门就该守那一份——钉成 `sqlite-vec` 就变成
    "守着一份产品不用的索引"，指标再漂亮也说明不了线上质量。

    代价：这一 job 需要配置里的向量库可达（默认是 Postgres）。数据库不可达时它会在预检
    阶段**响亮失败并说明原因**，而不是静默换一份索引去比——那会让基线彻底失去意义。
    想在完全离线（无数据库）的机器上守质量，就先用 `KUBEDOCS_STORE=sqlite-vec` 重建本地
    索引、再按它冻结基线（`eval --write-baseline`）。
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
        help="不传 --hf-offline（模型要联网校验时才用；默认传，理由见 README「环境约束」）",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留 pipeline job 的临时索引目录（排查失败时有用）",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="某个 job 失败后继续跑后面的（默认停：pipeline 与 gate 都以 quality 为前提）",
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
            # pipeline 与 gate 都以 quality 为前提：静态检查或单测就挂了，后面的结论没有意义。
            # 默认照此停止，--keep-going 可继续跑完。
            if not args.keep_going:
                print("\n  quality 是后续 job 的前提，停止；继续跑后面用 --keep-going", flush=True)
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
            print(f"  – {name}（未运行：前一个 job 失败）", flush=True)

    ran = len(finished) + len(failed)
    if failed:
        print(f"\n已运行的 {ran} 个 job 里有 {len(failed)} 个失败。", flush=True)
        return 1
    print(
        f"\n{ran}/{ran} 个 job 通过。"
        "\n注意：结论只对**这次运行的环境**成立 —— 用的是当前工作树、当前解释器、现成的索引与模型缓存，"
        "不重新检出、不重建环境；换了环境（干净检出、别的机器、别的 Postgres）要重跑。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
