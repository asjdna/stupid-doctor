#!/usr/bin/env python
"""生成 `requirements.lock.txt`：从 `pyproject.toml` 的依赖声明出发做**依赖闭包**遍历。

用法::

    python scripts/gen_lock.py            # 重新生成 requirements.lock.txt
    python scripts/gen_lock.py --check    # 只比对漂移，不写文件（有漂移 → 退出码 1）
    python scripts/gen_lock.py -o /tmp/x.txt
    python scripts/gen_lock.py --strict   # 有 optional 组没装时直接失败（而不是跳过）

为什么需要这个脚本（而不是人工维护锁文件）
------------------------------------------
`requirements.lock.txt` 是"实测跑通全量测试的那个环境"的冻结快照。dependabot 只看得懂
`pyproject.toml`、**不维护锁文件**，于是依赖升级会**静默漂移**：`pyproject.toml` 里写
`fastapi>=0.115`，锁文件却还钉着旧版本，没有任何东西会报警。把它做成仓库内可重复执行的
脚本，漂移就变成一条随时能跑的命令（`make lock` / `--check`）。

算法（与"声明的唯一权威是 pyproject"这一点对齐）
------------------------------------------------
1. 读 `pyproject.toml` 的 `[project].dependencies`（运行时）与
   `[project.optional-dependencies]` 各组（本项目是 `dev` / `eval`）作为**根**；
2. 每个根用 `importlib.metadata` 找**当前解释器里已装**的发行版，并校验版本满足声明
   （不满足就报错退出——那说明这个环境并不是"实测通过"的那套，硬写出来的锁文件是假的）；
3. 读它的 `Requires-Dist`，按 PEP 508 marker 求值（`extras` 逐组展开，例如
   `uvicorn[standard]` 的 watchfiles / httptools / websockets），递归遍历整棵依赖树；
4. 输出 `name==version`：PEP 503 归一化去重、按名字排序。

optional 组的处理（**不假装它装过**）
-------------------------------------
某个 optional 组的直接依赖在当前解释器里没装时，**跳过该组**并把组名与缺哪些包写进
生成的 header（同时打一条 stderr 警告）。理由：本脚本是"记录器"而不是"求解器"，
凭空写出没装过的包和版本等于伪造锁文件。加了 `--strict` 则跳过变成错误，
供"要求全组齐备"的场合（例如发版前）使用。

已知边界（如实写在这里，而不是假装它是 pip-compile）
----------------------------------------------------
* **不做版本求解**：版本取自当前环境。它复现的是"这台机器上跑通过"的那套版本，
  而锁文件要的正是这个。换环境时先 `python -m pip install -e ".[dev]"` 再跑本脚本。
* **不认** `-r` / `-e` 之类的 pip 指令（`pyproject.toml` 的依赖声明里也不该有）。
* **不写哈希**：本项目没有需要校验包来源的场景，而哈希会让每次 diff 噪声极大。
* 闭包**依赖解释器版本**（marker 求值，例如 `tomli` / `exceptiongroup` 只在 <3.11 装；
  `uvloop` 在 Windows 上不进闭包）：header 里记录了生成时的 `major.minor`，
  换大版本重跑即可。
* 闭包也**依赖平台**（`uvicorn[standard]` 的 `uvloop` 带 `sys_platform != "win32"`），
  所以 Windows 上生成的锁文件与 Linux CI 上生成的**不会逐字相同**。这一点必须知道：
  在 Windows 生成、拿到 Linux 装，得到的是"少了 uvloop"的环境。
"""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata as md
import pathlib
import platform
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # 3.10：标准库还没有 tomllib（本项目 requires-python = ">=3.10"）
    import tomli as tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_OUTPUT = REPO_ROOT / "requirements.lock.txt"

#: 生成的 header。**不含任何本机绝对路径**（读者会照抄这些命令）。
#: `make lock` 这句必须留在显眼处：锁文件的漂移只能靠"改完依赖就重跑"来防。
HEADER = """\
# ===========================================================================
# requirements.lock.txt —— 精确钉版本（锁文件），**不是**运行时依赖的声明处。
# ===========================================================================
#
# ⚠ 本文件由 `scripts/gen_lock.py` **自动生成，不要手改**。
#   改了 pyproject.toml 的依赖之后**必须重跑**：
#       make lock                    （没有 make 就 python scripts/gen_lock.py）
#   只检查有没有漂移、不写文件：
#       python scripts/gen_lock.py --check
#   为什么必须重跑：dependabot 只看得懂 pyproject.toml，**不维护本文件**；
#   不重跑的话依赖升级会在这里静默漂移（声明写 >=，锁文件钉着旧版本，没人会发现）。
#
# 用途
#   把一个"实测跑通全量测试"的环境原样复现出来。
#   声明依赖的**唯一权威**是 pyproject.toml：
#       [project].dependencies                 运行时
#       [project].optional-dependencies.dev    开发 / 测试 / 静态门禁
#       [project].optional-dependencies.eval   行业口径评测（ragas / deepeval）
#   本文件只是那份声明的**冻结快照**，供复现排查使用。
#
# 生成方式（由 scripts/gen_lock.py 实施，四条与该脚本的 1-4 步一一对应）
#   1) 以 pyproject 声明的运行时依赖 + 各 optional 组为根；
#   2) 用 importlib.metadata 读**当前解释器**里已装的发行版，并校验版本满足声明；
#   3) 读 Requires-Dist，按 PEP 508 marker 求值（含 extras，例如 uvicorn[standard] 的
#      watchfiles / httptools / websockets），递归遍历整棵依赖树；
#   4) 取精确版本，PEP 503 归一化去重后按名字排序输出。
#   pip freeze 的全量输出有 380+ 条（含大量与本项目无关的包），本文件只保留属于本项目
#   闭包的那些条目。版本**取自当前环境**：本脚本是"记录器"而不是 pip-compile 那样的
#   "求解器"，它复现的是"这台机器上实测通过"的那套版本，这正是锁文件要的东西。
#
# 本文件不含任何 API key / 凭据；.env 未被读取，其值不在导出范围内。
#
# 解释器与平台（两者都会改变闭包，换掉任一就要重跑）
#   生成解释器：Python {py_major_minor}
#     marker 求值随解释器版本变化（例如 tomli / exceptiongroup 只在 <3.11 进闭包）。
#   生成平台：{platform_tag}
#     `uvicorn[standard]` 的 uvloop 带 `sys_platform != "win32"`，所以本文件在 Windows
#     与 Linux 上**不会逐字相同**；拿它在另一个平台安装会得到与该平台匹配的那一套。
{groups_note}#
# 使用
#   python -m pip install -r requirements.lock.txt
# ===========================================================================
"""

GROUPS_NOTE_NONE = """\
# 覆盖的 optional 组：全部（本次生成时每组都装齐了）
"""


def _read_pyproject(path: pathlib.Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict):
        raise SystemExit(f"[gen_lock] {path} 里没有 [project] 表，无法确定依赖声明")
    return project


def _parse(text: str) -> Requirement:
    try:
        return Requirement(text)
    except InvalidRequirement as exc:
        raise SystemExit(f"[gen_lock] pyproject 里的依赖写法无法解析：{text!r}（{exc}）") from exc


def _applies(req: Requirement, extras: set[str]) -> bool:
    """这条 Requires-Dist 在当前 extras 组合下是否需要（PEP 508 marker 求值）。

    基础依赖的 marker 里 `extra` 是空串；`extra == "standard"` 这类只在
    对应 extra 被请求时才成立。marker 为 None 表示无条件依赖。
    """
    if req.marker is None:
        return True
    if req.marker.evaluate({"extra": ""}):
        return True
    return any(req.marker.evaluate({"extra": extra}) for extra in extras)


def _roots(
    project: dict[str, Any], *, groups: list[str] | None = None
) -> tuple[list[Requirement], list[str], dict[str, list[str]]]:
    """把 pyproject 声明的依赖解析成根 requirement。

    返回 `(根列表, 被跳过的组名, {组名: 缺哪些包})`。跳过只发生在"该组的直接依赖
    没装"时，且组名与缺失清单会写进锁文件 header——**不静默**。
    """
    raw: list[str] = list(project.get("dependencies") or [])
    roots: list[Requirement] = []
    for text in raw:
        req = _parse(text)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            # 声明自带 marker（例如仅 3.11+）：在当前解释器上不适用就不进闭包。
            continue
        roots.append(req)

    optional = project.get("optional-dependencies") or {}
    wanted = list(optional) if groups is None else groups
    unknown = [g for g in wanted if g not in optional]
    if unknown:
        raise SystemExit(
            f"[gen_lock] pyproject 里没有这些 optional 组：{', '.join(unknown)}；"
            f"可选：{', '.join(optional) or '（无）'}"
        )

    skipped: list[str] = []
    missing: dict[str, list[str]] = {}
    for group in wanted:
        parsed = [_parse(text) for text in optional[group]]
        absent = [r.name for r in parsed if not _installed(r.name)]
        if absent:
            # 这一组在当前解释器里不完整：跳过整组，但把缺什么记下来。
            # 部分收进闭包会更糟——闭包遍历要求根存在，缺根时只能改写声明，那是伪造。
            skipped.append(group)
            missing[group] = absent
            continue
        roots.extend(parsed)
    return roots, skipped, missing


def _installed(name: str) -> bool:
    try:
        md.distribution(name)
    except md.PackageNotFoundError:
        return False
    return True


def _closure(roots: list[Requirement]) -> dict[str, str]:
    """遍历依赖闭包，返回 {PEP 503 归一化名: 已装版本}。"""
    pins: dict[str, str] = {}
    # name -> 已经展开过的 extras 集合（extras 是逐组累加的：后到的 [standard] 也要展开）
    expanded: dict[str, set[str]] = {}
    # 键一律用 `str(...)` 收窄：packaging 的 canonicalize_name 返回 NormalizedName
    # （NewType of str），直接用它当键会让 mypy 认为 dict.get(str) 不匹配。
    queue: list[tuple[str, set[str], str]] = [
        (str(canonicalize_name(req.name)), set(req.extras), str(req)) for req in roots
    ]
    rooted: dict[str, Requirement] = {str(canonicalize_name(req.name)): req for req in roots}

    while queue:
        name, extras, origin = queue.pop()
        already = expanded.setdefault(name, set())
        if name in pins and extras <= already:
            continue
        expanded[name] = already | extras
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError as exc:
            raise SystemExit(
                f"[gen_lock] 依赖闭包里的 {name!r} 没有装（来自 {origin!r}）。\n"
                f'           先 `python -m pip install -e ".[dev]"` 再重跑本脚本。'
            ) from exc
        pins[name] = dist.version

        root_req = rooted.get(name)
        if (
            root_req is not None
            and root_req.specifier
            and not root_req.specifier.contains(dist.version, prereleases=True)
        ):
            # 声明的下界都没满足 → 这个环境不是"实测通过"的那套，别拿它生成锁文件。
            raise SystemExit(
                f"[gen_lock] 已装的 {name}=={dist.version} 不满足 pyproject 的声明 "
                f"{root_req.specifier}。先升级依赖，或先把 pyproject 的声明改对。"
            )

        for raw in dist.requires or []:
            try:
                req = Requirement(raw)
            except InvalidRequirement as exc:  # 第三方包的元数据坏了：不猜，直接报出来
                raise SystemExit(
                    f"[gen_lock] {name}=={dist.version} 的 Requires-Dist 无法解析：{raw!r}"
                ) from exc
            if _applies(req, extras):
                queue.append((str(canonicalize_name(req.name)), set(req.extras), raw))
    return pins


def _groups_note(skipped: list[str], missing: dict[str, list[str]]) -> str:
    """生成 header 里"覆盖了哪些 optional 组"那几行（**不隐瞒没装的组**）。"""
    if not skipped:
        return GROUPS_NOTE_NONE
    lines = ["# 覆盖的 optional 组：已装齐的那些组 + 运行时依赖；**下列组被跳过**（本环境未安装）：\n"]
    for group in skipped:
        lines.append(
            f"#   · {group}：缺 {', '.join(missing[group])}（本脚本不凭空写没装过的包——装齐后重跑即可纳入）\n"
        )
    return "".join(lines)


def render(
    pins: dict[str, str],
    *,
    skipped: list[str] | None = None,
    missing: dict[str, list[str]] | None = None,
    py_major_minor: str | None = None,
) -> str:
    """把闭包渲染成锁文件全文（LF 结尾，末尾一个换行）。"""
    version = py_major_minor or f"{sys.version_info.major}.{sys.version_info.minor}"
    tag = f"{platform.system()} {platform.machine()}"
    header = (
        HEADER.replace("{py_major_minor}", version)
        .replace("{platform_tag}", tag)
        .replace("{groups_note}", _groups_note(skipped or [], missing or {}))
    )
    lines = [header, ""]
    lines.extend(f"{name}=={pins[name]}" for name in sorted(pins))
    return "\n".join(lines) + "\n"


def _pins_of(text: str) -> dict[str, str]:
    """从锁文件全文里取出 `name==version` 行（忽略注释与空行）。"""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, _, ver = line.partition("==")
            pins[str(canonicalize_name(name))] = ver.strip()
    return pins


def _report_drift(expected: str, actual: str) -> None:
    """打印漂移的**可读**摘要（只列包差异，不打印整份 header diff）。"""
    exp, act = _pins_of(expected), _pins_of(actual)
    added = {k: v for k, v in exp.items() if act.get(k) != v}
    removed = {k: v for k, v in act.items() if exp.get(k) != v}
    for name, ver in sorted(added.items()):
        old = f"（原 {act[name]}）" if name in act else ""
        print(f"  + {name}=={ver}{old}")
    for name, ver in sorted(removed.items()):
        print(f"  - {name}=={ver}")
    if not added and not removed:
        print("  （包集合一致，差异只在 header 文字）")
        for line in difflib.unified_diff(
            actual.splitlines(), expected.splitlines(), "现有文件", "生成结果", lineterm="", n=1
        ):
            print(f"  {line}")


def _warn_skipped(skipped: list[str], missing: dict[str, list[str]]) -> None:
    for group in skipped:
        print(
            f"[gen_lock] 警告：optional 组 {group!r} 在本环境未安装"
            f"（{', '.join(missing[group])}），已跳过、未写入锁文件。"
            f"要让锁文件覆盖它：先安装该组再重跑。",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 requirements.lock.txt（依赖闭包冻结）")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pyproject", type=pathlib.Path, default=DEFAULT_PYPROJECT)
    parser.add_argument(
        "--groups", default="", help="只取这些 optional 组（逗号分隔）；默认取 pyproject 里全部"
    )
    parser.add_argument(
        "--strict", action="store_true", help="有 optional 组没装时直接失败（默认是跳过并在 header 里注明）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只比对现有锁文件是否有漂移（不写文件）；有漂移则退出码 1",
    )
    args = parser.parse_args(argv)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    roots, skipped, missing = _roots(_read_pyproject(args.pyproject), groups=groups)
    if skipped and args.strict:
        raise SystemExit(
            f"[gen_lock] --strict：optional 组 {', '.join(skipped)} 未安装齐全，拒绝生成不完整的锁文件"
        )
    _warn_skipped(skipped, missing)

    pins = _closure(roots)
    text = render(pins, skipped=skipped, missing=missing)
    output: pathlib.Path = args.output

    if args.check:
        if not output.exists():
            print(f"[gen_lock] {output} 不存在：跑 `make lock` 生成它", file=sys.stderr)
            return 1
        actual = output.read_text(encoding="utf-8")
        if actual == text:
            print(f"[gen_lock] 无漂移：{output.name} 与 pyproject 的依赖闭包一致（{len(pins)} 条）")
            return 0
        print(f"[gen_lock] 有漂移：{output.name} 与 pyproject 的依赖闭包不一致。")
        _report_drift(text, actual)
        print("[gen_lock] 跑 `make lock` 重跑生成（改了依赖就必须做这一步）。", file=sys.stderr)
        return 1

    # newline="\n"：Windows 上默认会被翻译成 CRLF，而 .gitattributes 要求 LF。
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"[gen_lock] 已写入 {output}（{len(pins)} 条，Python {platform.python_version()}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
