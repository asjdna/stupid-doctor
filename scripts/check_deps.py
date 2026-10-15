"""检查：`src/` 里直接 import 的第三方包，是否都写进了 `pyproject.toml`。

## 为什么需要这个脚本（两次实测撞出来的）

有一类缺陷**在本机开发环境永远看不见**：包是通过传递依赖凑巧装上的，
而 `pyproject.toml` 里没有声明。editable 安装 + 现成的 venv 一切正常，
`pip install .` 到干净环境（容器就是）才炸：

| 撞到的时间 | 缺的东西 | 后果 |
|---|---|---|
| 2026-10-15 重建镜像后 | `jieba` | 容器 `ingest --domain clinic` → `ModuleNotFoundError`。**没有任何已声明依赖会带上它** |
| 同日 | `system.md`（非 `.py`） | 容器 `/consult` → `FileNotFoundError`；那条由 `tests/unit/test_clinical.py::test_prompt_is_declared_as_package_data` 挡 |

所以把"import 与声明是否一致"做成一道**可复现的检查**，接在 CI 的 quality job 里。
它比"下次重建镜像时再发现"便宜得多。

## 判据

- 只看 `src/`（`scripts/`、`tests/` 的依赖允许更松：它们是开发期工具）
- 跳过标准库、`kubedocs` 自身、`src/` 下的相对导入
- import 名到分发包名的差异由 `ALIAS` 表补上（`sqlite_vec` → `sqlite-vec` 这类）
- **可选依赖不算缺**：`pyproject` 的 `[project.optional-dependencies]` 里声明过的
  也算已声明（例如 Milvus 后端的 `pymilvus`），因为它是刻意不装进默认路线的

用法::

    python scripts/check_deps.py            # 有缺就退出 1
    python scripts/check_deps.py --verbose  # 打印完整 import 清单
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PYPROJECT = REPO / "pyproject.toml"

ALIAS = {
    "sqlite_vec": "sqlite-vec",
    "sentence_transformers": "sentence-transformers",
    "prometheus_client": "prometheus-client",
    "opentelemetry": "opentelemetry-sdk",
    "sse_starlette": "sse-starlette",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
    "PIL": "pillow",
}
"""import 名 ≠ 分发包名时的映射。**只用确定知道的**——猜错会让检查假绿/假红。"""

EXTRA_PROVIDED = {
    "psycopg_pool": "psycopg[pool]",
    "psycopg_binary": "psycopg[binary]",
}
"""由**某个已声明依赖的 extra** 提供的 import。

显式列表而不是"自动展开 extra"：`pkg[extra]` 拉进来的分发包名没有通用规律
（`psycopg[pool]` 给的是 `psycopg-pool`，但别的包未必同名），自动推断会引入假绿。
所以这里逐条写清"这个 import 是靠哪个 extra 满足的"，一眼可核对。"""


def _norm(name: str) -> str:
    """PEP 503 归一化：`Python_DotEnv` / `python.dotenv` 都是 `python-dotenv`。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> tuple[set[str], set[str]]:
    """返回 (必装依赖, 可选依赖)。**逐行解析而不是 tomllib**：3.10 没有 tomllib，
    而引入 tomli 只为这个脚本不值当（它自己会变成又一个未声明依赖）。"""
    text = PYPROJECT.read_text(encoding="utf-8")

    def block(header: str) -> set[str]:
        out: set[str] = set()
        if header not in text:
            return out
        body = text.split(header, 1)[1]
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("") and not stripped:
                continue
            if stripped.startswith("[") and "dependencies" not in stripped:
                # 进了下一个顶层小节（`[tool.*]` 等），本块结束
                break
            if stripped.startswith("["):
                continue
            m = re.match(r'"([A-Za-z0-9_.\-]+)', stripped)
            if m:
                out.add(_norm(m.group(1)))
        return out

    required = block("dependencies = [")
    optional: set[str] = set()
    if "[project.optional-dependencies]" in text:
        for line in text.split("[project.optional-dependencies]", 1)[1].splitlines():
            m = re.match(r'\s*"([A-Za-z0-9_.\-]+)', line)
            if m:
                optional.add(_norm(m.group(1)))
            elif line.strip().startswith("[") and "dependencies" not in line:
                break
    return required, optional


def _imports() -> dict[str, set[str]]:
    """`{顶层模块名: {用到它的文件}}`，只统计第三方。"""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:  # pragma: no cover - 语法错该由 ruff/pytest 报
            print(f"[!] 解析失败：{path}（{e}）")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in stdlib or name == "kubedocs" or name.startswith("_"):
                    continue
                found.setdefault(name, set()).add(str(path.relative_to(REPO)))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="打印全部第三方 import")
    args = ap.parse_args(argv)

    required, optional = _declared()
    imports = _imports()
    if args.verbose:
        print(f"必装依赖 {len(required)} 个；可选 {len(optional)} 个")
        for name in sorted(imports):
            print(f"  {name:<24} {sorted(imports[name])}")

    missing: list[tuple[str, str, list[str]]] = []
    for name in sorted(imports):
        if name in EXTRA_PROVIDED:
            continue
        pkg = _norm(ALIAS.get(name, name))
        if pkg in required or pkg in optional:
            continue
        missing.append((name, pkg, sorted(imports[name])))

    if not missing:
        print(f"依赖声明一致：src/ 用到的 {len(imports)} 个第三方 import 都已声明")
        return 0

    print("[x] 下面这些包被 src/ 直接 import，但 pyproject 里没有声明：")
    for name, pkg, where in missing:
        print(f"    {name:<20} -> 包名应为 {pkg:<20} 用在 {where[:3]}")
    print(
        "\n为什么必须修：靠传递依赖凑巧可用 = 换到干净环境（容器 / CI）就 ModuleNotFoundError，"
        "而本机 editable 开发永远看不见。要么加进 [project] dependencies，"
        "要么明确放进 [project.optional-dependencies] 的某个 extra。"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
