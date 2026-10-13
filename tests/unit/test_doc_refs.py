"""`scripts/check_doc_refs.py` 的契约测试：**证明这个门禁真的会咬人**。

## 为什么必须有反例

一个只能"打印通过"的检查器等于没有检查器。本文件里最有价值的是
`TestWrongContentIsCaught`：它在 `tmp_path` 里造一份**锚点指向错误内容**的文档
（散文说 `run_tool_loop()`，锚点却指到另一个函数的行），检查器必须**非零退出**。
如果哪天有人把内容判据删掉、退化成"只看行号越界"，这条会立刻变红。

反例全部在 `tmp_path` 里造临时文档与临时目标文件，**不改真实文档**——
为了造反例去改 SECURITY.md，最后一定会把假锚点提交进去。
"真实文档现在全通过" 由 `TestCurrentDocs` 覆盖（只读，不写）。

## 覆盖的判据

- 正例：当前仓库的全部引用通过，白名单条目不报错；
- 反例：锚点指向错误内容（有候选标识符，但一个都没命中）；
- 反例：一句里既有普通词又有强候选时，只认强候选；
- 反例：目标文件不存在、行号越界、缩写路径有歧义；
- 反例：散文里没有可校验的标识符、又不在白名单里（不许静默跳过）；
- 反例：白名单钉住的 `expect` 内容漂走了（白名单不是免死金牌）；
- 功能：无文件名的续写引用沿用同段上文的目标文件；没上文时算失败。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from kubedocs.config import repo_root

SCRIPT = repo_root() / "scripts" / "check_doc_refs.py"

# 本文件自己也落在检查器的扫描面里（`tests/**/*.py`）。所以文档里那种
# `文件:行号` 引用必须**在运行时拼出来**，不能在源码里写成字面量——
# 否则这些测试夹具会先被检查器抓到（实测踩过：13 处"目标文件不存在"）。
_TICK = "`"


def _ref(target: str, spec: str) -> str:
    """拼出一个反引号引用（target 加冒号加行号，例如 src/svc/alpha.py 的 10-11 行）。"""
    return f"{_TICK}{target}:{spec}{_TICK}"


def _load_script() -> ModuleType:
    """按路径加载脚本模块（`scripts/` 不是包，没有 `__init__.py`）。

    ⚠️ 必须先把模块登记进 `sys.modules` 再 `exec_module`：脚本里有
    `@dataclass` 与 `from __future__ import annotations`，dataclasses 解析字符串注解时
    会去 `sys.modules[cls.__module__]` 里查；不登记就是 `None.__dict__` 的
    AttributeError（实测踩过）。
    """
    spec = importlib.util.spec_from_file_location("_check_doc_refs_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


def _write(root: Path, rel: str, text: str) -> Path:
    """写文件并**钉死 LF**：开发机是 Windows，默认会把 `\\n` 变成 CRLF。"""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _run(script: ModuleType, root: Path, *extra: str) -> int:
    """走真实的 `main()`（退出码才是这个门禁对 CI 的唯一契约）。"""
    return int(script.main(["--root", str(root), "--verbose", *extra]))


class TestCurrentDocs:
    """正例：**真实仓库**的引用现在全部站得住。"""

    def test_every_reference_passes(self, script: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
        code = _run(script, repo_root())

        out = capsys.readouterr().out
        assert code == 0, out
        assert "结果：通过" in out
        # 别让"零引用"伪装成通过：扫描面必须真的有东西
        assert "引用处数   : 0" not in out

    def test_whitelisted_entries_are_reported_not_swallowed(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """白名单必须**被打印出来**（含条数与理由），否则就是静默跳过。

        断言的是**不变量**（脚本里登记的每条 `reason` 都出现在输出里），
        而不是某个固定措辞——原先断言 `"白名单：第"`，措辞一改就变成假失败
        （实际发生过：重写白名单理由时 5 条都带了理由，这条测试却红了）。
        """
        _run(script, repo_root())

        out = capsys.readouterr().out
        assert "白名单     :" in out
        assert script.WHITELIST, "白名单不该是空的（空表会让这条测试退化成恒真）"
        for white in script.WHITELIST.values():
            assert white.reason in out, f"白名单条目没有把理由打印出来：{white.reason[:40]}"

    def test_whitelist_listing_matches_the_table(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert int(script.main(["--root", str(repo_root()), "--whitelist"])) == 0
        out = capsys.readouterr().out
        assert f"白名单 {len(script.WHITELIST)} 条" in out


class TestMissingFileRefsAreCaught:
    """反例：引用了**不存在的 `.md` 文件**必须非零退出。

    这是本检查器此前的**整类盲区**：`文件:行号` 校验要求"路径 + 冒号 + 数字"，
    所以只写文件名的引用完全不进入校验——`docs/adr/0005-milvus-后端.md` 曾经
    并不存在、12 处代码引用它，而门禁一路绿灯。下面的反例把这个能力钉住：
    如果哪天有人把文件引用检查删掉，这几条会立刻变红。
    """

    def test_dangling_docs_path_fails(self, script: ModuleType, tmp_path: Path) -> None:
        # ⚠️ 路径必须在**运行时拼**：本文件自己也落在检查器的扫描面里
        # （`tests/**/*.py`），写成字面量会让这些假引用先被真仓库的检查抓到。
        a = "docs/adr/" + "9999-nonexistent.md"
        b = "docs/adr/" + "8888-nope.md"
        _write(tmp_path, "README.md", f"# 说明\n\n见 `{a}` 与 [另一份]({b})。\n")
        assert _run(script, tmp_path) == 1

    def test_existing_docs_path_passes(self, script: ModuleType, tmp_path: Path) -> None:
        real = "docs/adr/" + "0001-real.md"
        _write(tmp_path, real, "# 真文件\n")
        _write(tmp_path, "README.md", f"# 说明\n\n见 `{real}`。\n")
        assert _run(script, tmp_path) == 0

    @pytest.mark.parametrize(
        "token",
        [
            "concepts/foo.md",  # k8s 语料里的示意路径：首段不是仓内顶层目录
            "_index.md",  # 裸文件名：文档里举例太常见
            "data/raw/other-repo.md",  # data/ 是被 gitignore 的语料，校验不了
            "docs/.../whatever.md",  # 省略号 = 示意
            "docs/*.md",  # 通配 = 示意
        ],
    )
    def test_schematic_paths_are_not_flagged(self, script: ModuleType, tmp_path: Path, token: str) -> None:
        """精度换召回：这几种"看着像路径、其实是示意/外部"的写法不报错。

        宽规则实测一次抓出 11 处、其中只有 1 处是真的——噪声门禁会被无视，
        那比没有门禁更糟。代价（裸文件名的悬空引用查不出）写在脚本的注释里。
        """
        _write(tmp_path, "README.md", f"# 说明\n\n例如 `{token}`。\n")
        assert _run(script, tmp_path) == 0


class TestWrongContentIsCaught:
    """反例（本次最关键的一类）：锚点**指到了别的函数**上。"""

    def test_anchor_pointing_at_the_wrong_function_fails(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(
            tmp_path,
            "docs/design.md",
            "# 设计\n\n工具循环由 `run_tool_loop()` 实现（" + _ref("src/svc/alpha.py", "10-11") + "）。\n",
        )
        _write(
            tmp_path,
            "src/svc/alpha.py",
            "".join(f"# 第 {i} 行\n" for i in range(1, 10)) + "def something_else():\n    return 42\n",
        )

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "不通过" in out
        assert "run_tool_loop" in out, "失败信息里要写清楚找了什么、没找到什么"

    def test_the_same_anchor_passes_once_it_points_at_the_right_lines(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """把锚点挪回 `run_tool_loop` 所在的那两行 ⇒ 立刻变绿。"""
        _write(
            tmp_path,
            "docs/design.md",
            "# 设计\n\n工具循环由 `run_tool_loop()` 实现（" + _ref("src/svc/alpha.py", "1-2") + "）。\n",
        )
        _write(
            tmp_path,
            "src/svc/alpha.py",
            "def run_tool_loop():\n    return 1\n\n\ndef something_else():\n    return 42\n",
        )

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 0, out
        assert "命中 'run_tool_loop'" in out

    def test_strong_candidates_win_over_generic_words(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """一句里同时有普通词与强候选时，必须由强候选命中：
        否则 `chat` 这种到处都有的词会让任何锚点都"通过"。"""
        _write(
            tmp_path,
            "docs/design.md",
            "# 设计\n\n`/chat` 的预算见 `ChatBudget`（" + _ref("src/svc/alpha.py", "1") + "）。\n",
        )
        _write(tmp_path, "src/svc/alpha.py", "# chat chat chat\n")

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "ChatBudget" in out


class TestStructuralFailures:
    """反例：文件不存在 / 行号越界 / 路径有歧义——连位置都不对，必须红。"""

    def test_missing_target_file_fails(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(tmp_path, "docs/design.md", "见 " + _ref("src/svc/nope.py", "1") + "。\n")

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "目标文件不存在" in out

    def test_line_number_out_of_range_fails(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(tmp_path, "docs/design.md", "见 " + _ref("src/svc/alpha.py", "9-12") + "。\n")
        _write(tmp_path, "src/svc/alpha.py", "def run_tool_loop():\n    return 1\n")

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "行号越界" in out

    def test_ambiguous_shorthand_path_fails(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """缩写路径对上两个同名文件时不许猜：猜错就是一条永远指错的引用。"""
        _write(tmp_path, "docs/design.md", "见 " + _ref("alpha.py", "1") + "。\n")
        _write(tmp_path, "src/a/alpha.py", "def run_tool_loop():\n    return 1\n")
        _write(tmp_path, "src/b/alpha.py", "def run_tool_loop():\n    return 1\n")

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "有歧义" in out

    def test_bare_line_reference_without_a_preceding_file_fails(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(tmp_path, "docs/design.md", "那里（" + _ref("", "12-13") + "）也有。\n")

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "无归属目标文件" in out

    def test_bare_line_reference_inherits_the_previous_file(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SECURITY.md 第 283-284 行就是这种写法（只有冒号与行号，沿用上一句的 compose 文件）。"""
        _write(
            tmp_path,
            "docs/design.md",
            "`run_tool_loop()` 见 " + _ref("src/svc/alpha.py", "1") + " 与 " + _ref("", "3") + " 两处。\n",
        )
        _write(
            tmp_path,
            "src/svc/alpha.py",
            "def run_tool_loop():\n    return 1\nrun_tool_loop()\n",
        )

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 0, out
        assert "src/svc/alpha.py:3" in out


class TestUnverifiableRefsNeedTheWhitelist:
    """机器判断不了的引用必须**显式登记**，否则红。"""

    def _doc(self) -> str:
        return "# 安全\n\n这一条只指向文件头的一句自述（" + _ref("svc/alpha.py", "1") + "）。\n"

    def test_ref_without_any_identifier_fails_when_not_whitelisted(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(tmp_path, "docs/security.md", self._doc())
        _write(tmp_path, "svc/alpha.py", "# 本文件尚未在本机实测。\n")

        code = _run(script, tmp_path)

        out = capsys.readouterr().out
        assert code == 1, out
        assert "白名单未登记" in out

    def test_whitelisted_ref_passes_and_carries_its_reason(self, script: ModuleType, tmp_path: Path) -> None:
        _write(tmp_path, "docs/security.md", self._doc())
        _write(tmp_path, "svc/alpha.py", "# 本文件尚未在本机实测。\n")
        table = {
            ("docs/security.md", 3, "svc/alpha.py:1"): script.White(
                reason="散文里只有这个锚点，没有任何符号名", expect="尚未在本机实测"
            )
        }

        report = script.check_paths(tmp_path, ["docs/security.md"], whitelist=table)

        assert [f.status for f in report.findings] == ["white"]
        assert not report.failures
        assert report.whitelisted[0].matched == "尚未在本机实测"
        assert "散文里只有这个锚点" in report.whitelisted[0].detail

    def test_whitelist_is_not_a_free_pass_when_its_expectation_drifts(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """白名单免除的是"散文里没有标识符"，不是"内容不用对"：
        `expect` 一旦在锚点范围里找不到，仍然必须红。"""
        _write(tmp_path, "docs/security.md", self._doc())
        _write(tmp_path, "svc/alpha.py", "# 完全无关的一行。\n")
        table = {
            ("docs/security.md", 3, "svc/alpha.py:1"): script.White(reason="同上", expect="尚未在本机实测")
        }

        report = script.check_paths(tmp_path, ["docs/security.md"], whitelist=table)

        assert [f.status for f in report.findings] == ["fail"]
        assert "不在锚点范围内" in report.failures[0].detail


class TestRealDocsAreUntouched:
    """测试本身不许碰真实文档：反例全部在 tmp_path 里造。"""

    def test_negative_cases_run_entirely_inside_tmp_path(self, script: ModuleType, tmp_path: Path) -> None:
        _write(tmp_path, "docs/design.md", "见 " + _ref("src/svc/alpha.py", "1") + "。\n")
        _write(tmp_path, "src/svc/alpha.py", "# 无关\n")

        _run(script, tmp_path)

        # 真实文档里不该出现测试造出来的这些字符串
        assert "src/svc/alpha.py" not in (repo_root() / "SECURITY.md").read_text(encoding="utf-8")
        assert not (repo_root() / "docs" / "design.md").exists()
        assert not (repo_root() / "src" / "svc" / "alpha.py").exists()
