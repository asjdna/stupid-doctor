"""`scripts/check_doc_refs.py` 的契约测试：门禁必须真的会咬人，而不只是"打印通过"。

正例是当前仓库的全部引用；反例全部在 `tmp_path` 里造临时文档与临时目标文件，
**不改真实文档**。覆盖的判据：

- 锚点指向错误内容、一句里有强候选时只认强候选；
- 目标文件不存在、行号越界、缩写路径有歧义、裸续写引用没有归属；
- 散文里没有可校验标识符时必须显式登记白名单；
- 白名单钉住的 `expect` 漂走了仍然失败；无文件名的续写引用沿用同段上文的目标文件。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from kubedocs.config import repo_root

SCRIPT = repo_root() / "scripts" / "check_doc_refs.py"

# 本文件自己也在扫描面里（`tests/**/*.py`）：`文件:行号` 引用必须在运行时拼出来，
# 写成字面量会先被检查器当成真引用校验。
_TICK = "`"


def _ref(target: str, spec: str) -> str:
    """拼出一个反引号引用（目标文件加冒号加行号）。"""
    return f"{_TICK}{target}:{spec}{_TICK}"


def _load_script() -> ModuleType:
    """按路径加载脚本模块（`scripts/` 不是包，没有 `__init__.py`）。

    必须先登记 `sys.modules` 再 `exec_module`：`@dataclass` 解析字符串注解时会去
    `sys.modules[cls.__module__]` 里查，不登记就是 `None.__dict__` 的 AttributeError。
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
        # 别让"扫描面塌了"伪装成通过：参与扫描的文件数必须真的成规模。
        # 刻意不校验引用条数——文档不写 `文件:行号` 是正常形态，条数为 0 不等于没在扫。
        match = re.search(r"扫描文档\s*:\s*(\d+) 个文件", out)
        assert match is not None, out
        assert int(match.group(1)) > 100, out

    def test_whitelist_listing_prints_every_reason(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """白名单必须**被打印出来**（含条数与理由），否则豁免就是静默的。

        断言的是不变量（脚本里登记的每条 `reason` 都出现在输出里），而不是某个固定措辞。
        """
        assert int(script.main(["--root", str(repo_root()), "--whitelist"])) == 0

        out = capsys.readouterr().out
        assert f"白名单 {len(script.WHITELIST)} 条" in out
        for white in script.WHITELIST.values():
            assert white.reason in out, f"白名单条目没有把理由打印出来：{white.reason[:40]}"


class TestWhitelistIsNotAFreePass:
    """白名单只免掉"散文里没有可校验标识符"这一条，**不免内容核对**。"""

    def test_whitelisted_anchor_must_satisfy_expect(self, script: ModuleType, tmp_path: Path) -> None:
        target = "src/svc/alpha.py"
        _write(tmp_path, target, "def alpha() -> int:\n    return 1\n")
        _write(tmp_path, "README.md", f"# 说明\n\n见 {_ref(target, '2')}。\n")
        key = ("README.md", f"{target}:2")

        hit = script.check_paths(
            tmp_path,
            ["README.md"],
            whitelist={key: script.White(reason="散文里只有这个锚点", expect="return 1")},
        )
        assert [f.status for f in hit.whitelisted] == ["white"]
        assert hit.whitelisted[0].anchor.text == f"{target}:2"

        drifted = script.check_paths(
            tmp_path,
            ["README.md"],
            whitelist={key: script.White(reason="同上", expect="return 2")},
        )
        assert [f.status for f in drifted.failures] == ["fail"]
        assert "锚点已漂移" in drifted.failures[0].detail


class TestMissingFileRefsAreCaught:
    """反例：引用了**不存在的 `.md` 文件**必须非零退出。

    只写文件名的引用不进入 `文件:行号` 校验，所以这一类必须单独把关；
    删掉这段检查，下面的反例会立刻变红。
    """

    def test_dangling_docs_path_fails(self, script: ModuleType, tmp_path: Path) -> None:
        # 路径必须在运行时拼：本文件也在扫描面里，字面量会被检查器先抓到
        a = "docs/" + "9999-nonexistent.md"
        b = "docs/" + "8888-nope.md"
        _write(tmp_path, "README.md", f"# 说明\n\n见 `{a}` 与 [另一份]({b})。\n")
        assert _run(script, tmp_path) == 1

    def test_existing_docs_path_passes(self, script: ModuleType, tmp_path: Path) -> None:
        real = "docs/" + "0001-real.md"
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

        宽规则会抓出大量示意路径，而噪声门禁会被无视——那比没有门禁更糟。
        """
        _write(tmp_path, "README.md", f"# 说明\n\n例如 `{token}`。\n")
        assert _run(script, tmp_path) == 0


class TestWrongContentIsCaught:
    """反例：锚点**指到了别的函数**上——内容判据的核心。"""

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
        """只有冒号与行号的续写引用沿用上一句的目标文件，是文档里的常见写法。"""
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
            ("docs/security.md", "svc/alpha.py:1"): script.White(
                reason="散文里只有这个锚点，没有任何符号名", expect="尚未在本机实测"
            )
        }

        report = script.check_paths(tmp_path, ["docs/security.md"], whitelist=table)

        assert [f.status for f in report.findings] == ["white"]
        assert not report.failures
        assert report.whitelisted[0].matched == "尚未在本机实测"
        assert "散文里只有这个锚点" in report.whitelisted[0].detail

    def test_whitelist_survives_lines_inserted_above_it(self, script: ModuleType, tmp_path: Path) -> None:
        """在文档上方插行不该让豁免失效：白名单的键不含"引用所在的行号"。

        键里放行号，任何上方插入内容的编辑都会让豁免失效并报"锚点已漂移"；
        真正防漂移的是 `expect`（它校验**目标文件**在锚点处的内容）。位置不是身份。
        """
        _write(tmp_path, "docs/security.md", self._doc())
        _write(tmp_path, "svc/alpha.py", "# 本文件尚未在本机实测。\n")
        table = {("docs/security.md", "svc/alpha.py:1"): script.White(reason="同上", expect="尚未在本机实测")}

        before = script.check_paths(tmp_path, ["docs/security.md"], whitelist=table)
        # 在引用上方插入若干行：引用被推到更靠下的行号
        _write(tmp_path, "docs/security.md", "新的一行。\n\n又一行。\n" + self._doc())
        after = script.check_paths(tmp_path, ["docs/security.md"], whitelist=table)

        assert [f.status for f in before.findings] == ["white"]
        assert [f.status for f in after.findings] == ["white"], "上方插行让豁免失效了（键里还有行号？）"
        assert not after.failures

    def test_whitelist_is_not_a_free_pass_when_its_expectation_drifts(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """白名单免除的是"散文里没有标识符"，不是"内容不用对"：
        `expect` 一旦在锚点范围里找不到，仍然必须红。"""
        _write(tmp_path, "docs/security.md", self._doc())
        _write(tmp_path, "svc/alpha.py", "# 完全无关的一行。\n")
        table = {("docs/security.md", "svc/alpha.py:1"): script.White(reason="同上", expect="尚未在本机实测")}

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
