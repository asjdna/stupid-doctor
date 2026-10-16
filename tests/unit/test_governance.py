"""`scripts/check_governance.py` 的契约测试：治理门禁必须**真的会咬人**。

正例是当前仓库（清单、不变量、测试覆盖三方一致）；反例全部在 `tmp_path` 里造临时清单
与临时资产，**不改真实文件**。覆盖的判据：

- 资产内容变了却不同步清单（哈希对不上）；
- 治理字段留白（version / effective_date / source / owner / rationale）；
- 不变量在代码里新增或删除却没进/没销清单；
- 不变量没有测试覆盖、或验证出处指向一个不存在的文件；
- `--update-hashes` 只刷哈希，**不动**版本号与理由。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from kubedocs.config import repo_root

SCRIPT = repo_root() / "scripts" / "check_governance.py"


def _load_script() -> ModuleType:
    """按路径加载脚本模块（`scripts/` 不是包，没有 `__init__.py`）。"""
    spec = importlib.util.spec_from_file_location("_check_governance_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


# --------------------------------------------------------------------------- 正例


def test_the_real_repository_passes(script: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    """真实清单必须自洽：资产哈希、不变量对应、实现路径、测试覆盖、验证出处全部一致。"""
    code = int(script.main(["--root", str(repo_root())]))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "结果：通过" in out


def test_known_gaps_are_printed_not_hidden(script: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    """已知治理缺口要打在输出里——留白会被读成"没有问题"。"""
    script.main(["--root", str(repo_root())])
    out = capsys.readouterr().out
    assert "已知治理缺口" in out
    assert "临床审阅人" in out


# --------------------------------------------------------------------------- 反例


def _asset(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _manifest(root: Path, *, sha256: str, **over: Any) -> dict[str, Any]:
    """写一份最小清单（一条资产 + 代码里那三条不变量）。"""
    from kubedocs.clinical.safety import INVARIANTS

    asset: dict[str, Any] = {
        "id": "prompt.test",
        "kind": "prompt",
        "path": "assets/system.md",
        "sha256": sha256,
        "version": "1.0.0",
        "effective_date": "2026-10-16",
        "source": "测试资产",
        "owner": "测试",
        "rationale": "测试",
    }
    asset.update(over)
    invariants = [
        {
            "id": inv.id,
            "owner": "测试",
            "effective_date": "2026-10-16",
            "rationale": "测试",
            "verified_by": ["tests/unit/test_governance.py"],
        }
        for inv in INVARIANTS
    ]
    data = {"schema": 1, "assets": [asset], "invariants": invariants, "governance_gaps": []}
    path = root / "data" / "governance" / "safety_assets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _write_tests(root: Path) -> None:
    """造一个"覆盖了全部不变量"的测试目录，并补齐不变量声明的实现文件。

    实现文件也要造：门禁会核对 `implemented_in` 的路径真的存在——
    "声明它实现于某处"而那个地方不存在，正是清单最容易漂的地方。
    """
    from kubedocs.clinical.safety import INVARIANTS

    blob = "\n".join(f"# {inv.id}" for inv in INVARIANTS)
    _asset(root, "tests/unit/test_governance.py", blob)
    for inv in INVARIANTS:
        for rel in inv.implemented_in:
            _asset(root, rel, f"# {inv.id} 的实现位置（测试替身）\n")


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_stale_hash_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """资产内容变了却不同步清单 ⇒ 失败（这就是变更控制）。"""
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _write_tests(tmp_path)
    _manifest(tmp_path, sha256=_sha("旧的哈希"))

    code = int(script.main(["--root", str(tmp_path)]))
    out = capsys.readouterr().out
    assert code == 1
    assert "内容已变但清单未同步" in out


def test_matching_hash_passes(script: ModuleType, tmp_path: Path) -> None:
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _write_tests(tmp_path)
    _manifest(tmp_path, sha256=_sha(text))
    assert int(script.main(["--root", str(tmp_path)])) == 0


@pytest.mark.parametrize("field", ["version", "effective_date", "source", "owner", "rationale"])
def test_blank_governance_fields_are_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    """治理字段不许留白：留白会被读成"这条资产不需要审"。"""
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _write_tests(tmp_path)
    _manifest(tmp_path, sha256=_sha(text), **{field: ""})

    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert f"缺少 {field}" in capsys.readouterr().out


def test_missing_asset_file_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tests(tmp_path)
    _manifest(tmp_path, sha256=_sha("x"))  # 资产文件根本没写
    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert "path 不存在" in capsys.readouterr().out


def test_stale_invariant_entry_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """清单里有、代码里没有的不变量（删了不变量没销账）⇒ 失败。"""
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _write_tests(tmp_path)
    data = _manifest(tmp_path, sha256=_sha(text))
    data["invariants"].append(
        {
            "id": "no_such_invariant",
            "owner": "测试",
            "effective_date": "2026-10-16",
            "rationale": "测试",
            "verified_by": ["tests/unit/test_governance.py"],
        }
    )
    (tmp_path / "data" / "governance" / "safety_assets.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )

    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert "在代码里不存在" in capsys.readouterr().out


def test_invariant_without_test_coverage_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """没有测试覆盖的不变量等于没人守。"""
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _asset(tmp_path, "tests/unit/test_governance.py", "# 一个不变量都没提到")
    _manifest(tmp_path, sha256=_sha(text))

    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert "找不到任何引用" in capsys.readouterr().out


def test_missing_verification_source_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _write_tests(tmp_path)
    data = _manifest(tmp_path, sha256=_sha(text))
    data["invariants"][0]["verified_by"] = ["tests/unit/no_such_test.py"]
    (tmp_path / "data" / "governance" / "safety_assets.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )

    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert "验证出处不存在" in capsys.readouterr().out


def test_missing_manifest_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert "治理清单缺失" in capsys.readouterr().out


def test_bad_schema_is_refused(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "data" / "governance" / "safety_assets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    assert int(script.main(["--root", str(tmp_path)])) == 1
    assert "schema" in capsys.readouterr().out


def test_update_hashes_only_touches_the_hash(script: ModuleType, tmp_path: Path) -> None:
    """`--update-hashes` 是机器可代填的那一列；版本号与理由必须由人来写。"""
    text = "提示词内容"
    _asset(tmp_path, "assets/system.md", text)
    _write_tests(tmp_path)
    _manifest(tmp_path, sha256=_sha("旧的哈希"), rationale="人写的理由")

    assert int(script.main(["--root", str(tmp_path), "--update-hashes"])) == 0
    saved = json.loads((tmp_path / "data" / "governance" / "safety_assets.json").read_text(encoding="utf-8"))
    assert saved["assets"][0]["sha256"] == _sha(text)
    assert saved["assets"][0]["rationale"] == "人写的理由", "刷哈希不许把理由冲掉"
    # 刷完之后门禁应当通过
    assert int(script.main(["--root", str(tmp_path)])) == 0
