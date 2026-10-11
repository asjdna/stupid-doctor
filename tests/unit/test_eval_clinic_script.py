"""`scripts/eval_clinic.py` 的 CLI 契约测试：退出码与输出分流。

为什么要单独测这个脚本（而不是只测 `gate_failures()`）：闸门的**用户可见形态**是
退出码。曾经这里有一条缺陷链：

- `gate_failures()` 不看分诊 ⇒ 分诊 0 分也退出 0；
- `--json` 曾（在 harness 里）抢在闸门之前返回，于是"退出码 0"与"质量达标"脱钩。

所以本文件钉住三件事，全部走**真实的 `main()`**、且**不加载卡片语料**
（`_triage_candidates` 被打桩，所以干净克隆上也能跑，且只花毫秒级）：

1. 分诊 0 分 ⇒ 退出码 **2**、打印「闸门：不通过」；
2. 分诊达标 ⇒ 退出码 **0**、打印「闸门：通过」；
3. `--gate --json` 的退出码与不带 `--json` **逐位相同**，且 stdout 是**纯 JSON**
   （旁白走 stderr）——与 `evaluation/harness.run_eval` 的 `emit()` 同一约定。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from kubedocs.config import repo_root

SCRIPT = repo_root() / "scripts" / "eval_clinic.py"


def _load_script() -> ModuleType:
    """按路径加载脚本模块（`scripts/` 不是包，没有 `__init__.py`）。"""
    spec = importlib.util.spec_from_file_location("_eval_clinic_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


@pytest.fixture
def gold(tmp_path: Path) -> Path:
    """最小金标集：一条正常话（红线不触发）+ 一条分诊用例。

    只用两条，是为了把被验证的东西锁在"闸门判据"上，而不是语料规模上。
    """
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"id": "s1", "kind": "safe_normal", "text": "今天天气不错", "expect_red_line": false}\n'
        '{"id": "t1", "kind": "triage_department", "text": "我咳嗽", "expect_department": "呼吸内科"}\n',
        encoding="utf-8")
    return path


def _stub_triage(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *,
                 top: str) -> None:
    """把 `_triage_candidates` 换成一个"永远给出 `top` 科室"的假实现。"""

    def fake(*, json_mode: bool = False) -> Any:
        return (
            {"咳嗽": ("呼吸内科",)},
            lambda text, n: [SimpleNamespace(symptom="咳嗽")],
            SimpleNamespace(score=lambda ev, links: {top: 1.0}),
            None,
        )

    monkeypatch.setattr(script, "_triage_candidates", fake)


def _run(script: ModuleType, gold: Path, tmp_path: Path,
         *extra: str) -> int:
    baseline = tmp_path / "does-not-exist-baseline.json"
    return int(script.main(["--gold", str(gold), "--baseline", str(baseline), *extra]))


class TestTriageDrivesTheExitCode:
    def test_zero_triage_accuracy_fails_the_gate(self, script, gold, tmp_path,
                                                 monkeypatch, capsys):
        """分诊一条都不中 ⇒ 退出码 2（缺陷回归：修复前是 0）。"""
        _stub_triage(script, monkeypatch, top="骨科")          # 期望科室命中不了
        code = _run(script, gold, tmp_path, "--gate")

        assert code == 2
        out = capsys.readouterr().out
        assert "闸门：不通过" in out
        assert "分诊准确率" in out
        assert "为什么分诊现在是红的" in out, "红的时候必须说明原因，否则会被当成新回归"

    def test_meeting_the_threshold_passes(self, script, gold, tmp_path, monkeypatch, capsys):
        """分诊达标（1/1 = 100%）⇒ 退出码 0：闸门不是"永远红"。"""
        _stub_triage(script, monkeypatch, top="呼吸内科")
        code = _run(script, gold, tmp_path, "--gate")

        assert code == 0
        assert "闸门：通过" in capsys.readouterr().out

    def test_without_gate_the_exit_code_is_zero_even_when_triage_is_zero(
            self, script, gold, tmp_path, monkeypatch):
        """不带 `--gate` 时退出码仍是 0：闸门只在被要求时才卡。"""
        _stub_triage(script, monkeypatch, top="骨科")
        assert _run(script, gold, tmp_path) == 0


class TestJsonDoesNotBypassTheGate:
    def test_json_exit_code_matches_the_plain_one(self, script, gold, tmp_path, monkeypatch):
        _stub_triage(script, monkeypatch, top="骨科")
        assert _run(script, gold, tmp_path, "--gate", "--json") == 2
        _stub_triage(script, monkeypatch, top="呼吸内科")
        assert _run(script, gold, tmp_path, "--gate", "--json") == 0

    def test_stdout_is_pure_json_and_gate_noise_goes_to_stderr(
            self, script, gold, tmp_path, monkeypatch, capsys):
        """`--json` 只改输出格式：stdout 必须能被 `json.loads` 直接吃掉。"""
        _stub_triage(script, monkeypatch, top="骨科")
        code = _run(script, gold, tmp_path, "--gate", "--json")

        captured = capsys.readouterr()
        assert code == 2
        payload = json.loads(captured.out)                 # 旁白混进来就会在这里炸
        assert payload["triage"]["accuracy"] == 0.0
        assert "闸门：不通过" in captured.err

    def test_without_json_the_gate_output_stays_on_stdout(
            self, script, gold, tmp_path, monkeypatch, capsys):
        """非 `--json` 时输出位置**逐字不变**（仍在 stdout），不制造无谓的破坏性改动。"""
        _stub_triage(script, monkeypatch, top="骨科")
        _run(script, gold, tmp_path, "--gate")

        captured = capsys.readouterr()
        assert "闸门：不通过" in captured.out
        assert captured.err == ""
