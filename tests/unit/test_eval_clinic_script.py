"""`scripts/eval_clinic.py` 的 CLI 契约测试：退出码与输出分流。

闸门的用户可见形态是退出码，本文件走真实的 `main()` 钉住三件事：
1. 分诊 0 分 ⇒ 退出码 2、打印「闸门：不通过」；
2. 分诊达标 ⇒ 退出码 0、打印「闸门：通过」；
3. `--gate --json` 的退出码与不带 `--json` 逐位相同，且 stdout 是纯 JSON
   （旁白走 stderr），与 harness 的 `emit()` 同一约定。

`_triage_candidates` 被打桩：不加载卡片语料，干净克隆上也能毫秒级跑完。
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
    """最小金标集：一条红线正例 + 一条正常话 + 一条分诊用例。

    只验证闸门判据、不涉及语料规模；红线正例必须存在，否则召回无从检验。
    """
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"id": "r1", "kind": "red_line", "text": "我现在胸痛", "expect_red_line": true}\n'
        '{"id": "s1", "kind": "safe_normal", "text": "今天天气不错", "expect_red_line": false}\n'
        '{"id": "t1", "kind": "triage_department", "text": "我咳嗽", "expect_department": "呼吸内科"}\n',
        encoding="utf-8",
    )
    return path


def _stub_triage(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *, top: str) -> None:
    """把 `_triage_candidates` 换成一个"永远给出 `top` 科室"的假实现。"""

    def fake(*, json_mode: bool = False) -> Any:
        return (
            {"咳嗽": ("呼吸内科",)},
            # 抽取器签名是 `(text, turn, *, suggested=(), links=None)`：替身必须接住
            # `links`，否则脚本一调就 TypeError（"分诊为 0"会被误读成"脚本崩了"）。
            lambda text, n, **_: [SimpleNamespace(symptom="咳嗽")],
            SimpleNamespace(score=lambda ev, links: {top: 1.0}),
            None,
        )

    monkeypatch.setattr(script, "_triage_candidates", fake)


def _run(script: ModuleType, gold: Path, tmp_path: Path, *extra: str) -> int:
    baseline = tmp_path / "does-not-exist-baseline.json"
    return int(script.main(["--gold", str(gold), "--baseline", str(baseline), *extra]))


class TestTriageDrivesTheExitCode:
    def test_zero_triage_accuracy_fails_the_gate(self, script, gold, tmp_path, monkeypatch, capsys):
        """分诊一条都不中 ⇒ 退出码 2（分诊指标必须真的参与闸门）。"""
        _stub_triage(script, monkeypatch, top="骨科")  # 期望科室命中不了
        code = _run(script, gold, tmp_path, "--gate")

        assert code == 2
        out = capsys.readouterr().out
        assert "闸门：不通过" in out
        assert "分诊准确率" in out
        # 断言"脚本登记的那段说明被原样打印出来"，而不是某个固定措辞：
        # 变红时必须自解释，否则读者无从判断是数据问题还是实现问题。
        assert script.TRIAGE_RED_NOTE in out

    def test_meeting_the_threshold_passes(self, script, gold, tmp_path, monkeypatch, capsys):
        """分诊达标（1/1 = 100%）⇒ 退出码 0：闸门不是"永远红"。"""
        _stub_triage(script, monkeypatch, top="呼吸内科")
        code = _run(script, gold, tmp_path, "--gate")

        assert code == 0
        assert "闸门：通过" in capsys.readouterr().out

    def test_without_gate_the_exit_code_is_zero_even_when_triage_is_zero(
        self, script, gold, tmp_path, monkeypatch
    ):
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
        self, script, gold, tmp_path, monkeypatch, capsys
    ):
        """`--json` 只改输出格式：stdout 必须能被 `json.loads` 直接吃掉。"""
        _stub_triage(script, monkeypatch, top="骨科")
        code = _run(script, gold, tmp_path, "--gate", "--json")

        captured = capsys.readouterr()
        assert code == 2
        payload = json.loads(captured.out)  # 旁白混进来就会在这里炸
        assert payload["triage"]["accuracy"] == 0.0
        assert "闸门：不通过" in captured.err

    def test_without_json_the_gate_output_stays_on_stdout(self, script, gold, tmp_path, monkeypatch, capsys):
        """非 `--json` 时输出位置**逐字不变**（仍在 stdout），不制造无谓的破坏性改动。"""
        _stub_triage(script, monkeypatch, top="骨科")
        _run(script, gold, tmp_path, "--gate")

        captured = capsys.readouterr()
        assert "闸门：不通过" in captured.out
        assert captured.err == ""


class TestRedLineMustBeEvaluated:
    """空集不得满足安全闸门：没有红线正例的运行必须判为未评测并失败。

    召回在分母为 0 时不许返回 1.0，否则"删光正例"就成了让闸门变绿的手段。
    """

    def _negative_only_gold(self, tmp_path: Path) -> Path:
        path = tmp_path / "no-redline.jsonl"
        path.write_text(
            '{"id": "s1", "kind": "safe_normal", "text": "今天天气不错", "expect_red_line": false}\n'
            '{"id": "t1", "kind": "triage_department", "text": "我咳嗽", "expect_department": "呼吸内科"}\n',
            encoding="utf-8",
        )
        return path

    def test_no_positive_case_fails_the_gate(self, script, tmp_path, monkeypatch, capsys):
        _stub_triage(script, monkeypatch, top="呼吸内科")  # 分诊满分，排除它的干扰
        gold = self._negative_only_gold(tmp_path)
        code = _run(script, gold, tmp_path, "--gate")

        out = capsys.readouterr().out
        assert code == 2, "没有红线正例的运行不该通过安全闸门"
        assert "红线未评测" in out

    def test_recall_is_none_when_the_denominator_is_empty(self):
        from kubedocs.evaluation.clinic import ClinicCase, evaluate_redline

        cases = [ClinicCase(id="s1", kind="safe_normal", text="今天天气不错", expect_red_line=False)]
        report = evaluate_redline(cases)
        assert report.redline_recall is None, "空分母必须返回 None，而不是伪装成 100%"


class TestBaselineIsBoundToTheGoldSet:
    """clinic 基线必须绑定金标指纹：只存指标的话，换尺子无人察觉。"""

    def test_baseline_records_the_gold_hash_and_case_count(self, script, gold, tmp_path):
        code = int(
            script.main(["--gold", str(gold), "--baseline", str(tmp_path / "b.json"), "--write-baseline"])
        )
        assert code == 0
        payload = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
        assert payload["gold_sha256"], "基线没有金标指纹"
        assert payload["n_cases"] == 2, "红线用例数（1 正 + 1 负）应被记进基线"

    def test_changed_gold_is_refused_not_silently_compared(self, script, gold, tmp_path, monkeypatch, capsys):
        baseline = tmp_path / "b.json"
        script.main(["--gold", str(gold), "--baseline", str(baseline), "--write-baseline"])

        # 改掉一把"尺子"（换掉那条正常话）后，闸门必须拒绝比较
        changed = tmp_path / "changed.jsonl"
        changed.write_text(
            gold.read_text(encoding="utf-8").replace("今天天气不错", "今天天气很好"),
            encoding="utf-8",
        )
        _stub_triage(script, monkeypatch, top="呼吸内科")
        code = int(script.main(["--gold", str(changed), "--baseline", str(baseline), "--gate"]))
        out = capsys.readouterr().out
        assert code == 2
        assert "标注集已变更" in out
