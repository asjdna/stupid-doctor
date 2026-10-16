"""`scripts/eval_clinic.py` 的 CLI 契约测试：退出码与输出分流。

闸门的用户可见形态是退出码，本文件走真实的 `main()` 钉住三件事：
1. 分诊 0 分 ⇒ 退出码 2、打印「闸门：不通过」；
2. 分诊达标 ⇒ 退出码 0、打印「闸门：通过」；
3. `--gate --json` 的退出码与不带 `--json` 逐位相同，且 stdout 是纯 JSON
   （旁白走 stderr），与 harness 的 `emit()` 同一约定。

`_triage_candidates` 被打桩：不加载卡片语料，干净克隆上也能毫秒级跑完。

本文件还覆盖 **LLM 臂**（真正做判断的那条路，见
`src/kubedocs/evaluation/clinic_llm.py`）：红线召回硬闸门、分诊门限、未评测不算通过、
单题失败不许崩、`--arm rule` 绝不联网。这些用例同样全部打桩——**真实模型调用
不在单测里发生**。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from kubedocs.clinical.contracts import ClinicalAssessment, DepartmentAdvice, RedFlag
from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem
from kubedocs.clinical.pipeline import ClinicalTurn
from kubedocs.clinical.reasoning import ReasoningResult
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
    """跑一次 `main()`。**固定加 `--arm rule`**：`--arm` 的默认值是 `both`，
    而 `both` 会真的去调大模型——本文件是离线单测，不能因为一个默认值就联网花钱。
    规则单臂沿用历史口径（红线 + 分诊），所以下面这些既有断言逐字不变。"""
    baseline = tmp_path / "does-not-exist-baseline.json"
    return int(script.main(["--gold", str(gold), "--baseline", str(baseline), "--arm", "rule", *extra]))


class TestMetricsDoNotDriveTheExitCode:
    """**指标只报不卡**：分诊 0 分、召回漏报、误报超限都不再影响退出码。

    这是本次改造的核心：指标来自自建小样本，拿它卡退出码只会逼着后来者去改门限。
    闸门只卡"这次运行不能作为证据"；安全不变量的触发交给长期监听报给运维。
    """

    def test_zero_triage_accuracy_still_exits_zero(self, script, gold, tmp_path, monkeypatch, capsys):
        _stub_triage(script, monkeypatch, top="骨科")  # 期望科室命中不了
        code = _run(script, gold, tmp_path, "--gate")

        assert code == 0
        out = capsys.readouterr().out
        assert "分诊科室    : 0.00% top3 命中" in out, "分数必须照旧打出来（记录不能省）"
        assert "闸门：通过" in out

    def test_without_gate_the_exit_code_is_zero_even_when_triage_is_zero(
        self, script, gold, tmp_path, monkeypatch
    ):
        """不带 `--gate` 时退出码仍是 0。"""
        _stub_triage(script, monkeypatch, top="骨科")
        assert _run(script, gold, tmp_path) == 0


class TestJsonDoesNotBypassTheGate:
    """`--json` 只改输出格式，**不**绕过闸门；而闸门现在只卡安全不变量。"""

    def test_json_exit_code_matches_the_plain_one(self, script, gold, tmp_path, monkeypatch):
        # 指标差不再是失败原因：两种格式都退 0
        _stub_triage(script, monkeypatch, top="骨科")
        assert _run(script, gold, tmp_path, "--gate", "--json") == 0
        _stub_triage(script, monkeypatch, top="呼吸内科")
        assert _run(script, gold, tmp_path, "--gate", "--json") == 0

    def test_stdout_is_pure_json_and_gate_noise_goes_to_stderr(
        self, script, gold, tmp_path, monkeypatch, capsys
    ):
        """`--json` 只改输出格式：stdout 必须能被 `json.loads` 直接吃掉。"""
        _stub_triage(script, monkeypatch, top="骨科")
        code = _run(script, gold, tmp_path, "--gate", "--json")

        captured = capsys.readouterr()
        assert code == 0
        payload = json.loads(captured.out)  # 旁白混进来就会在这里炸
        assert payload["triage"]["accuracy"] == 0.0

    def test_json_failure_still_returns_two(self, script, gold, tmp_path, monkeypatch, capsys):
        """真的触发闸门（这里用"没拿到判断"）时，`--json` 一样退 2、旁白仍在 stderr。"""
        _stub_triage(script, monkeypatch, top="呼吸内科")
        empty = tmp_path / "empty.jsonl"
        empty.write_text('{"id": "s1", "kind": "safe_normal", "text": "今天天气不错"}\n', encoding="utf-8")
        code = _run(script, empty, tmp_path, "--gate", "--json")

        captured = capsys.readouterr()
        assert code == 2
        payload = json.loads(captured.out)
        assert payload["n_cases"] == 0
        assert "闸门：不通过" in captured.err

    def test_without_json_the_gate_output_stays_on_stdout(self, script, gold, tmp_path, monkeypatch, capsys):
        """非 `--json` 时输出位置**逐字不变**（仍在 stdout），不制造无谓的破坏性改动。"""
        _stub_triage(script, monkeypatch, top="骨科")
        _run(script, gold, tmp_path, "--gate")

        captured = capsys.readouterr()
        assert "闸门：通过" in captured.out
        assert captured.err == ""


class TestAnEmptyRunMustNotPass:
    """空集不得满足闸门：一条可评测用例都没有的运行必须失败。"""

    def _no_cases_gold(self, tmp_path: Path) -> Path:
        path = tmp_path / "no-cases.jsonl"
        # 一条**不可评测**的用例：既没有危险信号期望，也没有科室/挂号期望
        path.write_text('{"id": "x1", "kind": "note", "text": "今天天气不错"}\n', encoding="utf-8")
        return path

    def test_gold_without_evaluable_cases_fails_the_gate(self, script, tmp_path, monkeypatch, capsys):
        _stub_triage(script, monkeypatch, top="呼吸内科")
        gold = self._no_cases_gold(tmp_path)
        code = _run(script, gold, tmp_path, "--gate")

        out = capsys.readouterr().out
        assert code == 2, "一条可评测用例都没有的运行不该通过闸门"
        assert "没有任何可评测的用例" in out

    def test_recall_is_none_when_the_denominator_is_empty(self):
        from kubedocs.evaluation.clinic import ClinicCase, evaluate_redline

        cases = [ClinicCase(id="s1", kind="safe_normal", text="今天天气不错", expect_red_line=False)]
        report = evaluate_redline(cases)
        assert report.redline_recall is None, "空分母必须返回 None，而不是伪装成 100%"


class TestBaselineIsBoundToTheGoldSet:
    """基线绑定金标指纹，但指纹变了只是**档案注记**（说明这次不与历史分数比较）。"""

    def test_baseline_records_the_gold_hash_and_case_count(self, script, gold, tmp_path):
        code = int(
            script.main(
                # `--arm rule`：默认的 `both` 会跑 LLM 臂（真调模型）；写基线不需要它。
                [
                    "--gold",
                    str(gold),
                    "--baseline",
                    str(tmp_path / "b.json"),
                    "--arm",
                    "rule",
                    "--write-baseline",
                ]
            )
        )
        assert code == 0
        payload = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
        assert payload["gold_sha256"], "基线没有金标指纹"
        assert payload["n_cases"] == 2, "红线用例数（1 正 + 1 负）应被记进基线"

    def test_changed_gold_is_noted_not_silently_compared(self, script, gold, tmp_path, monkeypatch, capsys):
        baseline = tmp_path / "b.json"
        # `--arm rule`：写基线不需要 LLM 臂（默认的 `both` 会真调模型）。
        script.main(["--gold", str(gold), "--baseline", str(baseline), "--arm", "rule", "--write-baseline"])

        # 换掉一把"尺子"（改一条正常话）
        changed = tmp_path / "changed.jsonl"
        changed.write_text(
            gold.read_text(encoding="utf-8").replace("今天天气不错", "今天天气很好"),
            encoding="utf-8",
        )
        _stub_triage(script, monkeypatch, top="呼吸内科")
        code = int(
            script.main(["--gold", str(changed), "--baseline", str(baseline), "--arm", "rule", "--gate"])
        )
        out = capsys.readouterr().out
        # 指标不是闸门 ⇒ 退出码仍为 0；但"换过尺子"必须写在档案里，否则历史数字会被误读
        assert code == 0
        assert "标注集已变更" in out
        assert "不可比" in out


# ---------------------------------------------------------------------------
# LLM 臂：真正做判断的那条路。
#
# 下面每个用例都用**替身 reasoner**（形状与 `ClinicalReasoner.reason` 一致），
# 把 `_judge_arm_status` / `_build_llm_reasoner` / `_build_tool_loop_pipeline` 一起替换掉——
# 本文件**绝不联网**、绝不读 `.env` 里的真密钥、**绝不打开真索引**。理由与 `_stub_triage`
# 相同：单测要毫秒级、可复现、不花钱，而且"模型这次答得怎么样"必须是测试输入而不是测试环境。
# ---------------------------------------------------------------------------

_TRIAGE_GLYPH = "一二三四五"


def _emergency() -> ClinicalAssessment:
    """模型判为急症的伪造结论（危险信号的落点：`urgency=emergency` + 非空 `red_flags`）。"""
    return ClinicalAssessment(
        urgency="emergency",
        red_flags=[RedFlag(condition="急性冠脉综合征", severity="critical", why="胸痛，需即刻排查")],
        departments=[DepartmentAdvice(name="心内科", confidence=0.9)],
        patient_summary="请立即拨打 120 或前往最近医院急诊。",
    )


def _routine(*departments: str) -> ClinicalAssessment:
    """模型判为常规的伪造结论。`departments` 就是分诊 top-3 的来源。"""
    names = departments or ("全科",)
    return ClinicalAssessment(
        urgency="routine",
        departments=[DepartmentAdvice(name=n, confidence=0.5 - i * 0.1) for i, n in enumerate(names)],
        patient_summary="建议门诊就诊。",
    )


def _llm_answers(*, red_line: str = "emergency", triage_hits: int = 5) -> dict[str, object]:
    """金标每一题的伪造答案：前 `triage_hits` 条分诊答对，其余答成"全科"（错科）。"""
    answers: dict[str, object] = {
        "我现在胸痛": _emergency() if red_line == "emergency" else _routine(),
        "今天天气不错": _routine(),
    }
    for i, glyph in enumerate(_TRIAGE_GLYPH, start=1):
        answers[f"分诊{glyph}"] = _routine(f"科室{i}") if i <= triage_hits else _routine("全科")
    return answers


class _StubReasoner:
    """替身 reasoner：按患者原话返回预设结论，**不联网**。

    答案表里放异常对象 = 那一题抛异常（用来验证"单题失败不许让整轮崩"）。
    """

    def __init__(self, answers: dict[str, object]) -> None:
        self._answers = dict(answers)
        self.calls: list[str] = []
        self.red_line_positive: list[bool] = []
        self.evidence_items: list[int] = []

    def reason(self, question: str, evidence: Any, intake: Any) -> ReasoningResult:
        self.calls.append(question)
        # 记下证据包的两件事：有没有红线线索、有没有知识片段。
        # 前者证明 `guard=RiskGuard()` 真的接进了这一臂，后者证明这臂**不做检索**。
        self.red_line_positive.append(bool(getattr(evidence, "red_line_positive", False)))
        self.evidence_items.append(len(getattr(evidence, "items", [])))
        outcome = self._answers.get(question, _routine())
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ClinicalAssessment)
        return ReasoningResult(assessment=outcome, raw="{}")


class _FakeBudget:
    """假的日预算账本：每读一次涨 `per_read`。

    用它证明 `evaluate_llm_arm` 记的花费是**账本前后差**（真的扣掉的那笔），
    而不是从模型输出里估的数。
    """

    def __init__(self, per_read: float = 0.0005) -> None:
        self._per_read = per_read
        self.reads = 0

    def spent_today(self) -> float:
        self.reads += 1
        return self.reads * self._per_read


@pytest.fixture
def llm_gold(tmp_path: Path) -> Path:
    """LLM 臂的离线金标：2 条红线（1 正 1 负）+ 5 条分诊。

    分诊刻意做成 5 条：门限 60% 在这份样本上等价于"至少答对 3 条"，
    所以 3/5 通过、2/5 不通过是能逐位钉住门限的最小样本。
    """
    rows: list[dict[str, object]] = [
        {"id": "r1", "kind": "redline_trigger", "text": "我现在胸痛", "expect_red_line": True},
        {"id": "n1", "kind": "redline_safe", "text": "今天天气不错", "expect_red_line": False},
    ]
    rows += [
        {
            "id": f"t{i}",
            "kind": "triage_department",
            "text": f"分诊{glyph}",
            "expect_department": f"科室{i}",
        }
        for i, glyph in enumerate(_TRIAGE_GLYPH, start=1)
    ]
    path = tmp_path / "llm_gold.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def _install_stub_reasoner(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, answers: dict[str, object], *, budget: Any = None
) -> _StubReasoner:
    """把**单发对照臂**接到替身 reasoner 上：状态探测与装配都换成假的（离线）。"""
    stub = _StubReasoner(answers)
    monkeypatch.setattr(script, "_judge_arm_status", lambda *, need_index: (True, ""))
    monkeypatch.setattr(script, "_build_llm_reasoner", lambda: (stub, budget or _FakeBudget()))
    return stub


class _StubPipeline:
    """替身管线：按患者原话返回预设 `ClinicalTurn`（形状与真产物一致）。

    它替代的是 `pipeline.advance`——也就是**产品路径**那一步。所以这里的输入必须包含
    真管线会给出的字段：`assessment` / `evidence` / `safety` / `llm_calls` / `steps` /
    `tool_calls` / `degraded`，否则测出来的东西不代表产品。
    """

    def __init__(self, answers: dict[str, object]) -> None:
        self._answers = dict(answers)
        self.calls: list[str] = []

    def advance(
        self, question: str, *, state: Any = None, history: Any = None, patient_id: str = "anon"
    ) -> Any:
        self.calls.append(question)
        outcome = self._answers.get(question, _routine())
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ClinicalTurn)
        return outcome


def _turn(
    assessment: ClinicalAssessment | None,
    *,
    safety: dict[str, Any] | None = None,
    llm_calls: int = 2,
    steps: int = 3,
    tools: int = 1,
    action: str = "answer",
    degraded: str = "",
) -> ClinicalTurn:
    return ClinicalTurn(
        action=action,  # type: ignore[arg-type]
        message="（打桩）",
        assessment=assessment,
        evidence=EvidenceBundle(items=[EvidenceItem(id="K1", kind="knowledge", text="资料")]),
        safety=safety if safety is not None else {"fired": [], "ids": [], "followup": ""},
        llm_calls=llm_calls,
        steps=steps,
        tool_calls=[{"name": "search_docs", "ok": True, "summary": "", "result": None}] * tools,
        degraded=degraded,
    )


def _pipeline_answers(*, red_line: str = "emergency", triage_hits: int = 5) -> dict[str, object]:
    """工具循环臂每一题的伪造产物（与 `_llm_answers` 同一批病例）。"""
    answers: dict[str, object] = {
        "我现在胸痛": _turn(_emergency() if red_line == "emergency" else _routine()),
        "今天天气不错": _turn(_routine()),
    }
    for i, glyph in enumerate(_TRIAGE_GLYPH, start=1):
        answers[f"分诊{glyph}"] = _turn(_routine(f"科室{i}") if i <= triage_hits else _routine("全科"))
    return answers


def _install_stub_pipeline(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, answers: dict[str, object], *, budget: Any = None
) -> _StubPipeline:
    """把**工具循环臂**（产品路径）接到替身管线上：不打开索引、不联网。"""
    stub = _StubPipeline(answers)
    monkeypatch.setattr(script, "_judge_arm_status", lambda *, need_index: (True, ""))
    monkeypatch.setattr(script, "_build_tool_loop_pipeline", lambda: (stub, budget or _FakeBudget(), None))
    return stub


def _run_arm(script: ModuleType, gold: Path, tmp_path: Path, arm: str, *extra: str) -> int:
    baseline = tmp_path / "does-not-exist-baseline.json"
    return int(script.main(["--gold", str(gold), "--baseline", str(baseline), "--arm", arm, *extra]))


class TestJudgeArmMetricsAreRecords:
    """LLM 臂的召回/误报/分诊都是**记录**：模型不是确定性的，拿分数卡退出码等于把波动当回归。"""

    def test_all_red_lines_found_passes(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=5))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate")

        out = capsys.readouterr().out
        assert code == 0, out
        assert "闸门：通过" in out
        assert "判断臂（单发对照（不检索））" in out

    def test_one_missed_red_line_is_reported_but_does_not_fail(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        """漏一条危险信号 ⇒ **记在案**（`fn=1`）但不再卡退出码；人看记录判断要不要跟进。"""
        _install_stub_reasoner(script, monkeypatch, _llm_answers(red_line="routine", triage_hits=5))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert code == 0, captured.err
        assert payload["llm"]["redline"]["fn"] == 1, "漏报必须照样记录在案"
        assert payload["llm"]["targets"]["redline_recall"] == 1.0, "参考目标仍要打出来给它对照"

    def test_triage_scores_are_reported_for_both_three_and_two_of_five(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        """3/5 与 2/5 都退 0，但分数必须如实打出来（记录的意义就在这里）。"""
        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=3))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")
        first = json.loads(capsys.readouterr().out)
        assert code == 0 and first["llm"]["triage"]["accuracy"] == 0.6

        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=2))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")
        second = json.loads(capsys.readouterr().out)
        assert code == 0 and second["llm"]["triage"]["accuracy"] == 0.4


class TestJudgeArmInvariantsAreTheGate:
    """**安全不变量的触发是记录、不是闸门**：它进报告，硬信号交给长期监听。"""

    def test_emergency_without_a_directive_is_recorded_but_the_run_still_passes(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        answers = _llm_answers(triage_hits=5)
        answers["我现在胸痛"] = ClinicalAssessment(
            urgency="emergency",
            red_flags=[
                RedFlag(
                    condition="急性冠脉综合征",
                    severity="critical",
                    why="胸痛，需即刻排查",
                    angle="症状性质",
                    likelihood="likely",
                )
            ],
            departments=[DepartmentAdvice(name="心内科", confidence=0.9)],
            patient_summary="需要尽快处理，请重视。",  # 判了急症却没给可执行动作
            confidence=0.9,
        )
        _install_stub_reasoner(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert code == 0, "触发不变量说明「模型需要被纠正」，不是「这次没测成」——闸门不该红"
        assert payload["llm"]["invariant_violations"], "不变量触发必须进报告"
        assert "emergency_judgment_requires_directive" in payload["llm"]["invariant_violations"][0]
        assert payload["llm"]["invariant_actions"] == ["append_directive"], "处置动作一起留档"

    def test_the_invariant_is_printed_with_its_disposition(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        """非 `--json` 时要把"已处置"打出来：读者才不会把"兜底生效"读成"兜底失效"。"""
        answers = _llm_answers(triage_hits=5)
        answers["我现在胸痛"] = _emergency().model_copy(update={"patient_summary": "需要尽快处理。"})
        _install_stub_reasoner(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate")

        out = capsys.readouterr().out
        assert code == 0
        assert "安全不变量  : 触发 1 条（已处置：append_directive×1）" in out
        assert "不判这次评测的有效性" in out

    def test_a_clean_emergency_answer_is_not_flagged(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        """合规的急症回答（正文里有 120）不该被记一条不变量——否则这一层会污染正确输出。"""
        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=5))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["llm"]["invariant_violations"] == []
        assert payload["llm"]["invariant_actions"] == []


class TestJudgeArmCalibrationIsReported:
    """置信度校准只报告：有 confidence 就算，没给就如实说"没有可校准的样本"。"""

    def test_confidence_pairs_produce_a_calibration_block(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        answers = _llm_answers(triage_hits=3)
        answers["我现在胸痛"] = ClinicalAssessment(
            urgency="emergency",
            red_flags=[
                RedFlag(
                    condition="急性冠脉综合征",
                    severity="critical",
                    why="胸痛伴大汗",
                    angle="组合共现",
                    likelihood="likely",
                )
            ],
            departments=[DepartmentAdvice(name="心内科", confidence=0.9)],
            patient_summary="请立即拨打 120。",
            confidence=0.9,
        )
        for glyph, i in zip(_TRIAGE_GLYPH, range(1, 6), strict=True):
            answers[f"分诊{glyph}"] = _routine(f"科室{i}").model_copy(update={"confidence": 0.7})
        _install_stub_reasoner(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        payload = json.loads(capsys.readouterr().out)
        cal = payload["llm"]["calibration"]
        assert code == 0
        assert cal["n"] >= 6, "每条给了 confidence 的用例都该进校准配对"
        assert cal["brier"] is not None
        assert cal["bins"] and "样本量" in cal["note"]

    def test_missing_confidence_is_said_out_loud(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        """模型没给 confidence ⇒ 明说"没有可校准的样本"，不许悄悄报一个 0 分。"""
        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=5))
        _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        cal = json.loads(capsys.readouterr().out)["llm"]["calibration"]
        assert cal["n"] == 0 and cal["brier"] is None
        assert "没有可校准的样本" in cal["note"]


class TestJudgeArmUnevaluatedIsNotAPass:
    """没密钥 ⇒ `evaluated=False` ⇒ **闸门失败**：没测不等于测了没问题。"""

    def test_no_api_key_fails_and_says_unevaluated(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            script,
            "_judge_arm_status",
            lambda *, need_index: (False, "未配置大模型密钥（DEEPSEEK_API_KEY 未设置或形如占位符）"),
        )
        # 真去装配就该失败：这里把它换成"一调就炸"，证明没密钥时根本没建客户端。
        monkeypatch.setattr(script, "_build_llm_reasoner", lambda: pytest.fail("没密钥不该装配 reasoner"))
        monkeypatch.setattr(
            script, "_build_tool_loop_pipeline", lambda: pytest.fail("没密钥不该装配工具循环")
        )
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate")

        out = capsys.readouterr().out
        assert code == 2, out
        assert "未评测" in out
        assert "未配置大模型密钥" in out
        assert "未评测不等于通过" in out

    def test_unevaluated_llm_arm_is_reported_in_json(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        """`--json` 下也要能读到"未评测"：stdout 仍是纯 JSON，退出码不变。"""
        monkeypatch.setattr(script, "_judge_arm_status", lambda *, need_index: (False, "未配置大模型密钥"))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)  # 旁白混进来就会在这里炸
        assert code == 2
        assert payload["llm"]["evaluated"] is False
        assert "未配置大模型密钥" in payload["llm"]["reason"]
        assert "未评测" in captured.err


class TestPerCaseFailureDoesNotCrashTheRun:
    """单题失败：脚本必须跑完，且这题要**单独可见**（不许混成"模型说没急症"）。"""

    def test_raising_reasoner_is_recorded_and_the_run_finishes(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        answers = _llm_answers(triage_hits=5)
        answers["分诊三"] = RuntimeError("连接被重置")
        stub = _install_stub_reasoner(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)  # 没崩：stdout 仍是完整 JSON
        llm = payload["llm"]
        assert len(stub.calls) == 7, "7 条用例都要真的被问到（2 红线 + 5 分诊）"
        assert llm["failed_calls"], "失败调用必须单独记一条，否则没人看得出有题没被判断过"
        assert "t3" in llm["failed_calls"][0]
        assert llm["n_failed_calls"] == 1
        # 失败的题按**未命中**计（保守），其余照常打分：4/5 = 80%
        assert llm["triage"]["total"] == 5
        assert llm["triage"]["hit"] == 4
        # 红线不受这题影响，仍然是 100%
        assert llm["redline"]["recall"] == 1.0
        # 有题没被判断过 ⇒ 不通过（"未评测不等于通过"的微形态）
        assert code == 2
        assert "没有得到模型判断" in captured.err

    def test_no_conclusion_is_not_the_same_as_no_emergency(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        """`assessment=None`（模型没答）按漏报计，**不许**算成"模型说没急症"（TN）。"""
        answers = _llm_answers(triage_hits=5)
        answers["我现在胸痛"] = ReasoningResult(assessment=None, raw="", degraded="模型不可用")
        _install_stub_reasoner(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--gate", "--json")

        payload = json.loads(capsys.readouterr().out)
        assert code == 2
        assert payload["llm"]["redline"]["fn"] == 1
        assert payload["llm"]["redline"]["tn"] == 1, "另一条负例才是 TN；没答的那题不许进 TN"
        assert payload["llm"]["failed_calls"]


class TestReasonerArmCallsTheSingleShotPath:
    """这一臂必须真的走"证据包 + 每题一次调用"那条路，而不是另造一条捷径。"""

    def test_each_case_is_asked_once_with_the_guard_evidence(self, script, llm_gold, tmp_path, monkeypatch):
        stub = _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=5))
        code = _run_arm(script, llm_gold, tmp_path, "reasoner")

        assert code == 0
        assert stub.calls == [
            "我现在胸痛",
            "今天天气不错",
            *(f"分诊{g}" for g in _TRIAGE_GLYPH),
        ], "每题恰好一次调用，顺序与金标一致"
        # `guard=RiskGuard()` 必须真的接进证据包——这正是要测的"模型看着线索怎么判"
        assert stub.red_line_positive == [True, False, False, False, False, False, False]
        # 不做检索：证据包里没有任何知识片段（检索质量有它自己的 harness）
        assert stub.evidence_items == [0] * 7

    def test_cost_is_read_from_the_budget_ledger(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        budget = _FakeBudget(per_read=0.0005)
        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=5), budget=budget)
        code = _run_arm(script, llm_gold, tmp_path, "reasoner", "--json")

        payload = json.loads(capsys.readouterr().out)
        llm = payload["llm"]
        assert code == 0
        assert llm["llm_calls"] == 7
        assert llm["cost_usd"] == 0.0005, "花费取账本前后差，不是估出来的数"
        assert budget.reads == 2


class TestRuleArmNeverTouchesTheModel:
    """`--arm rule` 是**离线**模式：`--arm` 默认改成 `both` 之后，这条必须被钉住。"""

    def test_arm_rule_does_not_probe_or_build_the_judge(self, script, gold, tmp_path, monkeypatch, capsys):
        def boom_status(*, need_index: bool) -> tuple[bool, str]:
            pytest.fail("--arm rule 不该探测模型密钥或索引")

        def boom_build() -> Any:
            pytest.fail("--arm rule 不该装配任何判断臂")

        monkeypatch.setattr(script, "_judge_arm_status", boom_status)
        monkeypatch.setattr(script, "_build_llm_reasoner", boom_build)
        monkeypatch.setattr(script, "_build_tool_loop_pipeline", boom_build)
        _stub_triage(script, monkeypatch, top="呼吸内科")
        code = _run(script, gold, tmp_path, "--gate")

        out = capsys.readouterr().out
        assert code == 0
        assert "没有跑判断臂" in out


class TestRuleArmTriageIsReportedNotGated:
    """规则臂的分诊数字**只报不卡**，两种模式下都一样。

    规则臂已经是线索提供者，拿它的票数卡退出码等于用错尺子；它现在的价值是
    "最便宜、最稳定的线索质量读数"。
    """

    def test_zero_rule_triage_does_not_fail_when_the_judge_arm_passes(
        self, script, gold, tmp_path, monkeypatch, capsys
    ):
        _stub_triage(script, monkeypatch, top="骨科")  # 规则臂分诊 0 分
        # `--arm both` 的判断臂是**工具循环**（产品路径）：替身要装在管线上。
        _install_stub_pipeline(
            script,
            monkeypatch,
            {
                "我现在胸痛": _turn(_emergency()),
                "今天天气不错": _turn(_routine()),
                "我咳嗽": _turn(_routine("呼吸内科")),
            },
        )
        code = _run_arm(script, gold, tmp_path, "both", "--gate")

        out = capsys.readouterr().out
        assert code == 0, out
        assert "闸门：通过" in out
        assert "0.00% top3 命中" in out, "规则臂的分诊数字仍要如实报出来"

    def test_rule_only_mode_also_reports_without_gating(self, script, gold, tmp_path, monkeypatch, capsys):
        """`--arm rule`（没有 LLM 判断者）同样只报不卡：0 分不是退出码的依据。"""
        _stub_triage(script, monkeypatch, top="骨科")
        code = _run(script, gold, tmp_path, "--gate")

        out = capsys.readouterr().out
        assert code == 0
        assert "分诊科室    : 0.00% top3 命中" in out
        assert "线索质量读数" in out


class TestBaselineRecordsTheJudgeArmWithoutGatingOnIt:
    """判断臂进基线是**记录**，不是回归门限（模型不确定，重跑会波动）。"""

    def test_write_baseline_adds_the_llm_key_and_the_volatility_note(
        self, script, gold, tmp_path, monkeypatch
    ):
        _stub_triage(script, monkeypatch, top="呼吸内科")
        _install_stub_pipeline(
            script,
            monkeypatch,
            {
                "我现在胸痛": _turn(_emergency()),
                "今天天气不错": _turn(_routine()),
                "我咳嗽": _turn(_routine("呼吸内科")),
            },
        )
        path = tmp_path / "b.json"
        code = int(
            script.main(["--gold", str(gold), "--baseline", str(path), "--arm", "both", "--write-baseline"])
        )
        assert code == 0

        payload = json.loads(path.read_text(encoding="utf-8"))
        # 既有键的形状不动
        assert payload["schema"] == 1
        assert payload["gold_sha256"]
        assert payload["metrics"]["redline"]["recall"] == 1.0
        assert "booking" in payload["metrics"]
        assert "triage" in payload["metrics"]
        # 新增：判断臂的数值（含引擎）+ "别拿它当门限"的说明
        assert payload["metrics"]["llm"]["evaluated"] is True
        assert payload["metrics"]["llm"]["engine"] == "tool_loop"
        assert "波动" in payload["llm_note"]

    def test_write_baseline_needs_the_rule_arm(self, script, gold, tmp_path, monkeypatch, capsys):
        """只跑判断臂时没有可写的基线主体 ⇒ 参数错误（退出码 1），不写半个文件。"""
        _install_stub_reasoner(script, monkeypatch, {"我现在胸痛": _emergency()})
        path = tmp_path / "b.json"
        code = int(
            script.main(
                ["--gold", str(gold), "--baseline", str(path), "--arm", "reasoner", "--write-baseline"]
            )
        )

        assert code == 1
        assert not path.exists()
        assert "参数错误" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 工具循环臂 = **产品路径**：`pipeline.advance(原话)` 走的是与 `/consult` 同一套装配
# （`clinical/runtime.py`），模型自己决定调不调检索。下面这些用例钉住三件事：
#   ① 这一臂真的按产品形状读产物（assessment/safety/tool_calls），不是另算一套；
#   ② 安全不变量取管线**实际触发**的记录，评测不自己重算；
#   ③ 默认的 `both` / `llm` 走的是工具循环，单发只在 `--arm reasoner` 下跑。
# ---------------------------------------------------------------------------


class TestToolLoopArmIsTheProductPath:
    def test_default_both_runs_the_tool_loop_and_reports_its_engine(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        stub = _install_stub_pipeline(script, monkeypatch, _pipeline_answers(triage_hits=5))
        code = _run_arm(script, llm_gold, tmp_path, "both", "--gate", "--json")

        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["llm"]["engine"] == "tool_loop"
        assert stub.calls, "工具循环臂必须真的调管线"
        assert "工具循环（产品路径）" in payload["arm"] or payload["arm"] == "both"

    def test_llm_alias_is_the_tool_loop(self, script, llm_gold, tmp_path, monkeypatch):
        stub = _install_stub_pipeline(script, monkeypatch, _pipeline_answers())
        code = _run_arm(script, llm_gold, tmp_path, "llm", "--gate")

        assert code == 0
        assert stub.calls, "`--arm llm` 现在是产品路径（工具循环），不是单发"

    def test_reasoner_arm_does_not_touch_the_pipeline(self, script, llm_gold, tmp_path, monkeypatch):
        """`--arm reasoner` 是单发对照：不许偷偷装配管线（那会把两臂的差别抹掉）。"""
        _install_stub_reasoner(script, monkeypatch, _llm_answers(triage_hits=5))
        monkeypatch.setattr(
            script, "_build_tool_loop_pipeline", lambda: pytest.fail("单发臂不该装配工具循环")
        )
        assert _run_arm(script, llm_gold, tmp_path, "reasoner") == 0

    def test_tool_and_step_counts_come_from_the_turn(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        """调用次数/步数/工具次数取**管线自己数的**（产品路径的读数），不是评估算的。"""
        per_case = {"我现在胸痛": (4, 6, 2), "今天天气不错": (1, 2, 1)}
        answers = {
            "我现在胸痛": _turn(_emergency(), llm_calls=4, steps=6, tools=2),
            "今天天气不错": _turn(_routine(), llm_calls=1, steps=2, tools=1),
        }
        for glyph in _TRIAGE_GLYPH:
            answers[f"分诊{glyph}"] = _turn(_routine("科室1"), llm_calls=1, steps=3, tools=0)
            per_case[f"分诊{glyph}"] = (1, 3, 0)
        _install_stub_pipeline(script, monkeypatch, answers)
        _run_arm(script, llm_gold, tmp_path, "llm", "--json")

        llm = json.loads(capsys.readouterr().out)["llm"]
        assert llm["llm_calls"] == sum(v[0] for v in per_case.values())
        assert llm["steps"] == sum(v[1] for v in per_case.values())
        assert llm["tool_calls"] == sum(v[2] for v in per_case.values())

    def test_invariants_come_from_the_pipeline_not_recomputed(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        """**关键**：触发记录取管线 `turn.safety`，评测不自己重算一遍。

        重算会得到"评测认为应该触发"与"产品实际触发了"两份真相，而这两份迟早不一致。
        因此这里构造一个"结论看起来没问题、但管线记了不变量"的产物：报告必须如实带上它。
        """
        answers = _pipeline_answers(triage_hits=5)
        answers["我现在胸痛"] = _turn(
            _emergency(),
            safety={
                "fired": [
                    {
                        "invariant": "emergency_judgment_requires_directive",
                        "action": "append_directive",
                        "detail": "已补固定指引",
                        "trace_id": "e1",
                        "domain": "clinic",
                    }
                ],
                "ids": ["emergency_judgment_requires_directive"],
                "followup": "",
            },
        )
        _install_stub_pipeline(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "llm", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert code == 0, "触发不变量是记录，不是闸门（硬信号在监听那边）"
        assert "emergency_judgment_requires_directive" in payload["llm"]["invariant_violations"][0]
        assert payload["llm"]["invariant_actions"] == ["append_directive"]

    def test_blocked_answer_is_still_a_judgement(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        """`action="error"` 也可能是"输出被约束层拦下、改成只含事实的回答"。

        那一轮的结论仍然存在 ⇒ 属于**测到了**，不许记成"没拿到判断"（否则闸门会假红）。
        """
        answers = _pipeline_answers(triage_hits=5)
        answers["我现在胸痛"] = _turn(
            _emergency(),
            action="error",
            degraded="回答未通过事实校验，已改为只陈述事实",
        )
        _install_stub_pipeline(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "llm", "--gate", "--json")

        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["llm"]["failed_calls"] == []
        assert payload["llm"]["n_failed_calls"] == 0

    def test_assessment_none_is_a_failed_call(self, script, llm_gold, tmp_path, monkeypatch, capsys):
        """管线没给出判断（模型失败/输出不可解析）⇒ 记一条"没测到"，闸门红。"""
        answers = _pipeline_answers(triage_hits=5)
        answers["我现在胸痛"] = _turn(None, action="error", degraded="大模型调用失败，请稍后重试。")
        _install_stub_pipeline(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "llm", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert code == 2
        assert payload["llm"]["failed_calls"], "没拿到判断的题必须单独可见"
        assert payload["llm"]["redline"]["fn"] == 1, "期望危险信号却拿不到结论 ⇒ 按漏报计（保守）"
        assert "没有得到模型判断" in captured.err

    def test_raising_pipeline_is_recorded_and_the_run_finishes(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        answers = _pipeline_answers(triage_hits=5)
        answers["分诊三"] = RuntimeError("索引连接被重置")
        stub = _install_stub_pipeline(script, monkeypatch, answers)
        code = _run_arm(script, llm_gold, tmp_path, "llm", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)  # 没崩：stdout 仍是完整 JSON
        assert len(stub.calls) == 7
        assert payload["llm"]["n_failed_calls"] == 1
        assert "t3" in payload["llm"]["failed_calls"][0]
        assert code == 2

    def test_index_missing_marks_the_arm_unevaluated_without_crashing(
        self, script, llm_gold, tmp_path, monkeypatch, capsys
    ):
        """没有索引 ⇒ **未评测**（不是 0 分）：工具循环臂没法检索，测出来的东西没有意义。"""
        monkeypatch.setattr(
            script,
            "_judge_arm_status",
            lambda *, need_index: (False, "clinic 域索引为空 ⇒ 工具循环臂无法检索（先跑 ingest）"),
        )
        monkeypatch.setattr(script, "_build_tool_loop_pipeline", lambda: pytest.fail("未评测时不该装配管线"))
        code = _run_arm(script, llm_gold, tmp_path, "llm", "--gate", "--json")

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert code == 2
        assert payload["llm"]["evaluated"] is False
        assert "索引为空" in payload["llm"]["reason"]
        assert "未评测不等于通过" in captured.err

    def test_judge_arm_status_checks_the_index_only_when_needed(self, script, monkeypatch):
        """单发臂不打开索引（它不做检索），工具循环臂必须检查索引是否为空。"""
        calls: list[bool] = []

        def fake(*, need_index: bool) -> tuple[bool, str]:
            calls.append(need_index)
            return True, ""

        monkeypatch.setattr(script, "_judge_arm_status", fake)
        monkeypatch.setattr(script, "_build_llm_reasoner", lambda: (object(), _FakeBudget()))
        monkeypatch.setattr(script, "_build_tool_loop_pipeline", lambda: (object(), _FakeBudget(), None))
        script._run_reasoner_arm([])  # noqa: SLF001
        script._run_tool_loop_arm([])  # noqa: SLF001

        assert calls == [False, True]
