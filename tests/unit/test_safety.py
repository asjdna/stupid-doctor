"""确定性安全门（第三层）的契约测试：**底线必须真的会咬人**。

这一层是三条不变量（`clinical/safety.py` 的 `INVARIANTS`）：

- `no_medical_advice_without_evidence`：没有检索到资料就不给医学内容；
- `no_fabricated_booking_action`：不得声称服务端没执行过的挂号动作；
- `emergency_judgment_requires_directive`：判了急症必须给出可执行的急救动作。

前两条由 `verify.py` 检出、这里只核对它们的**登记与留档**；第三条在这里直接执行。
测试的重点不是"函数返回了对的东西"，而是：**该补的补上了、该留的留下来了、
不该动的一个字都没动**——一个会误伤正确回答的安全层，比没有安全层更糟。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kubedocs.clinical.contracts import ClinicalAssessment, RedFlag
from kubedocs.clinical.safety import (
    EMERGENCY_DIRECTIVE,
    INVARIANTS,
    INVARIANTS_BY_ID,
    SafetyAudit,
    SafetyOutcome,
    append_audit,
    apply_gate,
    audit_path,
    ensure_emergency_directive,
    fallback_question,
    has_actionable_directive,
    prose_gate,
    read_audit,
    record_invariants,
)


def _assessment(**over: object) -> ClinicalAssessment:
    base: dict[str, object] = {
        "urgency": "routine",
        "departments": [{"name": "呼吸内科"}],
        "patient_summary": "建议到呼吸内科就诊。",
        "confidence": 0.8,
    }
    base.update(over)
    return ClinicalAssessment(**base)


def _signal(**over: object) -> RedFlag:
    base: dict[str, object] = {
        "condition": "急性冠脉综合征",
        "severity": "severe",
        "why": "胸痛伴气短",
        "angle": "组合共现",
        "likelihood": "possible",
    }
    base.update(over)
    return RedFlag(**base)


class TestInvariantRegistry:
    """注册表本身：它是治理清单与测试覆盖的锚点，所以形状要能被机械核对。"""

    def test_all_three_ids_are_registered(self) -> None:
        assert {i.id for i in INVARIANTS} == {
            "no_medical_advice_without_evidence",
            "no_fabricated_booking_action",
            "emergency_judgment_requires_directive",
        }

    def test_ids_are_unique_and_fields_are_filled(self) -> None:
        ids = [i.id for i in INVARIANTS]
        assert len(ids) == len(set(ids)), "不变量 id 重复：治理清单会指向两个东西"
        for inv in INVARIANTS:
            assert inv.title.strip() and inv.rationale.strip(), f"{inv.id} 缺说明"
            assert inv.implemented_in, f"{inv.id} 没写实现位置"

    def test_lookup_map_matches_the_registry(self) -> None:
        assert {i.id: i for i in INVARIANTS} == INVARIANTS_BY_ID

    def test_booking_invariant_covers_both_entry_points(self) -> None:
        """挂号动作幻觉是两条入口共有的底线；急救指引那条拿不到结构化判断，只在 /consult 上成立。

        这个差别必须写在注册表里，而不是假装两条路径一样。
        """
        assert set(INVARIANTS_BY_ID["no_fabricated_booking_action"].scope) == {"consult", "chat"}
        assert INVARIANTS_BY_ID["emergency_judgment_requires_directive"].scope == ("consult",)
        assert (
            "src/kubedocs/clinical/safety.py"
            in INVARIANTS_BY_ID["emergency_judgment_requires_directive"].implemented_in
        )


class TestActionableDirective:
    """ "可执行"是有判据的：只认去向（120 / 急诊 / 救护车 / 立刻就医），不认"要注意"。"""

    @pytest.mark.parametrize(
        "text",
        ["请立刻拨打 120", "建议尽快前往急诊科", "让家人叫救护车", "请立即就医", "马上送医"],
    )
    def test_real_actions_are_recognized(self, text: str) -> None:
        assert has_actionable_directive(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "请多喝热水，注意休息",
            "如果加重了要重视，不能大意",
            "建议尽快到呼吸内科门诊就诊",  # 门诊是常规去向，不是急救动作
        ],
    )
    def test_words_without_an_action_do_not_count(self, text: str) -> None:
        assert not has_actionable_directive(text)


class TestEmergencyDirectiveAmendment:
    """`emergency_judgment_requires_directive`：**只加不减**，而且幂等。"""

    def test_missing_directive_is_appended(self) -> None:
        out, appended = ensure_emergency_directive("您的情况需要立刻处理。")
        assert appended is True
        assert out.startswith("您的情况需要立刻处理。")
        assert EMERGENCY_DIRECTIVE in out

    def test_existing_directive_is_left_untouched(self) -> None:
        text = "请立刻拨打 120 或前往急诊科。"
        out, appended = ensure_emergency_directive(text)
        assert out == text, "已经有急救动作还被改写：这一层会污染正确回答"
        assert appended is False

    def test_amendment_is_idempotent(self) -> None:
        once, _ = ensure_emergency_directive("需要立刻处理。")
        twice, appended = ensure_emergency_directive(once)
        assert twice == once and appended is False

    def test_directive_contains_no_diagnosis(self) -> None:
        """补进去的话只能有动作，不能有断言——断言需要依据，而这条不变量正在依据之外兜底。"""
        for word in ("可能", "疑似", "诊断为"):
            assert word not in EMERGENCY_DIRECTIVE


class TestFallbackQuestion:
    """主动问诊层没给出追问时，服务端补一句**确定性**的追问（不含判断）。"""

    def test_prefers_the_model_discriminator(self) -> None:
        q = fallback_question([_signal(discriminator="有没有大汗或向左肩放射？")])
        assert q.startswith("关于「急性冠脉综合征」")
        assert "有没有大汗或向左肩放射？" in q

    def test_falls_back_to_a_generic_question_by_condition(self) -> None:
        q = fallback_question([_signal(), _signal(condition="脑卒中")])
        assert "急性冠脉综合征" in q and "脑卒中" in q
        assert "越来越重" in q

    def test_empty_signals_still_produce_a_question(self) -> None:
        """连病症都没有时也要给一句能问出口的话——空字符串会让前端显示一条空追问。"""
        assert fallback_question([]).strip()


class TestInvariantRecording:
    """违规类别 → 不变量的映射：安全事件只登记真正属于底线的那几类。"""

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("medical_without_evidence", "no_medical_advice_without_evidence"),
            ("fabricated_booking", "no_fabricated_booking_action"),
            ("fabricated_cancellation", "no_fabricated_booking_action"),
            ("unknown_ticket", "no_fabricated_booking_action"),
        ],
    )
    def test_blocking_kinds_map_to_invariants(self, kind: str, expected: str) -> None:
        fired = record_invariants([kind])
        assert [f.invariant for f in fired] == [expected]
        assert fired[0].action == INVARIANTS_BY_ID[expected].action

    @pytest.mark.parametrize("kind", ["unknown_amount", "red_line_unexplained", "fabricated_evidence"])
    def test_non_invariant_kinds_are_not_safety_events(self, kind: str) -> None:
        """金额/红线未解释/编造依据编号是回答质量问题，**不该冒充安全事件**：

        混进来会让"安全门触发次数"这个读数失真，而那是运维唯一能盯的读数。
        """
        assert record_invariants([kind]) == []

    def test_duplicate_kinds_are_recorded_once(self) -> None:
        """同一类别出现多次只留一条：留档要能数清"触发了几条不变量"。"""
        assert len(record_invariants(["fabricated_booking", "fabricated_booking"])) == 1


class TestApplyGate:
    """`apply_gate` 是第三层的执行体：该补的补、该问的问、不该动的别动。"""

    def test_emergency_without_directive_is_amended_and_recorded(self) -> None:
        a = _assessment(
            urgency="emergency",
            red_flags=[_signal(likelihood="likely")],
            patient_summary="需要立刻处理。",
        )
        out = apply_gate(a, a.patient_summary, violated_kinds=[])
        assert EMERGENCY_DIRECTIVE in out.message
        assert "emergency_judgment_requires_directive" in out.ids

    def test_emergency_with_directive_is_not_recorded(self) -> None:
        a = _assessment(urgency="emergency", red_flags=[_signal(likelihood="likely")])
        out = apply_gate(a, "请立刻拨打 120。", violated_kinds=[])
        assert out.message == "请立刻拨打 120。"
        assert out.ids == []

    def test_detected_violations_are_recorded_but_not_rewritten_here(self) -> None:
        """`replace_answer` 那两条的**改写**由 `verify.py` 完成，这里只登记。

        判定与执行写在两处会静默分歧，所以留档一定要在同一处收口。
        """
        a = _assessment()
        out = apply_gate(a, "好的。", violated_kinds=["fabricated_booking", "unknown_amount"])
        assert out.message == "好的。"
        assert out.ids == ["no_fabricated_booking_action"]

    def test_unresolved_signal_without_question_gets_a_fallback(self) -> None:
        a = _assessment(urgency="urgent", red_flags=[_signal()])
        out = apply_gate(a, "建议尽快就诊。", violated_kinds=[])
        assert out.followup, "未排除的危险信号没有下文时必须补一句追问"
        assert "急性冠脉综合征" in out.followup

    def test_signal_that_was_asked_about_gets_no_fallback(self) -> None:
        a = _assessment(
            urgency="urgent",
            red_flags=[_signal()],
            followup_questions=["有没有大汗或放射痛？"],
        )
        out = apply_gate(a, "想先确认一下：有没有大汗或放射痛？", violated_kinds=[])
        assert out.followup == ""
        assert out.message == "想先确认一下：有没有大汗或放射痛？"

    def test_ruled_out_signal_gets_no_fallback(self) -> None:
        a = _assessment(
            red_flags=[_signal(likelihood="unlikely", ruled_out_by="按压可诱发，无大汗")],
        )
        out = apply_gate(a, "建议门诊就诊。", violated_kinds=[])
        assert out.followup == ""

    def test_prose_gate_only_records(self) -> None:
        """/chat 的散文路径：没有结构化判断，所以只留档（急救指引那条不适用）。"""
        out = prose_gate(["fabricated_booking"], trace_id="p1")
        assert out.message == ""
        assert out.ids == ["no_fabricated_booking_action"]
        assert out.fired[0].domain == "chat"
        assert out.fired[0].trace_id == "p1"

    def test_payload_shape_is_stable(self) -> None:
        out = apply_gate(_assessment(), "好的。", violated_kinds=[])
        assert set(out.to_payload()) == {"fired", "ids", "followup"}


class TestAuditLog:
    """审计：**结构最小化**（不记患者原话），写失败不影响这一轮。"""

    def test_audit_path_lives_under_the_index_root(self, tmp_path: Path) -> None:
        assert audit_path(tmp_path) == tmp_path / "safety_audit.jsonl"

    def test_records_are_appended_as_jsonl(self, tmp_path: Path) -> None:
        outcome = SafetyOutcome(
            message="m",
            fired=[
                SafetyAudit(
                    "no_fabricated_booking_action", "replace_answer", "fabricated_booking", "tr1", "clinic"
                )
            ],
        )
        assert append_audit(tmp_path, outcome) == 1
        rows = read_audit(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["invariant"] == "no_fabricated_booking_action"
        assert row["trace_id"] == "tr1"
        assert row["ts"], "审计记录必须带时间戳（否则无法与日志/就诊记录对齐）"

    def test_no_patient_text_is_written(self, tmp_path: Path) -> None:
        """**患者原话不进审计日志**：临床文本已经落在临床库里，再抄一份只会扩大泄露面。"""
        secret = "我胸痛已经三天了，还咳血"
        outcome = SafetyOutcome(
            message=secret,
            fired=[
                SafetyAudit(
                    "emergency_judgment_requires_directive",
                    "append_directive",
                    "urgency=emergency",
                    "tr1",
                    "clinic",
                )
            ],
        )
        append_audit(tmp_path, outcome)
        raw = audit_path(tmp_path).read_text(encoding="utf-8")
        assert secret not in raw
        assert "胸痛" not in raw
        json.loads(raw.strip())  # 仍然是合法 JSONL

    def test_nothing_fired_means_nothing_written(self, tmp_path: Path) -> None:
        assert append_audit(tmp_path, SafetyOutcome(message="m")) == 0
        assert not audit_path(tmp_path).exists(), "没有触发就不该建文件：空文件会被误读成'跑过'"

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_audit(tmp_path) == []

    def test_read_skips_corrupt_lines_instead_of_raising(self, tmp_path: Path) -> None:
        path = audit_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"invariant": "x"}\n不是 json\n', encoding="utf-8")
        rows = read_audit(tmp_path)
        assert [r["invariant"] for r in rows] == ["x"]

    def test_write_failure_is_swallowed(self, tmp_path: Path) -> None:
        """磁盘问题不该影响这一轮应答（与落库同一条纪律），但也不能抛。"""
        blocked = tmp_path / "blocked"
        blocked.write_text("占位文件，不是目录", encoding="utf-8")
        outcome = SafetyOutcome(message="m", fired=[SafetyAudit("x", "record_only", "", "tr", "clinic")])
        assert append_audit(blocked, outcome) == 0

    def test_read_returns_the_most_recent_records(self, tmp_path: Path) -> None:
        for i in range(5):
            append_audit(
                tmp_path,
                SafetyOutcome(
                    message="m", fired=[SafetyAudit("x", "record_only", f"第{i}条", f"tr{i}", "clinic")]
                ),
            )
        rows = read_audit(tmp_path, limit=2)
        assert [r["trace_id"] for r in rows] == ["tr3", "tr4"]
