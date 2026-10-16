"""LLM 驱动的 clinic 路径测试：判断权归模型、面板进提示词、工具接线、确认策略。

## 本文件围绕的新契约

1. **没有规则短路**：红线词表 + 三维门控、症状词表、症状→科室票数都只进「参考标准」
   面板，随系统提示交给模型。把"我胸痛"发进来时，模型**必须真的被调用**（工具循环跑起来），
   判不判急症由模型给出——不再有"命中词表就回急诊话术、模型一次都不调"的分支。
2. `ClinicTurn.emergency` 是**展示层检测**（模型正文里有没有「120 / 急诊」），不是医学判断：
   词表命中而模型没判急症时它必须是 False。
3. **没有大模型密钥时只回一句话**（`NO_LLM_MESSAGE`）：不装配面板、不进循环、不做兜底。
4. 与规则无关的那条安全不变量照旧：副作用工具没有 `confirm` 回调一律被拒，
   模型只能提议，取号要等患者点按钮。

本文件不验证回答质量，只验证控制流归谁、以及模型看到了什么。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from kubedocs.agent.clinic_react import ClinicReactRuntime
from kubedocs.agent.tools import ToolRegistry
from kubedocs.clinical.reference import NO_BOOKING_FACT_NOTE
from kubedocs.config import Settings
from kubedocs.domains.clinic.booking import BookingService
from kubedocs.domains.clinic.mock import MockAppointmentProvider
from kubedocs.domains.clinic.pending import PendingBookings
from kubedocs.llm_status import NO_LLM_MESSAGE

LINKS: dict[str, tuple[str, ...]] = {
    "咳嗽": ("呼吸内科", "感染科"),
    "胸痛": ("心内科", "急诊科"),
    "发热": ("感染科", "呼吸内科"),
}
"""打桩的「症状 → 候选科室」关联表。

真实语料是 8MB / 几万张卡（解析一次秒级），单测不该读它；而这正好也验证了
"关联表由调用方提供"这一点——换一张表，面板里的候选就跟着换。
"""


@pytest.fixture(autouse=True)
def _stub_triage_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """面板一律用打桩关联表（理由见 `LINKS`）：单元测试不依赖磁盘语料。"""
    monkeypatch.setattr("kubedocs.agent.clinic_react._triage_links", lambda: LINKS)


@pytest.fixture
def runtime(tmp_path) -> ClinicReactRuntime:
    ToolRegistry.clear()
    settings = Settings(
        index_dir=str(tmp_path),
        domain="clinic",
        embedding_provider="hashing",
        embedding_dim=32,
        llm_model="deepseek/deepseek-chat",
    )
    # `pending` 与生产一致（`app.state.clinic_pending`）：没有它就不会注册
    # `propose_booking`，而"模型只能提议"这条策略正是靠那个工具成立的。
    rt = ClinicReactRuntime(settings, BookingService(MockAppointmentProvider()), pending=PendingBookings())
    yield rt
    ToolRegistry.clear()


def _capture_caller(monkeypatch: pytest.MonkeyPatch, answer: str) -> list[list[dict[str, Any]]]:
    """把 `ToolCaller` 换成**记录调用**的打桩实现，返回它收到的 messages 列表。

    打桩的 `choose` 直接给出最终回答（不调工具、不联网），但**它被调用了这件事本身**
    才是这些用例要断言的东西：判断权归模型的判据是"模型真的跑过"。
    """
    seen: list[list[dict[str, Any]]] = []

    class _Capturing:
        def __init__(self, _settings: Any) -> None:
            pass

        def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            seen.append(copy.deepcopy(messages))
            return {"content": answer, "tool_calls": []}

    monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: True)
    # `turn()` 在函数体内 `from ... import LiteLLMToolCaller`，所以补丁打在被导入的模块上
    monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _Capturing)
    return seen


class TestTheModelIsTheOnlyJudge:
    """红线不再短路：命中词表也要进模型，判不判急症由模型给出。"""

    def test_red_line_hit_still_calls_the_model(self, runtime, monkeypatch):
        """**回归（新契约）**：把"我胸痛"发给 `turn()`，模型必须被调用。

        以前这条路径是确定性短路：`RiskGuard.check` 一命中就回急诊话术，LLM 一次都不调
        （旧断言写的就是"红线命中后仍然进入了工具循环"要失败）。那条短路已经删掉——
        红线词只是面板里的一条线索，所以这里断言打桩的 `ToolCaller` **确实被调用过**，
        并且模型的正文里出现「120」时展示层会给出 `emergency=True`。
        """
        seen = _capture_caller(monkeypatch, "情况紧急：请立即拨打 120 或前往最近医院急诊。")

        turn = runtime.turn("我现在胸痛")

        assert seen, "红线命中时模型仍必须被调用：规则短路已经拆掉"
        assert "我现在胸痛" in seen[0][-1]["content"]
        assert turn.emergency is True
        assert "120" in turn.text

    def test_emergency_follows_the_model_not_the_keyword_table(self, runtime, monkeypatch):
        """词表命中、但模型没判急症 ⇒ `emergency=False`。

        横幅跟模型的话走：`emergency` 只是展示层从正文里读出来的（有没有「120 / 急诊」），
        它既不复述词表命中，也不参与医学推理。
        """
        _capture_caller(monkeypatch, "建议挂心内科门诊，方便时尽快来做进一步检查。")

        turn = runtime.turn("我现在胸痛")

        assert turn.reference is not None
        red = turn.reference["red_line"]
        assert red is not None and red["reportable"] is True, "面板里应当留着词表命中供审计"
        assert turn.emergency is False, "模型没判急症，展示层不该弹横幅"

    @pytest.mark.parametrize(
        "text,why",
        [
            ("我老公胸痛，我没事", "家属"),
            ("我去年胸痛过一次，现在好了", "既往"),
            ("我没有胸痛", "否认"),
            ("如果胸痛怎么办", "假设"),
        ],
    )
    def test_non_reportable_hits_do_not_pop_the_banner(self, runtime, monkeypatch, text, why):
        """四类"出现红线词但不作本人当前主诉"的输入：模型被叫到了，但正文里没有急症措辞。

        这里断言的是展示层（不给这四类弹横幅）；"三维门控怎么判"由规则层自己的用例覆盖
        （`test_clinic.py`），而判断本身已经不在这一层了。
        """
        seen = _capture_caller(monkeypatch, "建议挂心内科门诊，方便时来做进一步检查。")

        turn = runtime.turn(text)

        assert seen, f"这类输入同样要进模型（{why}）"
        assert turn.emergency is False, f"误弹急诊横幅（{why}）"
        assert turn.reference is not None


class TestReferencePanel:
    """参考标准面板：规则资产说的每一句话都要真的进到模型收到的 messages 里。"""

    def test_panel_and_the_no_booking_fact_reach_the_model(self, runtime, monkeypatch):
        """**关键断言**：系统提示里有「参考标准」，也有"本轮没有已确认的挂号动作"那句事实。

        挂号那句必须**显式出现**：模型看到空白会按对话语气补一句"已为您挂上"，
        而看到明确的否定才有一个可对照的事实（`verify.py` 的 `fabricated_booking` 也靠它）。
        """
        seen = _capture_caller(monkeypatch, "好的，请继续观察。")

        runtime.turn("我咳嗽", patient_id="p-panel")

        system = seen[0][0]
        assert system["role"] == "system"
        assert "参考标准" in system["content"], "面板没有拼进系统提示"
        assert NO_BOOKING_FACT_NOTE in system["content"], "本轮没有挂号的否定句必须出现"
        assert "挂号" in system["content"]

    def test_confirmed_booking_appears_as_a_server_side_fact(self, runtime, monkeypatch):
        """服务端真确认过的挂号要出现在面板里，模型才有一个可对照的事实。"""
        runtime.service.book("心内科", "p-book", time_slot="上午")
        seen = _capture_caller(monkeypatch, "帮您核对一下。")

        turn = runtime.turn("我挂上了吗", patient_id="p-book")

        assert "服务端事实：已确认挂号" in seen[0][0]["content"]
        assert turn.reference is not None
        assert [b["kind"] for b in turn.reference["booking"]] == ["confirmed"]
        assert NO_BOOKING_FACT_NOTE not in seen[0][0]["content"]

    def test_booking_lookup_failure_is_reported_instead_of_silently_empty(self, runtime, monkeypatch):
        """读挂号记录失败时**不许说成"没有动作"**：面板要写明这次业务事实不完整。

        静默返回空列表会把"查不到"读成"没发生过"，那是本模块最危险的一种沉默。
        """
        seen = _capture_caller(monkeypatch, "我们核实一下。")

        def _boom(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("号源库连不上")

        monkeypatch.setattr(runtime.service, "my_appointments", _boom)

        runtime.turn("我挂上了吗", patient_id="p-broken")

        assert "挂号记录读取失败" in seen[0][0]["content"]
        assert "业务事实可能不完整" in seen[0][0]["content"]

    def test_reference_is_populated_and_json_serializable(self, runtime, monkeypatch):
        """`ClinicTurn.reference` 非空且能 JSON 序列化（响应与审计都要用它）。"""
        _capture_caller(monkeypatch, "好的。")

        turn = runtime.turn("我咳嗽", patient_id="p-json")

        assert turn.reference, "面板留档不能是空的"
        payload = json.loads(json.dumps(turn.reference))
        assert payload["known_departments"], "可选科室表要一起留档（校验器也要用它）"
        assert "咳嗽" in [s["matched"] for s in payload["symptoms"]]
        assert {d["name"] for d in payload["department_candidates"]} <= set(LINKS["咳嗽"]) | {
            "呼吸内科",
            "感染科",
        }
        assert [b["kind"] for b in payload["booking"]] == []


class TestToolsAreWired:
    def test_runtime_registers_clinic_tools(self, runtime):
        """若注册没生效，run_tool_loop 会拿到空 tools 并静默退化成纯对话。"""
        names = [t.name for t in runtime.tools]
        assert "list_departments" in names
        assert "book_appointment" in names
        assert "get_availability" in names

    def test_tools_schemas_are_handed_over(self, runtime):
        schemas = [t.schema() for t in runtime.tools]
        assert all("parameters" in s["function"] for s in schemas)


class TestNoLlmIsOneSentence:
    def test_without_key_it_replies_with_the_single_message(self, runtime, monkeypatch):
        """无密钥：只回 `NO_LLM_MESSAGE`，不装配面板、不进循环、不做任何兜底。"""
        called: list[int] = []

        class _Boom:
            def __init__(self, *_a: Any, **_k: Any) -> None:
                called.append(1)

            def choose(self, *_a: Any, **_k: Any) -> dict[str, Any]:
                called.append(1)
                raise AssertionError("没有密钥却进入了工具循环")

        monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: False)
        monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _Boom)

        turn = runtime.turn("我咳嗽")

        assert turn.text == NO_LLM_MESSAGE
        assert turn.tool_calls == []
        assert turn.emergency is False
        assert turn.reference is None, "没有判断者时不该装配面板：它没有读者"
        assert turn.notes == ["未连接大模型"]
        assert called == [], "无密钥时一次模型调用都不该发生"


class TestConfirmationPolicy:
    """`/chat` 与 `/consult` 同一套确认策略：**模型只能提议，写操作由患者点按钮触发**。

    所以这条路径**永远不给** `confirm` 回调——包括用户原话里就写着"帮我挂号"的时候。
    "这句话里有挂号两个字"不等于"患者同意了这一张具体单子（科室、日期、时段、费用）"，
    而那才是确认要问的东西。
    """

    def test_no_confirm_callback_is_ever_handed_to_the_loop(self, runtime, monkeypatch):
        from kubedocs.agent.tool_loop import AgentTurn

        seen: list[Any] = []

        def fake_loop(question: str, **kw: Any) -> AgentTurn:
            seen.append(kw.get("confirm"))
            return AgentTurn(question=question, answer="（打桩）", steps=1, llm_calls=1)

        monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: True)
        monkeypatch.setattr("kubedocs.agent.clinic_react.run_tool_loop", fake_loop)
        runtime.turn("我最近咳嗽")
        runtime.turn("帮我挂个号")
        assert seen == [None, None], "任何一轮都不该给出确认回调：确认来自患者点按钮，不来自这句话"

    def test_the_model_cannot_book_even_when_it_tries(self, runtime):
        """模型硬调 `book_appointment` ⇒ 一律被拒，号源一个都不动。"""
        from kubedocs.agent.tool_loop import run_tool_loop

        class _Scripted:
            def __init__(self) -> None:
                self.n = 0

            def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
                self.n += 1
                if self.n == 1:
                    return {
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "book_appointment",
                                "arguments": {
                                    "department": "心内科",
                                    "date": "2026-01-01",
                                    "time_slot": "上午",
                                },
                            }
                        ],
                    }
                return {"content": "请确认后再取号", "tool_calls": []}

        turn = run_tool_loop("帮我挂号", tools=runtime.tools, llm=_Scripted())
        assert turn.tool_calls[0].error, "没有确认回调时副作用工具必须被拒"
        assert turn.tool_calls[0].result is None
        assert runtime.service.my_appointments("anon") == [], "被拒的调用不该真的占号"

    def test_proposing_a_booking_produces_a_pending_action(self, runtime):
        """`propose_booking` 只读：产出一张待确认单，**不占号**。"""
        from kubedocs.agent.tool_loop import run_tool_loop

        class _Scripted:
            def __init__(self) -> None:
                self.n = 0

            def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
                self.n += 1
                if self.n == 1:
                    return {
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "propose_booking",
                                "arguments": {"department": "心内科", "time_slot": "上午"},
                            }
                        ],
                    }
                return {"content": "请确认这张单子", "tool_calls": []}

        turn = run_tool_loop("帮我挂个号", tools=runtime.tools, llm=_Scripted(), context={"patient_id": "p9"})
        assert turn.tool_calls[0].error == ""
        proposal = (turn.tool_calls[0].result or {}).get("proposal") or {}
        assert proposal.get("proposal_id"), "提议必须留下一张可确认的单子"
        assert proposal.get("department") == "心内科"
        assert runtime.service.my_appointments("p9") == [], "提议**不占号**：取号要等患者点确认"


class TestHistoryReachesTheModel:
    """多轮上下文必须真的进到模型收到的 messages 里。

    只断言"history 被传进 turn 了"抓不到问题：tool loop 路径可能收下 history
    却从不使用它，所以这里直接检查模型收到的 messages。
    """

    def _capture(
        self, runtime, monkeypatch, message: str, history: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        seen = _capture_caller(monkeypatch, "好的，请继续测血压。")

        turn = runtime.turn(message, history=history)

        assert turn.text == "好的，请继续测血压。"
        assert seen, "模型一次都没被调用"
        return seen[0]

    def test_history_entries_are_sent_to_the_model_in_order(self, runtime, monkeypatch):
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "我最近总是头晕"},
            {"role": "assistant", "content": "头晕多久了？"},
        ]

        messages = self._capture(runtime, monkeypatch, "昨天量了 150/95", history)

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[1] == {"role": "user", "content": "我最近总是头晕"}
        assert messages[2] == {"role": "assistant", "content": "头晕多久了？"}
        assert messages[3] == {"role": "user", "content": "昨天量了 150/95"}
        # 系统提示仍是 index 0 那一条（history 不会插到它前面，也不会顶掉它）
        assert "预问诊助手" in messages[0]["content"]
        # 面板也拼在**同一条** system 消息里：不额外插 system 消息（序列不变量见 build_system_prompt）
        assert "参考标准" in messages[0]["content"]

    def test_without_history_only_system_and_the_current_question(self, runtime, monkeypatch):
        """回归：不带 history 时序列不变（system + 本轮 user）。"""
        messages = self._capture(runtime, monkeypatch, "我咳嗽")

        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1] == {"role": "user", "content": "我咳嗽"}

    def test_invalid_history_fails_loudly_instead_of_being_dropped(self, runtime, monkeypatch):
        """非法 role 一路报错到调用方，而不是被悄悄丢掉（丢掉的症状与"模型没记住"同形）。"""
        with pytest.raises(ValueError, match=r"history\[0\]"):
            self._capture(runtime, monkeypatch, "我咳嗽", [{"role": "system", "content": "顶掉系统提示"}])


class TestBudgetIsWiredIntoTheToolLoop:
    """`/chat` 的工具循环必须有成本上限，否则这一路径的花费没有日上限。

    预算必须在进入循环前收口：每轮只累加成本、从不比较的话，日上限形同虚设。
    """

    def test_runtime_builds_a_budget_from_settings(self, tmp_path):
        settings = Settings(
            index_dir=str(tmp_path),
            domain="clinic",
            embedding_provider="hashing",
            embedding_dim=32,
            llm_daily_budget_usd=1.5,
            llm_unpriced_call_usd=0.4,
        )
        rt = ClinicReactRuntime(settings, BookingService(MockAppointmentProvider()))

        assert rt.budget.limit == 1.5
        assert rt.budget.unpriced_usd == 0.4
        # 与 CLI `ask` / `POST /ask` 共用同一份花费文件，否则日上限会被按域翻倍。
        assert rt.budget.path == settings.llm_spend_path
        assert rt.budget.path == settings.index_root / "llm_spend.json"
        assert settings.domain not in str(rt.budget.path)

    def test_turn_hands_the_budget_to_the_loop_and_reports_the_cost(self, runtime, monkeypatch):
        from kubedocs.agent.tool_loop import AgentTurn

        captured: dict[str, Any] = {}

        def fake_loop(question: str, **kw: Any) -> AgentTurn:
            captured.update(kw)
            return AgentTurn(
                question=question, answer="（打桩）", steps=1, llm_calls=1, cost_usd=0.03, unpriced_calls=1
            )

        monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: True)
        monkeypatch.setattr("kubedocs.agent.clinic_react.run_tool_loop", fake_loop)
        turn = runtime.turn("我最近咳嗽")

        assert captured["budget"] is runtime.budget, "预算没有交给工具循环"
        assert turn.cost_usd == 0.03
        assert turn.unpriced_calls == 1, "未定价的调用次数必须向上透传（否则前端看不见）"

    def test_over_budget_stops_before_calling_the_model(self, runtime, monkeypatch):
        """**关键断言**：超限后 `choose()` 一次都没被调用（账单真的被挡住了）。"""
        calls: list[int] = []

        class _Counting:
            def __init__(self, _settings: Any) -> None:
                pass

            def choose(self, *_a: Any, **_k: Any) -> dict[str, Any]:
                calls.append(1)
                return {"content": "答", "tool_calls": []}

        from kubedocs.generate import BudgetExceeded

        monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: True)
        monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _Counting)
        runtime.budget.limit = 0.1
        runtime.budget.record(0.2)  # 今天已经花超了

        with pytest.raises(BudgetExceeded):
            runtime.turn("我最近咳嗽")

        assert calls == [], "超预算后仍然调用了模型"

    def test_empty_model_answer_becomes_the_failure_message(self, runtime, monkeypatch):
        """模型没给出内容（调用失败/步数耗尽）⇒ 只回 `LLM_FAILED_MESSAGE`，不给兜底内容。"""
        from kubedocs.llm_status import LLM_FAILED_MESSAGE

        _capture_caller(monkeypatch, "")

        turn = runtime.turn("我咳嗽")

        assert turn.text == LLM_FAILED_MESSAGE
        assert any("没有产出可用回答" in n for n in turn.notes or [])


def _sequence_caller(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> list[list[dict[str, Any]]]:
    """按顺序逐次给出回答的打桩 `ToolCaller`，返回它收到的 messages 列表。

    `_capture_caller` 每次都回同一句，验证不了"重生成后回正"这条分支；这里让第 N 次
    调用说第 N 句话（每次循环只调一次，因为打桩回答不带 tool_calls）。
    """
    seen: list[list[dict[str, Any]]] = []

    class _Sequenced:
        def __init__(self, _settings: Any) -> None:
            self.calls = 0

        def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            seen.append(copy.deepcopy(messages))
            text = answers[min(self.calls, len(answers) - 1)]
            self.calls += 1
            return {"content": text, "tool_calls": []}

    monkeypatch.setattr(ClinicReactRuntime, "has_llm", lambda self: True)
    monkeypatch.setattr("kubedocs.agent.llm_tool_caller.LiteLLMToolCaller", _Sequenced)
    return seen


HALLUCINATED_BOOKING = "已经为您挂上心内科上午的号了，凭据号 SN999999，请按时就诊。"
"""一段"没挂上却说挂上"的模型输出：完成态 + 服务端没发过的凭据号。"""

COMPLIANT_ANSWER = "目前没有查到您的挂号记录。需要我帮您预约心内科吗？"


class TestProseIsBoundByFacts:
    """`/chat` 的散文路径必须被事实约束：以前这一路直接上屏，没有任何校验。

    这些用例只验证**约束层有没有生效**（拦下、重生成、替换），不验证模型该说什么。
    """

    def test_clean_prose_passes_without_a_second_call(self, runtime, monkeypatch):
        """基线：守规矩的回答照原样上屏，不触发重生成（否则约束层会天天改写正确回答）。"""
        seen = _sequence_caller(monkeypatch, [COMPLIANT_ANSWER])

        turn = runtime.turn("我挂上了吗", patient_id="p-clean")

        assert len(seen) == 1, "没有违规却重生成了一次"
        assert turn.text == COMPLIANT_ANSWER
        assert turn.contract_violations == []

    def test_fabricated_booking_never_reaches_the_patient(self, runtime, monkeypatch):
        """**关键断言**：两轮都在编 ⇒ 患者看到的不是那段散文，而是只含事实的回答。"""
        seen = _sequence_caller(monkeypatch, [HALLUCINATED_BOOKING, HALLUCINATED_BOOKING])

        turn = runtime.turn("我挂上了吗", patient_id="p-halluc")

        assert len(seen) == 2, "第一次违规后必须重生成一次（提示词里带违规反馈）"
        assert "本轮没有产生任何挂号或退号动作" in turn.text
        assert "SN999999" not in turn.text, "编造的凭据号上屏了"
        assert "已经" not in turn.text
        kinds = {v["kind"] for v in turn.contract_violations or []}
        assert "fabricated_booking" in kinds
        assert "unknown_ticket" in kinds
        assert all(v["severity"] == "block" for v in turn.contract_violations or [])
        assert any("已改为只陈述事实" in n for n in turn.notes or [])

    def test_the_retry_prompt_carries_the_violation_feedback(self, runtime, monkeypatch):
        """重生成那一次的系统提示必须带上面板**和**违规明细，否则模型只能瞎猜哪里错了。"""
        seen = _sequence_caller(monkeypatch, [HALLUCINATED_BOOKING, COMPLIANT_ANSWER])

        runtime.turn("我挂上了吗", patient_id="p-feedback")

        retry_system = seen[1][0]["content"]
        assert "参考标准" in retry_system, "重生成时把面板丢了"
        assert "已被拦下" in retry_system
        assert "fabricated_booking" in retry_system or "完成态说法" in retry_system

    def test_a_recovered_retry_is_shown_but_the_first_pass_stays_on_record(self, runtime, monkeypatch):
        """第二轮回正 ⇒ 展示第二轮的合规回答；第一轮的违规**照样留档**。

        回正不等于没发生过："模型偶尔会编"正是这套架构最需要被观测到的事，
        抹掉第一次的违规会让运维看到一份"一切正常"的假账（与 `/consult` 同一口径）。
        """
        seen = _sequence_caller(monkeypatch, [HALLUCINATED_BOOKING, COMPLIANT_ANSWER])

        turn = runtime.turn("我挂上了吗", patient_id="p-recover")

        assert len(seen) == 2
        assert turn.text == COMPLIANT_ANSWER
        assert {v["kind"] for v in turn.contract_violations or []} == {
            "fabricated_booking",
            "unknown_ticket",
        }
        assert any("已重生成一次" in n for n in turn.notes or [])
        assert turn.reference is not None, "面板留档与违规留档要成对回传"
