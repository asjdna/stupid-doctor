"""回答生成测试：引用核对、拒答、预算、无密钥降级。

全部走打桩 LLM：不联网、不调真实模型、不需要密钥。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kubedocs.config import Settings
from kubedocs.generate import (
    UNPRICED_CALL_FALLBACK_USD,
    Answerer,
    BudgetExceeded,
    DailyBudget,
    LiteLLMClient,
    LLMResult,
    LLMUnavailable,
    build_context,
    context_text,
    has_api_key,
    required_api_key_env,
    validate_citations,
)
from kubedocs.store.base import ChunkHit


# --------------------------------------------------------------------- 打桩
class StubLLM:
    """返回固定文本，并记录是否被调用过。

    调用次数本身是断言对象：检索为空时调用 LLM 既花钱又给它凭空作答的机会。
    """

    def __init__(
        self,
        text: str = "答案 [1]",
        cost: float = 0.0,
        raises: Exception | None = None,
        cost_priced: bool = True,
    ) -> None:
        self.text = text
        self.cost = cost
        self.raises = raises
        self.cost_priced = cost_priced
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        return LLMResult(
            text=self.text,
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=self.cost,
            cost_priced=self.cost_priced,
        )


def _hit(
    i: int, title: str = "Doc", body: str = "Body text", crumb: str = "Sec", doc: str | None = None
) -> ChunkHit:
    d = doc if doc is not None else f"d{i}"
    return ChunkHit(
        chunk_id=f"{d}::{i:04d}",
        doc_id=d,
        ordinal=i,
        title=title,
        breadcrumb=crumb,
        source_url=f"https://kubernetes.io/docs/{d}/",
        body=body,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """小上下文窗口 + 无真实模型：让测试聚焦行为而不是规模。"""
    return Settings(
        index_dir=str(tmp_path / "index"),
        embedding_provider="hashing",
        embedding_dim=32,
        llm_model="deepseek/deepseek-chat",
        llm_daily_budget_usd=5.0,
        max_context_chars=400,
    )


# --------------------------------------------------------------------- 上下文
class TestContextText:
    def test_strips_the_context_header_line(self):
        assert context_text("[Pod Lifecycle › Probes]\n\nReal body") == "Real body"

    def test_keeps_body_without_header(self):
        assert context_text("Just a body") == "Just a body"

    def test_only_strips_one_header(self):
        assert context_text("[A]\n\n[B]\n\nbody") == "[B]\n\nbody"


class TestBuildContext:
    def test_numbers_chunks_and_returns_matching_citations(self):
        ctx, cites = build_context([_hit(1, "Doc A"), _hit(2, "Doc B")])
        assert "[1] Doc A › Sec" in ctx and "[2] Doc B › Sec" in ctx
        assert [c.index for c in cites] == [1, 2]
        assert cites[0].source_url.startswith("https://kubernetes.io/docs/")

    def test_dropped_chunk_is_not_citable(self):
        """契约：被丢弃的片段不进上下文，其编号也不可引用（引用错位的常见来源）。"""
        hits = [_hit(1, body="x" * 100), _hit(2, body="y" * 1000)]
        ctx, cites = build_context(hits, max_chars=200, per_chunk_chars=1000)
        assert len(cites) == 1
        assert "[2]" not in ctx

    def test_keeps_at_least_one_chunk(self):
        _ctx, cites = build_context([_hit(1, body="z" * 5000)], max_chars=50)
        assert len(cites) == 1

    def test_long_body_is_truncated_with_ellipsis(self):
        ctx, _ = build_context([_hit(1, body="w" * 500)], per_chunk_chars=50)
        assert "…" in ctx

    def test_empty_input(self):
        assert build_context([]) == ("", [])


class TestContextDiversity:
    """同一篇文档的相邻小节向量高度相似，容易一起挤进前几名。

    故默认每篇最多取 2 段：既给别的来源留位置，也降低上下文冗余。
    """

    def _hits(self) -> list[ChunkHit]:
        return [
            _hit(0, "A", doc="a"),
            _hit(1, "A", doc="a"),
            _hit(2, "A", doc="a"),
            _hit(0, "B", body="other", doc="b"),
        ]

    def test_default_caps_two_per_doc(self):
        ctx, cites = build_context(self._hits())
        assert [c.chunk_id for c in cites] == ["a::0000", "a::0001", "b::0000"]
        assert "[3] B" in ctx

    def test_cap_one_keeps_one_per_doc(self):
        _ctx, cites = build_context(self._hits(), max_per_doc=1)
        assert [c.chunk_id for c in cites] == ["a::0000", "b::0000"]

    def test_numbering_stays_contiguous_after_skipping(self):
        _ctx, cites = build_context(self._hits(), max_per_doc=1)
        assert [c.index for c in cites] == [1, 2]

    def test_never_returns_empty_when_there_are_hits(self):
        """限流再狠也必须留下至少一段：宁可冗余，不可空手。"""
        _ctx, cites = build_context([_hit(0, doc="a"), _hit(1, doc="a")], max_per_doc=1)
        assert len(cites) == 1
        ctx, cites2 = build_context([_hit(0, doc="a"), _hit(1, doc="a")], max_per_doc=1, max_chars=1)
        assert ctx and len(cites2) == 1

    def test_single_doc_still_gets_capped_in_context(self):
        ctx, cites = build_context([_hit(i, "A", doc="a") for i in range(5)], max_per_doc=2)
        assert len(cites) == 2
        assert "[3]" not in ctx


# --------------------------------------------------------------------- 引用核对
class TestValidateCitations:
    def test_valid_markers_are_kept(self):
        text, problems = validate_citations("Use limits [1] and requests [2].", 2)
        assert text == "Use limits [1] and requests [2]."
        assert problems == []

    def test_out_of_range_marker_is_stripped_and_reported(self):
        text, problems = validate_citations("See [1] and also [99].", 2)
        assert "[99]" not in text and "[1]" in text
        assert any("未提供的片段编号" in p for p in problems)

    def test_zero_marker_is_stripped(self):
        text, _ = validate_citations("See [0].", 2)
        assert "[0]" not in text

    def test_answer_without_any_citation_is_flagged(self):
        _text, problems = validate_citations("Trust me, it works.", 2)
        assert any("没有任何引用标记" in p for p in problems)

    def test_multiple_markers(self):
        text, problems = validate_citations("Both [1][2] apply.", 2)
        assert problems == []
        assert text.count("[") == 2


# --------------------------------------------------------------------- 预算
class TestDailyBudget:
    def test_records_and_accumulates(self, tmp_path: Path):
        b = DailyBudget(tmp_path / "spend.json", 1.0)
        assert b.spent_today() == 0.0
        b.record(0.25)
        assert b.spent_today() == pytest.approx(0.25)
        b.record(0.25)
        assert b.spent_today() == pytest.approx(0.5)

    def test_refuses_when_over_limit(self, tmp_path: Path):
        b = DailyBudget(tmp_path / "spend.json", 0.5)
        b.record(0.5)
        with pytest.raises(BudgetExceeded):
            b.check()

    def test_zero_limit_means_unlimited(self, tmp_path: Path):
        b = DailyBudget(tmp_path / "spend.json", 0.0)
        b.record(99.0)
        b.check()

    def test_corrupt_file_does_not_break_service(self, tmp_path: Path):
        p = tmp_path / "spend.json"
        p.write_text("{not json", encoding="utf-8")
        b = DailyBudget(p, 1.0)
        assert b.spent_today() == 0.0  # 保守当作没花过，而不是让服务挂掉

    def test_written_file_is_readable_json(self, tmp_path: Path):
        p = tmp_path / "spend.json"
        DailyBudget(p, 1.0).record(0.1)
        assert isinstance(json.loads(p.read_text(encoding="utf-8")), dict)


class TestUnknownCostIsNotFree:
    """契约：算不出价的调用不许按 0 记账。

    记 0 会让账本永远停在 0、`check()` 永不触发，"日预算上限"对未知模型形同虚设；
    所以未定价调用按保守上限计入，并单独计数。
    """

    def test_unpriced_call_is_charged_the_conservative_ceiling(self, tmp_path: Path):
        b = DailyBudget(tmp_path / "spend.json", 5.0, unpriced_usd=0.25)
        b.record(0.0, priced=False)
        assert b.spent_today() == pytest.approx(0.25), "未定价的调用被当成了免费"
        assert b.unpriced_calls_today() == 1
        assert b.record(0.0, priced=False) == pytest.approx(0.5)

    def test_unpriced_calls_reach_the_limit_and_then_refuse(self, tmp_path: Path):
        b = DailyBudget(tmp_path / "spend.json", 0.5, unpriced_usd=0.25)
        b.record(0.0, priced=False)
        b.check()  # 0.25 < 0.5：还放行
        b.record(0.0, priced=False)
        with pytest.raises(BudgetExceeded) as e:
            b.check()
        assert "算不出价" in str(e.value), "超限原因要说清是未定价调用累积出来的"

    def test_priced_zero_stays_zero(self, tmp_path: Path):
        """真免费（本地模型定价就是 0）不该被收"未定价"的罚金。"""
        b = DailyBudget(tmp_path / "spend.json", 5.0, unpriced_usd=0.25)
        b.record(0.0, priced=True)
        assert b.spent_today() == 0.0
        assert b.unpriced_calls_today() == 0

    def test_fallback_can_be_disabled_but_counting_continues(self, tmp_path: Path):
        """`unpriced_usd=0` 是运营者的显式选择：不计金额，但仍然计数。"""
        b = DailyBudget(tmp_path / "spend.json", 5.0, unpriced_usd=0.0)
        b.record(0.0, priced=False)
        assert b.spent_today() == 0.0
        assert b.unpriced_calls_today() == 1

    def test_unpriced_counter_survives_a_reload(self, tmp_path: Path):
        p = tmp_path / "spend.json"
        DailyBudget(p, 5.0, unpriced_usd=0.25).record(0.0, priced=False)
        again = DailyBudget(p, 5.0, unpriced_usd=0.25)
        assert again.unpriced_calls_today() == 1
        assert again.spent_today() == pytest.approx(0.25)

    def test_legacy_spend_file_still_loads(self, tmp_path: Path):
        """已落盘的老格式账本必须继续可读：升级不能丢掉已有的花费记录。"""
        p = tmp_path / "spend.json"
        today = time.strftime("%Y-%m-%d")
        p.write_text(json.dumps({today: 0.3}), encoding="utf-8")
        b = DailyBudget(p, 5.0)
        assert b.spent_today() == pytest.approx(0.3)
        assert b.unpriced_calls_today() == 0
        b.record(0.1)
        assert b.spent_today() == pytest.approx(0.4)

    def test_default_fallback_is_a_conservative_positive_number(self):
        """默认罚金必须为正数：0 等于把"未知"当成"免费"。"""
        assert UNPRICED_CALL_FALLBACK_USD > 0


class _FakeLitellm:
    """假 litellm：只回一个响应 + 一个可由用例指定的成本行为（不联网）。"""

    drop_params = True

    def __init__(self, resp: object, *, cost: object = None, cost_raises: Exception | None = None) -> None:
        self._resp = resp
        self._cost = cost
        self._cost_raises = cost_raises
        self.calls = 0

    def completion(self, **_kwargs: object) -> object:
        self.calls += 1
        return self._resp

    def completion_cost(self, completion_response: object = None) -> object:
        if self._cost_raises is not None:
            raise self._cost_raises
        return self._cost


def _fake_response() -> object:
    return SimpleNamespace(
        model="deepseek/deepseek-chat",
        choices=[SimpleNamespace(message=SimpleNamespace(content="打桩回答 [1]"))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


class TestRealClientFlagsUnknownPrice:
    """`LiteLLMClient` 必须把"算不出价"**如实标出来**，而不是悄悄记 0。"""

    def _client(
        self, settings: Settings, fake: _FakeLitellm, monkeypatch: pytest.MonkeyPatch
    ) -> LiteLLMClient:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        client = LiteLLMClient(settings)
        client._litellm = fake  # 绕开 import litellm（并避免它去拉价格表）
        return client

    def test_price_map_miss_is_flagged_not_zeroed(self, settings, monkeypatch):
        fake = _FakeLitellm(_fake_response(), cost_raises=Exception("This model isn't mapped yet"))
        result = self._client(settings, fake, monkeypatch).complete("sys", "user")
        assert result.cost_usd == 0.0
        assert result.cost_priced is False, "算不出价必须标成未定价（否则闸门永远拦不住）"

    def test_none_cost_is_flagged_not_zeroed(self, settings, monkeypatch):
        fake = _FakeLitellm(_fake_response(), cost=None)
        result = self._client(settings, fake, monkeypatch).complete("sys", "user")
        assert result.cost_priced is False

    def test_real_price_is_reported_as_priced(self, settings, monkeypatch):
        fake = _FakeLitellm(_fake_response(), cost=0.0012)
        result = self._client(settings, fake, monkeypatch).complete("sys", "user")
        assert result.cost_usd == pytest.approx(0.0012)
        assert result.cost_priced is True

    def test_priced_zero_from_the_provider_is_not_unpriced(self, settings, monkeypatch):
        """免费模型的定价就是 0.0：不能把它误判成"算不出价"。"""
        fake = _FakeLitellm(_fake_response(), cost=0.0)
        result = self._client(settings, fake, monkeypatch).complete("sys", "user")
        assert (result.cost_usd, result.cost_priced) == (0.0, True)


# --------------------------------------------------------------------- 密钥推断
class TestApiKeyDetection:
    def test_provider_to_env_mapping(self):
        assert required_api_key_env("deepseek/deepseek-chat") == "DEEPSEEK_API_KEY"
        assert required_api_key_env("openai/gpt-4o-mini") == "OPENAI_API_KEY"
        assert required_api_key_env("anthropic/claude-3-5-sonnet") == "ANTHROPIC_API_KEY"
        assert required_api_key_env("gemini/gemini-1.5-pro") == "GEMINI_API_KEY"

    def test_missing_key_is_detected(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert has_api_key("deepseek/deepseek-chat") is False

    def test_present_key_is_detected(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        assert has_api_key("deepseek/deepseek-chat") is True

    def test_whitespace_key_is_not_a_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
        assert has_api_key("deepseek/deepseek-chat") is False

    def test_env_example_placeholder_is_not_a_key(self, monkeypatch):
        """占位符文本不算密钥。

        判成"已配置"会让 `/ask` 拿它发起必然失败的请求，用户看到的是模型异常，
        而不是"你没配密钥"。
        """
        monkeypatch.setenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-placeholder-here")
        assert has_api_key("deepseek/deepseek-chat") is False

    def test_non_ascii_value_is_not_a_key(self, monkeypatch):
        """非 ASCII 值不算密钥：真密钥要进 `Authorization` 头，必然是可打印 ASCII。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "请把你的 DeepSeek 密钥填在这里，留空则不用大模型")
        assert has_api_key("deepseek/deepseek-chat") is False

    def test_non_ascii_check_matches_the_real_env_shape(self, monkeypatch):
        """词表判据挡不住的值：38 字符、非 ASCII，且不含任何占位符关键词。"""
        shape = "ab" + "填" * 16 + "a" * 11 + "填" * 3 + "a" * 5 + "填"
        assert len(shape) == 38
        assert not shape.isascii()
        assert not any(
            marker in shape.lower() for marker in ("your", "placeholder", "changeme", "example", "xxxx")
        )
        monkeypatch.setenv("DEEPSEEK_API_KEY", shape)
        assert has_api_key("deepseek/deepseek-chat") is False

    @pytest.mark.parametrize(
        "value",
        [
            "your-openai-key",
            "YOUR_API_KEY_HERE",
            "changeme",
            "CHANGE-ME",
            "sk-xxxxxxxxxxxxxxxx",
            "<your-key>",
            "my-example-key",
        ],
    )
    def test_placeholder_markers(self, monkeypatch, value):
        monkeypatch.setenv("DEEPSEEK_API_KEY", value)
        assert has_api_key("deepseek/deepseek-chat") is False

    def test_real_looking_key_still_passes(self, monkeypatch):
        """黑名单不能误伤真密钥：`sk-` + 32 位随机串仍须判为已配置。

        值在运行时拼装，文件里不留密钥形状的字面量，否则密钥扫描会把它当成真 key。
        """
        fake = "sk-" + "0123456789abcdef" * 2
        monkeypatch.setenv("DEEPSEEK_API_KEY", fake)
        assert has_api_key("deepseek/deepseek-chat") is True


# --------------------------------------------------------------------- 主流程（用打桩检索器）
class StubRetriever:
    def __init__(self, hits: list[ChunkHit], degraded: list[str] | None = None) -> None:
        from kubedocs.retrieve.hybrid import RetrievedChunk, SearchResult

        self._result = SearchResult(
            query="q",
            results=[RetrievedChunk.from_hit(h) for h in hits],
            diagnostics={"degraded": degraded or []},
        )

    def search(
        self, query: str, *, top_k: int | None = None, channels: tuple[str, ...] = ("bm25", "dense")
    ) -> object:
        return self._result


def _answerer(
    settings: Settings, hits: list[ChunkHit], llm: StubLLM, degraded: list[str] | None = None
) -> Answerer:
    return Answerer(StubRetriever(hits, degraded), llm, settings)  # type: ignore[arg-type]


class TestAnswerer:
    def test_happy_path_returns_answer_with_citations(self, settings):
        llm = StubLLM("先用 requests，再用 limits [1]")
        ans = _answerer(settings, [_hit(1, "Doc A")], llm).answer("q")
        assert ans.refused is False
        assert "[1]" in ans.text
        assert [c.index for c in ans.citations] == [1]
        assert ans.prompt_tokens == 100 and ans.completion_tokens == 20
        assert len(llm.calls) == 1

    def test_context_is_passed_to_the_model(self, settings):
        llm = StubLLM("ok [1]")
        _answerer(settings, [_hit(1, "Doc A", body="UNIQUEBODY")], llm).answer("my question")
        _system, user = llm.calls[0]
        assert "UNIQUEBODY" in user and "my question" in user
        assert "[1] Doc A" in user

    def test_empty_retrieval_refuses_without_calling_the_llm(self, settings):
        """检索为空还去问模型，既花钱又给了它凭空作答的机会。"""
        llm = StubLLM()
        ans = _answerer(settings, [], llm).answer("q")
        assert ans.refused is True
        assert "没有找到" in ans.refusal_reason
        assert llm.calls == []  # 一次都没调用

    def test_missing_key_degrades_but_still_returns_retrieval(self, settings, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        llm = StubLLM(raises=LLMUnavailable("未配置 DEEPSEEK_API_KEY"))
        ans = _answerer(settings, [_hit(1)], llm).answer("q")
        assert ans.refused is True
        assert "DEEPSEEK_API_KEY" in ans.refusal_reason
        assert ans.citations, "降级时仍应给出检索来源"
        assert any("仅返回检索结果" in d for d in ans.degraded)

    def test_provider_error_degrades_instead_of_crashing(self, settings):
        """供应商失败（401/429/超时/宕机）只降级，不让整个请求崩掉。

        用户仍应拿到检索结果和一句明确的失败原因；异常类型必须进诊断，否则就是静默吞异常。
        """

        class AuthenticationError(Exception):
            pass

        llm = StubLLM(raises=AuthenticationError("invalid api key"))
        ans = _answerer(settings, [_hit(1)], llm).answer("q")
        assert ans.refused is True
        assert "AuthenticationError" in ans.refusal_reason
        assert "invalid api key" in ans.refusal_reason
        assert ans.citations, "生成失败时仍应返回检索来源"
        assert any("仅返回检索结果" in d for d in ans.degraded)

    def test_unexpected_error_is_recorded_not_swallowed(self, settings):
        ans = _answerer(settings, [_hit(1)], StubLLM(raises=RuntimeError("boom"))).answer("q")
        assert ans.refused is True
        assert "RuntimeError" in ans.refusal_reason

    def test_budget_exceeded_refuses_before_calling(self, settings):
        settings.llm_daily_budget_usd = 0.1
        a = _answerer(settings, [_hit(1)], StubLLM())
        a.budget.record(0.2)
        llm = StubLLM()
        a.llm = llm
        ans = a.answer("q")
        assert ans.refused is True
        assert "上限" in ans.refusal_reason
        assert llm.calls == []

    def test_cost_is_recorded_in_budget(self, settings):
        a = _answerer(settings, [_hit(1)], StubLLM("ok [1]", cost=0.05))
        a.answer("q")
        assert a.budget.spent_today() == pytest.approx(0.05)

    def test_unpriced_cost_is_charged_and_flagged(self, settings):
        """`/ask` 路径同样不许把算不出价的调用当成免费。"""
        settings.llm_unpriced_call_usd = 0.25
        a = _answerer(settings, [_hit(1)], StubLLM("ok [1]", cost_priced=False))

        ans = a.answer("q")

        assert a.budget.spent_today() == pytest.approx(0.25), "未定价的调用没记账"
        assert a.budget.unpriced_calls_today() == 1
        assert any("算不出价" in d for d in ans.degraded), "降级说明里要能看见它"

    def test_priced_zero_does_not_touch_the_ledger(self, settings):
        """定价为 0 的调用不写账本文件。"""
        a = _answerer(settings, [_hit(1)], StubLLM("ok [1]", cost=0.0))
        a.answer("q")
        assert a.budget.spent_today() == 0.0
        assert not a.budget.path.exists()

    def test_hallucinated_citation_is_stripped(self, settings):
        ans = _answerer(settings, [_hit(1)], StubLLM("见 [1] 与 [7]")).answer("q")
        assert "[7]" not in ans.text
        assert any("未提供的片段编号" in d for d in ans.degraded)

    def test_uncited_answer_is_flagged(self, settings):
        ans = _answerer(settings, [_hit(1)], StubLLM("就是这么回事")).answer("q")
        assert any("没有任何引用标记" in d for d in ans.degraded)

    def test_empty_model_output_is_treated_as_refusal(self, settings):
        ans = _answerer(settings, [_hit(1)], StubLLM("   ")).answer("q")
        assert ans.refused is True
        assert "空内容" in ans.refusal_reason

    def test_retrieval_degradation_is_surfaced(self, settings):
        ans = _answerer(
            settings, [_hit(1)], StubLLM("ok [1]"), degraded=["向量路不可用，已降级为纯全文检索：boom"]
        ).answer("q")
        assert any("向量路不可用" in d for d in ans.degraded)

    def test_render_includes_answer_and_sources(self, settings):
        ans = _answerer(settings, [_hit(1, "Doc A")], StubLLM("答案 [1]")).answer("q")
        out = ans.render()
        assert "答案 [1]" in out
        assert "参考来源：" in out
        assert "https://kubernetes.io/docs/d1/" in out

    def test_json_export_has_usage_and_citations(self, settings):
        ans = _answerer(settings, [_hit(1)], StubLLM("答案 [1]", cost=0.01)).answer("q")
        payload = ans.to_json()
        assert payload["refused"] is False
        assert payload["usage"]["prompt_tokens"] == 100
        assert payload["citations"][0]["index"] == 1
