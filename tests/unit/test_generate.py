"""回答生成测试：引用核对、拒答、预算、无密钥降级。

全部用打桩 LLM：CI 与单测**不调用真实模型、不花一分钱、不依赖密钥**。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kubedocs.config import Settings
from kubedocs.generate import (
    Answerer,
    BudgetExceeded,
    DailyBudget,
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

    "是否被调用"是可断言的**成本与诚实性指标**：检索为空时调用 LLM
    既浪费钱，又给了模型凭空作答的机会。
    """

    def __init__(self, text: str = "答案 [1]", cost: float = 0.0,
                 raises: Exception | None = None) -> None:
        self.text = text
        self.cost = cost
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        return LLMResult(text=self.text, prompt_tokens=100, completion_tokens=20,
                         cost_usd=self.cost)


def _hit(i: int, title: str = "Doc", body: str = "Body text", crumb: str = "Sec") -> ChunkHit:
    return ChunkHit(chunk_id=f"d{i}::0000", doc_id=f"d{i}", ordinal=0, title=title,
                    breadcrumb=crumb, source_url=f"https://kubernetes.io/docs/d{i}/",
                    body=body)


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
        """只给了模型一部分片段，却允许它引用全部编号——引用错位的常见来源。"""
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
        b.check()          # 不抛

    def test_corrupt_file_does_not_break_service(self, tmp_path: Path):
        p = tmp_path / "spend.json"
        p.write_text("{not json", encoding="utf-8")
        b = DailyBudget(p, 1.0)
        assert b.spent_today() == 0.0     # 保守当作没花过，而不是让服务挂掉

    def test_written_file_is_readable_json(self, tmp_path: Path):
        p = tmp_path / "spend.json"
        DailyBudget(p, 1.0).record(0.1)
        assert isinstance(json.loads(p.read_text(encoding="utf-8")), dict)


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


# --------------------------------------------------------------------- 主流程（用打桩检索器）
class StubRetriever:
    def __init__(self, hits: list[ChunkHit], degraded: list[str] | None = None) -> None:
        from kubedocs.retrieve.hybrid import RetrievedChunk, SearchResult
        self._result = SearchResult(
            query="q",
            results=[RetrievedChunk.from_hit(h) for h in hits],
            diagnostics={"degraded": degraded or []},
        )

    def search(self, query: str, *, top_k: int | None = None,
               channels: tuple[str, ...] = ("bm25", "dense")) -> object:
        return self._result


def _answerer(settings: Settings, hits: list[ChunkHit], llm: StubLLM,
              degraded: list[str] | None = None) -> Answerer:
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
        assert llm.calls == []          # 关键断言：一次都没调用

    def test_missing_key_degrades_but_still_returns_retrieval(self, settings, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        llm = StubLLM(raises=LLMUnavailable("未配置 DEEPSEEK_API_KEY"))
        ans = _answerer(settings, [_hit(1)], llm).answer("q")
        assert ans.refused is True
        assert "DEEPSEEK_API_KEY" in ans.refusal_reason
        assert ans.citations, "降级时仍应给出检索来源"
        assert any("仅返回检索结果" in d for d in ans.degraded)

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
        ans = _answerer(settings, [_hit(1)], StubLLM("ok [1]"),
                        degraded=["向量路不可用，已降级为纯全文检索：boom"]).answer("q")
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
