"""结构化路由与 RAG 工具的单元测试（全部打桩：不联网、不调真实模型）。

两条被守护的约定：单候选时不调模型（省钱，也少一次把唯一合法值答错的机会）；
模型给出非法域时拒绝并回退，而不是把不存在的域放进下游。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from kubedocs.agent.rag_tools import (
    CitationCheckParams,
    make_citation_check_tool,
    make_search_tool,
)
from kubedocs.agent.routing import RouteDecision, build_candidates, route
from kubedocs.agent.tools import (
    EmptyParams,
    SearchDocsParams,
    ToolError,
    ToolRegistry,
    ToolSpec,
)


# --------------------------------------------------------------------- 打桩
class _StubRouter:
    """记录调用次数的路由模型替身：单候选分支若仍去问模型，测试必须立刻失败。"""

    def __init__(self, decision: Any = None, raises: Exception | None = None) -> None:
        self.calls = 0
        self._decision = decision
        self._raises = raises

    def decide(self, query: str, candidates: Sequence[dict[str, Any]]) -> RouteDecision:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        assert self._decision is not None, "替身没有配置返回值，测试写错了"
        return self._decision  # type: ignore[no-any-return]


class _DuckDecision:
    """鸭子类型的决定对象：字段齐但不是 `RouteDecision` 实例（实现方可能这样返回）。"""

    def __init__(self, domain: str, confidence: float, reason: str) -> None:
        self.domain = domain
        self.confidence = confidence
        self.reason = reason


class _StubDomain:
    """最小域包替身：路由只要求 `name` / `description` 两个属性。"""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


class _StubChunk:
    """只暴露工具真正用到的字段，证明 rag_tools 走鸭子类型而不是绑定具体实现。"""

    def __init__(self, i: int, body: str = "正文") -> None:
        self.chunk_id = f"d1::{i:04d}"
        self.doc_id = "d1"
        self.title = f"Doc {i}"
        self.breadcrumb = "Sec"
        self.source_url = f"https://kubernetes.io/docs/d1/#{i}"
        self.body = body


class _StubResult:
    """`SearchResult` 的最小替身：工具只需要 `.results`。"""

    def __init__(self, chunks: Sequence[_StubChunk]) -> None:
        self.results = list(chunks)


class _StubRetriever:
    """打桩检索器：记录 (query, top_k)，可配置成抛异常。"""

    def __init__(self, chunks: Sequence[_StubChunk], raises: Exception | None = None) -> None:
        self.chunks = list(chunks)
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int | None = None) -> _StubResult:
        self.calls.append((query, top_k or 0))
        if self.raises is not None:
            raise self.raises
        return _StubResult(self.chunks)


def _handler(spec: ToolSpec) -> Any:
    """取出执行体。`None` 表示只声明不实现——测试里出现就是测试写错了。"""
    assert spec.handler is not None, f"工具 {spec.name} 没有执行体"
    return spec.handler


def _candidates() -> list[dict[str, Any]]:
    """两个候选域：只有多候选时"选择"才有意义，也才需要模型。"""
    return build_candidates(
        [
            _StubDomain("clinic", "中文医疗问诊与挂号"),
            _StubDomain("k8s_docs", "Kubernetes 官方文档问答（英文语料）"),
        ]
    )


class TestBuildCandidates:
    def test_reads_name_and_description_from_domain_packs(self) -> None:
        """域包的 description 是路由提示的唯一来源：内核不硬编码任何域知识。"""
        cands = build_candidates(
            [_StubDomain("clinic", "中文医疗问诊"), _StubDomain("k8s_docs", "K8s 文档问答")]
        )
        assert cands == [
            {"name": "clinic", "description": "中文医疗问诊"},
            {"name": "k8s_docs", "description": "K8s 文档问答"},
        ]

    def test_accepts_domain_registry_describe_output(self) -> None:
        """`DomainRegistry.describe()` 给的是 dict：同一份清单不该因为容器形状不同丢掉。"""
        described = [{"name": "clinic", "description": "中文医疗问诊", "normalizer": "chinese"}]
        assert build_candidates(described) == [{"name": "clinic", "description": "中文医疗问诊"}]

    def test_domain_without_name_is_a_programming_error(self) -> None:
        """没有 name 的域包永远选不中；静默跳过会让它以"神秘失踪"的形式出现。"""
        with pytest.raises(ValueError, match="name"):
            build_candidates([_StubDomain("", "缺名字")])

    def test_missing_description_becomes_empty_string(self) -> None:
        """description 缺失/为 None 时补空串，而不是字符串 "None" 混进候选。"""

        class _Bare:
            name = "clinic"
            description = None

        assert build_candidates([_Bare()]) == [{"name": "clinic", "description": ""}]

    def test_empty_input(self) -> None:
        """没有域包就是空清单，交给 route 去报编程错误。"""
        assert build_candidates([]) == []


class TestRouteSingleCandidate:
    def test_single_candidate_short_circuits_without_llm(self) -> None:
        """只有一个候选时模型没有选择空间：这一次调用纯属浪费。"""
        llm = _StubRouter(RouteDecision(domain="k8s_docs", confidence=0.3, reason="随便"))
        decision = route(
            "kubelet 怎么配",
            llm=llm,
            candidates=build_candidates([_StubDomain("k8s_docs", "K8s 文档")]),
            default="clinic",
        )
        assert decision.domain == "k8s_docs"
        assert decision.confidence == 1.0
        assert llm.calls == 0

    def test_single_candidate_ignores_default(self) -> None:
        """唯一候选就是答案：连 default 都不该影响结果。"""
        decision = route(
            "q", llm=None, candidates=build_candidates([_StubDomain("clinic", "问诊")]), default="k8s_docs"
        )
        assert decision.domain == "clinic"
        assert decision.confidence == 1.0
        assert decision.reason


class TestRouteIllegalDomain:
    def test_hallucinated_domain_is_rejected_and_falls_back(self) -> None:
        """模型能编出不存在的域：静默接受会让下游全部崩掉。"""
        llm = _StubRouter(RouteDecision(domain="travel_advisor", confidence=0.97, reason="看起来像旅行"))
        decision = route("怎么挂号", llm=llm, candidates=_candidates(), default="k8s_docs")
        assert decision.domain == "k8s_docs"
        assert decision.confidence == 0.0
        assert "travel_advisor" in decision.reason
        assert "候选" in decision.reason
        assert llm.calls == 1

    def test_domain_is_matched_exactly_not_fuzzily(self) -> None:
        """大小写不同也算非法：一个"差一点"的域名说明模型没按 schema 选，猜它更危险。"""
        llm = _StubRouter(RouteDecision(domain="K8S_DOCS", confidence=0.9))
        decision = route("q", llm=llm, candidates=_candidates(), default="clinic")
        assert decision.domain == "clinic"
        assert decision.confidence == 0.0
        assert "K8S_DOCS" in decision.reason

    def test_valid_domain_from_llm_is_kept(self) -> None:
        """合法选择原样返回，置信度与理由都保留下来供审计。"""
        llm = _StubRouter(RouteDecision(domain="clinic", confidence=0.8, reason="问的是症状"))
        decision = route("头疼三天了", llm=llm, candidates=_candidates(), default="k8s_docs")
        assert decision.domain == "clinic"
        assert decision.confidence == pytest.approx(0.8)
        assert decision.reason == "问的是症状"


class TestRouteDegradation:
    def test_llm_exception_falls_back_without_crashing(self) -> None:
        """供应商 429/超时不该让整轮对话崩掉：带记录地降级，并把异常类型写进理由。"""
        llm = _StubRouter(raises=RuntimeError("rate limited"))
        decision = route("q", llm=llm, candidates=_candidates(), default="clinic")
        assert decision.domain == "clinic"
        assert decision.confidence == 0.0
        assert "RuntimeError" in decision.reason
        assert "clinic" in decision.reason

    def test_missing_llm_falls_back_to_default(self) -> None:
        """没配模型（离线 CI、没密钥）时路由仍要出结果，不能把服务卡死。"""
        decision = route("q", llm=None, candidates=_candidates(), default="k8s_docs")
        assert decision.domain == "k8s_docs"
        assert decision.confidence == 0.0
        assert decision.reason

    def test_default_outside_candidates_is_pointed_out(self) -> None:
        """default 自己也不在候选里是配置错误：照常返回，但必须让人看得见。"""
        decision = route("q", llm=None, candidates=_candidates(), default="ghost_domain")
        assert decision.domain == "ghost_domain"
        assert "配置错误" in decision.reason

    def test_duck_typed_decision_is_clamped_and_truncated(self) -> None:
        """替身对象的越界置信度/超长理由会被收进 schema 允许的范围，而不是抛校验错。"""
        llm = _StubRouter(_DuckDecision("clinic", 5.0, "很长的理由" * 100))
        decision = route("q", llm=llm, candidates=_candidates(), default="k8s_docs")
        assert decision.domain == "clinic"
        assert decision.confidence == 1.0
        assert len(decision.reason) <= 200


class TestRouteNoCandidates:
    def test_no_candidates_is_a_programming_error(self) -> None:
        """没有候选域说明域包没注册：不能悄悄返回一个域糊过去，也不能去问模型。"""
        llm = _StubRouter(RouteDecision(domain="clinic", confidence=1.0))
        with pytest.raises(ValueError, match="候选"):
            route("q", llm=llm, candidates=[], default="k8s_docs")
        assert llm.calls == 0


class TestSearchTool:
    def test_schema_and_when_not_are_declared(self) -> None:
        """when_not 必填（注册表会拒空），且它必须真的进到 function description 里。"""
        spec = make_search_tool(_StubRetriever([]))
        assert spec.when_not.strip()
        assert spec.side_effect is False
        schema = spec.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_docs"
        assert spec.when_not in schema["function"]["description"]
        parameters = schema["function"]["parameters"]
        assert set(parameters["properties"]) == {"query", "top_k"}

    def test_handler_returns_expected_shape(self) -> None:
        """返回形状是内核与生成段的契约：query / n / hits，字段一个不多一个不少。"""
        retriever = _StubRetriever([_StubChunk(1), _StubChunk(2)])
        payload = _handler(make_search_tool(retriever))(SearchDocsParams(query="probe", top_k=2))
        assert payload["query"] == "probe"
        assert payload["n"] == 2
        assert len(payload["hits"]) == 2
        first = payload["hits"][0]
        assert set(first) == {"chunk_id", "doc_id", "title", "breadcrumb", "source_url", "body"}
        assert first["chunk_id"] == "d1::0001"
        assert first["doc_id"] == "d1"
        assert first["source_url"].startswith("https://")
        assert retriever.calls == [("probe", 2)]

    def test_long_body_is_truncated(self) -> None:
        """整块正文塞进上下文会挤掉对话历史：截断并留 `…`，让模型知道后面还有内容。"""
        retriever = _StubRetriever([_StubChunk(1, body="x" * 5000)])
        payload = _handler(make_search_tool(retriever, body_limit=100))(SearchDocsParams(query="q"))
        body = payload["hits"][0]["body"]
        assert body.endswith("…")
        assert len(body) < 200

    def test_short_body_is_left_intact(self) -> None:
        """没有超限就不要加省略号，否则模型会以为内容被截了。"""
        retriever = _StubRetriever([_StubChunk(1, body="短正文")])
        payload = _handler(make_search_tool(retriever))(SearchDocsParams(query="q"))
        assert payload["hits"][0]["body"] == "短正文"

    def test_query_is_stripped_before_search(self) -> None:
        """首尾空白对检索没有意义，先剥掉，让缓存/日志里的查询形状稳定。"""
        retriever = _StubRetriever([_StubChunk(1)])
        _handler(make_search_tool(retriever))(SearchDocsParams(query="  kubelet  "))
        assert retriever.calls == [("kubelet", 5)]

    def test_top_k_is_capped_by_the_configured_budget(self) -> None:
        """模型可以在 schema 里要 20 条，但工具构造时的 top_k 才是内核给的预算上限。"""
        retriever = _StubRetriever([_StubChunk(1)])
        _handler(make_search_tool(retriever, top_k=2))(SearchDocsParams(query="q", top_k=20))
        assert retriever.calls == [("q", 2)]

    def test_explicit_small_top_k_is_honored(self) -> None:
        """模型要得比预算少就按它要的来——省下来的预算不该被强行用掉。"""
        retriever = _StubRetriever([_StubChunk(1)])
        _handler(make_search_tool(retriever, top_k=5))(SearchDocsParams(query="q", top_k=1))
        assert retriever.calls == [("q", 1)]

    def test_blank_query_raises_tool_error(self) -> None:
        """长度合法的空白串（" "）也会空转一次检索，必须转成 ToolError。"""
        spec = make_search_tool(_StubRetriever([]))
        with pytest.raises(ToolError, match="空白"):
            _handler(spec)(SearchDocsParams(query=" "))

    def test_retrieval_failure_raises_tool_error(self) -> None:
        """检索失败是可预期失败：转成一句人话（ToolError），不要抛裸异常让图 traceback。"""
        spec = make_search_tool(_StubRetriever([], raises=RuntimeError("db locked")))
        with pytest.raises(ToolError, match="RuntimeError"):
            _handler(spec)(SearchDocsParams(query="q"))

    def test_missing_search_method_raises_tool_error(self) -> None:
        """检索器不符合鸭子类型（没 search）是配置错误，也要以 ToolError 的形式暴露。"""
        spec = make_search_tool(object())
        with pytest.raises(ToolError, match="search"):
            _handler(spec)(SearchDocsParams(query="q"))

    def test_malformed_params_raise_tool_error(self) -> None:
        """内核若把原始 JSON 直接递进来，参数不合法要给 ToolError 而不是 AttributeError。"""
        spec = make_search_tool(_StubRetriever([]))
        with pytest.raises(ToolError, match="参数不合法"):
            _handler(spec)({"query": ""})

    def test_invalid_top_k_at_construction_is_a_programming_error(self) -> None:
        """构造期的坏参数是代码问题：当场 ValueError，而不是等到线上第一次调用。"""
        with pytest.raises(ValueError, match="top_k"):
            make_search_tool(_StubRetriever([]), top_k=0)


class TestCitationCheckTool:
    def test_when_not_and_schema_are_declared(self) -> None:
        """引用核对工具同样要写清楚"什么时候不要用"，否则会被当成万能复核器乱调。"""
        spec = make_citation_check_tool()
        assert spec.when_not.strip()
        schema = spec.schema()
        assert schema["function"]["name"] == "check_citations"
        assert set(schema["function"]["parameters"]["properties"]) == {"text", "n_available"}

    def test_valid_markers_pass(self) -> None:
        payload = _handler(make_citation_check_tool())(
            CitationCheckParams(text="先 requests [1]，再 limits [2]", n_available=2)
        )
        assert payload == {"ok": True, "problems": []}

    def test_out_of_range_marker_is_reported(self) -> None:
        """越界编号必须被指出：打不开的引用比没有引用更误导。"""
        payload = _handler(make_citation_check_tool())(
            CitationCheckParams(text="见 [1] 与 [99]", n_available=1)
        )
        assert payload["ok"] is False
        assert any("[99]" in p for p in payload["problems"])

    def test_answer_without_markers_is_flagged(self) -> None:
        """一个引用都没有就等于无法核对来源，不能算通过。"""
        payload = _handler(make_citation_check_tool())(
            CitationCheckParams(text="就是这么回事", n_available=3)
        )
        assert payload["ok"] is False
        assert any("没有任何引用标记" in p for p in payload["problems"])

    def test_zero_available_chunks_is_flagged(self) -> None:
        """片段数为 0 时任何引用都无从谈起。"""
        payload = _handler(make_citation_check_tool())(CitationCheckParams(text="答案 [1]", n_available=0))
        assert payload["ok"] is False
        assert any("没有提供任何片段" in p for p in payload["problems"])

    def test_zero_marker_is_out_of_range(self) -> None:
        """`[0]` 不是合法编号（片段从 1 开始编号）。"""
        payload = _handler(make_citation_check_tool())(CitationCheckParams(text="见 [0]", n_available=3))
        assert payload["ok"] is False
        assert any("[0]" in p for p in payload["problems"])

    def test_blank_text_raises_tool_error(self) -> None:
        """空答案没有可核对的内容：报 ToolError，而不是返回一个"通过"。"""
        spec = make_citation_check_tool()
        with pytest.raises(ToolError, match="空白"):
            _handler(spec)(CitationCheckParams(text="   ", n_available=2))


class TestRegistryContract:
    def test_register_rejects_empty_when_not(self) -> None:
        """when_not 是约束模型的主要手段，注册表必须拒空；注册表是类级共享状态，用完清空。"""
        ToolRegistry.clear()
        try:
            with pytest.raises(ValueError, match="when_not"):
                ToolRegistry.register(
                    ToolSpec(name="bad_tool", description="没有 when_not", params=EmptyParams)
                )
            assert ToolRegistry.names() == []

            ToolRegistry.register(make_search_tool(_StubRetriever([])))
            ToolRegistry.register(make_citation_check_tool())
            assert ToolRegistry.names() == ["check_citations", "search_docs"]
            assert ToolRegistry.get("search_docs").when_not
            assert ToolRegistry.schemas("k8s_docs"), "注册后应当能产出 function calling schema"
        finally:
            ToolRegistry.clear()
