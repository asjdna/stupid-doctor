"""追踪插桩的契约：属性必须在 span 结束前写入，trace id 必须在 span 内取。

这两个坑都真实踩过：把 `span_attrs()` / `current_trace_id()` 写在 `with span(...)` **外面**
时，OpenTelemetry 只会打一句 `Setting attribute on ended span` 警告、值进不了 span，
trace id 则直接返回空串——而调用方看起来"代码写了、字段也有了"，只是永远是空的。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kubedocs import observability
from kubedocs.config import Settings
from kubedocs.generate import LiteLLMClient, LLMResult
from kubedocs.observability import current_trace_id, span, span_attrs


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """把追踪器换成内存导出器。

    直接改 `observability._TRACER` 而不是 `trace.set_tracer_provider()`：后者全局只能设一次，
    在测试里会互相污染；而 `span()` 读的就是这个模块级变量，换掉它即等价于"开了追踪"。
    """
    out = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(out))
    monkeypatch.setattr(observability, "_TRACER", provider.get_tracer("test"))
    yield out
    out.clear()


def test_span_records_attributes(exporter: InMemorySpanExporter) -> None:
    with span("demo", channel="bm25") as current:
        span_attrs(current, limit=50)

    got = exporter.get_finished_spans()
    assert [s.name for s in got] == ["demo"]
    assert got[0].attributes is not None
    assert got[0].attributes["channel"] == "bm25"
    assert got[0].attributes["limit"] == 50, "写在 span 结束之后的属性会丢，这条断言就是防它"


def test_current_trace_id_only_inside_a_span(exporter: InMemorySpanExporter) -> None:
    assert current_trace_id() == "", "没有活跃 span 时必须是空串，而不是伪造一个 id"

    with span("demo"):
        inside = current_trace_id()

    assert len(inside) == 32, f"trace id 应为 32 位十六进制，实际 {inside!r}"
    assert set(inside) <= set("0123456789abcdef")

    # 同一 trace 内的子 span 共享 trace id
    with span("parent"):
        outer = current_trace_id()
        with span("child"):
            inner = current_trace_id()
    assert outer == inner


def test_span_is_noop_when_tracing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """关着追踪时是零开销空操作：`as` 拿到 None，写属性与取 id 都不能炸。"""
    monkeypatch.setattr(observability, "_TRACER", None)

    with span("demo", a=1) as current:
        assert current is None
        span_attrs(current, b=2)
        assert current_trace_id() == ""


class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7


class _FakeMessage:
    content = "ok"
    tool_calls: list[Any] = []


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeLiteLLM:
    def completion(self, **kwargs: Any) -> _FakeResponse:
        del kwargs
        return _FakeResponse()

    def completion_cost(self, **kwargs: Any) -> float:
        del kwargs
        return 0.01


def test_llm_call_span_carries_token_usage(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实调用点（`LiteLLMClient.complete`）必须把 token 数写进 llm.call span。

    用假 litellm 而不是真调模型：这条测的是插桩位置对不对，不该依赖网络与密钥。
    """
    monkeypatch.setattr("kubedocs.generate.has_api_key", lambda model: True)
    client = LiteLLMClient(Settings(domain="clinic"))
    client._litellm = _FakeLiteLLM()  # noqa: SLF001 - 测试替身，就是要替掉它

    result = client.complete("sys", "user")

    assert isinstance(result, LLMResult)
    spans = [s for s in exporter.get_finished_spans() if s.name == "llm.call"]
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes["model"] == client.model
    assert attributes["purpose"] == "text"
    assert attributes["prompt_tokens"] == 11
    assert attributes["completion_tokens"] == 7
