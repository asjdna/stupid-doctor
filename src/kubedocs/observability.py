"""可观测性：结构化日志级别、OTel 追踪、Prometheus 指标。

三块配置都在这里落到真实代码路径上：

| 配置 | 作用 |
|---|---|
| `log_level` | `configure_logging()` 设置 root logger 级别 |
| `otel_enable` / `otel_exporter` | `setup_tracing()` 建 `TracerProvider` 并导出（console / otlp）；检索、模型调用与工具执行真的开 span |
| `/metrics` | Prometheus 文本格式端点，计数器与直方图在真实请求路径上被 +1 / observe |

指标口径：

- `kubedocs_http_requests_total{endpoint,status}`：按端点与状态码计数
- `kubedocs_http_request_seconds{endpoint}`：请求耗时直方图
- `kubedocs_retrieve_seconds{channel}`：检索单路耗时（`search --explain` 里那个数字的指标版）
- `kubedocs_llm_calls_total{priced}`：LLM 调用次数，按能否定价分解
- `kubedocs_llm_cost_usd_total`：累计花费（与 `llm_spend.json` 同一口径，便于对账）
- `kubedocs_llm_budget_usd`：当前日上限（gauge，**0 表示不限**）

不引第三方 instrumentation 包：手工 span + 手工计数，依赖少、离线可跑。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kubedocs.config import Settings

logger = logging.getLogger(__name__)

_PREFIX = "kubedocs"

HTTP_REQUESTS = Counter(
    f"{_PREFIX}_http_requests_total",
    "HTTP 请求数（按端点与状态码）",
    ["endpoint", "status"],
)
HTTP_LATENCY = Histogram(
    f"{_PREFIX}_http_request_seconds",
    "HTTP 请求耗时（秒）",
    ["endpoint"],
)
RETRIEVE_LATENCY = Histogram(
    f"{_PREFIX}_retrieve_seconds",
    "单路检索耗时（秒）",
    ["channel"],
)
LLM_CALLS = Counter(
    f"{_PREFIX}_llm_calls_total",
    "LLM 调用次数（priced 标签区分能否定价）",
    ["priced"],
)
LLM_COST = Counter(
    f"{_PREFIX}_llm_cost_usd_total",
    "累计 LLM 花费（美元）",
)
LLM_BUDGET = Gauge(
    f"{_PREFIX}_llm_budget_usd",
    "当日 LLM 花费上限（美元；0 表示不限）",
)
SAFETY_INVARIANT_FIRED = Counter(
    f"{_PREFIX}_safety_invariant_fired_total",
    "确定性安全不变量被触发的次数（按不变量与入口）",
    ["invariant", "domain"],
)
"""安全门触发计数。

它是**运行期长期监听**的抓手：审计日志（`safety_audit.jsonl`）能回答"发生了什么"，
但要发现"最近触发变多了"得靠 Prometheus 这种可聚合的计数——告警规则写在它上面，
而不是让人去 tail 一个 JSONL 文件。每条不变量单独一个标签值，所以"哪一条开始频繁触发"
一眼可见；`domain` 标签区分 `/consult`（临床管线）与 `/chat`（散文路径）。
"""


def configure_logging(settings: Settings) -> None:
    """把 `settings.log_level` 应用到 root logger。

    root logger 已经有 handler 时只改级别、不动已有格式（`force=False`）：
    库代码不该抢宿主程序（例如 uvicorn）的日志配置；作为进程入口（CLI /
    `create_app`）调用时它就是那份配置。级别名无法解析时退回 `INFO`。
    """
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)
    logger.debug("日志级别设为 %s", logging.getLevelName(level))


_TRACER: Any | None = None


def setup_tracing(settings: Settings) -> Any | None:
    """按配置建 `TracerProvider` 并返回 tracer；关闭时返回 `None`。

    幂等：重复调用返回同一个 provider（`uvicorn --reload` 会重复 import）。
    """
    global _TRACER
    if not settings.otel_enable or settings.otel_exporter == "none":
        logger.info(
            "OTel 追踪未启用（otel_enable=%s exporter=%s）", settings.otel_enable, settings.otel_exporter
        )
        return None
    if _TRACER is not None:
        return _TRACER

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": "kubedocs"}))
    if settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # OTLP 端点由标准环境变量 OTEL_EXPORTER_OTLP_ENDPOINT 提供
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("kubedocs")
    logger.info("OTel 追踪已启用（exporter=%s）", settings.otel_exporter)
    return _TRACER


@contextmanager
def span(name: str, **attributes: object) -> Iterator[Any]:
    """开一个 span 并写入属性；未启用追踪时是**零开销的空操作**。

    用法一律是 `with span("retrieve", channel="bm25"):`——调用方不需要知道追踪
    有没有开，也就不必写 `if tracing_enabled` 这类容易漏路径的分支。
    """
    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current


def span_attrs(current: Any | None, **attributes: object) -> None:
    """给已经开着的 span 补属性；未启用追踪时（`current is None`）是空操作。

    调用点拿到的是 `with span(...) as current` 的返回值——追踪关着时它是 `None`，
    所以这里必须容忍 `None`，否则每个调用点都要写一遍 `if current is not None`。
    """
    if current is None:
        return
    for key, value in attributes.items():
        current.set_attribute(key, value)


def current_trace_id() -> str:
    """当前 span 的 trace id（32 位十六进制）；未启用追踪时返回空串。

    回写进响应，让"一次请求"与"一条 trace"能对上——否则追踪开了也只是一堆没有归属的
    span：日志里有、报告里没有，谁也无法用它回答"这次问诊到底走了哪些步骤"。
    """
    if _TRACER is None:
        return ""
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return ""
    return format(ctx.trace_id, "032x")


def metrics_payload() -> tuple[bytes, str]:
    """Prometheus 文本格式的指标快照 + 正确的 Content-Type。"""
    return generate_latest(), CONTENT_TYPE_LATEST


def observe_llm_call(*, cost_usd: float, priced: bool, budget_usd: float | None = None) -> None:
    """记一次 LLM 调用（纯文本与工具循环两条路径共用）。

    与 `DailyBudget.record()` 是**互补**关系：账本是"花费的事实来源"，指标是
    "可被 Prometheus 抓取的视图"。`budget_usd` 只有调用方知道时才设 gauge
    （工具循环不持有配置，就不设）。
    """
    LLM_CALLS.labels(priced="yes" if priced else "no").inc()
    if cost_usd:
        LLM_COST.inc(cost_usd)
    if budget_usd is not None:
        LLM_BUDGET.set(budget_usd)


def publish_config_gauges(settings: Settings) -> None:
    """把配置里"应当可见"的值在**启动时**发布成 gauge。

    `LLM_BUDGET` 若只在真有 LLM 调用时才被 `observe_llm_call` 设值，那么"配了上限
    但一次都还没调用过"的进程里它是 0，而 0 的约定是"不限"——控制台会把真实存在的
    上限显示成"不限"，这是显示错的金额控制，所以在装配点补一次，而不是等第一次调用。
    没密钥、纯检索的部署同样受益：预算上限与有没有密钥无关。
    """
    LLM_BUDGET.set(float(settings.llm_daily_budget_usd or 0.0))
