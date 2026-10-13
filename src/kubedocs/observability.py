"""可观测性：结构化日志级别、OTel 追踪、Prometheus 指标。

## 为什么这个模块是"接上"而不是"声明"

`config.py` 里曾有 6 个**没有人读**的观测字段（`redis_url`、`otel_*`、`langfuse_*`），
而 `config.py` 自己在另一处注释里立过规矩：**"假开关比缺配置更糟"**。
所以这里把其中真正该有的三个接上真实分支：

| 配置 | 现在的行为 |
|---|---|
| `log_level` | `configure_logging()` 真正设 root logger 级别（此前 `KUBEDOCS_LOG_LEVEL` 无任何效果） |
| `otel_enable` / `otel_exporter` | `setup_tracing()` 真的建 `TracerProvider` 并导出（console / otlp），检索与 LLM 调用真的开 span |
| `/metrics` | Prometheus 文本格式端点，计数器与直方图在真实请求路径上被 +1 / observe |

**没用上的那两个（`redis_url` 与 `langfuse_*`）已被删除**，连同 `redis` 依赖与
compose 里那个没人连的 redis 服务——而不是留着它们继续假装生效。

## 指标口径

- `kubedocs_http_requests_total{endpoint,status}`：按端点与状态码计数
- `kubedocs_http_request_seconds{endpoint}`：请求耗时直方图
- `kubedocs_retrieve_seconds{channel}`：检索单路耗时（`search --explain` 里那个数字的指标版）
- `kubedocs_llm_calls_total{priced}`：LLM 调用次数，按"能否定价"分解
- `kubedocs_llm_cost_usd_total`：累计花费（与 `llm_spend.json` 同一口径，便于对账）
- `kubedocs_llm_budget_usd`：当前日上限（gauge，便于算余量）

不引第三方 instrumentation 包：手工 span + 手工计数，依赖少、可读、离线可跑。
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


def configure_logging(settings: Settings) -> None:
    """把 `settings.log_level` 真正应用到 root logger。

    此前 `--log-level` 只进了 `Settings`（并只被 `uvicorn.run(log_level=...)` 用来设
    uvicorn 自己的级别），`KUBEDOCS_LOG_LEVEL=DEBUG` 对本项目自己的 logger 没有任何效果。
    `force=False`：已经配置过就**不覆盖**——库代码不该抢宿主程序的日志配置，
    但作为进程入口（CLI / create_app）调用时它就是那份配置。
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

        # 端点走标准环境变量 OTEL_EXPORTER_OTLP_ENDPOINT（不在这里再发明一个配置项）
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("kubedocs")
    logger.info("OTel 追踪已启用（exporter=%s）", settings.otel_exporter)
    return _TRACER


@contextmanager
def span(name: str, **attributes: object) -> Iterator[Any]:
    """开一个 span；未启用追踪时是**零开销的空操作**。

    用法一律是 `with span("retrieve", channel="bm25"):`——调用方不需要知道
    追踪有没有开，也就不需要写 `if tracing_enabled` 这类分支（那种分支最终
    会有一半的路径忘了加）。
    """
    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current


def metrics_payload() -> tuple[bytes, str]:
    """Prometheus 文本格式的指标快照 + 正确的 Content-Type。"""
    return generate_latest(), CONTENT_TYPE_LATEST


def observe_llm_call(*, cost_usd: float, priced: bool, budget_usd: float | None = None) -> None:
    """记一次 LLM 调用（两条 LLM 路径共用）。

    与 `DailyBudget.record()` 是**互补**关系，不是替代：账本是"花费的事实来源"，
    指标是"可被 Prometheus 抓取的视图"。两者都在真实调用路径上被触达。
    `budget_usd` 只有调用方知道时才设 gauge（工具循环不持有配置，就不设）。
    """
    LLM_CALLS.labels(priced="yes" if priced else "no").inc()
    if cost_usd:
        LLM_COST.inc(cost_usd)
    if budget_usd is not None:
        LLM_BUDGET.set(budget_usd)
