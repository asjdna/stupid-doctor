"""观测面测试：日志级别、预算 gauge、`/metrics` 的暴露文本与标签口径。

## 为什么这个文件此前缺失是个真问题

`configure_logging` / `setup_tracing` / `publish_config_gauges` 与 `/metrics`
在 README 与 SECURITY 里都被当作**已实现的**能力对外声明（"开关不再是假开关"），
但全仓没有任何一行测试碰过它们——声明的证据只有"我手工 curl 过一次"。
手工 curl 证明不了回归：把中间件的标签从路由模板改回原始 URL，页面照样出数，
只有 Prometheus 侧的聚合会悄悄碎掉。所以这里把口径**写死成断言**。

不联网、不起真实服务：一律走 `TestClient` + `hashing` 嵌入。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kubedocs.api.app import create_app
from kubedocs.config import Settings
from kubedocs.observability import (
    LLM_BUDGET,
    configure_logging,
    metrics_payload,
    publish_config_gauges,
)

DIM = 32


def _sample(name: str, text: str) -> float | None:
    """从 Prometheus 暴露文本里取一个无标签样本的值。

    刻意解析**暴露文本**而不是读 `LLM_BUDGET._value`：前者是 Prometheus 真正抓到的
    东西（也就是外部唯一能看到的东西），后者是内部状态。测内部状态会放过
    "值设对了但没被 expose" 这类问题。
    """
    pattern = re.compile(rf"^{re.escape(name)}\s+([0-9.eE+-]+)\s*$", re.MULTILINE)
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "index_dir": str(tmp_path / "index"),
        "domain": "k8s_docs",
        "embedding_provider": "hashing",
        "embedding_dim": DIM,
        "embedding_model": "hashing-test",
        "rerank_enable": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------- 日志级别
class TestConfigureLogging:
    def test_level_is_applied_to_root_logger(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        before = root.level
        try:
            configure_logging(_settings(tmp_path, log_level="WARNING"))
            assert root.level == logging.WARNING
            configure_logging(_settings(tmp_path, log_level="DEBUG"))
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(before)

    def test_unknown_level_falls_back_to_info(self, tmp_path: Path) -> None:
        """拼错的级别不该让进程起不来，退到 INFO 即可（而不是抛异常）。"""
        root = logging.getLogger()
        before = root.level
        try:
            configure_logging(_settings(tmp_path, log_level="NOT-A-LEVEL"))
            assert root.level == logging.INFO
        finally:
            root.setLevel(before)


# --------------------------------------------------------------------- 预算 gauge
class TestPublishConfigGauges:
    def test_budget_is_published_without_any_llm_call(self, tmp_path: Path) -> None:
        """**核心回归**：一次 LLM 都没调用过时，上限也必须已经在指标里。

        修之前 `LLM_BUDGET` 只在 `observe_llm_call()` 里被设值，于是"配了 $2.00、
        但还没调用过"的进程暴露的是 gauge 默认值 0——而 0 的约定是"不限"，
        控制台就会把一个真实存在的花费上限显示成"不限"。这是显示假信息，
        比少一个字段严重。
        """
        publish_config_gauges(_settings(tmp_path, llm_daily_budget_usd=2.0))
        text = metrics_payload()[0].decode("utf-8")
        assert _sample("kubedocs_llm_budget_usd", text) == 2.0

    def test_zero_budget_means_unlimited_and_stays_zero(self, tmp_path: Path) -> None:
        """0 是"不限"的**合法**取值，不能被当成"没设"而不发出去。"""
        publish_config_gauges(_settings(tmp_path, llm_daily_budget_usd=0.0))
        text = metrics_payload()[0].decode("utf-8")
        assert _sample("kubedocs_llm_budget_usd", text) == 0.0

    def test_create_app_publishes_budget(self, tmp_path: Path) -> None:
        """装配点必须真的调用它——单测函数本身通过、但没人调用，是白测。"""
        with TestClient(create_app(_settings(tmp_path, llm_daily_budget_usd=7.5))):
            text = metrics_payload()[0].decode("utf-8")
        assert _sample("kubedocs_llm_budget_usd", text) == 7.5

    def test_gauge_object_reflects_value(self, tmp_path: Path) -> None:
        """顺带钉住对象本身，避免"文本里有、对象里没有"这种诡异不一致。"""
        publish_config_gauges(_settings(tmp_path, llm_daily_budget_usd=1.25))
        assert LLM_BUDGET._value.get() == 1.25  # noqa: SLF001 - 就是要读这个内部量做交叉验证


# --------------------------------------------------------------------- /metrics
class TestMetricsEndpoint:
    def test_content_type_is_prometheus(self, tmp_path: Path) -> None:
        with TestClient(create_app(_settings(tmp_path))) as client:
            resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_exposes_the_documented_metric_names(self, tmp_path: Path) -> None:
        """README/模块文档列出的名字必须真的出现在暴露文本里——名字就是对外契约。"""
        with TestClient(create_app(_settings(tmp_path))) as client:
            resp = client.get("/metrics")
        for name in (
            "kubedocs_http_requests_total",
            "kubedocs_http_request_seconds",
            "kubedocs_retrieve_seconds",
            "kubedocs_llm_calls_total",
            "kubedocs_llm_cost_usd_total",
            "kubedocs_llm_budget_usd",
        ):
            assert name in resp.text, f"{name} 没出现在 /metrics 里"

    def test_request_counter_labels_by_route_template(self, tmp_path: Path) -> None:
        """端点标签必须是**路由模板**而不是原始 URL。

        用原始 URL 的话 `/search?q=...` 每次查询都会造出一个新时间序列——
        Prometheus 侧的基数和聚合会一起碎掉，而页面看上去一切正常。
        这里用一个带查询串/带路径参数的请求来钉住这条口径。
        """
        with TestClient(create_app(_settings(tmp_path))) as client:
            client.get("/healthz")
            client.get("/healthz?probe=1")  # 同一个模板，不该产生第二条序列
            resp = client.get("/metrics")

        assert 'endpoint="/healthz"' in resp.text
        assert 'endpoint="/healthz?probe=1"' not in resp.text
        # 同一个模板的两次请求应当累加到**同一个**序列上（值 >= 2）
        counts = re.findall(
            r'kubedocs_http_requests_total\{endpoint="/healthz",status="200"\} ([0-9.]+)', resp.text
        )
        assert counts, "没有找到 /healthz 的计数样本"
        assert float(counts[0]) >= 2

    def test_status_label_separates_failures(self, tmp_path: Path) -> None:
        """状态码必须进标签：否则 404 会混进"正常流量"，仪表盘看不出异常。

        控制台的"非 2xx 响应"那一行完全依赖这条口径。
        """
        with TestClient(create_app(_settings(tmp_path))) as client:
            client.get("/no-such-route")
            resp = client.get("/metrics")
        assert re.search(r'kubedocs_http_requests_total\{[^}]*status="404"\}', resp.text)


@pytest.mark.parametrize("path", ["/metrics", "/healthz"])
def test_metrics_never_requires_llm_or_network(tmp_path: Path, path: str) -> None:
    """观测端点必须与 LLM/模型完全无关：没密钥、没索引时也要能拿到。"""
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get(path).status_code == 200
