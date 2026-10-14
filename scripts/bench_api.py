"""API 性能基准：并发下的延迟分位、吞吐、错误率（客户端 + 服务端两个口径）。

## 为什么要它：这是仓库里唯一从没测过的一面

`SECURITY.md` 第 6 节一直写着"**没有做过任何渗透测试、模糊测试或负载测试**"。
检索延迟有 `cli eval`（p50/p95）与 `bench_store_latency.py`（存储层），
但**HTTP 层在并发下是什么样，一次都没量过** —— 而"并发下会不会崩、会不会把
CPU/预算吃光"恰恰是部署时最先被问到的问题。

## 两个口径，别混为一谈

| 口径 | 含什么 | 怎么取 |
|---|---|---|
| **客户端** | 应用耗时 + 序列化 + 环回网络 + 客户端调度 | 本脚本直接掐表 |
| **服务端** | 只有应用耗时（中间件里的 `time.perf_counter()`） | 读 `/metrics` 的 `kubedocs_http_request_seconds` 直方图，**按窗口差值**从 `_bucket` 算分位 |

两个都报：差得远说明瓶颈在序列化/网络或客户端，差得近说明时间真的花在应用里。
（直方图桶是固定的，服务端分位报的是**桶上界** ⇒ 只会高估；默认桶在 100→250 ms 之间有个大空档。）

## 方法

- 每档并发先跑 `--warmup` 个请求丢弃，再计 `--requests` 个；避免把"首次加载模型"
  混进稳态数字（冷启动单独由 `--cold` 模式测量，两者**绝不能混报**）；
- 用 `httpx` 线程池发请求（项目本来就依赖 httpx，不引入 k6/locust —— 本机也没有，
  且"为了压测装一个工具"与"什么都不装到 C 盘"的约束冲突）；
- 每档前后读一次 `/metrics`，报服务端计数增量与**非 2xx** 增量：客户端超时会把
  请求算成失败，而服务端计数能区分"没到服务端"与"服务端返回了错"。

## 用法

    python scripts/bench_api.py                              # 默认打 127.0.0.1:8080
    python scripts/bench_api.py --base-url http://127.0.0.1:8080 --requests 40
    python scripts/bench_api.py --endpoint /healthz          # 极轻的对照端点
    python scripts/bench_api.py --cold                       # 只测冷启动（首个请求）
    python scripts/bench_api.py --json out.json              # 落盘，便于写进文档

## 读结果时的三个前提

1. **本机不是基准机**：16 GB / 20 逻辑核，且 `--base-url` 可能指向 Docker 容器
   （CPU 与宿主共享）。数字只对"这台机器这一次"负责，不要当容量规划依据。
2. **`/search` 的第一枪要加载嵌入模型**（本机实测冷启动 20~70 s），所以稳态测试
   必须预热；`--cold` 是**单独**的模式。
3. **别把这份数字当 SLO**：没有多机、没有长稳压、没有内存泄漏观测。
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_QUERY = "how do I stop a container using too much memory"
"""与 README / 金标集里用的同一句，便于和历史数字对齐。"""


@dataclass
class Level:
    """一档并发的结果。"""

    concurrency: int
    n_ok: int = 0
    n_err: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def throughput(self) -> float:
        return self.n_ok / self.wall_s if self.wall_s > 0 else 0.0

    def pct(self, q: float) -> float:
        if not self.latencies_ms:
            return float("nan")
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, max(0, round(q / 100.0 * len(ordered) + 0.5) - 1))
        return ordered[idx]


def _hist_buckets(text: str, metric: str, endpoint: str) -> dict[float, float]:
    """抓某个端点的直方图桶（`le` -> 累计计数）。

    必须按**前后差值**用（见 `_quantile_from_delta`）：Prometheus 直方图是
    **进程生命周期累计**的，而进程里第一枪 `/search` 要加载嵌入模型（本机 20~70 s），
    它落在高桶里。直接对累计值取分位会得到"服务端 p95 = 750ms"这种**假数字**——
    实测第一次跑就是这么被骗的（客户端只有 100ms）。
    """
    out: dict[float, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(f"{metric}_bucket"):
            continue
        if f'endpoint="{endpoint}"' not in line:
            continue
        try:
            labels, value = line.rsplit(" ", 1)
            le = float(labels.split('le="')[1].split('"')[0])
        except (IndexError, ValueError):
            continue
        out[le] = float(value)
    return out


def _quantile_from_delta(before: dict[float, float], after: dict[float, float], q: float) -> float | None:
    """窗口内分位（秒 → 毫秒）：先逐桶相减，再在差值上取分位。

    返回的是命中桶的 `le`，即**桶上界**：真分位 ≤ 报出的数。也就是说这个数字
    只会**高估**、不会低估（默认桶在 100ms → 250ms 之间有个大空档，所以报 250
    的实际含义是"p50 落在 100~250 ms 之间"）。报表必须写明这一点，否则会被
    当成"服务端比客户端还慢"。
    """
    delta = {le: after.get(le, 0.0) - before.get(le, 0.0) for le in after}
    total = delta.get(float("inf"), 0.0)
    if total <= 0:
        return None
    target = q / 100.0 * total
    for le, cum in sorted(delta.items()):
        if cum >= target:
            return le * 1000.0
    return None


def _server_counters(text: str, endpoint: str) -> tuple[int, int]:
    """服务端口径：该端点的请求总数与**非 2xx** 数。"""
    total = 0
    bad = 0
    for line in text.splitlines():
        if not line.startswith("kubedocs_http_requests_total{"):
            continue
        if f'endpoint="{endpoint}"' not in line:
            continue
        try:
            labels, value = line.rsplit(" ", 1)
            status = labels.split('status="')[1].split('"')[0]
            count = int(float(value))
        except (IndexError, ValueError):
            continue
        total += count
        if not status.startswith("2"):
            bad += count
    return total, bad


def _one_request(client: httpx.Client, method: str, url: str, payload: dict[str, Any]) -> tuple[float, bool]:
    started = time.perf_counter()
    try:
        resp = client.get(url) if method == "GET" else client.post(url, json=payload)
        ok = resp.status_code < 400
    except httpx.HTTPError:
        ok = False
    return (time.perf_counter() - started) * 1000.0, ok


def run_level(
    client: httpx.Client, method: str, url: str, payload: dict[str, Any], concurrency: int, n: int
) -> Level:
    lvl = Level(concurrency=concurrency)
    lock = threading.Lock()

    def worker(_: int) -> None:
        latency, ok = _one_request(client, method, url, payload)
        with lock:
            lvl.latencies_ms.append(latency)
            if ok:
                lvl.n_ok += 1
            else:
                lvl.n_err += 1

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(n)))
    lvl.wall_s = time.perf_counter() - started
    return lvl


def _wait_until_ready(client: httpx.Client, base: str, timeout_s: float) -> float:
    """轮询 `/healthz` 直到通，返回耗时（秒）。用于冷启动测量。"""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        try:
            if client.get(f"{base}/healthz", timeout=5).status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise SystemExit(f"等了 {timeout_s}s 服务仍不可达：{base}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/search")
    parser.add_argument(
        "--method", default=None, help="HTTP 方法；默认按端点推断（/search /ask /chat 用 POST，其余 GET）"
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", default="1,2,4,8", help="逗号分隔的并发档")
    parser.add_argument("--requests", type=int, default=32, help="每档的请求数")
    parser.add_argument("--warmup", type=int, default=4, help="每档丢弃的预热请求数")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", type=Path, default=None, help="把结果落盘（便于写进文档）")
    parser.add_argument("--cold", action="store_true", help="只测冷启动：等到 /healthz 通 + 第一枪 /search")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    # /healthz、/metrics、/domains、/tools、/console 都是 GET；POST 它们会拿到 405，
    # 于是"延迟"测的是 405 的生成开销、而且 64 个请求全被算成失败
    # （实测踩到：服务端非 2xx +216 的交叉核对把它抓了出来）。
    method = args.method or ("POST" if args.endpoint in ("/search", "/ask", "/chat") else "GET")
    payload: dict[str, Any] = {"q": args.query, "top_k": args.top_k}
    url = f"{base}{args.endpoint}"
    result: dict[str, Any] = {
        "base_url": base,
        "endpoint": args.endpoint,
        "method": method,
        "payload": payload,
        "levels": [],
    }

    with httpx.Client(timeout=args.timeout) as client:
        if args.cold:
            # 冷启动：调用方自己负责把服务停在"刚起来、还没被请求过"的状态。
            ready_s = _wait_until_ready(client, base, timeout_s=args.timeout)
            first_ms, ok = _one_request(client, method, url, payload)
            print(
                f"冷启动：/healthz 就绪 {ready_s:.1f}s；首个 {args.endpoint} {first_ms / 1000:.1f}s（ok={ok}）"
            )
            result["cold"] = {"ready_s": ready_s, "first_request_ms": first_ms, "ok": ok}
            if args.json:
                args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0 if ok else 1

        # 服务端计数基线
        before_text = client.get(f"{base}/metrics").text
        srv_before, bad_before = _server_counters(before_text, args.endpoint)

        print(
            f"{'并发':>4} {'请求':>5} {'成功':>5} {'失败':>5} {'吞吐/s':>8} "
            f"{'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'服务端p50/p95':>14}"
        )
        print("-" * 80)
        for level in [int(x) for x in args.concurrency.split(",") if x.strip()]:
            for _ in range(args.warmup):
                _one_request(client, method, url, payload)
            # 服务端直方图按**窗口差值**取分位：累计值会被冷启动那一枪污染。
            hist_before = _hist_buckets(
                client.get(f"{base}/metrics").text, "kubedocs_http_request_seconds", args.endpoint
            )
            lvl = run_level(client, method, url, payload, level, args.requests)
            hist_after = _hist_buckets(
                client.get(f"{base}/metrics").text, "kubedocs_http_request_seconds", args.endpoint
            )
            srv_p50 = _quantile_from_delta(hist_before, hist_after, 50)
            srv_p95 = _quantile_from_delta(hist_before, hist_after, 95)
            print(
                f"{level:>4} {args.requests:>5} {lvl.n_ok:>5} {lvl.n_err:>5} {lvl.throughput:>8.2f} "
                f"{lvl.pct(50):>8.1f} {lvl.pct(95):>8.1f} {lvl.pct(99):>8.1f} "
                f"{(f'{srv_p50:.0f}/{srv_p95:.0f}' if srv_p95 else '—'):>14}"
            )
            result["levels"].append(
                {
                    "concurrency": level,
                    "requests": args.requests,
                    "ok": lvl.n_ok,
                    "err": lvl.n_err,
                    "throughput_per_s": round(lvl.throughput, 2),
                    "p50_ms": round(lvl.pct(50), 1),
                    "p95_ms": round(lvl.pct(95), 1),
                    "p99_ms": round(lvl.pct(99), 1),
                    "mean_ms": round(statistics.fmean(lvl.latencies_ms), 1) if lvl.latencies_ms else None,
                    "server_p50_ms_bucket": srv_p50,
                    "server_p95_ms_bucket": srv_p95,
                }
            )

        after_text = client.get(f"{base}/metrics").text
        srv_after, bad_after = _server_counters(after_text, args.endpoint)
        result["server_side"] = {
            "requests_total_delta": srv_after - srv_before,
            "non_2xx_delta": bad_after - bad_before,
        }
        print(f"\n服务端口径：本端点请求 +{srv_after - srv_before}，其中非 2xx +{bad_after - bad_before}")
        print("（服务端分位取直方图**桶上界** ⇒ 只会高估不会低估；客户端分位含序列化与环回）")

    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
