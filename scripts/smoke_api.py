"""端到端验收：把服务层每个端点真的打一遍，并检查工具链路是活的。

这是跨模块的冒烟：真的构造 app、真的发请求、真的看响应体。单测容易漏的三件事——
每个端点都返回 200（含索引为空）、`/tools` 里真的能看到工具、`/` 与 `/console` 是真的页面。

用法：
    python scripts/smoke_api.py                 # 用真实索引
    python scripts/smoke_api.py --empty-index   # 故意用空索引，验证降级路径

前置条件：服务层依赖可导入，否则退出码 3；不需要任何模型密钥。

输出：每项检查一行 `[PASS]`/`[FAIL]` 与详情，末行是「通过数/总数」。
退出码：0 全部通过；1 有失败项；3 服务层不可用。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="服务层端到端冒烟")
    p.add_argument("--empty-index", action="store_true", help="用空索引，验证「索引为空也不能 500」")
    p.add_argument("--domain", default="k8s_docs")
    args = p.parse_args(argv)

    try:
        from fastapi.testclient import TestClient

        from kubedocs.api.app import create_app
        from kubedocs.config import Settings
    except ImportError as e:
        print(f"服务层不可用：{type(e).__name__}: {e}", file=sys.stderr)
        return 3

    if args.empty_index:
        tmp = tempfile.mkdtemp(prefix="kubedocs-smoke-")
        settings = Settings(index_dir=tmp, domain=args.domain, embedding_provider="hashing", embedding_dim=32)
        print(f"（使用空索引：{tmp}）")
    else:
        settings = Settings(domain=args.domain)
        print(f"（使用真实索引：{settings.sqlite_path}，存在={settings.sqlite_path.exists()}）")

    app = create_app(settings)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    with TestClient(app) as client:
        r = client.get("/healthz")
        check("GET /healthz 200", r.status_code == 200, f"{r.status_code}")
        if r.status_code == 200:
            body = r.json()
            check("healthz 含 n_chunks", "n_chunks" in body, str(body)[:120])

        r = client.get("/domains")
        ok = r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) >= 2
        check("GET /domains 返回 ≥2 个域", ok, f"{r.status_code} {str(r.text)[:120]}")

        r = client.get("/tools")
        names: list[str] = []
        if r.status_code == 200:
            names = [t.get("name", "") for t in r.json().get("tools", [])]
        check("GET /tools 200", r.status_code == 200, f"{r.status_code}")
        check("工具链路是活的（至少 2 个工具）", len(names) >= 2, f"tools={names}")

        r = client.get(f"/tools?domain={args.domain}")
        check("GET /tools?domain= 200", r.status_code == 200, f"{r.status_code}")

        r = client.post("/search", json={"q": "pod cpu limit", "top_k": 5})
        check("POST /search 200", r.status_code == 200, f"{r.status_code}")
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            if args.empty_index:
                check("空索引 /search 返回空而非 500", hits == [], f"hits={len(hits)}")
            else:
                ok = bool(hits) and all(h.get("source_url") for h in hits)
                check("有索引时 hits 带 source_url", ok, f"hits={len(hits)}")

        r = client.post("/search", json={"q": "x", "top_k": 99})
        check("top_k 越界被拒（422）", r.status_code == 422, f"{r.status_code}")

        r = client.post("/search", json={"q": "x", "domain": "no_such_domain"})
        check("非法域被拒（400）", r.status_code == 400, f"{r.status_code}")

        r = client.post("/ask", json={"q": "pod cpu limit", "top_k": 5})
        check("POST /ask 不因无密钥而 500", r.status_code == 200, f"{r.status_code}")

        # SSE：只验证"是流且收到了事件"，不验证内容（内容需要真实模型）
        try:
            with client.stream("POST", "/chat", json={"message": "我咳嗽", "thread_id": "smoke"}) as s:
                ctype = s.headers.get("content-type", "")
                body = "".join(s.iter_text())
            check("POST /chat 是 SSE", "text/event-stream" in ctype, ctype)
            check("SSE 里有 event:", "event:" in body, body[:80].replace("\n", " "))
        except Exception as e:  # noqa: BLE001 - 冒烟脚本要把异常变成失败项而不是崩掉
            check("POST /chat 可流式返回", False, f"{type(e).__name__}: {e}")

        r = client.get("/")
        check("GET / 200", r.status_code == 200, f"{r.status_code}")
        check(
            "GET / 返回真正的前端页（不是兜底说明）",
            "html" in r.headers.get("content-type", "").lower()
            and ("<script" in r.text or "<form" in r.text),
            f"{len(r.text)} 字节",
        )

        # 容器里没有外网：`/console` 必须自带样式与脚本、不挂任何外链。
        r = client.get("/console")
        check("GET /console 200", r.status_code == 200, f"{r.status_code}")
        check(
            "GET /console 自带样式与脚本且无外链",
            "<style" in r.text and "<script" in r.text and "//cdn" not in r.text,
            f"{len(r.text)} 字节",
        )

        # 患者端不该混进开发/运维控件：改 API 地址、域、top_k 对患者是纯粹的坏事。
        r = client.get("/patient")
        check("GET /patient 200", r.status_code == 200, f"{r.status_code}")
        check(
            "GET /patient 无开发/运维控件",
            all(x not in r.text for x in ("#domainSel", 'id="apiBase"', 'id="topK"', "/metrics", "/tools")),
            f"{len(r.text)} 字节",
        )
        check(
            "GET /patient 保留急诊指引与免责声明",
            "emergency" in r.text and "120" in r.text and "不能替代医生诊断" in r.text,
            "急诊/免责在",
        )

    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name:<{width}}  {detail}")
    print(f"\n{len(checks) - failed}/{len(checks)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
