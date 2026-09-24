"""取 Kubernetes 官方文档语料。

环境实测结论（决定了本脚本的做法）：
- git 直连 github.com 被墙（Failed to connect ... 21s），无法 clone
- 逐文件 CDN（jsDelivr / statically / githack）全部连接被拒
- GitHub API 可用但限流 60/h，取 1720 个文件必然撞墙
- ⇒ 只有一条路：下 `codeload` 的 tarball，流式解包只抽 content/en/docs/**/*.md

用 `tarfile` 的 `r|gz`（流式、不 seek、不吃内存），边下边抽，抽完删 tarball。
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import time

import httpx

URL = "https://codeload.github.com/kubernetes/website/tar.gz/refs/heads/main"
ROOT = r"F:\86189\Documents\kubedocs"
TARBALL = os.path.join(ROOT, ".cache", "k8s-website.tar.gz")
OUT_DIR = os.path.join(ROOT, "data", "raw", "k8s-docs")
# tarball 内的路径形如 `kubernetes-website-<sha>/content/en/docs/concepts/foo.md`，
# 所以**不能**用 startswith("content/en/docs/") 匹配（第一版就栽在这，抽出 0 个文件）。
INNER_PREFIX = "content/en/docs/"
MAX_BYTES = 2_600_000_000          # 保险：超过 2.6GB 说明判断有误，中止而不是塞满磁盘


def _iter_docs(member_name: str) -> str | None:
    """从 tarball 成员名解析出 (语料内相对路径)。不是目标文档则返回 None。

    剥掉两层：仓库目录前缀（`kubernetes-website-<sha>/`）与 `content/en/docs/`，
    使语料根 == 官方文档根，source_url 映射才是直通的。
    """
    _, _, rest = member_name.partition("/")          # 去掉仓库目录那一层
    if not rest.startswith(INNER_PREFIX) or not rest.endswith(".md"):
        return None
    return rest[len(INNER_PREFIX):]


def main() -> int:
    os.makedirs(os.path.dirname(TARBALL), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(TARBALL) and os.path.getsize(TARBALL) > 1_000_000:
        print(f"[1/3] 复用已下载的 tarball（{os.path.getsize(TARBALL)/1e6:.1f} MB）", flush=True)
    else:
        print(f"[1/3] 下载 tarball -> {TARBALL}", flush=True)
        t0 = time.perf_counter()
        got = 0
        last_report = t0
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as c, \
                c.stream("GET", URL) as resp:
            resp.raise_for_status()
            with open(TARBALL, "wb") as f:
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if got > MAX_BYTES:
                        print(f"  中止：已超过 {MAX_BYTES/1e9:.1f}GB 上限", flush=True)
                        return 2
                    now = time.perf_counter()
                    if now - last_report > 20:
                        mb = got / 1e6
                        print(f"  … {mb:7.1f} MB  ({mb/(now-t0):.2f} MB/s)", flush=True)
                        last_report = now
        dt = time.perf_counter() - t0
        print(f"  完成：{got/1e6:.1f} MB，用时 {dt/60:.1f} 分钟（{got/1e6/dt:.2f} MB/s）", flush=True)

    print(f"[2/3] 流式解包，只抽 {INNER_PREFIX}**.md -> {OUT_DIR}", flush=True)
    n = 0
    total = 0
    samples: list[str] = []
    with tarfile.open(TARBALL, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            rel = _iter_docs(member.name)
            if rel is None:
                continue
            if len(samples) < 3:
                samples.append(member.name)
            dst = os.path.join(OUT_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            data = src.read()
            with open(dst, "wb") as f:
                f.write(data)
            n += 1
            total += len(data)
    print(f"  抽出 {n} 个 markdown，{total/1e6:.1f} MB", flush=True)
    for s in samples:
        print(f"    例：{s}", flush=True)

    if n == 0:
        # 抽不到东西说明匹配逻辑又错了：保留 tarball 以便重试，且以非零码退出（不要静默成功）
        print("  ✗ 没有抽到任何文档，保留 tarball 供排查/重试", flush=True)
        return 3

    print("[3/3] 清理 tarball", flush=True)
    try:
        os.remove(TARBALL)
    except OSError as e:
        print(f"  删除失败（非致命）：{e}", flush=True)
    free = shutil.disk_usage(ROOT).free
    print(f"完成。语料目录：{OUT_DIR}  剩余磁盘 {free/1e9:.0f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
