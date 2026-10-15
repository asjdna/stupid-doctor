"""在真实 Kubernetes 文档上体检切块质量（离线，不依赖 Postgres）。

用法：`python scripts/chunk_stats.py [--limit N] [--samples N]`（`--limit 0` 为全量）。
前置条件：`Settings.corpus_dir` 下有 markdown 语料（先跑 scripts/fetch_corpus.py）。
输出：文件/块数、块长均值与 p95、代码块占比、链接与非法 URL 数、重复块数、耗时、目录分布、样例块；末行 `RESULT ...` 是机器可读摘要。
退出码：0 正常；2 语料目录不存在；3 没切出任何块。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubedocs.chunker import chunk_document  # noqa: E402
from kubedocs.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（0=全量）")
    ap.add_argument("--samples", type=int, default=2, help="打印几个样例块")
    args = ap.parse_args()

    s = get_settings()
    root = Path(s.corpus_dir)
    if not root.is_dir():
        print(f"语料目录不存在：{root}（先跑 scripts/fetch_corpus.py）")
        return 2

    files = sorted(root.rglob("*.md"))
    if args.limit:
        files = files[: args.limit]
    print(f"语料：{root}")
    print(f"文件数：{len(files)}\n")

    t0 = time.perf_counter()
    chunks = []
    empty_files = 0
    sizes: list[int] = []
    by_dir: Counter = Counter()
    with_code = 0
    urls: set[str] = set()
    bad_urls = 0
    text_counter: Counter = Counter()

    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"  跳过 {rel}：{type(e).__name__}")
            continue
        cs = chunk_document(
            rel,
            text,
            max_chars=s.chunk_max_chars,
            overlap=s.chunk_overlap_chars,
            min_chars=s.chunk_min_chars,
            url_prefix=s.docs_url_prefix,
        )
        if not cs:
            empty_files += 1
            continue
        for c in cs:
            chunks.append(c)
            sizes.append(len(c.text))
            by_dir[rel.split("/", 1)[0] if "/" in rel else "(root)"] += 1
            if "```" in c.text:
                with_code += 1
            urls.add(c.source_url)
            if "//docs" in c.source_url or c.source_url.endswith("//"):
                bad_urls += 1
            text_counter[c.content_hash] += 1
    dt = time.perf_counter() - t0

    if not chunks:
        print("没有切出任何块——检查语料或切块参数")
        return 3

    sizes_sorted = sorted(sizes)
    dup = sum(v - 1 for v in text_counter.values() if v > 1)

    print("=" * 74)
    print("切块体检（真实语料）")
    print("=" * 74)
    print(f"  文件          {len(files)} 个（空文档 {empty_files} 个被跳过）")
    print(f"  块数          {len(chunks)}")
    print(f"  平均块长      {statistics.mean(sizes):.0f} 字符")
    print(f"  中位 / p95    {statistics.median(sizes):.0f} / {sizes_sorted[int(len(sizes) * 0.95) - 1]} 字符")
    print(f"  含代码块的块  {with_code} （{with_code / len(chunks):.1%}）")
    print(f"  去重后链接    {len(urls)} 个（非法 URL {bad_urls} 个，应为 0）")
    print(f"  内容重复块    {dup} 个（{dup / len(chunks):.2%}）")
    print(f"  耗时          {dt:.1f}s（{len(files) / dt:.0f} 文件/s）")
    print("\n  按二级目录分布（前 10）：")
    for k, v in by_dir.most_common(10):
        print(f"    {k:22s} {v:6d} 块")

    if args.samples:
        print("\n  样例块：")
        step = max(1, len(chunks) // (args.samples + 1))
        for i in range(step, len(chunks), step):
            c = chunks[i]
            head = c.text[:150].replace("\n", " / ")
            print(f"\n    [{c.chunk_id}] {c.title!r}")
            print(f"      面包屑: {c.breadcrumb_text}")
            print(f"      链接  : {c.source_url}")
            print(f"      正文  : {head}…")
            if len([1 for _ in range(1)]) and i > step * args.samples:
                break

    print(
        f"\n  RESULT files={len(files)} chunks={len(chunks)} "
        f"avg={statistics.mean(sizes):.0f} bad_urls={bad_urls} dup={dup}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
