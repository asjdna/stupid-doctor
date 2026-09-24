"""标注集预检：在跑评测之前，用原始语料核对标注是否站得住。

## 为什么需要它（而不是只靠 `cli eval` 的自检）

`cli eval` 的自检用的是**已建好的索引**，必须先等摄取跑完才能发现问题。
这个脚本直接读原始 markdown，秒级给出结果，适合在**写标注的过程中**反复跑：
路径写错了、关键词记错了，立刻就能看到。

两者互补：
    写标注时  -> scripts/check_gold.py      （快，查路径与关键词）
    正式评测  -> python -m kubedocs.cli eval（慢，查索引一致性并产出分数）

用法：
    python scripts/check_gold.py
    python scripts/check_gold.py --gold data/eval/retrieval_gold.jsonl --corpus data/raw/k8s-docs

退出码：0 全部通过；1 有标注站不住；3 找不到文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def check(gold_path: Path, corpus: Path) -> int:
    if not gold_path.exists():
        print(f"找不到标注集：{gold_path}", file=sys.stderr)
        return 3
    if not corpus.exists():
        print(f"找不到语料目录：{corpus}", file=sys.stderr)
        return 3

    rows: list[dict] = []
    with open(gold_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ✗ 第 {lineno} 行不是合法 JSON：{e}")
                return 1

    bad_path: list[tuple[str, str]] = []
    bad_term: list[tuple[str, str, str]] = []
    kinds: Counter[str] = Counter()
    text_cache: dict[str, str] = {}

    for row in rows:
        kinds[row.get("kind", "concept")] += 1
        for doc, term in (row.get("must_contain") or {}).items():
            f = corpus / f"{doc}.md"
            if not f.exists():
                bad_path.append((row["id"], doc))
                continue
            if doc not in text_cache:
                text_cache[doc] = f.read_text(encoding="utf-8", errors="replace").lower()
            if term.lower() not in text_cache[doc]:
                bad_term.append((row["id"], doc, term))

    ids = [r["id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    targets = Counter(d for r in rows for d in (r.get("must_contain") or {}))

    print(f"标注集   : {gold_path}")
    print(f"语料目录 : {corpus}")
    print(f"标注条数 : {len(rows)}")
    print(f"类型分布 : {dict(kinds)}")
    print(f"目标文档 : {len(targets)} 篇不同文档"
          + (f"，被标注最多的：{targets.most_common(3)}" if targets else ""))
    if dupes:
        print(f"重复 id  : ✗ {dupes}")
    print(f"路径不存在 : {len(bad_path)}")
    for rid, doc in bad_path:
        print(f"  ✗ {rid}: {doc}")
    print(f"关键词找不到 : {len(bad_term)}")
    for rid, doc, term in bad_term:
        print(f"  ✗ {rid}: {term!r} 不在 {doc}")

    ok = not (bad_path or bad_term or dupes) and bool(rows)
    print(f"\n结果：{'通过' if ok else '不通过'}"
          + ("" if rows else "（没有任何标注行）"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="标注集预检（对原始语料）")
    p.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    p.add_argument("--corpus", default="data/raw/k8s-docs")
    args = p.parse_args(argv)
    return check(Path(REPO_ROOT / args.gold), Path(REPO_ROOT / args.corpus))


if __name__ == "__main__":
    sys.exit(main())
