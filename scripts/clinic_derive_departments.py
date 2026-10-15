"""从 OpenCMKG 原始三元组推导卡片缺失的科室标签，写 `data/clinic/department_map.json`。

`data/clinic/cards.json` 里大量卡片的「建议就诊科室」是「未标注」（`diseases` 也「暂无」），
因为卡片生成阶段丢掉了关系；本脚本回到原始 `triples.txt`（`head,relation,tail` CSV）
重新推导，输出可审计的 `key -> departments` 映射表，供分诊检索使用。

用法::

    python scripts/clinic_derive_departments.py --report-only   # 只打印覆盖率与样例
    python scripts/clinic_derive_departments.py                 # 写出映射表

前置条件：`--triples` 默认 `data/raw/OpenCMKG/triples.txt`（`data/raw/` 不入库，新克隆里
没有，需用 `--triples` 指定自己的副本）；`--cards` 指向已构建的卡片语料。

输出：`--out`（默认 `data/clinic/department_map.json`）以 `meta` 记录各补全来源计数与置信度
说明，`map` 是 `key -> {departments, source, confidence}`。`source` 给出推导路径便于抽检：
`direct`/`via_disease`/`alias:<别名>`（`confidence: high`）或 `fuzzy:<key>:<相似度>`
（图谱补不齐时才用，Jaccard ≥ 0.50，`confidence: medium`，需人工抽检）。

退出码：0 成功 / 1 找不到 `--triples`。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_TRIPLES = REPO / "data" / "raw" / "OpenCMKG" / "triples.txt"
DEFAULT_CARDS = REPO / "data" / "clinic" / "cards.json"
DEFAULT_OUT = REPO / "data" / "clinic" / "department_map.json"

_DEPT_REL_SUFFIX = "_belong_department"


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            # 尾部可能自带逗号（如治疗描述），故只把最后两段当成 relation/tail
            rel, tail = parts[-2], parts[-1]
            head = ",".join(parts[:-2])
            head, rel, tail = head.strip(), rel.strip(), tail.strip()
            if head and rel and tail:
                rows.append((head, rel, tail))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.triples.exists():
        print(f"[x] 找不到原始三元组：{args.triples}")
        return 1

    triples = load_triples(args.triples)
    print(f"三元组总数：{len(triples):,}")

    rel_counts = Counter(r for _, r, _ in triples)
    print("\n关系种类（前 20）：")
    for rel, n in rel_counts.most_common(20):
        print(f"  {n:>9,}  {rel}")

    dept_rel = [r for r in rel_counts if r.endswith(_DEPT_REL_SUFFIX)]
    print(f"\n科室类关系：{dept_rel}")

    ent_depts: dict[str, list[str]] = defaultdict(list)
    for head, rel, tail in triples:
        if rel.endswith(_DEPT_REL_SUFFIX) and tail not in ent_depts[head]:
            ent_depts[head].append(tail)

    # 实体 -> 疾病：两个方向的关系都归一到这个方向
    ent_diseases: dict[str, list[str]] = defaultdict(list)
    for head, rel, tail in triples:
        if rel.endswith("_belong_disease") or rel in ("disease_has_symptom",):
            key = head if rel.endswith("_belong_disease") else tail
            val = tail if rel.endswith("_belong_disease") else head
            if val not in ent_diseases[key]:
                ent_diseases[key].append(val)

    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    from kubedocs.clinical.triage import departments_in_card

    unlabeled = sorted({c["key"] for c in cards if not departments_in_card(c.get("text") or "")})
    total = len(unlabeled)
    print(f"\n卡片数：{len(cards):,} | 未标注的唯一 key：{total:,}")

    result: dict[str, dict] = {}
    stats: Counter[str] = Counter()
    samples: list[str] = []
    for key in unlabeled:
        if key in ent_depts:
            result[key] = {"departments": ent_depts[key], "source": "direct"}
            stats["direct"] += 1
        elif key in ent_diseases:
            ds = [d for d in ent_diseases[key] if d in ent_depts]
            if ds:
                merged: list[str] = []
                for d in ds:
                    for dep in ent_depts[d]:
                        if dep not in merged:
                            merged.append(dep)
                result[key] = {"departments": merged, "source": "via_disease"}
                stats["via_disease"] += 1
        else:
            stats["unresolved"] += 1
        if key in result and len(samples) < 15:
            samples.append(f"{key} -> {'/'.join(result[key]['departments'])} ({result[key]['source']})")

    # 未标注 key 有多少根本不在图谱里（决定补全的天花板）
    heads_any = {h for h, _, _ in triples}
    in_kg = sum(1 for k in unlabeled if k in heads_any)
    print(f"\n未标注 key 中出现在三元组头部的：{in_kg:,} / {total:,} ({in_kg / max(total, 1):.1%})")
    print(f"  => 完全没有图谱记录的：{total - in_kg:,} ({1 - in_kg / max(total, 1):.1%})")

    samas: dict[str, list[str]] = defaultdict(list)
    for head, rel, tail in triples:
        if rel == "SameAs":
            samas[head].append(tail)
            samas[tail].append(head)
    alias_fixed = 0
    for key in unlabeled:
        if key in result:
            continue
        for alias in samas.get(key, []):
            deps = ent_depts.get(alias)
            if not deps:
                continue
            ds = [d for d in ent_diseases.get(alias, []) if d in ent_depts]
            for d in ds:
                deps = deps + [x for x in ent_depts[d] if x not in deps]
            if deps:
                result[key] = {"departments": deps, "source": f"alias:{alias}"}
                alias_fixed += 1
                break
    print(f"SameAs 别名可补全：{alias_fixed:,} ({alias_fixed / max(total, 1):.1%})")

    # 模糊归并：图谱补不齐的回溯到最相似的已标注 key，记录相似度供人工抽检
    from kubedocs.clinical.triage import RetrievalTriage

    cand = [k for k in unlabeled if k not in result]
    lab_depts: dict[str, list[str]] = {}
    for c in cards:
        deps = departments_in_card(c.get("text") or "")
        if deps:
            lab_depts.setdefault(c["key"], deps)
    lab_tok = {k: set(RetrievalTriage._tokens(k)) for k in lab_depts}
    tiers: Counter[str] = Counter()
    for key in cand:
        kt = set(RetrievalTriage._tokens(key))
        best, score = None, 0.0
        if kt:
            for lk, lt in lab_tok.items():
                if not lt:
                    continue
                j = len(kt & lt) / len(kt | lt)
                if j > score:
                    best, score = lk, j
        if score >= 0.5 and best is not None:
            tiers["fuzzy>=0.50"] += 1
            result[key] = {
                "departments": list(lab_depts.get(best, [])),
                "source": f"fuzzy:{best}:{score:.2f}",
            }
        elif score >= 0.34:
            tiers["fuzzy>=0.34"] += 1
        else:
            tiers["fuzzy<0.34"] += 1
    print("\n模糊归并（最相似已标注 key 的 Jaccard）：")
    for k in ("fuzzy>=0.50", "fuzzy>=0.34", "fuzzy<0.34"):
        print(f"  {k:<14} {tiers[k]:>7,}  ({tiers[k] / max(total, 1):.1%})")

    total = len(unlabeled)
    print("\n=== 补全结果（图谱 + 别名 + 模糊）===")
    by_source = Counter(v["source"].split(":")[0] for v in result.values())
    for src, n in by_source.most_common():
        print(f"  {src:<12} {n:>7,}  ({n / max(total, 1):.1%})")
    print(f"  {'未解决':<12} {total - len(result):>7,}  ({1 - len(result) / max(total, 1):.1%})")
    print(f"  {'合计可补':<12} {len(result):>7,}  ({len(result) / max(total, 1):.1%})")
    print("\n样例：")
    for s in samples:
        print("   ", s)

    if not args.report_only:
        high = {"direct", "via_disease", "alias"}
        for rec in result.values():
            src = rec["source"].split(":")[0]
            rec["confidence"] = "high" if src in high else "medium"
        payload = {
            "meta": {
                "generated_by": "scripts/clinic_derive_departments.py",
                "triples": str(args.triples),
                "cards": str(args.cards),
                "unlabeled_keys": total,
                "filled": len(result),
                "by_source": dict(by_source),
                "confidence_note": (
                    "high = 图谱直接关系或 SameAs 别名；medium = 最近邻 key 的 Jaccard>=0.5，需人工抽检"
                ),
            },
            "map": result,
        }
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(f"\n已写出 {args.out.relative_to(REPO)} （{len(result):,} 条，含来源与置信度）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
