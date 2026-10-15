r"""用原型那套 500 例分层金标评本项目分诊：`data/eval/prototype/patients.json`。

金标由原型（已冻结）用 LLM 生成，每条带 `severity` / `style` / `description` /
`gold_department` / `gold_diseases`；严重度 mild 100 / risky 300 / severe 100，
语言风格 formal / natural / vague / erroneous 各 125。17 个金标科室全部落在本项目
科室词表内，可直接使用（`--check-vocab` 只验覆盖、不跑分诊）。

口径（**必须分层报，不许合并成一个总数**）：

| 口径 | 含义 |
|---|---|
| `dept top-1 / top-3` | 期望科室是否出现在前 1 / 前 3 |
| `emergency_as_dept` | 对 severe 档，top-3 里有没有「急诊科」——重症患者该拿到的动作 |
| `strict top-1` | 只在同一 severity 档内比；各档期望科室不同，合并会把两类失败平均掉 |

用法::

    python scripts/eval_prototype_patients.py                # 离线、免费、确定
    python scripts/eval_prototype_patients.py --check-vocab   # 只验词表覆盖
    python scripts/eval_prototype_patients.py --json out.json

前置条件：金标与 `data/clinic/cards.json` 存在、clinic 索引已建好。退出码恒为 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GOLD = REPO / "data" / "eval" / "prototype" / "patients.json"
CARDS = REPO / "data" / "clinic" / "cards.json"

EMERGENCY_DEPT = "急诊科"
"""重症档的目标科室：原型 severe 档 90/100 的金标就是它。"""


def _load_gold() -> list[dict[str, Any]]:
    if not GOLD.is_file():
        raise SystemExit(f"金标缺失：{GOLD}（应从原型 data/patients.json 复制而来）")
    rows: list[dict[str, Any]] = json.loads(GOLD.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"金标为空：{GOLD}")
    return rows


def _project_departments() -> set[str]:
    if not CARDS.is_file():
        return set()
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    return {d for c in cards for d in (c.get("departments") or [])}


def _build_backend() -> Any:
    from kubedocs.cli import apply_domain_defaults, open_store
    from kubedocs.clinical.rag import LocalIndexBackend
    from kubedocs.config import Settings
    from kubedocs.domains import load_all
    from kubedocs.domains.base import DomainRegistry
    from kubedocs.embed import build_embedder
    from kubedocs.retrieve.hybrid import HybridRetriever

    load_all()
    settings = apply_domain_defaults(Settings(domain="clinic"))
    store = open_store(settings)
    embedder = build_embedder(settings)
    try:
        retriever = DomainRegistry.get(settings.domain).retriever(store, embedder, settings)
    except KeyError:
        retriever = HybridRetriever(store, embedder, settings)
    return LocalIndexBackend(retriever)


def _triage_engine(department_map: dict[str, list[str]], precision_k: int) -> Any:
    from kubedocs.clinical.triage import RetrievalTriage

    return RetrievalTriage(known_departments=None, precision_k=precision_k, department_map=department_map)


def _run_one(engine: Any, backend: Any, text: str, *, coarse_k: int) -> tuple[list[str], list[str]]:
    """跑一条，返回 (粗排命中的标题, 分诊出的科室)。

    `engine.rank(query, hits)` 收的是整个检索结果对象而非 chunks 列表（同 `eval_triage.py`）。
    """
    hits = backend.search(text, top_k=coarse_k)
    ranked = engine.rank(text, hits)
    titles = [str(getattr(c, "title", "") or "") for c in (getattr(hits, "results", None) or [])]
    return titles, [d.name for d in ranked]


def _rate(n: int, d: int) -> float:
    return d / n if n else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse-k", type=int, default=50, help="粗排候选池大小")
    ap.add_argument("--precision-k", type=int, default=0, help="精排后参与投票的条数（0=不截断）")
    ap.add_argument("--department-map", choices=("off", "high"), default="high")
    ap.add_argument("--check-vocab", action="store_true", help="只验金标科室是否落在项目词表内")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = _load_gold()
    proj = _project_departments()
    gold_depts = sorted({str(r.get("gold_department") or "") for r in rows} - {""})
    inside = [d for d in gold_depts if d in proj]
    outside = [d for d in gold_depts if d not in proj]
    print(f"金标 {len(rows)} 例；金标科室 {len(gold_depts)} 种")
    print(f"落在项目 42 科室词表内：{len(inside)}/{len(gold_depts)}；未覆盖：{outside or '无'}")
    covered = sum(1 for r in rows if str(r.get("gold_department") or "") in proj)
    print(f"按病例计可用：{covered}/{len(rows)} = {covered / len(rows):.1%}")
    print(f"严重度分布：{_counts(rows, 'severity')}")
    print(f"语言风格分布：{_counts(rows, 'style')}")
    if args.check_vocab:
        return 0

    from kubedocs.clinical.triage import load_department_map

    dep_map = load_department_map(min_confidence="high") if args.department_map == "high" else {}
    print(f"科室补全映射表：{args.department_map} 档，{len(dep_map):,} 条")
    backend = _build_backend()
    engine = _triage_engine(dep_map, args.precision_k)

    results: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("description") or "")
        gold = str(row.get("gold_department") or "")
        titles, preds = _run_one(engine, backend, text, coarse_k=args.coarse_k)
        results.append(
            {
                "id": row.get("id"),
                "severity": row.get("severity"),
                "style": row.get("style"),
                "gold_department": gold,
                "pred": preds,
                "coarse_hits": len(titles),
                "top1": bool(preds[:1] and preds[0] == gold),
                "top3": gold in preds[:3],
                "emergency_as_dept": EMERGENCY_DEPT in preds[:3],
            }
        )

    print()
    print("=" * 78)
    print("原型 500 例分层金标 · kubedocs 分诊（离线、免费、确定）")
    print("=" * 78)
    print(f"{'分层':<22}{'n':>5}{'top-1':>9}{'top-1 率':>11}{'top-3':>8}{'top-3 率':>11}{'含急诊科':>10}")
    print("-" * 78)
    _row("全部", results)
    for sev in ("mild", "risky", "severe"):
        _row(f"severity={sev}", [r for r in results if r["severity"] == sev])
    for st in ("formal", "natural", "vague", "erroneous"):
        _row(f"style={st}", [r for r in results if r["style"] == st])

    print()
    print("分层口径说明：`severity` 与 `style` 的期望科室不同（severe 档 90/100 金标是急诊科），")
    print("所以各行**不能合并成一个总数**——原型自己也吃了这个亏（severe 档 dept_hit 12% 而")
    print("emergency 98%，只看总数会读成'分诊不行'，其实问题是'该转急诊时没给急诊科'）。")
    print()
    print("**这不是上线准确率**：金标由 LLM 生成、未经医学人员复核（原型自己的 README 已注明），")
    print("它衡量的是'对这套分层用例的符合度'。")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "gold": str(GOLD.relative_to(REPO)),
                    "n": len(results),
                    "coarse_k": args.coarse_k,
                    "precision_k": args.precision_k,
                    "department_map": args.department_map,
                    "cases": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已写入 {args.json}")
    return 0


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key) or "")
        out[k] = out.get(k, 0) + 1
    return out


def _row(label: str, sub: list[dict[str, Any]]) -> None:
    n = len(sub)
    t1 = sum(1 for r in sub if r["top1"])
    t3 = sum(1 for r in sub if r["top3"])
    em = sum(1 for r in sub if r["emergency_as_dept"])
    print(f"{label:<22}{n:>5}{t1:>9}{_rate(n, t1):>10.1%}{t3:>8}{_rate(n, t3):>10.1%}{_rate(n, em):>10.1%}")


if __name__ == "__main__":
    raise SystemExit(main())
