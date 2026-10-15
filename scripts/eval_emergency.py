r"""急诊判定评测：用 `data/eval/prototype/patients.json` 的 `severity` 当急重症金标。

`red_flags` / `emergency` 走模型路径，所以这里必须真调模型；按 severity 分档：

| severity | 期望 | 样本 |
|---|---|---|
| `severe` | 应当判出急重症（漏判 = 危险） | 100 例 |
| `mild` | 不应当判（误报 = 让人对提示麻木） | 100 例 |
| `risky` | 不参与判定，本就该由模型酌情 | 300 例 |

两个口径：**检出率**（severe 里 `emergency=true` 的比例，漏一个就是一次危险的失败）
与**误报率**（mild 里 `emergency=true` 的比例）。返回 `action=ask`（先追问、本轮没判）
算已判——`ask` 轮同样带完整结论，只是同时列了追问；只有 `action=error` 才计入解析失败。
对 severe 案例，`ask` 不等于漏判，但也不等于安全：患者拿到的是追问而非急救指引。

前置条件：`--base` 上的服务在跑（默认 `http://127.0.0.1:8082`）、模型密钥可用。
每个用例都会真调模型（含工具轮），默认每类只跑 12 例（24 次问诊）；`--per-class 100` 明显花钱。

用法::

    python scripts/eval_emergency.py                    # 默认每类 12 例
    python scripts/eval_emergency.py --per-class 30 --json out.json
    python scripts/eval_emergency.py --dry-run          # 只列样本，不调模型

退出码：0 正常结束 / 2 某例请求失败（服务不可用时立即中止，不当成"未检出"）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GOLD = REPO / "data" / "eval" / "prototype" / "patients.json"
SEED = 42
CASES = (("severe", True), ("mild", False))
"""`(severity, 期望是否急重症)`。risky 档刻意不参与——它本就该由模型酌情。"""


def _load(per_class: int) -> list[dict[str, Any]]:
    if not GOLD.is_file():
        raise SystemExit(f"金标缺失：{GOLD}")
    rows: list[dict[str, Any]] = json.loads(GOLD.read_text(encoding="utf-8"))
    rng = random.Random(SEED)  # noqa: S311
    out: list[dict[str, Any]] = []
    for severity, _expect in CASES:
        pool = [r for r in rows if r.get("severity") == severity]
        out.extend(rng.sample(pool, min(per_class, len(pool))))
    return out


def _ask(base: str, text: str, sid: str) -> dict[str, Any]:
    import urllib.request

    body = json.dumps({"question": text, "session_id": sid}).encode("utf-8")
    # S310：URL 由 `--base` 给出、默认是本地服务，本脚本只用于本地评测。
    req = urllib.request.Request(  # noqa: S310
        base + "/consult", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
        payload: dict[str, Any] = json.loads(r.read().decode("utf-8"))
        return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8082", help="/consult 所在的地址")
    ap.add_argument("--per-class", type=int, default=12, help="每类抽多少例（会真花钱）")
    ap.add_argument("--dry-run", action="store_true", help="只抽样本，不调模型")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    cases = _load(args.per_class)
    print(f"抽样 {len(cases)} 例（severe/mild 各 {args.per_class}，种子 {SEED}）")
    for c in cases[:2]:
        print(f"  [{c['severity']}] {c['description'][:60]}")
    if args.dry_run:
        print("\n--dry-run：未调用模型。")
        return 0

    results: list[dict[str, Any]] = []
    for i, c in enumerate(cases, start=1):
        t0 = time.perf_counter()
        try:
            p = _ask(args.base, str(c["description"]), f"emg-{c['severity']}-{i}")
        except Exception as e:  # noqa: BLE001 - 服务不可用要立刻说清，不要静默当成"未检出"
            print(f"  [x] 第 {i} 例请求失败：{type(e).__name__}: {e}")
            return 2
        results.append(
            {
                "id": c.get("id"),
                "severity": c["severity"],
                "expect_emergency": c["severity"] == "severe",
                "description": c["description"],
                "action": p.get("action"),
                "emergency": bool(p.get("emergency")),
                "degraded": p.get("degraded") or "",
                "keyword_hit": p.get("keyword_hit") or "",
                "n_red_flags": len(p.get("red_flags") or []),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        r = results[-1]
        flag = "✓" if r["emergency"] is r["expect_emergency"] else "✗"
        print(
            f"  {flag} [{r['severity']:<6}] action={r['action']:<9} "
            f"emergency={r['emergency']!s:<5} red_flags={r['n_red_flags']} "
            f"关键词={r['keyword_hit']!r} {r['seconds']}s"
        )

    print()
    print("=" * 74)
    print("急诊判定（金标：原型 severity；**只有 severe 该报、mild 不该报**）")
    print("=" * 74)
    for severity, expect in CASES:
        sub = [r for r in results if r["severity"] == severity]
        n = len(sub)
        # 判据是 `action != "error"` 而非 `action != "ask"`：`ask` 轮同样带完整结论
        # （`emergency`/`red_flags` 都在），只是同时列了追问；当成"未判"会凭空压低检出率。
        broken = [r for r in sub if r["action"] == "error"]
        judged = [r for r in sub if r["action"] != "error"]
        hit = sum(1 for r in judged if r["emergency"] is expect)
        label = "severe ⇒ 应报" if expect else "mild ⇒ 不应报"
        rate = hit / len(judged) if judged else 0.0
        print(
            f"  {label:<18} n={n:<4} 可判 {len(judged):<4} 正确 {hit:<4} ({rate:.0%})  解析失败 {len(broken)}"
        )
    sev = [r for r in results if r["severity"] == "severe"]
    kw = sum(1 for r in sev if r["keyword_hit"])
    print()
    print(f"  severe 里被 15 词表命中的：{kw}/{len(sev)}")
    print("  ⇒ 表没命中的那一部分**只能靠模型**；这正是这次转向的意义，也是它的风险。")
    broken_all = [r for r in results if r["action"] == "error"]
    if broken_all:
        print()
        print(f"  ⚠️ 解析失败 {len(broken_all)}/{len(results)} 例（模型输出不合契约 ⇒ 走兜底）。")
        print("     对 severe 案例这意味着**急症患者拿到兜底文案**，是要盯的稳定性指标。")
    print()
    print("**这不是上线准确率**：金标文本是 LLM 生成的、未经医学复核；样本默认很小（见 --per-class）。")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"per_class": args.per_class, "seed": SEED, "cases": results}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
