#!/usr/bin/env python
"""临床治理门禁：安全资产清单 ↔ 代码里的不变量 ↔ 测试覆盖，三者必须一致。

这是"改动安全资产要留痕"这条纪律的**机器可查形式**（清单位于
`data/governance/safety_assets.json`）。校验四件事：

1. **资产完整性**：清单里每条资产的 `path` 存在、`sha256` 与文件当前内容一致，
   并且 `version` / `effective_date` / `source` / `owner` / `rationale` 都写了。
   被哈希钉住的资产改了却不同步清单 ⇒ 失败——那就是"改了安全资产而没人知道"。
2. **不变量一一对应**：清单里的不变量 id 与 `kubedocs.clinical.safety.INVARIANTS`
   **完全相同**（多一条少一条都失败），而且注册表里声明的 `implemented_in` 路径真的存在。
3. **不变量有测试**：每个不变量 id 必须在 `tests/` 下至少出现一次。没有测试覆盖的
   不变量等于没人守——它会悄悄失效，而清单上还写着它是底线。
4. **验证出处可查**：每条不变量的 `verified_by` 非空，且其中每一项的文件部分真的存在。

## 为什么清单里要有 `governance_gaps`

治理最怕的是"清单看起来很完整"。已知缺口（没有具名临床审阅人、资产来源没有外部标准版本号、
校准样本量不足）必须写在清单里而不是留白：留白会被读成"没有问题"。

用法：
    python scripts/check_governance.py                  # 用脚本所在仓库
    python scripts/check_governance.py --root DIR       # 指定仓库根
    python scripts/check_governance.py --update-hashes  # 把清单里的哈希刷成当前值（改完资产用它）

`--update-hashes` 只动 `assets[].sha256`，**不动版本号与理由**：哈希是机器可以代填的部分，
"这次改了什么、为什么"必须由人写。

退出码：0 全部通过；1 有地方对不上。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MANIFEST_REL = Path("data") / "governance" / "safety_assets.json"

ASSET_KINDS = ("prompt", "rule_asset", "goldset")
"""允许的资产类别。`prompt` = 指令层；`rule_asset` = 线索/规则表；`goldset` = 评测尺子。"""

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_REL
    if not path.is_file():
        raise FileNotFoundError(f"治理清单缺失：{path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"治理清单 schema 不是 1（收到 {data.get('schema')!r}）：无法解释这份清单")
    return data


def _check_assets(root: Path, manifest: dict[str, Any]) -> list[str]:
    """资产：路径存在、哈希一致、治理字段齐全。"""
    fails: list[str] = []
    seen: set[str] = set()
    for asset in manifest.get("assets", []):
        aid = str(asset.get("id", ""))
        if not aid:
            fails.append("资产条目缺少 id")
            continue
        if aid in seen:
            fails.append(f"资产 id 重复：{aid}")
        seen.add(aid)

        rel = str(asset.get("path", ""))
        path = root / rel
        if not rel or not path.is_file():
            fails.append(f"资产 {aid} 的 path 不存在：{rel!r}")
            continue

        kind = str(asset.get("kind", ""))
        if kind not in ASSET_KINDS:
            fails.append(f"资产 {aid} 的 kind 不合法：{kind!r}（允许 {list(ASSET_KINDS)}）")

        for field in ("version", "effective_date", "source", "owner", "rationale"):
            if not str(asset.get(field, "")).strip():
                fails.append(f"资产 {aid} 缺少 {field}（治理记录不能留白）")
        if not _DATE_RE.match(str(asset.get("effective_date", ""))):
            fails.append(f"资产 {aid} 的 effective_date 不是 YYYY-MM-DD：{asset.get('effective_date')!r}")

        got = _sha256(path)
        want = str(asset.get("sha256", ""))
        if got != want:
            fails.append(
                f"资产 {aid} 内容已变但清单未同步（{rel}）：\n"
                f"      清单 {want}\n      实际 {got}\n"
                "      ⇒ 更新清单里的 sha256 / version / rationale 并写清改了什么"
            )
    if not manifest.get("assets"):
        fails.append("清单里一条资产都没有：治理清单不能是空壳")
    return fails


def _check_invariants(root: Path, manifest: dict[str, Any]) -> list[str]:
    """不变量：与代码注册表一一对应、实现路径存在、有测试覆盖与验证出处。"""
    from kubedocs.clinical.safety import INVARIANTS

    fails: list[str] = []
    code_ids = {i.id for i in INVARIANTS}
    manifest_rows = manifest.get("invariants", [])
    manifest_ids = [str(row.get("id", "")) for row in manifest_rows]

    for missing in sorted(code_ids - set(manifest_ids)):
        fails.append(f"代码里的不变量没有进治理清单：{missing}（新增不变量必须同时登记）")
    for stale in sorted(set(manifest_ids) - code_ids):
        fails.append(f"治理清单里的不变量在代码里不存在：{stale}（删掉不变量要同时销账）")
    if len(manifest_ids) != len(set(manifest_ids)):
        fails.append("治理清单里的不变量 id 有重复")

    tests_dir = root / "tests"
    test_blob = (
        "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in tests_dir.rglob("*.py"))
        if tests_dir.is_dir()
        else ""
    )

    for inv in INVARIANTS:
        for rel in inv.implemented_in:
            if not (root / rel).exists():
                fails.append(f"不变量 {inv.id} 声明的实现文件不存在：{rel}")

    for row in manifest_rows:
        inv_id = str(row.get("id", ""))
        if inv_id not in code_ids:
            continue
        for field in ("owner", "effective_date", "rationale"):
            if not str(row.get(field, "")).strip():
                fails.append(f"不变量 {inv_id} 缺少 {field}")
        if not _DATE_RE.match(str(row.get("effective_date", ""))):
            fails.append(f"不变量 {inv_id} 的 effective_date 不是 YYYY-MM-DD")

        verified = [str(v) for v in row.get("verified_by", [])]
        if not verified:
            fails.append(f"不变量 {inv_id} 没有 verified_by：没人验过的底线不算底线")
        for entry in verified:
            rel = entry.split("::", 1)[0]
            if not (root / rel).exists():
                fails.append(f"不变量 {inv_id} 的验证出处不存在：{rel}")

        if inv_id not in test_blob:
            fails.append(f"不变量 {inv_id} 在 tests/ 里找不到任何引用：没有测试覆盖的底线会悄悄失效")
    return fails


def _update_hashes(root: Path, manifest: dict[str, Any]) -> int:
    """把清单里的 sha256 刷成当前值（只动机器可代填的那一列）。"""
    changed = 0
    for asset in manifest.get("assets", []):
        path = root / str(asset.get("path", ""))
        if not path.is_file():
            continue
        got = _sha256(path)
        if asset.get("sha256") != got:
            asset["sha256"] = got
            changed += 1
            print(f"  更新哈希：{asset.get('id')} -> {got}")
    (root / MANIFEST_REL).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写回 {MANIFEST_REL}（更新 {changed} 条哈希）")
    print("注意：哈希刷好了，但 version / rationale 仍要人来写——机器只填得了哈希。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="临床治理门禁：资产清单 / 不变量 / 测试覆盖三方一致")
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--update-hashes", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        manifest = _load_manifest(root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"  ✗ {e}")
        return 1

    if args.update_hashes:
        return _update_hashes(root, manifest)

    asset_fails = _check_assets(root, manifest)
    inv_fails = _check_invariants(root, manifest)
    fails = asset_fails + inv_fails

    print(f"治理清单    : {MANIFEST_REL}")
    print(f"安全资产    : {len(manifest.get('assets', []))} 条")
    print(f"不变量      : {len(manifest.get('invariants', []))} 条")
    print(f"已知治理缺口: {len(manifest.get('governance_gaps', []))} 条（写在清单里，不藏着）")
    for gap in manifest.get("governance_gaps", []):
        print(f"  · {gap}")
    for f in fails:
        print(("  ✗ " if not f.startswith("      ") else "") + f)
    if fails:
        print(f"\n结果：不通过（{len(fails)} 处）")
        return 1
    print("\n结果：通过（资产哈希、不变量对应、实现路径、测试覆盖、验证出处全部一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
