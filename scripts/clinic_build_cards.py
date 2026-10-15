"""构建 clinic 域知识卡片语料 `data/clinic/cards.json`（向量索引的上游）。

用法：
    python scripts/clinic_build_cards.py --source <上游产物目录> --dry-run
    python scripts/clinic_build_cards.py --source <上游产物目录> --out data/clinic

前置条件：`--source` 指向跑过上游 `scripts/build_knowledge.py`（OpenCMKG → 中间产物）
得到的目录，默认取环境变量 `KUBEDOCS_CLINIC_SOURCE`，脚本不写死本机路径。目录须含
`symptoms.json`、`diseases.json`、`disease_to_department.json`、`symptom_to_diseases.json`、
`disease_to_drugs.json`、`drugs.json`，缺任一即报错退出。clinic 域的规则轨
（`data/` 下的词典）不依赖该语料，没有源目录也能跑通问诊/挂号/紧急。

输出：`<--out>/cards.json`，JSON 数组，元素为 `{kind, key, text, ...}`。`kind` 取
`symptom`/`disease`/`drug`；症状卡另有 `diseases` 与 `departments`，疾病卡另有
`departments` 与 `drugs`，药物卡只有 `text`。该语料是派生数据且上游标注「仅用于学术
研究，不得用于商用」，不入库（`.gitignore` 已忽略 `data/clinic/`），不要提交到公开仓库。

结果解读：打印各 `kind` 的卡片数与带科室的卡片数——后者是分诊落地的依据，其余只能给疾病
方向；"跳过条目"是非字符串条目数（上游混有列表/字典，过滤但不静默丢弃）。

退出码：0 成功 / 3 未指定源目录或源目录不可用（含缺少必需文件）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kubedocs.domains.clinic.data.departments import normalize_department  # noqa: E402

# 不写死本机路径：默认取环境变量，缺失时由调用方显式给 --source。
DEFAULT_SOURCE = os.environ.get("KUBEDOCS_CLINIC_SOURCE", "")
OUT_DIR = REPO_ROOT / "data" / "clinic"

# 源目录必须齐备的中间文件；缺任一说明上游还没构建完
REQUIRED = (
    "symptoms.json",
    "diseases.json",
    "disease_to_department.json",
    "symptom_to_diseases.json",
    "disease_to_drugs.json",
    "drugs.json",
)


def _load(src: Path, name: str) -> object:
    with open(src / name, encoding="utf-8") as fh:
        return json.load(fh)


def _names(obj: object) -> tuple[list[str], int]:
    """取出名字列表，返回 (名字, 被跳过的条目数)。

    上游是抓取聚合来的数据，条目不保证是字符串（混有列表/字典）；这里过滤掉非字符串，
    但把跳过数量报出去——静默丢数据只表现为"检索覆盖变小"，不会有任何报错。
    """
    if isinstance(obj, list):
        raw: list[object] = list(obj)
    elif isinstance(obj, dict):
        raw = list(obj)
    else:
        raw = []
    names = [x for x in raw if isinstance(x, str) and x.strip()]
    return names, len(raw) - len(names)


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str) and x.strip()]


def build_symptom_cards(src: Path) -> tuple[list[dict], int]:
    sym_names, skipped = _names(_load(src, "symptoms.json"))
    disease_of = _load(src, "symptom_to_diseases.json")
    dept_of = _load(src, "disease_to_department.json")
    cards: list[dict] = []
    for name in sym_names:
        diseases = _str_list(disease_of.get(name, []) if isinstance(disease_of, dict) else [])[:8]
        depts: list[str] = []
        for d in diseases:
            for dep in _str_list(dept_of.get(d, []) if isinstance(dept_of, dict) else []):
                norm = normalize_department(dep)
                if norm and norm not in depts:
                    depts.append(norm)
        cards.append(
            {
                "kind": "symptom",
                "key": name,
                "text": f"{name}。可能相关的疾病：{'、'.join(diseases) or '暂无'}。"
                f"建议就诊科室：{'、'.join(depts) or '未标注'}。",
                "diseases": diseases,
                "departments": depts,
            }
        )
    return cards, skipped


def build_disease_cards(src: Path) -> tuple[list[dict], int]:
    dis_names, skipped = _names(_load(src, "diseases.json"))
    dept_of = _load(src, "disease_to_department.json")
    drug_of = _load(src, "disease_to_drugs.json")
    cards: list[dict] = []
    for name in dis_names:
        depts = [
            d
            for d in (
                normalize_department(x)
                for x in _str_list(dept_of.get(name, []) if isinstance(dept_of, dict) else [])
            )
            if d
        ]
        drugs = _str_list(drug_of.get(name, []) if isinstance(drug_of, dict) else [])[:10]
        cards.append(
            {
                "kind": "disease",
                "key": name,
                "text": f"{name}。建议就诊科室：{'、'.join(depts) or '未标注'}。"
                f"常用药物：{'、'.join(drugs) or '暂无'}。",
                "departments": depts,
                "drugs": drugs,
            }
        )
    return cards, skipped


def build_drug_cards(src: Path) -> tuple[list[dict], int]:
    names, skipped = _names(_load(src, "drugs.json"))
    return [{"kind": "drug", "key": n, "text": f"{n}。"} for n in names], skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="重建 clinic 域知识卡片（上游派生语料）")
    p.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="含 build_knowledge.py 产物的目录（必填，或用环境变量 "
        "KUBEDOCS_CLINIC_SOURCE 指定；本脚本不猜机器上的路径）",
    )
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not str(args.source).strip():
        print(
            "未指定源目录：本脚本不设默认值（它指向另一个项目的产物，写死路径只对"
            "一台机器成立）。请用 --source <上游产物目录>，或设环境变量 "
            "KUBEDOCS_CLINIC_SOURCE。\n"
            "（clinic 域的规则轨不依赖它，问诊/挂号/紧急仍可运行）",
            file=sys.stderr,
        )
        return 3
    src = Path(args.source)
    if not src.exists():
        print(
            f"源目录不存在：{src}\n"
            "请先在上游项目里跑 scripts/build_knowledge.py 生成中间产物，"
            "或用 --source 指向同结构的目录。\n"
            "（clinic 域的规则轨不依赖它，问诊/挂号/紧急仍可运行）",
            file=sys.stderr,
        )
        return 3
    missing = [n for n in REQUIRED if not (src / n).exists()]
    if missing:
        print(f"源目录缺少 {missing}；它看起来不是 build_knowledge.py 的产物目录", file=sys.stderr)
        return 3

    sym, skip_s = build_symptom_cards(src)
    dis, skip_d = build_disease_cards(src)
    drug, skip_g = build_drug_cards(src)
    cards = sym + dis + drug
    kinds: dict[str, int] = {}
    for c in cards:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    out = Path(args.out) / "cards.json"
    print(f"源目录   : {src}")
    print(f"卡片合计 : {len(cards)}  （{kinds}）")
    skipped = skip_s + skip_d + skip_g
    if skipped:
        print(f"跳过条目 : {skipped}（非字符串条目，已在上面按类别过滤；不静默丢弃）")
    with_dept = sum(1 for c in sym + dis if c.get("departments"))
    print(f"带科室的卡片 : {with_dept}（分诊落地的依据；其余只能给疾病方向）")
    print(f"输出     : {out}")
    if args.dry_run:
        print("[dry-run] 未写文件")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(cards, fh, ensure_ascii=False)
    print(f"已写出 {out.stat().st_size / 1024:.0f} KB")
    print("注意：该语料上游标注「仅用于学术研究」，请勿提交到公开仓库（.gitignore 已忽略 data/clinic/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
