"""构建分诊金标 C 集 `data/eval/triage_gold.jsonl`，答案取自标准文档的「症状 → 科室」判据。

用法::

    python scripts/build_triage_gold.py --report-only        # 只打印覆盖率与排除项
    python scripts/build_triage_gold.py --per-symptom 2      # 写 `--out`

前置条件：标准文档 `data/eval/prototype/department_scope.md` 与现象表
`data/eval/prototype/phenomenon_to_symptom.json`（口语现象 → 规范症状），缺任一返回 2；
`manual_annotation.csv` 可选，用来给口语变体加人工同义词背书；`--cards` 指向 clinic 语料，
用于交叉校验并优先挑选语料里真实存在的查询词。

构造规则（确定性、无随机）：从标准文档抄录 `(科室, 症状关键词)` 对；同时归属多科室、或
互为子串且归属不同科室的关键词整条丢弃——金标答案必须唯一；用现象表把关键词换成口语变体，
按 `(不在语料, 长度, 字典序)` 排序并跨科室轮转，每对最多取 `--per-symptom` 个。

输出：`--out`（默认 `data/eval/triage_gold.jsonl`）为 JSONL。首行 `kind=meta`，含
`built_from`（卡片语料 sha/数量、标准文件、规则数、抽样策略、被丢弃的歧义关键词）、
`human_verified: false` 与 `disclaimer`；其余行 `kind=triage_department`，字段 `text`
（口语查询）、`expect_department`（标准文档给出的科室）、`canonical`（规范症状）、
`synonym_verified`（是否有 `manual_annotation.csv` 背书）、`source`、`note`。

结果解读：打印按科室分布、查询词在语料中存在的比例，以及标准文档与语料卡片科室的两来源
一致率——冲突项逐条列出，需人工裁决。`human_verified=false` 说明答案来自标准文档规则而非
医学人员逐条复核：它测的是「对原型标准的符合度」，不是诊断正确性，也不是上线准确率。
生成不足 50 条时会提示：上限由标准文档的规则数量决定，不要为凑数放宽规则。

退出码：0 成功 / 2 缺少标准文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROTO = REPO / "data" / "eval" / "prototype"
DEFAULT_CARDS = REPO / "data" / "clinic" / "cards.json"
DEFAULT_OUT = REPO / "data" / "eval" / "triage_gold.jsonl"

STANDARD_DOC = PROTO / "department_scope.md"
PHENOMENON = PROTO / "phenomenon_to_symptom.json"
MANUAL_ANNOTATION = PROTO / "manual_annotation.csv"

STANDARD_SECTION = "department_scope.md#科室导诊范围说明"
"""依据段落：`source` 字段回指到标准文档的这一节，便于人工核对。"""

# (科室, [症状关键词], 文档小节)：逐条抄自标准文档，未作引申。
RULES: list[tuple[str, list[str], str]] = [
    ("呼吸内科", ["咳嗽", "咳痰", "咯血", "气短", "胸痛", "发热"], "内科系统"),
    ("消化内科", ["腹痛", "腹胀", "反酸", "恶心", "呕吐", "腹泻", "便秘", "呕血", "黑便"], "内科系统"),
    ("心内科", ["心悸", "胸闷", "胸痛", "气促", "水肿", "血压"], "内科系统"),
    ("神经内科", ["头痛", "头晕", "肢体麻木", "无力", "言语不清", "抽搐", "意识"], "内科系统"),
    ("内分泌科", ["多饮", "多尿", "体重", "怕冷", "怕热", "血糖", "甲状腺"], "内科系统"),
    ("骨外科", ["外伤", "骨折", "关节", "腰腿", "活动受限"], "外科与其他"),
    ("泌尿科", ["尿频", "尿急", "尿痛", "血尿", "绞痛"], "外科与其他"),
    ("妇产科", ["月经", "下腹痛", "妊娠"], "外科与其他"),
    ("眼科", ["眼", "视力"], "外科与其他"),
    ("耳鼻喉科", ["耳", "耳鸣", "听力", "鼻塞", "鼻涕", "咽", "喉"], "外科与其他"),
    ("皮肤科", ["皮疹", "瘙痒", "瘙痒", "皮肤"], "外科与其他"),
    ("口腔科", ["牙", "口腔", "牙龈"], "外科与其他"),
]

MIN_VARIANT_LEN = 3
MAX_VARIANT_LEN = 10

_GENERIC_CHARS = set("的了个是有很都好一二三五几多量些点儿时天前后中上下来去着过感些")
"""虚字/量词：判定"实义字重合"时先剔掉，否则「一直咳」与「咳嗽」只能靠虚字沾边。"""

_BODY_PARTS = set("头面眼耳鼻咽喉口牙舌唇颈肩胸背腰腹肚脐臀腿脚踝膝手胳膊皮肤阴乳肛")
"""身体部位字：用于「部位冲突」判定。"""

_LEAD_STOPWORDS = ("全", "所有", "整个", "无", "不", "没")
"""起首的总括/否定词：「全部痛」「无痛」不是「头痛」的口语说法。"""


def _clean_variant(v: str) -> str:
    return v.strip().strip("·。，,.、;；:：!！?？ \t")


def _plausible_variant(variant: str, canonical: str) -> bool:
    """口语变体必须是**可辨认的同义说法**，否则丢掉。

    现象表是挖掘出来的，含大量伪变体（如 `腰痛 -> 腿疼`、`口臭 -> 胃热`、OCR 噪声
    `通痛`）：拿它们当查询文本，金标答案会与文本实际含义对不上，整份金标被假错污染。
    三道判定：① 起首是总括/否定词则丢（`全部痛`/`无痛`）；② 起首部位字与规范症状不同
    则丢（`头痛 -> 乳头疼`）；③ 同形、互为子串或实义字有交集则留（须先过前两道）。
    """
    if variant == canonical:
        return True
    if len(canonical) < 2:
        return False
    """单字规范症状（`眼`/`耳`/`咽`/`牙`）语义太宽：其变体常造出文本里没提部位的假题，一律不建。"""
    if variant.startswith(_LEAD_STOPWORDS):
        return False
    if canonical[0] in _BODY_PARTS and variant[0] in _BODY_PARTS and variant[0] != canonical[0]:
        return False
    if canonical in variant or variant in canonical:
        return True
    a = set(variant) - _GENERIC_CHARS
    b = set(canonical) - _GENERIC_CHARS
    return bool(a & b)


def _conflicting_keywords() -> set[str]:
    """互为子串、且归属**不同**科室的症状关键词：两边一起丢。

    例：`下腹痛`(妇产科) 含 `腹痛`(消化内科)——按哪个都不算错，进金标只会制造假错。
    """
    flat = [(kw, dept) for dept, kws, _ in RULES for kw in kws]
    bad: set[str] = set()
    for i, (ka, da) in enumerate(flat):
        for kb, db in flat[i + 1 :]:
            if da == db:
                continue
            if ka in kb or kb in ka:
                bad.update({ka, kb})
    return bad


def _load_manual_synonyms(path: Path) -> set[frozenset[str]]:
    """人工实体等价标注（`实体1,实体2,标注结果`，标注结果=1 表示同一实体）。

    第二份独立依据：口语变体与规范症状在这份表里被判为同一实体时，该条金标的
    "查询文本↔规范症状"就有人工背书，不只是形态学判定。
    """
    if not path.is_file():
        return set()
    out: set[frozenset[str]] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[2] == "1" and parts[0] and parts[1]:
            out.add(frozenset((parts[0], parts[1])))
    return out


def _cards_sha(path: Path) -> tuple[str, int]:
    if not path.is_file():
        return "", 0
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16], len(json.loads(data))


def _load_colloquial(path: Path) -> dict[str, list[str]]:
    """口语变体表；键是规范症状，值是该症状的口语说法。"""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for canonical, variants in raw.items():
        if isinstance(variants, list):
            out[str(canonical)] = [str(v) for v in variants]
    return out


def _match_canonical(keyword: str, table: dict[str, list[str]]) -> list[str]:
    """把标准文档的症状关键词对到现象表的规范症状上：只做前缀匹配，不做子串包含。

    包含匹配会把泌尿科的「绞痛」接到规范症状「心绞痛」，造出 `「心绞痛」-> 泌尿科` 这种
    概念不同的假金标；前缀匹配下「绞痛」找不到规范症状，就诚实记为建不出用例。
    """
    if keyword in table:
        return [keyword]
    return [k for k in table if k.startswith(keyword)]


def _pick_variants(canonical: str, table: dict[str, list[str]], card_keys: set[str], limit: int) -> list[str]:
    """确定性挑选口语变体：先过 `_plausible_variant` 闸，再优先语料里真实存在的。"""
    seen: set[str] = set()
    cands: list[str] = []
    for v in [canonical, *table.get(canonical, [])]:
        v = _clean_variant(v)
        if not (MIN_VARIANT_LEN <= len(v) <= MAX_VARIANT_LEN) or v in seen:
            continue
        if not _plausible_variant(v, canonical):
            continue
        seen.add(v)
        cands.append(v)
    cands.sort(key=lambda v: (v not in card_keys, len(v), v))
    return cands[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-symptom", type=int, default=2, help="每个症状最多取几个口语变体")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    for need in (STANDARD_DOC, PHENOMENON):
        if not need.is_file():
            print(f"[x] 缺标准文件：{need}")
            return 2

    table = _load_colloquial(PHENOMENON)
    synonyms = _load_manual_synonyms(MANUAL_ANNOTATION)
    print(f"原型人工等价标注：{len(synonyms):,} 对（用于给口语变体加人工背书）")
    card_keys: set[str] = set()
    if args.cards.is_file():
        card_keys = {str(c.get("key", "")) for c in json.loads(args.cards.read_text(encoding="utf-8"))}

    # 同时归属多科室的症状关键词丢弃：金标答案必须唯一
    owner: dict[str, set[str]] = {}
    for dept, keywords, _ in RULES:
        for kw in keywords:
            owner.setdefault(kw, set()).add(dept)
    dropped = sorted({kw for kw, depts in owner.items() if len(depts) > 1} | _conflicting_keywords())
    print(f"标准文档共 {len(owner)} 个症状关键词；因多科室归属/互为子串被丢弃 {len(dropped)} 个：{dropped}")

    # 交叉校验用：卡片 key -> 语料里这张卡自己标注的科室（独立于标准文档）
    corpus_depts: dict[str, list[str]] = {}
    if args.cards.is_file():
        from kubedocs.clinical.triage import departments_in_card

        for c in json.loads(args.cards.read_text(encoding="utf-8")):
            deps = departments_in_card(str(c.get("text") or ""))
            if deps:
                corpus_depts.setdefault(str(c.get("key", "")), deps)

    # 先给每条 (科室, 症状) 备好候选，再跨科室轮转取用，避免单个科室/症状主导样本
    per_pair: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    unmapped: list[str] = []
    for dept, keywords, section in RULES:
        for kw in keywords:
            if kw in dropped:
                continue
            canon = _match_canonical(kw, table)
            if not canon:
                unmapped.append(f"{dept}/{kw}")
                continue
            # 变体既要像它真正来源的规范症状（_pick_variants 保证），也要像标准文档的
            # 关键词，否则读者看不出这条文本属于哪条规则。
            picked: list[tuple[str, str]] = []
            for c in canon[:3]:
                for v in _pick_variants(c, table, card_keys, args.per_symptom * 2):
                    if _plausible_variant(v, kw) and all(v != p[0] for p in picked):
                        picked.append((v, c))
            picked.sort(key=lambda p: (p[0] not in card_keys, len(p[0]), p[0]))
            if picked:
                per_pair.append((dept, kw, section, picked[: args.per_symptom]))

    print(f"可造句的 (科室,症状) 对：{len(per_pair)}；现象表里查不到的症状：{len(unmapped)} {unmapped[:8]}")

    cases: list[dict] = []
    seen_texts: set[str] = set()
    round_idx = 0
    while any(len(v) > round_idx for _, _, _, v in per_pair):
        for dept, kw, section, variants in per_pair:
            if len(variants) <= round_idx:
                continue
            text, canonical = variants[round_idx]
            if text in seen_texts:  # 同一条口语变体可能挂在多个规范症状下
                continue
            seen_texts.add(text)
            backed = frozenset((text, kw)) in synonyms
            cases.append(
                {
                    "kind": "triage_department",
                    "id": f"c{len(cases) + 1:03d}",
                    "text": text,
                    "expect_department": dept,
                    "style": "colloquial",
                    "synonym_verified": backed,
                    "canonical": canonical,
                    "source": (
                        f"{STANDARD_SECTION}/{section} + phenomenon_to_symptom.json:{canonical}"
                        + (" + manual_annotation.csv:1" if backed else "")
                    ),
                    "note": f"标准文档「{kw}」归属 {dept}；口语变体「{text}」（规范症状 {canonical}）",
                }
            )
        round_idx += 1

    in_corpus = sum(1 for c in cases if c["text"] in card_keys)
    print(
        f"\n生成 {len(cases)} 条；其中查询词在 clinic 语料里存在的 {in_corpus} 条"
        f"（{in_corpus / max(len(cases), 1):.1%}）"
    )
    by_dept: dict[str, int] = {}
    for c in cases:
        by_dept[c["expect_department"]] = by_dept.get(c["expect_department"], 0) + 1
    print("按科室分布：", dict(sorted(by_dept.items(), key=lambda kv: -kv[1])))

    # 两来源交叉校验（只诊断、不筛选）：标准文档 vs 语料卡片自己的科室标注。
    # 冲突不是"金标错了"，而是语料与标准打架，正是分诊会答错的题——逐条列出供人工裁决。
    agree = disagree = unknown = 0
    conflicts: list[str] = []
    for c in cases:
        card_deps = corpus_depts.get(c["text"])
        if not card_deps:
            unknown += 1
        elif c["expect_department"] in card_deps:
            agree += 1
        else:
            disagree += 1
            if len(conflicts) < 12:
                conflicts.append(
                    f"{c['id']} 「{c['text']}」 标准={c['expect_department']} 语料={'/'.join(card_deps)}"
                )
    known = agree + disagree
    print(
        f"\n两来源交叉校验：一致 {agree}，冲突 {disagree}，语料无该卡 {unknown}"
        f"（在可校验的 {known} 条里一致率 {agree / max(known, 1):.1%}）"
    )
    if conflicts:
        print("冲突样例（标准 vs 语料，需人工裁决）：")
        for line in conflicts:
            print("   ", line)

    print("样例：")
    for c in cases[:6]:
        print(f"   {c['id']} 「{c['text']}」 -> {c['expect_department']}")
    backed_n = sum(1 for c in cases if c.get("synonym_verified"))
    print(f"其中带人工同义词背书的：{backed_n} / {len(cases)}（其余靠形态学判定，需人工抽检）")

    if len(cases) < 50:
        print(f"\n[!] 只生成 {len(cases)} 条 (<50)：标准文档的规则数量决定了上限。")
        print("    要更多需要补标准文档的「症状 → 科室」条目，**不要**为了凑数放宽规则。")

    if args.report_only:
        return 0

    sha, count = _cards_sha(args.cards)
    meta = {
        "kind": "meta",
        "version": 1,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "built_by": "scripts/build_triage_gold.py",
        "built_from": {
            "cards_sha": sha,
            "card_count": count,
            "standard": str(STANDARD_DOC.relative_to(REPO)),
            "phenomenon": str(PHENOMENON.relative_to(REPO)),
            "rules": len(RULES),
            "per_symptom": args.per_symptom,
            "sample_strategy": "标准文档规则 → 现象表口语变体；跨科室轮转，排序键 (不在语料,长度,字典序)，无随机",
            "dropped_ambiguous": dropped,
        },
        "human_verified": False,
        "disclaimer": (
            "本文件是**按原型标准文档派生的金标**，答案来自 department_scope.md 的规则，"
            "**未经医学人员逐条复核**（human_verified=false）。它测的是「对原型标准的符合度」，"
            "不是诊断正确性，也**不是上线准确率**。"
        ),
    }
    lines = [json.dumps(meta, ensure_ascii=False), *[json.dumps(c, ensure_ascii=False) for c in cases]]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写出 {args.out.relative_to(REPO)}（meta + {len(cases)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
