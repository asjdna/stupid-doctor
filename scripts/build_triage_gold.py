"""按原型的**科室导诊标准**构建分诊金标 C 集：`data/eval/triage_gold.jsonl`。

## 为什么要扩金标

原 C 集只有 **5 条**人工手写用例。5 条里 1 条就值 20%，
"分诊准确率 80%"在这个样本量上**不可复现、不该写进结论**。
方案 §5.1 已经把"扩到 ≥50 条"列为前置条件。

## 依据的"标准"来自原型（不是我编的）

| 来源文件 | 作用 |
|---|---|
| `data/eval/prototype/department_scope.md` | **科室导诊范围标准**：显式给出「症状 → 科室」判据 |
| `data/eval/prototype/phenomenon_to_symptom.json` | **口语现象 → 规范症状**（300 个规范症状，含大量口语变体） |
| `data/eval/prototype/disease_to_department.json` | 疾病 → 科室（8,819 条，用于交叉校验） |

原型 `scripts/build_rag_gold.py` 的**金标规范**也照搬：文件首行是 `kind=meta`
头（含 `built_from` 的语料指纹、抽样策略、`human_verified`、`disclaimer`），
其余行是 `kind=<类型>` 的条目。原型的 `rag_gold.jsonl` 明确写着
`"human_verified": false`——**本文件同样如此**，见下面的"诚实边界"。

## 构造规则（确定性，可复现）

1. 从标准文档的「症状 → 科室」规则里取 `(科室, 症状关键词)` 对；
2. **一个症状同时归属多个科室时直接丢弃**——金标答案必须唯一，
   （例：`胸痛` 同时出现在呼吸内科"胸痛伴呼吸相关症状"与心内科"胸闷胸痛"，
   `发热` 在呼吸内科是"发热**伴呼吸道症状**"；这类模糊项进金标只会制造假错）；
3. 用 `phenomenon_to_symptom.json` 把每个症状关键词换成**口语变体**作查询文本
   （C 集的意义就是"真实问法"）；
4. 每个 `(科室, 症状)` 最多取 `--per-symptom` 个变体，跨科室**轮转**选取，
   避免某个科室/症状主导样本；
5. 变体优先取**在 clinic 语料里真实存在**的（检索才可能命中），
   排序键 `(不在语料, 长度, 字典序)` —— 完全确定，无随机。

## 诚实边界（照原型的写法）

- `human_verified: false`：条目的**答案来自标准文档的规则**，不是医学人员逐条复核过；
- 这是**导诊参考**，不是诊断正确性判定；
- 用它测出来的数字是"**对原型标准的符合度**"，不是"上线准确率"。

用法::

    python scripts/build_triage_gold.py --report-only   # 只看覆盖率与排除项
    python scripts/build_triage_gold.py                  # 写金标
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
"""依据段落：`科室导诊范围.md` 的「本文说明各科室大致覆盖的就诊方向……」

规则按**该文档的原文**逐条抄录，`source` 字段回指到这一节，便于人工核对。
"""

# (科室, [症状关键词], 文档小节) —— 逐条抄自 department_scope.md，未作引申。
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
"""身体部位字。用于「部位冲突」判定——这是第二道闸要拦的东西。"""

_LEAD_STOPWORDS = ("全", "所有", "整个", "无", "不", "没")
"""起首的总括/否定词：「全部痛」「无痛」不是「头痛」的口语说法。"""


def _clean_variant(v: str) -> str:
    return v.strip().strip("·。，,.、;；:：!！?？ \t")


def _plausible_variant(variant: str, canonical: str) -> bool:
    """口语变体必须是**可辨认的同义说法**，否则丢掉。

    **这道闸是必须的**：原型的 `phenomenon_to_symptom.json` 是**挖掘**出来的，
    含大量伪变体——实测见过 `腰痛 -> 腿疼`、`口臭 -> 胃热/喉咙有痰`、
    `咳痰 -> 吐好多`、以及 `通痛` 这种 OCR 噪声。拿它们当查询文本，
    金标答案（标准文档给的科室）会和文本实际含义对不上，
    整份金标就会被假错污染——**宁可少几条，也不要错的**。

    三道判定，逐条都是被实际错例逼出来的：

    1. **部位冲突**：与规范症状**起首部位字不同**的直接丢。
       `头痛 -> 乳头疼`（乳≠头）就是这么错的——只按"实义字重合"判它会通过，
       因为二者都有 `头/疼`。这一条把它拦掉。
    2. **起首总括/否定词**：`全部痛`、`无痛` 不是症状口语说法。
    3. 完全同形、互为子串、或实义字有交集（兜底，且要过上面两条）。
    """
    if variant == canonical:
        return True
    if len(canonical) < 2:
        return False
    """单字规范症状（`眼`/`耳`/`咽`/`牙`）语义太宽：其变体实测出了
    `干涩感 -> 眼科`、`有异物 -> 耳鼻喉科`、`出牙血 -> 口腔科` 这类
    文本里根本没提部位的假题，一律不建。"""
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
    """原型的人工实体等价标注（`实体1,实体2,标注结果`，标注结果=1 表示同一实体）。

    这是**第二份独立的人工依据**：如果口语变体与规范症状在这份表里被判为同一实体，
    该条金标的"查询文本↔规范症状"这一环就有人工背书，不只是我的形态学判定。
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
    """把标准文档里的症状关键词对到现象表的规范症状上。

    **只用前缀匹配，不用子串包含**——这条是被实测纠过的：用包含匹配时，
    泌尿科的「绞痛」会匹配到规范症状「心绞痛」，于是造出
    `「心绞痛」-> 泌尿科` 这种概念完全不同的假金标。
    前缀匹配下「绞痛」找不到规范症状，就诚实记为"建不出用例"，而不是硬造。
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

    # 规则 2：症状同时归属多科室 ⇒ 丢弃（金标答案必须唯一）
    owner: dict[str, set[str]] = {}
    for dept, keywords, _ in RULES:
        for kw in keywords:
            owner.setdefault(kw, set()).add(dept)
    dropped = sorted({kw for kw, depts in owner.items() if len(depts) > 1} | _conflicting_keywords())
    print(f"标准文档共 {len(owner)} 个症状关键词；因多科室归属/互为子串被丢弃 {len(dropped)} 个：{dropped}")

    # 语料交叉校验用：卡片 key -> 该卡片自己标注的科室（独立于标准文档的第二个来源）
    corpus_depts: dict[str, list[str]] = {}
    if args.cards.is_file():
        from kubedocs.clinical.triage import departments_in_card

        for c in json.loads(args.cards.read_text(encoding="utf-8")):
            deps = departments_in_card(str(c.get("text") or ""))
            if deps:
                corpus_depts.setdefault(str(c.get("key", "")), deps)

    # 每条 (科室, 症状) 先备好候选，再跨科室轮转取用（规则 4）
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
            # 变体要同时满足两件事：
            #   (a) 与它**真正来源**的那个规范症状像（由 _pick_variants 保证）；
            #   (b) 与**标准文档的关键词**也像——否则文本读者看不出它属于这一条规则。
            # 只做 (a) 会漏进 `月经量少 -> 怕量少`：它对「月经量少」成立，
            # 但把答案记成「月经→妇产科」时，查询文本「怕量少」谁都读不出是月经问题。
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

    # **两来源交叉校验**（诊断，不用于筛选）：标准文档 vs 语料卡片自己的科室标注。
    # 不一致不是"金标错了"，而是"语料与标准打架"——正是分诊会答错的那些题，
    # 必须逐条列出来给人看，不能藏。
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
