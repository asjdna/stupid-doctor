"""极性判断（正/反方向）：区分「确有症状 present」vs「否认症状 absent」vs「可能有 possible」。

混合双轨：
- 规则快路径（本模块）：否定词 + 作用域（列举延续 + 转折截断 + 假否定/非否定白名单 + 双否定识别），
  确定性判断某短语的极性，零延迟、可离线。
- LLM 慢路径（由 clinic 域的问诊节点在复杂否定时调用）：规则拿不准的复杂否定交 LLM。

> 移植说明：本模块自 clinic 域原样移植（原 medical-agent 项目），零外部导入、判定逻辑一字未改。
> 它是急诊红线安全性的根基——`is_reportable` 的三维门控决定了「我老公胸痛」「去年胸痛」
> 不会把患者本人送进急诊。移植时只改了本段对 LLM 慢路径的路径引用。

## ConText 的三个维度

临床语境判定不是"一个极性"就够的。参考 ConText（Chapman 2009，medspaCy 的
`medspacy.context` 是它的 Python 实现）与 i2b2/VA 2010 断言标注指南，本模块把语境拆成三维：

| 维度 | 取值 | 不建模的后果（实测） |
|---|---|---|
| **断言状态 status** | present / absent / possible / hypothetical | 「如果发烧的话要不要去医院」被当成现症 |
| **经验者 experiencer** | self / other | 「我老公胸痛，我没事」**触发急诊红线**，把患者本人送进急诊 |
| **时间性 temporality** | present / past | 「我去年胸痛过一次，现在好了」按现症**触发急诊红线** |

⚠️ 中文没有现成的 ConText 规则可用（medspaCy 的语言表只有 en/fr/nl），所以三维都得自己写。

**作用域约定**：三维一律采用「**所有出现处**都命中才算数」——与 `is_negated` 同一约定。
医学安全偏向"宁可留着"：只要有一处是本人/现在/非假设，就按本人/现在/非假设处理，
避免把患者自己的症状误判成别人的或过去的而漏掉。

对外主 API：
- is_negated(text, phrase) -> bool：True = 该短语在**所有出现处**均为 absent（应剔除）；
  False = 至少一处为 present 或 possible（保留）。医学安全偏向保守——不轻易丢弃「可能有」。
- polarity(text, phrase) -> str：首处出现的三值极性 present/absent/possible，供单测与自省。
- assertion(text, phrase) -> dict：四值状态 + 经验者 + 时间性（三维一次拿全）。
- is_reportable(text, phrase) -> bool：能否作为「患者当前主诉」上报（急诊红线/病历用它）。
"""

from __future__ import annotations

from collections.abc import Iterable

# 否定词（长词在前，避免「无明显」被「无」抢先匹配）。
# 不含「非/否」——「非常疼」「否则」里的非/否是程度/连词，不是否定。
NEGATION_WORDS = [
    "无明显",
    "并无",
    "没有",
    "未见",
    "否认",
    "毫无",
    "完全无",
    "无",
    "没",
    "不",
    "未",
]

# 转折/重启词：否定作用域到此为止（"无发热，但是咳嗽" -> 咳嗽为正）。
REVERSAL_WORDS = [
    "但是",
    "但",
    "不过",
    "只是",
    "就是",
    "而是",
    "可是",
    "然而",
    "却",
    "反而",
    "倒",
    "还有",
    "伴有",
    "伴",
    "出现",
    "存在",
    "主要",
    "倒是",
    "又有",
    "并有",
    # 下面两个是**用原型用例实测补上的**（`data/eval/prototype/symptom_status_gold.jsonl`）：
    #   · ss-034「无腹泻，其实是肚子疼」——缺 "其实"，否定作用域盖过逗号，把"肚子疼"判成 absent；
    #   · ss-035「无胸痛、无呼吸困难，仅有轻微头晕」——缺 "仅"，"头晕"同样被判成 absent。
    # 补这两个词后 status 一致率 43/45 → 45/45（见 `scripts/eval_prototype_cases.py`）。
    "其实",
    "仅",
]

# 假否定复合词：「无X」里 X 是修饰而非被否定的症状（「无痛性血尿」= painless hematuria，血尿仍在）。
FAKE_NEGATION = ["无痛性", "无创", "无诱因"]

# 非否定复合词：「不/没」在这些词里不是否定（「差不多/不错/没准」）。
# 「没劲」= 乏力，本身是个症状词；其中的「没」不是否定触发词，
# 不排除的话它的作用域会把后面的症状吞掉（「没劲犯困」-> 犯困被判否定而丢掉）。
NON_NEGATION = [
    "差不多",
    "不错",
    "没准",
    "不过",
    "不仅",
    "不但",
    "不止",
    "没关系",
    "不如",
    "不料",
    "不妨",
    "不由得",
    "没劲",
    "没力气",
    "没精神",
]

# 双否定 / 不确定 cue：后面的症状是 possible（可能有），不是 absent。
# 临床上「不能排除胸痛」意味着胸痛方向需要警惕，而非「无胸痛」。
UNCERTAIN_CUES = ["不能排除", "不除外", "不排除", "疑似", "怀疑", "不能完全排除"]

# 假设/条件 cue：症状是"设想"而非"现症"，不应据以分诊（i2b2 的 hypothetical 类）。
HYPOTHETICAL_CUES = ["如果", "假如", "要是", "万一", "一旦", "倘若", "若是"]

# 经验者 = 第三方（家属/他人）。这些标记**本身含「我」**，所以必须长词在前 + 整词匹配，
# 否则「我老公」会被切成「我」(本人) + 「老公」，家属的主诉就记到患者头上了。
#
# 刻意**不收**「他/她」：口语里「他」可能指医生、也可能指本人，误判会把患者自己的症状
# 当成别人的而漏掉 —— 漏急诊比假急诊更危险，所以歧义时一律按本人处理。
EXPERIENCER_OTHER = [
    "我老公",
    "我丈夫",
    "我老婆",
    "我妻子",
    "我儿子",
    "我女儿",
    "我家孩子",
    "我的孩子",
    "我爸爸",
    "我爸",
    "我妈妈",
    "我妈",
    "我父亲",
    "我母亲",
    "我爷爷",
    "我奶奶",
    "我姥姥",
    "我姥爷",
    "我外公",
    "我外婆",
    "我哥哥",
    "我哥",
    "我姐姐",
    "我姐",
    "我弟弟",
    "我弟",
    "我妹妹",
    "我妹",
    "我孙子",
    "我孙女",
    "家里人",
    "家人",
    "孩子",
    "小孩",
    "宝宝",
    "婴儿",
    "同事",
    "朋友",
    "邻居",
    "别人",
    "他人",
    "亲戚",
    # 不带「我」的亲属称谓：预问诊里用户就是患者本人，所以这些一律指别人。
    # （长词在前，所以「我父亲」仍优先于「父亲」。）
    "爸爸",
    "妈妈",
    "父亲",
    "母亲",
    "儿子",
    "女儿",
    "哥哥",
    "姐姐",
    "弟弟",
    "妹妹",
    "爷爷",
    "奶奶",
    "姥姥",
    "姥爷",
    "外公",
    "外婆",
    "老公",
    "老婆",
    "丈夫",
    "妻子",
    "孙子",
    "孙女",
]
EXPERIENCER_SELF = ["我自己", "本人", "我"]

# 时间性：既往 vs 现在。既往症状不进「现病史」，也不该触发急诊红线。
TEMPORAL_PAST = [
    "大前年",
    "多年前",
    "上个月",
    "上个星期",
    "去年",
    "前年",
    "以前",
    "从前",
    "曾经",
    "小时候",
    "上周",
    "上次",
    "之前",
    "当初",
    "当年",
    "当时",
    "得过",
    "有过",
    "已经好了",
    "早好了",
]
TEMPORAL_NOW = [
    "这几天",
    "这两天",
    "这会儿",
    "现在",
    "目前",
    "当前",
    "今天",
    "昨天",
    "刚刚",
    "刚才",
    "最近",
]

# 分句边界：否定作用域不跨越这些标点（逗号/顿号在列举中继续延续，不截断）
CLAUSE_BOUNDARY = "。！？；\n"

# 否定作用域最长延伸字符数（无转折/分句时的兜底，防止一个「无」吞掉整句）
NEG_SCOPE_MAX = 14


def _scan_scope(text: str, start: int, limit: int) -> int:
    """从 start 起找作用域结束位置（遇到转折词或分句边界即止）。

    ⚠️ 逗号**不是**分句边界，这是有意的：`无发热、无咳嗽` 与 `无发热，无咳嗽`
    都该让"无"盖住后面那一段。真正该收束作用域的是**转折词**（`但没有`、`其实是`）。
    所以缺的转折词会让否定"越界"——用原型用例实测到两处（见下表），补词即可，
    不要把逗号当成边界（那会把 `无发热，无咳嗽` 判反）。
    """
    k = start
    while k < limit:
        for r in REVERSAL_WORDS:
            if text.startswith(r, k):
                return k
        if text[k] in CLAUSE_BOUNDARY:
            return k
        k += 1
    return limit


def _negation_ranges(text: str) -> list[tuple[int, int, str]]:
    """返回 [(neg_end, scope_end, kind), ...]，kind ∈ {absent, possible}。"""
    ranges: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        # 1) 假否定 / 非否定复合词：跳过，不产生作用域
        compound = None
        for w in FAKE_NEGATION + NON_NEGATION:
            if text.startswith(w, i):
                compound = w
                break
        if compound:
            i += len(compound)
            continue

        # 2) 双否定 / 不确定 cue -> possible 作用域
        cue = None
        for w in UNCERTAIN_CUES:
            if text.startswith(w, i):
                cue = w
                break
        if cue:
            neg_end = i + len(cue)
            scope_end = _scan_scope(text, neg_end, min(n, neg_end + NEG_SCOPE_MAX))
            ranges.append((neg_end, scope_end, "possible"))
            i = neg_end
            continue

        # 3) 普通否定词 -> absent 作用域
        neg = None
        for w in NEGATION_WORDS:
            if text.startswith(w, i):
                neg = w
                break
        if neg:
            neg_end = i + len(neg)
            scope_end = _scan_scope(text, neg_end, min(n, neg_end + NEG_SCOPE_MAX))
            ranges.append((neg_end, scope_end, "absent"))
            i = neg_end
            continue

        i += 1
    return ranges


def _find_all(text: str, phrase: str) -> list[int]:
    positions = []
    start = 0
    while True:
        p = text.find(phrase, start)
        if p == -1:
            break
        positions.append(p)
        start = p + 1
    return positions


def _polarity_at(pos: int, ranges: list[tuple[int, int, str]]) -> str:
    for neg_end, scope_end, kind in ranges:
        if neg_end > pos:
            break
        if neg_end <= pos < scope_end:
            return kind  # "absent" 或 "possible"
    return "present"


def polarity(text: str, phrase: str) -> str:
    """首处出现的三值极性；短语不在文本中返回 absent（供单测/自省）。"""
    positions = _find_all(text, phrase)
    if not positions:
        return "absent"
    return _polarity_at(positions[0], _negation_ranges(text))


def verdicts(text: str, phrases: Iterable[str]) -> dict[str, str]:
    """批量返回 {短语: 三值极性}（作用域只算一次）。

    供抗噪层的「删除影响守卫」使用：它要在原句与清洗句之间逐个比对十几个症状候选的极性，
    逐个调 polarity() 会把否定作用域重复计算十几遍。
    """
    ranges = _negation_ranges(text)
    out: dict = {}
    for phrase in phrases:
        positions = _find_all(text, phrase)
        out[phrase] = _polarity_at(positions[0], ranges) if positions else "absent"
    return out


def is_negated(text: str, phrase: str) -> bool:
    """True = 所有出现处均为 absent（应剔除）；False = 至少一处 present 或 possible。"""
    positions = _find_all(text, phrase)
    if not positions:
        return False
    ranges = _negation_ranges(text)
    return all(_polarity_at(p, ranges) == "absent" for p in positions)


# ==================== ConText 三维：经验者 / 时间性 / 假设 ====================
# 标记 -> 标签的映射；键按**长度降序**用于扫描，保证「我老公」优先于「我」。
_EXPERIENCER_MAP = dict.fromkeys(EXPERIENCER_OTHER, "other")
_EXPERIENCER_MAP.update(dict.fromkeys(EXPERIENCER_SELF, "self"))
_TEMPORAL_MAP = dict.fromkeys(TEMPORAL_PAST, "past")
_TEMPORAL_MAP.update(dict.fromkeys(TEMPORAL_NOW, "present"))
_HYPOTHETICAL_MAP = dict.fromkeys(HYPOTHETICAL_CUES, "hypothetical")

EXPERIENCER_MARKERS = tuple(sorted(_EXPERIENCER_MAP, key=len, reverse=True))
TEMPORAL_MARKERS = tuple(sorted(_TEMPORAL_MAP, key=len, reverse=True))
HYPOTHETICAL_MARKERS = tuple(sorted(_HYPOTHETICAL_MAP, key=len, reverse=True))

# 归因维度的硬边界：。！？；\n 截断作用域。
# 刻意**不**把转折词（但是/就是）当边界 —— 转折改变的是否定作用域，不是"这话在说谁"。
# 逗号也不是边界：「我老公胸痛，我没事」靠后面出现的「我」把归属切回来。
_HARD_BOUNDARY = "。！？；\n"


def _marker_hits(text: str, markers: tuple[str, ...], mapping: dict[str, str]) -> list[tuple[int, int, str]]:
    """按出现顺序找标记，长词优先且不重叠：[(_start, _end, label), ...]。"""
    hits = []
    i, n = 0, len(text)
    while i < n:
        for w in markers:
            if text.startswith(w, i):
                hits.append((i, i + len(w), mapping[w]))
                i += len(w)
                break
        else:
            i += 1
    return hits


def _hard_boundary_end(text: str, start: int) -> int:
    """start 之后第一个硬边界的位置（没有则到句尾）。"""
    for k in range(start, len(text)):
        if text[k] in _HARD_BOUNDARY:
            return k
    return len(text)


def _label_at(text: str, pos: int, hits: list, default: str, combined: list | None = None) -> str:
    """pos 处生效的标记标签：取「pos 之前最近的一个标记」，且其作用域未被截断。

    `combined` 是**三个维度合并后**的标记序列。归因标记之间互相截断作用域：
    「我父亲得过心梗，我胸口有点闷」里，`得过`(既往) 的作用域必须被后面的 `我`(本人) 打断，
    否则患者**自己的**胸闷会被误判成既往 —— 那正是本模块要避免的那类错误。
    """
    prev = None
    for hit in hits:
        if hit[0] > pos:
            break
        prev = hit
    if prev is None:
        return default
    _start, end, label = prev
    scope_end = _hard_boundary_end(text, end)
    if combined:
        for c in combined:
            # `end < c[0]`（而非 <=）：紧挨着的**另一个维度**的标记不算打断归属。
            # 「我父亲得过心梗」里 `得过`(时间) 紧接 `我父亲`(经验者) 之后，
            # 若算打断，心梗就会从"父亲的既往"变成"本人的"——正好判反。
            if end < c[0] < scope_end and c[0] <= pos:
                scope_end = c[0]
                break
    return label if pos < scope_end else default


def _combined_hits(text: str) -> list:
    """三个维度的标记合并成一个按位置排序的序列（用于互相截断作用域）。"""
    merged = (
        _marker_hits(text, EXPERIENCER_MARKERS, _EXPERIENCER_MAP)
        + _marker_hits(text, TEMPORAL_MARKERS, _TEMPORAL_MAP)
        + _marker_hits(text, HYPOTHETICAL_MARKERS, _HYPOTHETICAL_MAP)
    )
    return sorted(merged, key=lambda h: h[0])


def _attribution(
    text: str, phrase: str, markers: tuple[str, ...], mapping: dict[str, str], default: str
) -> str:
    """短语的归因标签。

    **所有出现处**都命中该标签才算数（与 is_negated 同一约定）：只要有一处不是，
    就返回 default。医学安全偏向"宁可留着"——避免把患者自己的症状误判成别人的/过去的。
    """
    positions = _find_all(text, phrase)
    if not positions:
        return default
    hits = _marker_hits(text, markers, mapping)
    combined = _combined_hits(text)
    for p in positions:
        if _label_at(text, p, hits, default, combined) != default:
            continue
        return default  # 有一处不是该标签 -> 按 default（保守）
    return _label_at(text, positions[0], hits, default, combined)


def experiencer_of(text: str, phrase: str) -> str:
    """经验者：self（本人）/ other（家属、他人）。歧义与缺省一律 self。"""
    return _attribution(text, phrase, EXPERIENCER_MARKERS, _EXPERIENCER_MAP, "self")


def temporality_of(text: str, phrase: str) -> str:
    """时间性：present（现在）/ past（既往）。歧义与缺省一律 present。"""
    return _attribution(text, phrase, TEMPORAL_MARKERS, _TEMPORAL_MAP, "present")


def status_of(text: str, phrase: str) -> str:
    """四值断言状态：absent / hypothetical / possible / present。

    优先级：否认 > 假设 > 可能有 > 确有。前两者都不作为"患者当前主诉"上报。
    polarity() 仍保持三值（向后兼容），hypothetical 只有本函数会返回。
    """
    if is_negated(text, phrase):
        return "absent"
    if _attribution(text, phrase, HYPOTHETICAL_MARKERS, _HYPOTHETICAL_MAP, "present") == "hypothetical":
        return "hypothetical"
    return polarity(text, phrase)


def assertion(text: str, phrase: str) -> dict[str, str]:
    """一次拿全三维：{status, experiencer, temporality}。"""
    return {
        "status": status_of(text, phrase),
        "experiencer": experiencer_of(text, phrase),
        "temporality": temporality_of(text, phrase),
    }


def is_reportable(text: str, phrase: str) -> bool:
    """能否作为「患者当前主诉」上报（急诊红线与病历用它）。

    True 需同时满足：状态是 present/possible（不是否认、不是假设）、
    经验者是本人、时间性是现在。
    """
    a = assertion(text, phrase)
    return (
        a["status"] in ("present", "possible")
        and a["experiencer"] == "self"
        and a["temporality"] == "present"
    )


def assertion_verdicts(text: str, phrases: Iterable[str]) -> dict[str, tuple[str, str, str]]:
    """批量 {短语: (status, experiencer, temporality)}（供抗噪层的删除影响守卫复用）。"""
    hits = {
        "exp": _marker_hits(text, EXPERIENCER_MARKERS, _EXPERIENCER_MAP),
        "tmp": _marker_hits(text, TEMPORAL_MARKERS, _TEMPORAL_MAP),
        "hyp": _marker_hits(text, HYPOTHETICAL_MARKERS, _HYPOTHETICAL_MAP),
    }
    combined = _combined_hits(text)
    neg_ranges = _negation_ranges(text)
    out: dict = {}
    for phrase in phrases:
        positions = _find_all(text, phrase)
        if not positions:
            out[phrase] = ("absent", "self", "present")
            continue
        if all(_polarity_at(p, neg_ranges) == "absent" for p in positions):
            status = "absent"
        elif all(_label_at(text, p, hits["hyp"], "present", combined) == "hypothetical" for p in positions):
            status = "hypothetical"
        else:
            status = _polarity_at(positions[0], neg_ranges)
        out[phrase] = (
            status,
            _attribution(text, phrase, EXPERIENCER_MARKERS, _EXPERIENCER_MAP, "self"),
            _attribution(text, phrase, TEMPORAL_MARKERS, _TEMPORAL_MAP, "present"),
        )
    return out
