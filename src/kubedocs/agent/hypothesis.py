"""假设-检验打分引擎：候选（疾病/科室）的规则加权打分 + 主动试探性提问。

问诊不是"等患者把话说完再检索"，而是一轮轮**抛假设、收证据、修候选**（MYCIN 式鉴别诊断）：
系统问"我猜您是胸口发闷，像压着东西？"，用户答是/否，候选分数随之升降，直到某个候选
明显领先（收敛）或轮次用尽。

## 为什么是加权求和，不是贝叶斯

贝叶斯要可靠先验 P(症状|疾病) 与 P(疾病)，而本仓库的数据是人工整理的症状-候选关联表
（`symptom_to_diseases.json` 形制），**没有可信的频率/先验**。在没有先验的数据上套贝叶斯，
只会把拍脑袋的数字包装成"概率"，比显式的权重更难审计也更难调。所以这里只做规则加权求和：
每一分的来源都能追到"哪条证据、什么来源、什么权重"，病历里也照实写出来。

## 为什么权重表长这样（`WEIGHTS`）

| 来源 | 权重 | 为什么 |
|---|---|---|
| VOLUNTEERED | +1.0 | 用户自己说的，最可信，作为基准 |
| CONFIRMED | +0.5 | 系统抛出假设后用户确认——**建议性偏倚**：人会顺着问题答"是"，只算一半 |
| NEGATED | -0.5 | 排除与命中同等重要；但口语否认可能是没听准，所以只按半权扣 |
| NEGATED_AFTER_SUGGESTION | -0.25 | 建议后被否，可能是被问烦了随口否，最弱 |
| RULE | 0.0 | 急诊红线走确定性门控，**不参与候选打分**，不在这里和鉴别诊断混为一谈 |

## 为什么红线症状绝不能由系统主动抛出（`is_safe_to_suggest`）

"系统主动建议症状"是把双刃剑：它提高采集效率，但也**制造症状**——用户会因为被问而
承认本没有的东西（建议性偏倚，`state.py` 里已论证）。对急诊红线而言这个风险不可接受：
一旦系统问出"您是不是胸痛"，用户顺口答"是"，就等于系统**自己制造了**一个急诊信号，
后续的确定性门控再怎么严密，处理的也是被污染的证据。所以红线症状一律不在建议池里——
这条约束施加在**所有建议路径**（`next_question` / `suggest_hypothesis`）上，
而不是只在措辞层过滤。

## 为什么选题要"切得最均匀"（`next_question`）

选题的目标不是确认当前领先者（那是回声室），而是**尽快把候选集劈开**：一个症状如果所有
候选都有，或在所有候选里都没有，问了也白问；"一半候选有、一半没有"的症状才最有鉴别力。
所以判据是 |正例候选数 - 负例候选数| 最小（信息增益的零阶近似：不依赖任何概率估计，
只需要集合计数）。注意切分是在**当前候选集**上做，且不看分数——低分候选正是最需要被
区分开的，按分数加权会让选题偏向已经领先的候选。

## 为什么 LLM 只在并列时兜底（`tie_breaker`）

绝大多数轮次候选分差明显，规则求和足够；只有 Top1 与 Top2 几乎同分（< `tie_epsilon`）时，
才值得花一次模型调用去仲裁。这样 LLM 是**兜底**而不是主路径：问诊的默认行为可离线、
可复现、零延迟、零成本，模型只在真正拿不准的地方介入。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress

from kubedocs.agent.state import EvidenceSource, SymptomEvidence
from kubedocs.domains.clinic.redline import EMERGENCY_RED_LINE

# 证据来源 -> 分数权重。数值是**产品决策**，不是可调超参：改动会直接改变分诊倾向，
# 所以这里的每一个数字都有上面的理由，且被单测逐项锁死。
WEIGHTS: dict[EvidenceSource, float] = {
    EvidenceSource.VOLUNTEERED: 1.0,
    EvidenceSource.CONFIRMED: 0.5,
    EvidenceSource.NEGATED: -0.5,
    EvidenceSource.NEGATED_AFTER_SUGGESTION: -0.25,
    # RULE 是"占位 0"而不是缺键：让"新增来源忘了配权重"在单测里就暴露，
    # 也让调用方查表时永远拿得到数（不用写 .get(src, 0.0)）。
    EvidenceSource.RULE: 0.0,
}


def weight_of(source: EvidenceSource) -> float:
    """按来源取权重。

    只认 `WEIGHTS` 这一张表：证据自带的 `weight` 字段是**快照**（`SymptomEvidence`
    的 dataclass 默认值是 0.0，调用方常常不填），若以字段为准，未填的证据会被静默
    当成零权，分数看起来"对"却少了分。所以打分一律回表，字段只用于展示与留档。
    """
    return WEIGHTS.get(source, 0.0)


def score(
    evidence: Sequence[SymptomEvidence],
    links: Mapping[str, Sequence[str]],
    *,
    tie_breaker: Callable[[str, Sequence[str]], str] | None = None,
    tie_epsilon: float = 1e-9,
) -> dict[str, float]:
    """把证据按权重累加到它关联的候选上，返回 候选 -> 分数。

    返回的键**只包含被证据触及的候选**：本函数只拿到 `links`（症状 -> 候选），拿不到
    完整候选集，因此"零分候选"无法凭空出现；调用方要完整候选表现状，请自行并入初始候选表。

    `tie_breaker` 的调用约定：仅在 Top1 与 Top2 的分差 < `tie_epsilon` 时调用，
    入参是（已有证据的上下文摘要, 并列候选名单），返回它认为最可能的那个候选名；
    返回不在并列名单里的名字时按"模型幻觉"忽略，抛异常时保持原分数——兜底失败不该
    让整轮问诊崩掉，宁可不打破并列。
    """
    scores: dict[str, float] = {}
    for item in evidence:
        targets = links.get(item.symptom)
        # 症状词典比知识库大得多：查不到关联是常态，跳过而不是报错。
        if not targets:
            continue
        weight = weight_of(item.source)
        for candidate in targets:
            scores[candidate] = scores.get(candidate, 0.0) + weight
    if tie_breaker is not None:
        _break_tie(scores, evidence, tie_breaker, tie_epsilon)
    return scores


def rank(scores: Mapping[str, float], *, top: int | None = None) -> list[tuple[str, float]]:
    """按分数降序排名，同分按候选名升序。

    同分必须有一个**确定性**次序：否则同一份证据在不同进程里会给出不同的 Top1，
    问诊结果与回归测试都会随字典插入顺序抖动。`top` 为 None 时返回全部；
    `top <= 0` 返回空表（而不是 Python 切片那种"负数从尾部取"的意外语义）。
    """
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top is None:
        return ordered
    if top <= 0:
        return []
    return ordered[:top]


def leading_margin(scores: Mapping[str, float]) -> float:
    """Top1 与 Top2 的分差；候选少于 2 个时返回 0.0。

    少于两个候选时"领先"没有意义（没有对手可比），返回 0.0 让调用方自己决定
    单候选算不算收敛——而不是在这里替它猜，也不是抛 IndexError。
    """
    ordered = rank(scores, top=2)
    if len(ordered) < 2:
        return 0.0
    return ordered[0][1] - ordered[1][1]


def should_stop(
    scores: Mapping[str, float],
    turn: int,
    *,
    max_turns: int = 6,
    margin: float = 1.0,
    user_stop: bool = False,
) -> tuple[bool, str]:
    """三个停止条件任一满足即停，返回 (是否停止, 停止原因)。

    检查顺序即优先级：**用户意愿 > 系统预算 > 收敛度**。
    1. 用户主动要求停止——最重要，用户说不问了就立刻停，不该再问一句"再确认一下"；
    2. 轮次达到上限——问诊必须收敛，领先度不够也不许无限追问（6 轮的默认值来自
       实际问诊的可承受长度，超过之后用户的回答质量本身就会下降）；
    3. 领先度达标（`leading_margin >= margin`）——已经有结论了，继续问只增加成本与偏倚。
    """
    if user_stop:
        return True, "用户主动要求停止"
    if turn >= max_turns:
        return True, f"已达轮次上限（{turn}/{max_turns}）"
    gap = leading_margin(scores)
    if gap >= margin:
        return True, f"候选领先度达标（领先 {gap:.2f} ≥ {margin}）"
    return False, "继续追问"


def is_safe_to_suggest(symptom: str) -> bool:
    """False 表示该症状属于急诊红线，不允许由系统主动抛出。

    **双向子串匹配**，两种方向都要拦：
    - 红线词嵌在症状名里（"突发的胸痛伴出汗" ⊃ "胸痛"）——只做全等比较会放过它；
    - 症状名只是红线词的片段（"看不清" ⊂ "突然看不清"）——临床上指同一件事，
      保守起见同样禁问。
    空串也返回 False：建议一个空症状没有意义，且 `"" in w` 会命中任何红线词。

    红线词表直接引用 `EMERGENCY_RED_LINE`（唯一真相源），不在这里复制一份——
    两份词表必然会漂移，而漂移的方向一定是"漏掉某个急诊词"。
    """
    text = (symptom or "").strip()
    if not text:
        return False
    return not any(word in text or text in word for word in EMERGENCY_RED_LINE)


def next_question(
    candidates: Mapping[str, float],
    symptom_of: Mapping[str, Sequence[str]],
    *,
    asked: Sequence[str] = (),
    known: Sequence[SymptomEvidence] = (),
) -> str | None:
    """挑下一个最有鉴别力的问题（返回症状名，None 表示没有可问的了）。

    判据：|有这个症状的候选数 - 没有的候选数| 最小，即把当前候选集切得最均匀的那个。
    三个约定：
    - 正例只数**当前候选集里**的候选：关联表里那些已被淘汰的候选不该影响切分；
    - 不看分数：低分候选正是最需要被区分开的，按分数加权会让选题偏向已领先者；
    - 红线症状与已问过/已知的症状先排除，且**红线优先于均匀度**——宁可多问一轮，
      也不能由系统主动抛出"您胸痛吗"。
    同分时按症状名升序取定，保证同一会话的选择可复现。
    `asked` 约定存的是症状名（问题由症状生成），不是整句问话。
    """
    pool = set(candidates)
    if not pool:
        return None
    size = len(pool)
    skip = set(asked) | {item.symptom for item in known}
    balanced: list[tuple[int, str]] = []
    for symptom, linked in symptom_of.items():
        if symptom in skip or not is_safe_to_suggest(symptom):
            continue
        positive = sum(1 for candidate in set(linked) if candidate in pool)
        # |正例 - 负例| == |positive - (size - positive)| == |2 * positive - size|
        balanced.append((abs(2 * positive - size), symptom))
    if not balanced:
        return None
    return min(balanced)[1]


def suggest_hypothesis(
    candidates: Mapping[str, float],
    symptom_of: Mapping[str, Sequence[str]],
    *,
    asked: Sequence[str] = (),
) -> str | None:
    """抛出一个试探性假设（症状名），None 表示没有安全的可建议症状。

    与 `next_question` 的区别是**目标不同**：这里不是"最能劈开候选集"，而是"最能把
    当前领先的候选问实"——取关联候选**分数之和**最大的症状，按分数押注在当前最被
    证据支持的方向上。两者配合使用：先 `suggest_hypothesis` 确认领先候选，问不出来
    再用 `next_question` 做鉴别。

    红线症状同样排除（这是所有建议路径的硬约束）；已被 `asked` 问过的不再重复抛。
    想同时排除"已知症状"，调用方应把它们放进 `asked`，或改用 `next_question`。
    对外措辞由调用方加"我猜…不是请直接说"的软化语：本函数只负责选，不负责说。
    """
    if not candidates:
        return None
    skip = set(asked)
    ranked: list[tuple[float, str]] = []
    for symptom, linked in symptom_of.items():
        if symptom in skip or not is_safe_to_suggest(symptom):
            continue
        # 取负号是为了用 min() 同时完成"分数最大"与"同分按名字升序"两件事。
        mass = sum(candidates[candidate] for candidate in set(linked) if candidate in candidates)
        ranked.append((-mass, symptom))
    if not ranked:
        return None
    return min(ranked)[1]


def explain(scores: Mapping[str, float], evidence: Sequence[SymptomEvidence]) -> list[str]:
    """人类可读的打分明细，用于调试与病历留档。

    为什么证据与候选分两段列：本函数拿不到 `links`，无法知道哪条证据打到了哪个候选，
    硬猜会把病历写错（医疗留档宁可少写也不能写错），所以照实分成"候选得分"与
    "计入的证据"两段。
    为什么这里**不调** `tie_breaker`：明细要的是确定性快照，同一个状态两次导出必须一致，
    不该因为再问一次模型而变化。
    """
    lines: list[str] = []
    if scores:
        lines.append(
            "候选得分（高→低）：" + "，".join(f"{name} {value:+.2f}" for name, value in rank(scores))
        )
    else:
        lines.append("候选为空：没有可解释的候选得分。")
    if not evidence:
        lines.append("证据为空：候选分数未被任何证据改动。")
        return lines
    lines.append("计入的证据：")
    for item in evidence:
        weight = weight_of(item.source)
        note = "（红线规则，权重 0，不计入候选打分）" if item.source is EvidenceSource.RULE else ""
        lines.append(
            f"  {item.symptom} {'阳性' if item.present else '阴性'}"
            f" 来源={item.source.value} 权重={weight:+.2f} 第 {item.turn} 轮{note}"
        )
    return lines


def _context_of(evidence: Sequence[SymptomEvidence]) -> str:
    """给兜底回调的一句话上下文：把已有证据压成人类可读摘要，供模型仲裁时参考。"""
    if not evidence:
        return "尚无证据"
    parts = [f"{item.symptom}（{'有' if item.present else '无'}，{item.source.value}）" for item in evidence]
    return "；".join(parts)


def _break_tie(
    scores: dict[str, float],
    evidence: Sequence[SymptomEvidence],
    tie_breaker: Callable[[str, Sequence[str]], str],
    tie_epsilon: float,
) -> None:
    """并列时请兜底回调裁决，并把胜出者抬到严格第一（**原地修改** `scores`）。

    回调选中的若正好是同分时名次最前的那个，则**不动分数**：结果与"没兜底"完全一致，
    没必要往留档里写一个假差值。

    为什么用 `math.nextafter` 而不是"加一个小的 epsilon"：分数可能是 1e10 量级，
    加 1e-9 会被浮点直接吸收（结果仍相等），并列根本没被打破。`nextafter` 给出的是
    "刚好大一点"的可表示值，任何量级下都成立，也避免把一个人为的差值写进分数明细。
    """
    ordered = rank(scores)
    if len(ordered) < 2:
        return
    top_name, top_score = ordered[0]
    tied = [name for name, value in ordered if top_score - value < tie_epsilon]
    if len(tied) < 2:
        return
    # 兜底回调通常是一次 LLM 往返：超时/报错时保持原分数。宁可不打破并列，
    # 也不要因为一次外部调用失败丢掉整轮问诊（并列会在下一轮由新证据自己打破）。
    with suppress(Exception):
        picked = tie_breaker(_context_of(evidence), tied)
        if picked in tied and picked != top_name:
            scores[picked] = math.nextafter(top_score, math.inf)
