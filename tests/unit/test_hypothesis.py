"""假设-检验打分引擎的单元测试，钉住三个契约：

1. **权重表数值**：建议性偏倚折扣（CONFIRMED 是 VOLUNTEERED 的一半）与否定降权是
   产品决策，逐项断言而非区间判断——数值被改动必须当场失败；
2. **选题判据**：`next_question` 取 |正例 - 负例| 最小的症状；
3. **红线保护**：红线症状的切分再均匀也不许被建议路径选中。

纯内存、纯标准库，不联网、不调真实模型（`tie_breaker` 全是本地假回调）。
"""

from collections.abc import Sequence

import pytest

from kubedocs.agent.hypothesis import (
    WEIGHTS,
    explain,
    is_safe_to_suggest,
    leading_margin,
    next_question,
    rank,
    score,
    should_stop,
    suggest_hypothesis,
    weight_of,
)
from kubedocs.agent.state import EvidenceSource, SymptomEvidence
from kubedocs.domains.clinic.redline import EMERGENCY_RED_LINE


def _ev(
    symptom: str,
    source: EvidenceSource,
    *,
    present: bool = True,
    turn: int = 1,
) -> SymptomEvidence:
    """构造一条证据。刻意不填 `weight`：打分必须自己回表查权重，而不是信这个字段。"""
    return SymptomEvidence(symptom=symptom, present=present, source=source, turn=turn)


# --------------------------------------------------------------------- 权重表
def test_权重表逐项正确():
    """五个来源的权重逐个断言：这些数字是产品决策，被改动必须立刻失败。"""
    assert WEIGHTS[EvidenceSource.VOLUNTEERED] == 1.0
    assert WEIGHTS[EvidenceSource.CONFIRMED] == 0.5
    assert WEIGHTS[EvidenceSource.NEGATED] == -0.5
    assert WEIGHTS[EvidenceSource.NEGATED_AFTER_SUGGESTION] == -0.25
    assert WEIGHTS[EvidenceSource.RULE] == 0.0
    # 权重表必须覆盖枚举全集：新增来源漏配权重要在单测炸掉，而不是线上静默得 0
    assert set(WEIGHTS) == set(EvidenceSource)
    for source in EvidenceSource:
        assert weight_of(source) == WEIGHTS[source]


def test_建议后确认的权重是自发的一半():
    """CONFIRMED 只有 VOLUNTEERED 的一半——这就是"建议性偏倚降权"的落地方式。"""
    assert weight_of(EvidenceSource.CONFIRMED) == weight_of(EvidenceSource.VOLUNTEERED) / 2
    links = {"心慌": ["甲"]}
    assert score([_ev("心慌", EvidenceSource.VOLUNTEERED)], links) == {"甲": 1.0}
    assert score([_ev("心慌", EvidenceSource.CONFIRMED)], links) == {"甲": 0.5}


def test_否定证据按负半权计入():
    """用户自发否认按 -0.5 计入（反证降权）；系统建议后否认更弱，只有 -0.25。"""
    assert weight_of(EvidenceSource.NEGATED) == -0.5
    links = {"胸闷": ["心梗"]}
    assert score([_ev("胸闷", EvidenceSource.NEGATED, present=False)], links) == {"心梗": -0.5}
    weaker = score([_ev("胸闷", EvidenceSource.NEGATED_AFTER_SUGGESTION, present=False)], links)
    assert weaker == {"心梗": -0.25}


# --------------------------------------------------------------------- 打分累加
def test_打分累加与手算一致():
    """手算对照：同症状多证据累加、跨症状叠加、RULE 不改分。

    links: 发热 -> [甲, 乙]；咳嗽 -> [甲]；胸痛 -> [丙]
    甲 = 发热(自发 +1.0) + 咳嗽(建议后确认 +0.5) + 咳嗽(自发否认 -0.5) = +1.0
    乙 = 发热(自发 +1.0) = +1.0；丙 = 胸痛(红线规则 +0.0) = 0.0
    "咳嗽"两条证据都打在"甲"上，故本条同时覆盖同症状多证据累加。
    """
    links = {"发热": ["甲", "乙"], "咳嗽": ["甲"], "胸痛": ["丙"]}
    evidence = [
        _ev("发热", EvidenceSource.VOLUNTEERED),
        _ev("咳嗽", EvidenceSource.CONFIRMED),
        _ev("咳嗽", EvidenceSource.NEGATED, present=False),
        _ev("胸痛", EvidenceSource.RULE, turn=2),
    ]
    assert score(evidence, links) == {"甲": 1.0, "乙": 1.0, "丙": 0.0}


def test_links缺键只跳过不报错():
    """关联表里没有的症状直接跳过：症状词典比知识库大，缺键是常态而非异常。"""
    links = {"发热": ["甲"]}
    evidence = [
        _ev("发热", EvidenceSource.VOLUNTEERED),
        _ev("知识库没收录的症状", EvidenceSource.VOLUNTEERED),
    ]
    assert score(evidence, links) == {"甲": 1.0}


def test_空证据返回空表():
    """没有证据就没有分数：返回空 dict，不是"每个候选 0 分"（本函数根本不知道完整候选集）。"""
    assert score([], {"发热": ["甲"]}) == {}


# ----------------------------------------------------------------- 排名与领先度
def test_排名降序且同分按名字定序():
    """排序必须确定性：同分按候选名的**码位**升序（乙 U+4E59 < 甲 U+7532），
    否则同一份证据在不同进程里会给出不同的 Top1，问诊结果与回归测试都会抖动。
    """
    assert rank({"乙": 1.0, "甲": 1.0, "丙": 2.0}) == [("丙", 2.0), ("乙", 1.0), ("甲", 1.0)]
    assert rank({"甲": 1.0, "乙": 0.5}, top=1) == [("甲", 1.0)]
    assert rank({"甲": 1.0}, top=0) == []
    assert rank({}) == []


def test_领先度是前两名分差():
    """leading_margin = Top1 - Top2；候选少于 2 个返回 0.0，不抛 IndexError。"""
    assert leading_margin({"甲": 2.0, "乙": 0.5, "丙": -1.0}) == 1.5
    assert leading_margin({"甲": 1.0}) == 0.0
    assert leading_margin({}) == 0.0


# --------------------------------------------------------- 停止条件（三个条件各一条）
def test_停止条件之一_用户主动要求停止():
    """用户说"不用了"立刻停：优先级最高，领先度再高也要看用户意愿。"""
    stop, reason = should_stop({"甲": 5.0, "乙": 0.0}, 1, max_turns=6, margin=1.0, user_stop=True)
    assert stop is True
    assert "用户主动" in reason


def test_停止条件之二_达到轮次上限():
    """轮次用尽就停：领先度不够也不许无限追问，问诊必须收敛。"""
    stop, reason = should_stop({"甲": 0.4, "乙": 0.3}, 6, max_turns=6, margin=1.0)
    assert stop is True
    assert "轮次上限" in reason


def test_停止条件之三_领先度达标():
    """Top1 拉开足够差距就停：此时继续问只增加成本与建议性偏倚。"""
    stop, reason = should_stop({"甲": 2.0, "乙": 1.0}, 2, max_turns=6, margin=1.0)
    assert stop is True
    assert "领先度达标" in reason


def test_三个条件都没满足时继续追问():
    """分差不够、轮次没用完、用户没喊停 -> 继续问，并给出可留档的原因串。"""
    stop, reason = should_stop({"甲": 1.2, "乙": 1.0}, 2, max_turns=6, margin=1.0)
    assert stop is False
    assert reason


def test_空候选时停止判定不炸():
    """一开局候选为空（比如检索还没跑）也要给确定答案：不按领先度停，按轮次停。"""
    assert should_stop({}, 0) == (False, "继续追问")
    assert should_stop({}, 6)[0] is True


# --------------------------------------------------------------------- 红线保护
@pytest.mark.parametrize("word", EMERGENCY_RED_LINE)
def test_红线词一律不可由系统建议(word: str):
    """EMERGENCY_RED_LINE 里的每个真实红线词都必须返回 False——词表是唯一真相源。"""
    assert is_safe_to_suggest(word) is False


def test_红线词嵌在症状名里也算红线():
    """子串匹配：建议用的症状名常是"突发的胸痛伴出汗"这类长串，红线词嵌在里面同样禁问。"""
    assert is_safe_to_suggest("突发的胸痛伴出汗") is False
    assert is_safe_to_suggest("一阵一阵的喘不上气") is False


def test_症状名是红线词片段时保守禁问():
    """反向包含也拦：'看不清' ⊂ '突然看不清'，临床上是同一件事，宁可少建议一个。"""
    assert is_safe_to_suggest("看不清") is False
    assert is_safe_to_suggest("") is False


def test_普通症状可以建议():
    """非红线的常见症状正常放行（'胸闷'不在红线词表里，不要误伤）。"""
    assert is_safe_to_suggest("发热") is True
    assert is_safe_to_suggest("胸闷") is True


# --------------------------------------------------------------- 选题：切得最均匀
def test_选题取切分最均匀的症状():
    """用具体数字验证判据 |正例 - 负例| 最小，而不是按插入顺序取第一个。

    候选 = 甲/乙/丙/丁（n = 4）
      症状A -> 甲,乙,丙     -> 正 3 / 负 1 -> 2
      症状B -> 甲,乙,丙,丁  -> 正 4 / 负 0 -> 4
      症状Z -> 甲,乙       -> 正 2 / 负 2 -> 0   唯一对半切的，必须选它
    症状Z 放在字典最后、名字也排最靠后：只有它胜出才能证明实现是照数字挑的。
    """
    candidates = {"甲": 0.9, "乙": 0.8, "丙": 0.2, "丁": 0.1}
    symptom_of = {
        "症状A": ["甲", "乙", "丙"],
        "症状B": ["甲", "乙", "丙", "丁"],
        "症状Z": ["甲", "乙"],
    }
    assert next_question(candidates, symptom_of) == "症状Z"


def test_选题同分时按症状名定序():
    """两个症状一样均匀时取名字升序者：选题必须可复现，否则同一会话两轮问不同问题。"""
    candidates = {"甲": 1.0, "乙": 1.0, "丙": 1.0, "丁": 1.0}
    symptom_of = {"症状B": ["甲", "乙"], "症状A": ["丙", "丁"]}
    assert next_question(candidates, symptom_of) == "症状A"


def test_选题只数当前候选集里的候选():
    """关联表提到但已被淘汰的候选不计入正例数，否则切分会偏向"已经出局的病"。"""
    candidates = {"甲": 1.0, "乙": 1.0}
    symptom_of = {
        "症状A": ["甲", "丙", "丁"],  # 正例只有 甲 -> |1-1| = 0
        "症状B": ["甲", "乙", "丙"],  # 正例 甲,乙 -> |2-0| = 2
    }
    assert next_question(candidates, symptom_of) == "症状A"


def test_选题跳过已问过和已知的症状():
    """asked 与 known 都要排除：重复追问既浪费轮次又惹人烦；全排完则返回 None。"""
    candidates = {"甲": 1.0, "乙": 1.0, "丙": 1.0, "丁": 1.0}
    symptom_of = {"症状A": ["甲", "乙"], "症状B": ["丙", "丁"]}
    assert next_question(candidates, symptom_of) == "症状A"
    assert next_question(candidates, symptom_of, asked=["症状A"]) == "症状B"
    known = [_ev("症状A", EvidenceSource.VOLUNTEERED)]
    assert next_question(candidates, symptom_of, known=known) == "症状B"
    assert next_question(candidates, symptom_of, asked=["症状A", "症状B"]) is None


def test_选题绝不为均匀度牺牲红线保护():
    """红线症状哪怕切分完美（|2-2| = 0）也必须跳过，退而求其次问次优的安全症状。"""
    candidates = {"甲": 1.0, "乙": 1.0, "丙": 1.0, "丁": 1.0}
    symptom_of = {"胸痛": ["甲", "乙"], "症状A": ["甲", "乙", "丙"]}
    assert is_safe_to_suggest("胸痛") is False
    assert next_question(candidates, symptom_of) == "症状A"


# ------------------------------------------------------------------- 试探性假设
def test_假设建议押在分数之和最高的症状上():
    """assume 的目标是"问实领先候选"：症状关联候选的分数之和最大者胜出。

    症状A -> 心梗(2.0) = 2.0；症状B -> 胃炎(0.5) = 0.5；症状C -> 心梗+胃炎 = 2.5
    """
    candidates = {"心梗": 2.0, "胃炎": 0.5}
    symptom_of = {"症状A": ["心梗"], "症状B": ["胃炎"], "症状C": ["心梗", "胃炎"]}
    assert suggest_hypothesis(candidates, symptom_of) == "症状C"


def test_假设建议跳过红线与已问过的():
    """支持度最高的症状若是红线照样跳过；已问过的不再抛；无候选或无可用症状返回 None。"""
    candidates = {"心梗": 2.0}
    symptom_of = {"胸痛": ["心梗"], "出汗": ["心梗"]}
    assert suggest_hypothesis(candidates, symptom_of) == "出汗"
    assert suggest_hypothesis(candidates, symptom_of, asked=["出汗"]) is None
    assert suggest_hypothesis({}, symptom_of) is None


# --------------------------------------------------------------------- LLM 兜底
def test_并列时才调兜底回调并把胜出者抬到第一():
    """Top1/Top2 分差 < tie_epsilon 才调 tie_breaker；它选中谁，谁就成为严格第一。

    这里让它选同分时的第二名（同分按码位排：乙在前），这样"被抬到第一"才可观测。
    """
    calls: list[tuple[str, tuple[str, ...]]] = []

    def breaker(context: str, tied: Sequence[str]) -> str:
        calls.append((context, tuple(tied)))
        return "甲"

    links = {"发热": ["甲", "乙"]}
    scores = score([_ev("发热", EvidenceSource.VOLUNTEERED)], links, tie_breaker=breaker)
    assert len(calls) == 1
    assert "发热" in calls[0][0]
    assert set(calls[0][1]) == {"甲", "乙"}
    assert scores["甲"] > scores["乙"]
    assert rank(scores)[0][0] == "甲"


def test_兜底回调选中本就领先者时不改分数():
    """回调选了同分时名次第一的那个 -> 分数原样保留：没必要往留档里写一个假差值。"""
    links = {"发热": ["甲", "乙"]}
    scores = score([_ev("发热", EvidenceSource.VOLUNTEERED)], links, tie_breaker=lambda context, tied: "乙")
    assert scores == {"甲": 1.0, "乙": 1.0}
    assert rank(scores)[0][0] == "乙"


def test_分差不小就不打扰兜底回调():
    """领先度明显时不调 LLM：兜底只是并列时的仲裁，不该变成每轮都花钱的默认路径。"""

    def breaker(context: str, tied: Sequence[str]) -> str:
        raise AssertionError("分差明显时不该调用 tie_breaker")

    links = {"发热": ["甲"], "咳嗽": ["乙"]}
    evidence = [_ev("发热", EvidenceSource.VOLUNTEERED), _ev("咳嗽", EvidenceSource.NEGATED, present=False)]
    assert score(evidence, links, tie_breaker=breaker) == {"甲": 1.0, "乙": -0.5}


def test_兜底回调返回野名字或抛异常都不影响打分():
    """模型幻觉出一个不存在的候选、或者直接超时，打分必须保持可用（宁可不打破并列）。"""
    links = {"发热": ["甲", "乙"]}
    evidence = [_ev("发热", EvidenceSource.VOLUNTEERED)]
    assert score(evidence, links, tie_breaker=lambda context, tied: "不存在的候选") == {"甲": 1.0, "乙": 1.0}

    def boom(context: str, tied: Sequence[str]) -> str:
        raise RuntimeError("模型超时")

    assert score(evidence, links, tie_breaker=boom) == {"甲": 1.0, "乙": 1.0}


# ------------------------------------------------------------------------- 明细
def test_打分明细可读且覆盖证据():
    """明细要能直接贴进病历：候选按分降序，每条证据都有来源/权重/轮次，RULE 标注不计分。"""
    evidence = [
        _ev("发热", EvidenceSource.VOLUNTEERED, turn=1),
        _ev("胸痛", EvidenceSource.RULE, turn=2),
    ]
    text = "\n".join(explain({"甲": 1.0, "乙": -0.5}, evidence))
    assert "甲 +1.00" in text
    assert text.index("甲") < text.index("乙")
    assert "volunteered" in text and "第 1 轮" in text
    assert "不计入候选打分" in text


def test_空输入的所有入口都有确定行为():
    """空候选/空证据是常态（第一轮、检索没命中），每个入口都必须给确定返回值而不是抛异常。"""
    assert score([], {}) == {}
    assert score([_ev("发热", EvidenceSource.VOLUNTEERED)], {}) == {}
    assert next_question({}, {"发热": ["甲"]}) is None
    assert next_question({"甲": 1.0}, {}) is None
    assert suggest_hypothesis({}, {"发热": ["甲"]}) is None
    assert explain({}, [])  # 至少一行说明，不返回空列表让调用方去猜
