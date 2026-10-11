"""意图分类词表（样例数据）。

意图大类 -> 触发词（口语为主，覆盖老人/家长/初诊患者）。
"""

INTENT_CLASSES = {
    "症状自查": [
        "不舒服",
        "难受",
        "疼",
        "痛",
        "晕",
        "咳嗽",
        "发烧",
        "发热",
        "没劲",
        "心慌",
        "发慌",
        "胸闷",
        "拉肚子",
        "睡不着",
        "没胃口",
        "肿",
        "恶心",
    ],
    "就医导诊": [
        "挂号",
        "看哪个科",
        "挂什么科",
        "去哪看",
        "挂急诊",
        "看医生",
        "去哪个医院",
        "怎么挂号",
        "挂不上号",
    ],
    "用药咨询": [
        "吃什么药",
        "这药怎么吃",
        "药的副作用",
        "忘吃药",
        "记不清药名",
        "药名忘了",
        "药片",
        "一天吃几次",
        "饭前饭后",
    ],
    "健康科普": ["吃什么好", "要注意什么", "怎么预防", "饮食", "运动", "忌口", "能不能吃", "多喝水", "休息"],
}


def classify_intent(text: str) -> dict:
    """规则路意图分类：返回命中最多的大类与得分（0~1）。"""
    scores = {}
    for cls, words in INTENT_CLASSES.items():
        hit = sum(1 for w in words if w in text)
        if hit:
            scores[cls] = hit
    if not scores:
        return {"intent_class": "unknown", "confidence": 0.0}
    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    return {"intent_class": best, "confidence": scores[best] / total}
