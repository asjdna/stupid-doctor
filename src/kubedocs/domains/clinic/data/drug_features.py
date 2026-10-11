"""药物特征库（样例数据）：用于"记不清药名"时按特征检索。

生产可替换为公开结构化药品数据（丁香园 / 无码科技 / BCP 药数 API），
字段扩充为：适应症、用法用量、禁忌、不良反应等。
"""

# 每条药物含：通用名 + 可回忆特征（用途/颜色/形状/服用时间/类别）
DRUGS = [
    {
        "name": "硝苯地平",
        "purpose": "降血压",
        "color": "黄色",
        "shape": "圆片",
        "time": "每日",
        "class": "降压药",
    },
    {
        "name": "氨氯地平",
        "purpose": "降血压",
        "color": "白色",
        "shape": "长条",
        "time": "每日",
        "class": "降压药",
    },
    {
        "name": "二甲双胍",
        "purpose": "降血糖",
        "color": "白色",
        "shape": "圆片",
        "time": "随餐",
        "class": "降糖药",
    },
    {
        "name": "阿司匹林",
        "purpose": "防血栓",
        "color": "白色",
        "shape": "圆片",
        "time": "每日",
        "class": "抗凝药",
    },
    {
        "name": "美托洛尔",
        "purpose": "降心率/治心慌",
        "color": "白色",
        "shape": "圆片",
        "time": "每日",
        "class": "β受体阻滞剂",
    },
    {
        "name": "阿托伐他汀",
        "purpose": "降血脂",
        "color": "白色",
        "shape": "长条",
        "time": "每日",
        "class": "降脂药",
    },
]

# 可检索的特征字段
FEATURE_FIELDS = ["purpose", "color", "shape", "time", "class"]

# 口语 -> 特征值（用于从患者描述中抽取可检索特征，解决"记不清药名"）
_COLOR_WORDS = {"白色": "白色", "黄色": "黄色", "红色": "红色", "蓝色": "蓝色", "绿色": "绿色"}
_SHAPE_WORDS = {"圆片": "圆片", "圆粒": "圆片", "圆的": "圆片", "长条": "长条", "胶囊": "胶囊"}
_PURPOSE_WORDS = {
    "降血压": "降血压",
    "降压": "降血压",
    "降血糖": "降血糖",
    "降糖": "降血糖",
    "降血脂": "降血脂",
    "降脂": "降血脂",
    "防血栓": "防血栓",
    "治心慌": "降心率/治心慌",
    "心慌": "降心率/治心慌",
    "降心率": "降心率/治心慌",
}
_TIME_WORDS = {"随餐": "随餐", "饭后": "饭后", "每日": "每日", "一天一次": "每日", "睡前": "睡前"}


def extract_features(text: str) -> dict:
    """从患者口语描述中抽取可检索特征（用途/颜色/形状/时间）。

    只返回能确定命中的字段，未提到的不返回。供「记不清药名」场景按特征召回。
    """
    feats = {}
    for w, v in _COLOR_WORDS.items():
        if w in text:
            feats["color"] = v
            break
    for w, v in _SHAPE_WORDS.items():
        if w in text:
            feats["shape"] = v
            break
    for w, v in _PURPOSE_WORDS.items():
        if w in text:
            feats["purpose"] = v
            break
    for w, v in _TIME_WORDS.items():
        if w in text:
            feats["time"] = v
            break
    return feats


def search_by_features(**features: str) -> list:
    """按特征检索候选药物（模糊匹配）。

    用法：search_by_features(color="白色", purpose="降血压")
    返回命中的药物列表。
    """
    results = []
    for drug in DRUGS:
        matched = True
        for key, val in features.items():
            if key in drug and val not in drug[key]:
                matched = False
                break
        if matched:
            results.append(drug)
    return results
