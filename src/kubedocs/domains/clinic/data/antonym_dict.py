"""反义症状词典（对立词对库）：用于跨轮冲突检测。

场景：患者第一轮说"冷"，第二轮说"热"——同一槽位出现对立值，
需要识别为冲突并触发"追问澄清"（重点轮次，不占信息追问的轮次池）。
"""

# 每组是一个"同义症状集合"，组与组两两对立。
ANTONYM_GROUPS = [
    (["怕冷", "畏寒", "发冷", "手脚冰凉", "发寒"], ["怕热", "发热", "燥热", "发烫", "特别热", "热得慌"]),
    (["腹泻", "拉肚子", "窜稀", "拉稀"], ["便秘", "拉不出来", "大便干结"]),
    (["失眠", "睡不着", "睡不好", "半夜醒"], ["嗜睡", "总想睡", "睡不醒", "老犯困"]),
    (["没胃口", "吃不下", "食欲差", "不想吃饭"], ["胃口好", "吃得多", "食欲好", "饿得快"]),
    (["消瘦", "掉秤", "变瘦", "瘦了"], ["发胖", "长胖", "变胖", "胖了"]),
]


def is_antonym(a: str, b: str) -> bool:
    """判断两个症状短语是否对立（用于冲突检测）。"""
    for group_a, group_b in ANTONYM_GROUPS:
        a_in_a = any(s in a for s in group_a)
        a_in_b = any(s in a for s in group_b)
        b_in_a = any(s in b for s in group_a)
        b_in_b = any(s in b for s in group_b)
        if (a_in_a and b_in_b) or (a_in_b and b_in_a):
            return True
    return False


def find_conflict(tokens: list, new_tokens: list) -> list[tuple]:
    """在旧词元与新词元之间找出冲突对。

    tokens / new_tokens: list of dict，至少含 {'text': str, 'slot': str}
    返回 [(旧词元, 新词元), ...]
    """
    conflicts = []
    for old in tokens:
        for new in new_tokens:
            # 同一槽位才可能冲突（症状 vs 症状）
            if old.get("slot") == new.get("slot") and is_antonym(old["text"], new["text"]):
                conflicts.append((old, new))
    return conflicts
