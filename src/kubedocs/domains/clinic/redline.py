"""急诊红线：确定性规则 + 三维门控（`is_reportable`）。**仅图路径**（`/chat`）。

## ⚠️ 两条路径的急诊设计是相反的

| 路径 | 急重症怎么判 | 本模块的作用 |
|---|---|---|
| **图路径**（`agent/kernel.py`，入口 `/chat`） | 本模块关键词命中即强制转急诊 | **是**那个强制边 |
| **临床管线**（`clinical/pipeline.py`，入口 `/consult`） | **模型**按病症 + 严重程度判 `red_flags` | **已退出判断链**：不进提示词、不影响 `urgency`、不触发横幅 |

改急诊相关代码前先确认走的是哪条。

## 图路径上红线必须是确定性规则，LLM 无权绕过

急诊识别是**安全属性**，不是检索质量。模型这次觉得危险、下次不觉得，这种不确定性
在医疗场景不可接受；规则短路还有两个好处：零延迟（不引入 LLM 往返）、
可审计（每个判断都能追到具体词与作用域）。

所以本模块放在图上一条**LLM 不可达的强制边**上：命中即走紧急分支，模型没有机会否决。
代价是**词表覆盖不到的表述会漏**（例如教科书级心梗表述），所以临床管线改由模型判断。

## 不能"命中词表就触发"

红线词只说明"这句话里出现了胸痛"，不说明**这是谁、什么时候**的胸痛：

| 输入 | 只看词表的后果 |
|---|---|
| 我**老公**胸痛，我没事 | 把**患者本人**送进急诊 |
| 我**去年**胸痛过一次，现在好了 | 按现症处理已缓解的旧症状 |
| 我**没有**胸痛 | 否认的症状被当成主诉 |

接上 `polarity.is_reportable` 的三维门控（状态 ∈ {present, possible} + 经验者本人 +
时间性现在）后，这三类都不再触发；而「我现在胸痛」「胸痛，喘不上气」照常触发。
方向上是**只收紧明显错的**：无法归因（缺省本人/现在）时行为与词表匹配一致——
漏掉急诊比误报急诊危险得多。
"""

from __future__ import annotations

from dataclasses import dataclass

from kubedocs.domains.clinic.polarity import assertion, is_reportable

# 口语红线词：命中即强制转急诊，零延迟。
# 书面医学术语（胸骨后疼痛 / 呕血 / 端坐呼吸等）不在词表内，靠模型画像分析兜底。
EMERGENCY_RED_LINE: tuple[str, ...] = (
    "胸痛",
    "胸口压着疼",
    "心口疼",
    "喘不上气",
    "呼吸困难",
    "憋得慌",
    "意识不清",
    "晕倒",
    "昏迷",
    "大出血",
    "吐了好多血",
    "口吐白沫",
    "半边身子不能动",
    "说话不清楚",
    "突然看不清",
)

# 用户在急诊二选一里选择"直接就医"的信号词
_WANTS_CARE = ("120", "急救", "救护车", "去医院", "急诊", "拨打", "叫车", "马上到")


@dataclass(frozen=True)
class RedLineHit:
    """一条红线命中：词 + 三维判定（留档便于审计与回归）。"""

    word: str
    status: str
    experiencer: str
    temporality: str

    def describe(self) -> str:
        return (
            f"「{self.word}」 status={self.status} "
            f"experiencer={self.experiencer} temporality={self.temporality}"
        )


class RiskGuard:
    """急诊红线判定。**无状态、无 IO、可直接单测**。"""

    name = "risk_guard"

    def check(self, text: str) -> RedLineHit | None:
        """命中返回 `RedLineHit`，否则 None。

        只有**可上报**的红线才算数：被否认的（无胸痛）、他人的（我老公胸痛）、
        既往的（去年胸痛现在好了）、假设的（如果胸痛怎么办）都不触发。
        """
        for word in EMERGENCY_RED_LINE:
            if word in text and is_reportable(text, word):
                a = assertion(text, word)
                return RedLineHit(
                    word=word, status=a["status"], experiencer=a["experiencer"], temporality=a["temporality"]
                )
        return None

    def explain(self, text: str) -> list[str]:
        """诊断用：列出文本里出现的全部红线词及其三维判定。

        调参或复现误报时用它——只看 `check()` 的结果无法回答"为什么这条没触发"，
        而这个问题在安全回归里必然会被问。
        """
        out: list[str] = []
        for word in EMERGENCY_RED_LINE:
            if word in text:
                a = assertion(text, word)
                mark = "触发" if is_reportable(text, word) else "不触发"
                out.append(
                    f"{mark} 「{word}」 status={a['status']} exp={a['experiencer']} tmp={a['temporality']}"
                )
        return out


def emergency_menu(word: str) -> str:
    """红线命中时的二选一提示（**不强制短路**：把选择权交回患者）。"""
    return (
        f"您描述的情况（「{word}」）可能比较紧急。请您选择：\n"
        "① 立即拨打 120 / 前往急诊；\n"
        "② 暂不就医，我继续帮您追问判断。"
    )


def emergency_advice(word: str) -> str:
    """用户选择"立即就医"时的回复。"""
    return (
        f"您提到「{word}」，这可能是比较紧急的情况，请您或家人立即拨打 120，"
        "或尽快到最近的医院急诊，不要耽搁。"
    )


def wants_emergency_care(text: str) -> bool:
    """用户在二选一里是否选了"直接就医"。"""
    return any(w in text for w in _WANTS_CARE)
