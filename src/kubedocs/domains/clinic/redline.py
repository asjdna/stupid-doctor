"""危险信号关键词表 + 三维门控（`is_reportable`）：**提供线索，不下判断**。

## 它在系统里的位置

| 角色 | 是谁 |
|---|---|
| **判断者** | 大模型：按病症 + 严重程度判 `red_flags` / `urgency`，多角度扫描（见 `prompts/system.md` R1） |
| **线索提供者** | 本模块：命中词 + 三维取值进参考标准面板，随系统提示交给模型 |
| **底线** | `clinical/safety.py` 的三条确定性不变量（本模块不在其中——它不执行任何动作） |

`check()` 的命中**不是**急症结论：它只说明"这句话里出现了某个词"。三条不相干的事因此分开：

- **命中了要不要紧**：由模型结合三维取值与整句话判断，本模块不覆盖 `urgency`、不短路流程；
- **能不能作为判断依据**：`polarity.is_reportable` 的三维门控说了算（见下）；
- **命中与判断不一致时**：留一条可审计的痕迹（`keyword_hit_conflict`），但**不**替模型改结论。

## 不能"命中词表就算数"

红线词只说明"这句话里出现了胸痛"，不说明**这是谁、什么时候**的胸痛：

| 输入 | 只看词表的后果 |
|---|---|
| 我**老公**胸痛，我没事 | 把**患者本人**送进急诊 |
| 我**去年**胸痛过一次，现在好了 | 按现症处理已缓解的旧症状 |
| 我**没有**胸痛 | 否认的症状被当成主诉 |

接上 `polarity.is_reportable` 的三维门控（状态 ∈ {present, possible} + 经验者本人 +
时间性现在）后，这三类都不再触发；而「我现在胸痛」「胸痛，喘不上气」照常触发。
方向上是**只收紧明显错的**：无法归因（缺省本人/现在）时行为与词表匹配一致——
漏报比多问一句危险得多。同一套语义在模型侧由提示词要求（别人 / 既往 / 假设不算本人当前急症）。
"""

from __future__ import annotations

from dataclasses import dataclass

from kubedocs.domains.clinic.polarity import assertion, is_reportable

# 口语危险信号词：命中只是"这句话里出现了这个说法"，是不是本人当前的急症由三维门控与模型判断。
# 书面医学术语（胸骨后疼痛 / 呕血 / 端坐呼吸等）不在词表内，靠模型读整句话兜底。
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
    """一条危险信号命中：词 + 三维判定（留档便于审计与回归）。"""

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
    """危险信号关键词 + 三维门控。**无状态、无 IO、可直接单测**。"""

    name = "risk_guard"

    def check(self, text: str) -> RedLineHit | None:
        """命中返回 `RedLineHit`，否则 None。

        只有**可上报**的命中才算数：被否认的（无胸痛）、他人的（我老公胸痛）、
        既往的（去年胸痛现在好了）、假设的（如果胸痛怎么办）都不算——它们是"系统看过这个词、
        判断不该上报"的`reportable=False` 留痕，见 `evidence.red_line_from_guard`。
        """
        for word in EMERGENCY_RED_LINE:
            if word in text and is_reportable(text, word):
                a = assertion(text, word)
                return RedLineHit(
                    word=word, status=a["status"], experiencer=a["experiencer"], temporality=a["temporality"]
                )
        return None

    def explain(self, text: str) -> list[str]:
        """诊断用：列出文本里出现的全部危险信号词及其三维判定。

        调参或复现误报时用它——只看 `check()` 的结果无法回答"为什么这条没触发"，
        而这个问题在回归里必然会被问。命中明细也会进参考标准面板。
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
    """关键词命中时的二选一提示（**不强制短路**：把选择权交回患者）。

    只给**对照臂**（`agent/kernel.py` 的确定性图，不在请求路径上）用：
    请求路径上的急诊判断与动作由模型给出、由安全门兜底（见 `clinical/safety.py`）。
    """
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
