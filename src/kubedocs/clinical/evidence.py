"""证据装配：三路信号 → 一段可直接插进提示词的上下文（方案 §2）。

| 来源 | 角色 | 谁算 | 能不能决定结论 |
|---|---|---|---|
| 急诊红线 | **证据**（按决定 2b，不是闸门） | 规则，毫秒级 | 否。但为阳性时提示词**强制**给紧急指引 |
| 分诊 | 辅助信号 | 检索式 | 否。它只提供候选与置信度 |
| RAG | **唯一知识来源** | 可替换后端 | 检查/用药建议只能引用它 |

## 为什么红线不在这里"直接短路到急诊"

按你的决定 2b：判断主体是大模型，红线只帮它找对答案与紧急处理。所以本模块
**不算**紧急档、**不拦截**流程，只把红线做成证据包里**最高优先级**的一条。

工程上的补偿（不是偷偷把它改回闸门）：

1. 红线在 `render()` 里排在最前面并标注 **MUST**；
2. 提示词对它的处理是硬性条款，不是建议语气；
3. `pipeline` 会**检测**"红线阳性但模型给了非紧急档"并把它作为**契约问题**上报——
   模型可以决定，但决定与证据冲突时**必须留下痕迹**，不能静默过去。
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kubedocs.clinical.contracts import DepartmentAdvice
from kubedocs.clinical.rag import RetrievalBackend

EvidenceKind = Literal["red_line", "triage", "knowledge"]

_KNOWLEDGE_BODY_CHARS = 400
"""进上下文的单条正文上限。取 400 是因为它够判定"这段讲不讲这个问题"，
又不至于让 10 条命中把上下文塞满（`Settings.max_context_chars` 另有一层总量控制）。"""


class EvidenceItem(BaseModel):
    """一条证据。`id` 是**模型引用它的凭据**（如 `K3`），所以必须稳定且短。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    text: str
    score: float = 0.0
    source_url: str = ""


class RedLineSignal(BaseModel):
    """红线信号。字段取自 `domains.clinic.redline.RedLineHit`（三维判定）。"""

    model_config = ConfigDict(extra="forbid")

    word: str
    category: str = ""
    experiencer: str = ""
    temporality: str = ""


class EvidenceBundle(BaseModel):
    """进模型上下文的全部依据 + 这一轮检索链路上出过的问题。"""

    model_config = ConfigDict(extra="forbid")

    items: list[EvidenceItem] = Field(default_factory=list)
    red_line: RedLineSignal | None = None
    triage: list[DepartmentAdvice] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    """检索降级/后端异常。**必须**随证据一起进上下文：让模型知道"这次依据不全"，
    比让它以为"知识库里就是没有"要好得多。"""

    @property
    def red_line_positive(self) -> bool:
        return self.red_line is not None

    def ids(self) -> set[str]:
        return {item.id for item in self.items}

    def render(self, *, max_items: int = 10) -> str:
        """渲染成提示词里的一段。编号与 `EvidenceItem.id` 一致，模型引用它们即可。"""
        lines: list[str] = []

        if self.red_line is not None:
            # 排在最前 + MUST：这是安全关键路径，位置本身就是信息。
            lines.append(
                "【红线信号｜MUST 处理】"
                f"命中词「{self.red_line.word}」"
                f"（类别 {self.red_line.category or '未标注'}，"
                f"主体 {self.red_line.experiencer or '未判定'}，"
                f"时间性 {self.red_line.temporality or '未判定'}）。"
                "本次回答**必须**包含明确的紧急处理指引。"
            )

        if self.triage:
            ranked = "、".join(f"{d.name}（{d.confidence:.2f}）" for d in self.triage[:3])
            lines.append(f"【分诊信号｜仅供参考】候选科室：{ranked}")

        knowledge = [i for i in self.items if i.kind == "knowledge"][:max_items]
        if knowledge:
            lines.append("【知识依据｜只能据此给检查与用药建议】")
            lines.extend(f"[{i.id}] {i.text}" for i in knowledge)
        else:
            lines.append(
                "【知识依据】本次**没有检索到任何资料**。"
                "因此不得给出检查建议或用药建议——检索不到就不给，不要凭空补。"
            )

        if self.degraded:
            lines.append("【检索链路问题】" + "；".join(self.degraded))

        return "\n".join(lines)


class Guard(Protocol):
    """红线判定器。与 `domains.clinic.redline.RiskGuard` 形状一致（便于直接注入）。"""

    def check(self, text: str) -> object | None: ...


class TriageEngine(Protocol):
    """分诊引擎：给问题与检索结果，产出候选科室。"""

    def rank(self, question: str, chunks: list[object]) -> list[DepartmentAdvice]: ...


def _red_line_signal(hit: object) -> RedLineSignal:
    """把 `RedLineHit` 读成信号。字段用 `getattr` 兜底：三维判定将来可能增字段，
    这里不该因为多/少一个字段就把整条链弄崩。"""
    return RedLineSignal(
        word=str(getattr(hit, "word", "") or ""),
        category=str(getattr(hit, "category", "") or ""),
        experiencer=str(getattr(hit, "experiencer", "") or ""),
        temporality=str(getattr(hit, "temporality", "") or ""),
    )


def build_evidence(
    question: str,
    *,
    backend: RetrievalBackend,
    top_k: int = 10,
    guard: Guard | None = None,
    triage: TriageEngine | None = None,
) -> EvidenceBundle:
    """跑一次完整的证据装配。**任何一路失败都不许让整轮崩**：

    - 检索抛异常 → 记进 `degraded`，证据包照样返回（模型会看到"这次没有依据"）；
    - 红线/分诊失败 → 同样记 `degraded`，不吞掉、也不放大。

    这个取舍的理由：问诊对话的中断成本远高于"这一轮依据不全"。
    """
    degraded: list[str] = []
    chunks: list[object] = []
    items: list[EvidenceItem] = []

    try:
        hits = backend.search(question, top_k=top_k)
    except Exception as e:  # noqa: BLE001 - 后端是注入的，任何失败都只降级不崩
        hits = []
        degraded.append(f"检索后端 {backend.name} 失败：{type(e).__name__}: {e}")

    for i, hit in enumerate(hits, start=1):
        body = (hit.body or "").strip().replace("\n", " ")
        items.append(
            EvidenceItem(
                id=f"K{i}",
                kind="knowledge",
                text=f"{hit.title} › {hit.breadcrumb}：{body[:_KNOWLEDGE_BODY_CHARS]}",
                score=hit.score,
                source_url=hit.source_url,
            )
        )
        chunks.append(hit)

    red_line: RedLineSignal | None = None
    if guard is not None:
        # 变量名刻意不叫 `hit`：上面那个循环已经用过它了。复用同名变量会让
        # mypy 的类型推断对不上（实测就是这么被抓出来的），也让人读错。
        red_hit: object | None = None
        try:
            red_hit = guard.check(question)
        except Exception as e:  # noqa: BLE001
            red_hit = None
            degraded.append(f"红线判定失败：{type(e).__name__}: {e}")
        if red_hit is not None:
            red_line = _red_line_signal(red_hit)

    departments: list[DepartmentAdvice] = []
    if triage is not None:
        try:
            departments = triage.rank(question, chunks)
        except Exception as e:  # noqa: BLE001
            degraded.append(f"分诊失败：{type(e).__name__}: {e}")

    return EvidenceBundle(items=items, red_line=red_line, triage=departments, degraded=degraded)
