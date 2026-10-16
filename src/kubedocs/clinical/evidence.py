"""证据装配：三路信号 → 一段可直接插进提示词的上下文。

| 来源 | 角色 | 能不能决定结论 |
|---|---|---|
| 急诊红线 | **证据**（不是闸门） | 否。但为阳性时提示词**强制**给紧急指引 |
| 分诊 | 辅助信号 | 否。它只提供候选与置信度 |
| RAG | **唯一知识来源** | 检查/用药建议只能引用它 |

判断主体是**大模型**，所以红线在这里**不算**紧急档、**不拦截**流程，
只作为证据包里**最高优先级**的一条。配套约束：

1. 提示词对它的处理是硬性条款，不是建议语气；
2. `pipeline` 会**检测**"红线阳性但模型给了非紧急档"并作为**契约问题**上报——
   模型可以决定，但决定与证据冲突时**必须留下痕迹**，不能静默过去。
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kubedocs.clinical.contracts import DepartmentAdvice
from kubedocs.clinical.rag import RetrievalBackend

EvidenceKind = Literal["red_line", "triage", "knowledge"]

_KNOWLEDGE_BODY_CHARS = 400
"""单条**命中片段**进上下文的正文上限。400 字够判定"这段讲不讲这个问题"，
又不至于让 10 条命中把上下文塞满（`Settings.max_context_chars` 另有一层总量控制）。"""

_KNOWLEDGE_PARENT_CHARS = 900
"""单条补充的**整节**（父块）正文上限，比命中片段宽得多。

为什么两条都要给：`text` 是**命中的那一句**（精确，但可能只有半句话），`context` 是它
所在的**整节**（读得到前后文）。只给前者，模型会拿着被切断的话答题；只给后者再截断，
就可能把命中那句整段截掉——因为命中位置在小节里的哪里是不确定的。
所以宁可每条多花几百字，也不要让"读了资料"变成"读了一句断章"。"""


class EvidenceItem(BaseModel):
    """一条证据。`id` 是**模型引用它的凭据**（如 `K3`），所以必须稳定且短。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    text: str
    """**命中的那一段**（子块，已截断到 `_KNOWLEDGE_BODY_CHARS`）。"""

    context: str = ""
    """命中块所属**整节**（父块）的正文，截断到 `_KNOWLEDGE_PARENT_CHARS`；空串 = 没有父块。

    与 `text` 是两种用途：`text` 回答"命中的是哪一句"，`context` 给模型读前后文。
    """

    score: float = 0.0
    source_url: str = ""

    source_id: str = ""
    """这条依据在**索引里的真实 id**（`chunk_id`，形如 `disease/咳嗽::0000`）。

    为什么必须留着：`search_docs` 工具返回给模型的就是这个 id，模型很自然地会拿它当
    "依据编号"写进 `evidence_ids`（实测如此）。只认我们自己分配的 `K#`，会把一条**真实
    命中**判成"编造依据"并丢掉整条建议——那是纯粹的误伤，而且看起来像模型在撒谎。
    校验因此接受两种写法：我们的 `K#`，或索引里的真实 `chunk_id`。
    """


class RedLineSignal(BaseModel):
    """红线信号。字段取自 `domains.clinic.redline.RedLineHit`（三维判定）。

    它现在是**给模型的线索**（不再是强制边）：`assertion` / `experiencer` / `temporality`
    一起进参考面板，由模型判断这是不是急症。`reportable` 记下三维门控的结论，
    而 `notes` 保留**全部出现处**的判定明细——包括"不触发"的那些，
    因为「我老公胸痛」「去年胸痛」为什么不触发，恰恰是模型最需要看到的解释。
    """

    model_config = ConfigDict(extra="forbid")

    word: str
    category: str = ""
    experiencer: str = ""
    temporality: str = ""
    assertion: str = ""
    """三维里的断言状态：present / absent / possible / hypothetical（取自 `RedLineHit.status`）。"""

    reportable: bool = True
    """三维门控是否认为它可作**患者本人当前的主诉**上报（False = 家属/既往/否定/假设）。"""

    notes: list[str] = Field(default_factory=list)
    """`RiskGuard.explain()` 的逐条明细：`触发/不触发 「词」 status=… exp=… tmp=…`。"""


class EvidenceBundle(BaseModel):
    """进模型上下文的全部依据 + 这一轮检索链路上出过的问题。"""

    model_config = ConfigDict(extra="forbid")

    items: list[EvidenceItem] = Field(default_factory=list)
    red_line: RedLineSignal | None = None
    """关键词扫描的命中结果（`domains.clinic.redline.RiskGuard`）。

    **它不进模型上下文，也不影响 `urgency`**（见 `render`）。保留它只为两件事：
    主判不可用时给一句保守的对话提示，以及"命中而模型未判急重症"时留一条不一致痕迹。
    """

    triage: list[DepartmentAdvice] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    """检索降级/后端异常。**必须**随证据一起进上下文：让模型知道"这次依据不全"，
    比让它以为"知识库里就是没有"要好得多。"""

    @property
    def red_line_positive(self) -> bool:
        """关键词命中。**只是"表里出现过某个词"，不等于"这是急重症"**。

        调用方不要拿它当医学判断用：急诊结论由模型的 `red_flags` 给出。
        """
        return self.red_line is not None

    def ids(self) -> set[str]:
        return {item.id for item in self.items}

    def render(self, *, max_items: int = 10) -> str:
        """渲染成提示词里的一段。编号与 `EvidenceItem.id` 一致，模型引用它们即可。

        **红线命中故意不渲染进提示词**：否则那行字会牵着模型走，
        "判断权交给模型"就是假的。判断依据由模型的 `red_flags` 给出。
        """
        lines: list[str] = []

        if self.triage:
            ranked = "、".join(f"{d.name}（{d.confidence:.2f}）" for d in self.triage[:3])
            lines.append(f"【分诊信号｜仅供参考】候选科室：{ranked}")

        knowledge = [i for i in self.items if i.kind == "knowledge"][:max_items]
        if knowledge:
            lines.append("【知识依据｜只能据此给检查与用药建议】")
            for i in knowledge:
                lines.append(f"[{i.id}] {i.text}")
                if i.context:
                    # 标注清楚哪一段是"整节"：模型引用编号时引的是这一节，
                    # 而它答题时依据的应当是整节，不是被截断的那一句。
                    lines.append(f"  （{i.id} 所在小节全文）{i.context}")
        else:
            lines.append(
                "【知识依据】本次**没有检索到任何资料**。"
                "因此不得给出检查建议或用药建议——检索不到就不给，不要凭空补。"
            )

        if self.degraded:
            lines.append("【检索链路问题】" + "；".join(self.degraded))

        return "\n".join(lines)


class Guard(Protocol):
    """红线判定器。与 `domains.clinic.redline.RiskGuard` 形状一致（便于直接注入）。

    `explain()` 是可选的：有它就能把"为什么没触发"也讲给模型（三维判定的明细），
    没有它只是线索少一层，判定链不受影响。
    """

    def check(self, text: str) -> object | None: ...

    def explain(self, text: str) -> list[str]: ...


class TriageEngine(Protocol):
    """分诊引擎：给问题与检索结果，产出候选科室。"""

    def rank(self, question: str, chunks: list[object]) -> list[DepartmentAdvice]: ...


def _red_line_signal(
    hit: object, *, reportable: bool = True, notes: list[str] | None = None
) -> RedLineSignal:
    """把 `RedLineHit` 读成信号。字段用 `getattr` 兜底：三维判定将来可能增字段，
    这里不该因为多/少一个字段就把整条链弄崩。

    `notes` 是 `RiskGuard.explain()` 的明细（含"不触发"的出现处）。它经 `getattr` 取
    ——`Guard` 协议只强制 `check()`，没有 `explain()` 的实现照样能用。
    """
    return RedLineSignal(
        word=str(getattr(hit, "word", "") or ""),
        category=str(getattr(hit, "category", "") or ""),
        experiencer=str(getattr(hit, "experiencer", "") or ""),
        temporality=str(getattr(hit, "temporality", "") or ""),
        assertion=str(getattr(hit, "status", "") or ""),
        reportable=bool(reportable),
        notes=list(notes or []),
    )


def red_line_from_guard(
    guard: Guard, question: str, *, degraded: list[str] | None = None
) -> RedLineSignal | None:
    """跑一次红线判定，返回 `RedLineSignal`（或 None）。**两个调用点共用这一份**。

    - `check()` 命中 ⇒ `reportable=True` 的信号（三维门控认为可作本人当前主诉）；
    - 只出现红线词但门控判定"不触发"（家属/既往/否定/假设）⇒ 仍然返回一个
      `reportable=False` 的信号，**明细照给**。"系统看到过这个词、但认为不该上报"
      本身就是要交给模型的线索：抹掉它等于让模型不知道系统担心过。
    - 完全没出现 ⇒ None（面板会如实写"未命中"）。

    `degraded` 传进来就记失败原因：判定失败**不许让整轮崩**，但也不能无声。
    """
    notes: list[str] = []
    red_hit: object | None = None
    try:
        red_hit = guard.check(question)
    except Exception as e:  # noqa: BLE001 - 注入的实现可能任何异常，都不该炸掉整条链
        if degraded is not None:
            degraded.append(f"红线判定失败：{type(e).__name__}: {e}")
    explain = getattr(guard, "explain", None)
    if callable(explain):
        try:
            notes = [str(x) for x in explain(question)]
        except Exception as e:  # noqa: BLE001 - 诊断信息取不到不影响判定
            if degraded is not None:
                degraded.append(f"红线明细取不到：{type(e).__name__}: {e}")
    if red_hit is not None:
        return _red_line_signal(red_hit, reportable=True, notes=notes)
    if notes:
        return RedLineSignal(word="", assertion="", reportable=False, notes=notes)
    return None


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
        # 整节正文与命中片段一并给出：前者是语境，后者是命中位置。两者相同（没有父块时）
        # 就不重复一遍——那会白占一倍的上下文。
        parent = (getattr(hit, "parent_text", "") or "").strip().replace("\n", " ")
        if parent[: len(body)] == body:
            parent = ""
        items.append(
            EvidenceItem(
                id=f"K{i}",
                kind="knowledge",
                text=f"{hit.title} › {hit.breadcrumb}：{body[:_KNOWLEDGE_BODY_CHARS]}",
                context=parent[:_KNOWLEDGE_PARENT_CHARS],
                score=hit.score,
                source_url=hit.source_url,
                # 索引里的真实 id：模型在 `search_docs` 结果里看到的就是它，
                # 校验要接受这个写法（见 `EvidenceItem.source_id`）。
                source_id=str(getattr(hit, "chunk_id", "") or ""),
            )
        )
        chunks.append(hit)

    red_line: RedLineSignal | None = None
    if guard is not None:
        # 明细（含**不触发**的出现处）一并取回：这是参考面板里最有价值的一条线索——
        # 「我老公胸痛」为什么不触发，只有 explain() 说得清，而模型需要这个解释。
        red_line = red_line_from_guard(guard, question, degraded=degraded)

    departments: list[DepartmentAdvice] = []
    if triage is not None:
        try:
            departments = triage.rank(question, chunks)
        except Exception as e:  # noqa: BLE001
            degraded.append(f"分诊失败：{type(e).__name__}: {e}")

    return EvidenceBundle(items=items, red_line=red_line, triage=departments, degraded=degraded)
