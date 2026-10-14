"""分诊引擎（方案 §5）。目标指标：**准确率 ≥ 80%**。

## 现状与方向

现有实现是 `agent/hypothesis.py` 里手写权重的证据打分，实测准确率 **0.0**，
且评测只有 **5 条**用例。本模块给出的是**检索式分诊的 v0**：

> 用患者原话去检索知识卡片，命中卡片里若出现科室名（或其别名），就为该科室投票，
> 票数按检索分数加权；再叠加"原话里直接提到科室名"的强信号。

## 刻意做不到的事（写清楚，免得被误读为"已完成"）

- **v0 用"科室名出现在检索文本里"当判据**，这比"卡片带科室标签"弱。要提升到 80%，
  真正该做的是给知识卡片补**结构化科室标签**（原型 `medical-agent` 的 OpenCMKG 派生卡片
  已带科室/药物/疾病字段，可作为数据源），然后按标签投票——那时这个模块的
  `rank()` 实现会换掉，**协议不变**。
- **别名与层级映射表**目前只有极小的内置表（"心内科" ↔ "心血管内科" 这类），
  需要按真实金标集补。
- **本模块不宣称任何准确率**：在分诊金标扩到 ≥50 条并实测之前，
  任何"达到 80%"的说法都是编的（见方案 §5 的"度量先修"）。
"""

from __future__ import annotations

from typing import Protocol

from kubedocs.clinical.contracts import DepartmentAdvice
from kubedocs.clinical.rag import RetrievedChunk

ALIASES: dict[str, tuple[str, ...]] = {
    "心血管内科": ("心内科", "心脏内科"),
    "消化内科": ("胃病科", "肠胃科"),
    "呼吸内科": ("呼吸科", "肺科"),
    "耳鼻喉科": ("五官科", "耳鼻咽喉科"),
    "皮肤科": ("皮科",),
    "神经内科": ("神内",),
    "骨科": ("骨外科",),
    "泌尿外科": ("泌尿科",),
    "妇产科": ("妇科", "产科"),
    "眼科": ("眼科",),
    "口腔科": ("牙科",),
    "急诊科": ("急诊",),
}
"""科室别名。**明显不完整**，只为让 v0 能跑通并被测试驱动出正确形状。

为什么不一次写全：别名表的价值取决于真实患者的说法，那要等分诊金标集有了真实问法
才能定。先写一张小表 + 一条"按金标补表"的路径，比凭空写 200 条更诚实。
"""


class TriageBackend(Protocol):
    """分诊所需的科室清单来源。与 `domains.clinic.booking.BookingService.departments()` 对齐。"""

    def departments(self) -> list[dict]: ...


class RetrievalTriage:
    """检索式分诊 v0。

    打分规则（三条，按贡献排序）：

    1. **原话直呼**：患者原话里出现科室名/别名 ⇒ +3.0（最强信号，几乎不会错）；
    2. **卡片命中**：检索到的卡片正文里出现科室名/别名 ⇒ 累加该卡片的检索分数；
    3. **标题命中**：科室名出现在卡片标题/面包屑里 ⇒ 该条按 1.5 倍计权
       （标题里的科室名比正文里的一次偶然提及更可信）。

    置信度用"最高分 / 总分"归一，天花板 1.0。**它只用于排序**，不是概率。
    """

    _DIRECT_WEIGHT = 3.0
    _TITLE_BOOST = 1.5
    _MAX_DEPARTMENTS = 3

    def __init__(self, service: TriageBackend) -> None:
        self._service = service

    def _department_names(self) -> list[str]:
        out: list[str] = []
        for row in self._service.departments():
            name = str(row.get("name") or row.get("department") or "").strip()
            if name:
                out.append(name)
        return out

    @staticmethod
    def _variants(name: str) -> tuple[str, ...]:
        return (name, *ALIASES.get(name, ()))

    def rank(self, question: str, chunks: list[RetrievedChunk] | list[object]) -> list[DepartmentAdvice]:
        names = self._department_names()
        if not names:
            return []

        scores: dict[str, float] = dict.fromkeys(names, 0.0)
        reasons: dict[str, list[str]] = {n: [] for n in names}

        for name in names:
            for variant in self._variants(name):
                if variant and variant in question:
                    scores[name] += self._DIRECT_WEIGHT
                    reasons[name].append(f"原话提到「{variant}」")
                    break

        for chunk in chunks:
            body = str(getattr(chunk, "body", "") or "")
            title = f"{getattr(chunk, 'title', '')} {getattr(chunk, 'breadcrumb', '')}"
            weight = float(getattr(chunk, "score", 0.0) or 0.0)
            if weight <= 0:
                weight = 1.0  # 后端没给分数时按等权计，免得整个信号被 0 抹掉
            for name in names:
                for variant in self._variants(name):
                    if not variant:
                        continue
                    if variant in title:
                        scores[name] += weight * self._TITLE_BOOST
                        reasons[name].append(f"卡片标题含「{variant}」")
                        break
                    if variant in body:
                        scores[name] += weight
                        reasons[name].append(f"卡片正文含「{variant}」")
                        break

        ranked = sorted(((n, s) for n, s in scores.items() if s > 0), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return []
        top = ranked[0][1]
        total = sum(s for _, s in ranked) or 1.0
        out: list[DepartmentAdvice] = []
        for name, score in ranked[: self._MAX_DEPARTMENTS]:
            # 置信度 = 相对最强候选的占比（不是绝对概率）；见类文档说明。
            confidence = min(1.0, score / total) if score < top else 1.0
            out.append(
                DepartmentAdvice(
                    name=name,
                    confidence=round(confidence, 4),
                    reason="；".join(dict.fromkeys(reasons[name]))[:200],
                )
            )
        return out
