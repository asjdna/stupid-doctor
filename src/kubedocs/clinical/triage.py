"""分诊引擎（方案 §5）。目标指标：**准确率 ≥ 80%**。

## v1 与 v0 的区别（v0 是猜的，v1 用上了真实数据结构）

v0 只会"看科室名有没有出现在检索文本里"——那是当时**不知道卡片结构**时的权宜做法。
后来查清了 `data/clinic/cards.json`（31,592 张）的真实形状：

    {"kind":"symptom","key":"纤毛结构缺陷","departments":["呼吸内科","儿科"],
     "text":"纤毛结构缺陷。可能相关的疾病：……。建议就诊科室：呼吸内科、儿科。"}

- **14,440 张（45.7%）带 `departments` 字段**；
- 正文里那句「建议就诊科室：X、Y。」**可稳定解析**（前 3000 条与字段零不一致）；
- 共 **42 个科室**。

所以 v1 的做法是**按召回卡片的科室标签投票**，而不是靠名字碰巧出现在文本里。

## 打分规则（每条都写清为什么）

1. **原话直呼科室名**（"+3.0"）：患者自己说"我要挂心内科"，这是最强的信号，几乎不会错。
2. **卡片科室标签投票**：每条命中的卡片给出它的 `departments`，权重 = 该卡片的检索分数。
3. **标题命中加成**（×1.5）：卡片标题（症状/疾病规范名）直接出现在患者原话里时，
   该卡片比"正文里偶然提到"更可信。
4. **多科室卡片按科室数均分**：一张列出 3 个科室的卡片，不该比只列 1 个的卡片多 3 票。

## 做不到 / 未验证的事（写清楚，免得被当成"已完成"）

- **准确率尚未在真实问法上测过**：人工金标只有 **5 条**（`data/eval/clinic_gold.jsonl`），
  在 5 条上谈 80% 没有意义。`scripts/eval_triage.py` 会用**派生的大样本**给出一个
  可复现的数字，并**明确标注它是"规范名/去答案描述"这种偏简单的口径**，不是真实问法。
- **别名表很小**（只够让 v1 跑通）：真正的提升要靠按真实问法补表。
- **42 个科室名与挂号系统的科室名未必一致**：本模块只输出"知识库认为该去哪个科"，
  到号源系统还要一次映射（`BookingService.resolve_department` 负责，见挂号 skill）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from kubedocs.clinical.contracts import DepartmentAdvice
from kubedocs.clinical.rag import RetrievedChunk

_PLACEHOLDER_DEPARTMENTS = ("未标注", "无", "不详", "未知", "暂无", "其他综合", "其他科室")
"""**占位符**科室名，一律当"这张卡没有科室信息"处理。

为什么必须挡（实测数据）：`data/clinic/cards.json` 里正文含科室子句的卡有 27,303 张，
其中 **12,863 张（47.1%）只写着「未标注」**。不挡的话，这些卡会把「未标注」投成
一个"科室"——实测它就真的出现在了分诊结果里（人工金标 5 条中 3 条的 top-3 是「未标注」），
而那是**一个看起来像答案、实际什么都没有的值**。

⚠️ 顺带记下这个语料事实：**近一半卡片没有科室信息**。这是"真实问法分诊只有 40%"的
主要原因之一——不是排序不行，是**依据本身缺失**。补语料才是根治手段。
"""

_DEPARTMENT_CLAUSE = re.compile(r"建议就诊科室：([^。]+)。")
"""卡片正文里那句科室建议。格式由 `scripts/clinic_build_cards.py` 生成，稳定。"""

_DEPARTMENT_SPLIT = re.compile(r"[、,，/]")

ALIASES: dict[str, tuple[str, ...]] = {
    "心血管内科": ("心内科", "心脏内科"),
    "消化内科": ("胃病科", "肠胃科"),
    "呼吸内科": ("呼吸科", "肺科"),
    "耳鼻喉科": ("五官科", "耳鼻咽喉科"),
    "皮肤科": ("皮科", "皮肤性病科"),
    "神经内科": ("神内",),
    "神经外科": ("神外",),
    "骨科": ("骨外科",),
    "泌尿外科": ("泌尿科",),
    "妇产科": ("妇科", "产科"),
    "眼科": ("眼科",),
    "口腔科": ("牙科",),
    "急诊科": ("急诊",),
    "内分泌科": ("内分泌",),
    "风湿免疫科": ("风湿科",),
    "精神科": ("心理科", "精神心理科"),
}
"""科室别名。**明显不完整**，只为让 v1 能跑通并被测试驱动出正确形状。

为什么不一次写全：别名表的价值取决于**真实患者的说法**，那要等分诊金标集有了真实问法
才能定。先写一张小表 + 一条"按金标补表"的路径，比凭空写 200 条更诚实。
"""


def departments_in_card(text: str) -> list[str]:
    """从卡片正文里取「建议就诊科室」。取不到返回空列表（没有科室的卡片是正常的）。

    **占位符会被滤掉**（见 `_PLACEHOLDER_DEPARTMENTS`）：`未标注` 不是科室，
    放它进投票会产出一个"看起来像答案、实际什么都没有"的结果。
    """
    match = _DEPARTMENT_CLAUSE.search(text or "")
    if not match:
        return []
    out: list[str] = []
    for raw in _DEPARTMENT_SPLIT.split(match.group(1)):
        name = raw.strip()
        if not name or any(p in name for p in _PLACEHOLDER_DEPARTMENTS):
            continue
        out.append(name)
    return out


DEFAULT_DEPARTMENT_MAP = "data/clinic/department_map.json"
"""语料补全映射表（`key -> departments`），由 `scripts/clinic_derive_departments.py` 生成。

**为什么需要它**：clinic 语料里 12,863 张卡片的「建议就诊科室」是 `未标注`，
根因在卡片生成阶段丢掉了关系（这些卡片的 `diseases` 就是 `暂无`）。
回到原始 `triples.txt`（`head,relation,tail`）能补回 27.6% 的未标注 key：

| 来源 | 条数 | 占未标注 key | 性质 |
|---|---|---|---|
| `SameAs` 别名归并 | 3,009 | 17.5% | 权威（图谱自带别名） |
| 最近邻模糊归并 ≥0.5 | 1,252 | 7.3% | **有错例，需抽检** |
| 实体→疾病→科室 | 466 | 2.7% | 权威 |
| 无解（纯口语变体，图谱无记录） | 12,425 | 72.4% | 需外部标注 |

## 补全之后**没有变好**——这一项同样如实留档

`scripts/eval_triage.py --department-map {off,high,medium}`，`--limit 200`（A/B），
C 为 5 条人工金标；括号里是映射表命中次数（**非零 ⇒ 代码路径确实被触发**，不是空转）：

| 档位 | A top-1 | A top-3 | B top-1 | B top-3 | C top-1 | C top-3 |
|---|---|---|---|---|---|---|
| off | **164** | 197 | 153 | 196 | 2 | 4 |
| high | 162 | 197 | 153 | 196 | 3 | 4 |
| medium | 160 | **196** | 153 | 196 | **4** | 4 |

⇒ **主口径 B 一动不动，A 的 top-1 反降 4 例（164→160）**。机制是清楚的：
补全让候选池里的**旁观卡片**也开始投票，在"查询词就是卡片名"的 A 集里
这只会稀释正确卡片的票。C 上 2→4 看着好，但 **5 条样本里 1 条就值 20%，不构成结论**。

⇒ 由此得到的判断（这比数字本身重要）：**当前瓶颈不是语料的标签覆盖率，而是投票机制本身**。
在把 C 金标集从 5 条扩到 ≥50 条之前，任何"分诊准确率 xx%"都不可复现、不该写进结论。
"""


def load_department_map(
    path: str | Path = DEFAULT_DEPARTMENT_MAP,
    *,
    min_confidence: str = "high",
) -> dict[str, list[str]]:
    """载入补全映射表；失败时返回空表并**不抛异常**（退化为正文解析，分诊仍可用）。

    `min_confidence="high"` 只信任图谱直接关系 / `SameAs` 别名；
    传 `"medium"` 会额外纳入最近邻模糊归并（Jaccard>=0.5，有错例，**需抽检**）。
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    raw = payload.get("map") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {}
    allow = {"high"} if min_confidence == "high" else {"high", "medium"}
    out: dict[str, list[str]] = {}
    for key, rec in raw.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("confidence") not in allow:
            continue
        deps = [d for d in (rec.get("departments") or []) if d]
        if deps:
            out[key] = deps
    return out


DEFAULT_COARSE_K = 50
"""**粗排**候选池大小。

为什么分两阶段（按建议实测调整）：分诊本质是**投票**——"这些候选卡片分别建议去哪个科"。
既然是投票，候选池就不该只取 5 条：池子越大，投出正确科室的机会越大。
而实测告诉我们，粗排取 5 条与取 50 条**代价几乎相同**（瓶颈在生成候选，不在返回条数：
clinic 语料 31,592 张卡上一次混合检索约 1.2s，返回 5 条还是 50 条都是这个量级）。
所以"多拿候选、再精排"几乎免费。

**精排不需要模型**：用 jieba 词元的 IDF 加权重合度给候选重新定序
（见 `_refine`）。`retrieve/rerank.py` 里那个 `CrossEncoderReranker` 是更强的选择，
但它要额外 1.1GB 模型；先把不需模型的一档做出来并能测，再谈换更强的精排。
"""

DEFAULT_PRECISION_K = 0
"""**精排**之后**参与投票**的候选条数；`0` = 不截断（默认）。

## 这一项是被实测**否证**过的，如实留档

最初的假设是"精排之后必须截断，否则远端噪声把票摊薄"。实测（人工金标 5 条，离线）：

| 粗排池 | 投票池 | top-1 | top-3 |
|---|---|---|---|
| **50** | **不截断** | **2** | **4（80%）** |
| 50 | 20 | 2 | 3（60%） |
| 50 | 10 | 1 | 2（40%） |
| 20 | 20 | 1 | 2（40%） |
| 100 | 30 | 2 | 3（60%） |

⇒ **截断有害**：我的 IDF 精排还不足以支撑"只看前 10"这么硬的判断（正确的卡片常常排在
10 名之外）。真正把数字从 40% 抬到 80% 的是**粗排池从 5 扩到 50**，而不是精排截断。

所以默认不截断，参数保留是为了让后来者能重跑这张表。
> ⚠️ 也如实记下：**IDF 精排本身的独立贡献尚未被隔离**——
> 缺一组"粗排=50 但用原始检索分投票"的对照。这是已知的未测项，不是已验证的收益。
"""


class TriageBackend(Protocol):
    """分诊所需的科室清单来源。与 `domains.clinic.booking.BookingService.departments()` 对齐。"""

    def departments(self) -> list[dict]: ...


class RetrievalTriage:
    """检索式分诊 v1：按召回卡片的**科室标签**投票。

    `known_departments` 只用于约束输出（把投票结果收敛到系统里真实存在的科室）；
    留空表示不约束——金标里的科室名与号源系统的科室名未必一致，评测时不该因为
    一个名字对不上就把正确答案判错。
    """

    _DIRECT_WEIGHT = 3.0
    _MAX_DEPARTMENTS = 3

    def __init__(
        self,
        service: TriageBackend | None = None,
        *,
        known_departments: list[str] | None = None,
        precision_k: int = DEFAULT_PRECISION_K,
        department_map: dict[str, list[str]] | None = None,
    ) -> None:
        self._service = service
        self._known = list(known_departments or [])
        self.precision_k = max(0, precision_k)
        self.department_map = dict(department_map or {})
        self.map_hits = 0

    def _departments_of(self, title: str, body: str) -> list[str]:
        """优先查补全映射表（卡片正文的「未标注」是源图谱缺关系造成的），再退回正文解析。

        映射表由 `scripts/clinic_derive_departments.py` 生成，条目自带来源与置信度；
        调用方负责决定信任哪一档（见该脚本 `meta.confidence_note`）。
        """
        if title:
            hit = self.department_map.get(title)
            if hit:
                self.map_hits += 1
                return list(hit)
        return departments_in_card(body)

    # ------------------------------------------------------------------ 内部
    def _names(self) -> list[str]:
        if self._known:
            return self._known
        if self._service is None:
            return []
        out: list[str] = []
        for row in self._service.departments():
            name = str(row.get("name") or row.get("department") or "").strip()
            if name:
                out.append(name)
        return out

    @staticmethod
    def _variants(name: str) -> tuple[str, ...]:
        return (name, *ALIASES.get(name, ()))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """jieba 切词，失败时退化为字符二元组。

        为什么退路是字符二元组而不是"整串当一个词"：中文里"皮肤痒"与"皮肤红疹"共有的信息
        在字上，退化成一整串会让它们完全不重合。二元组是保底的相似度来源。
        """
        text = (text or "").strip()
        if not text:
            return []
        try:
            import jieba  # 与 clinic 域包进 FTS5 用的是同一个分词器

            tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        except Exception:  # noqa: BLE001 - 分词是可选增强，缺了也要能跑
            tokens = []
        if tokens:
            return tokens
        return [text[i : i + 2] for i in range(max(1, len(text) - 1))]

    def _refine(
        self, question: str, chunks: list[RetrievedChunk] | list[object]
    ) -> list[tuple[float, object]]:
        """**精排**：用查询词元的 IDF 加权重合度给候选重新定序（不需要模型）。

        规则（每条都有理由）：

        - 词元出现在**卡片标题**里 ⇒ 权重 ×3：clinic 语料里标题就是症状/疾病的规范名，
          标题命中比正文偶然提及强得多；
        - 词元只出现在正文里 ⇒ ×1；
        - 词元用 **IDF 加权**：`红疹`、`烧心` 这类少见词才是判别信号，
          而 `有点`、`还有` 到处都是，不该主导排序；
        - IDF 只在本批候选池内统计（局部 IDF）——不引全局词表，少一个需要维护的真相源。
        """
        query_tokens = set(self._tokens(question))
        if not query_tokens:
            return [(float(getattr(c, "score", 0.0) or 0.0), c) for c in chunks]

        pool_tokens: list[set[str]] = []
        for chunk in chunks:
            title = str(getattr(chunk, "title", "") or "")
            body = str(getattr(chunk, "body", "") or "")
            pool_tokens.append(set(self._tokens(f"{title} {body}")))

        n = max(1, len(chunks))
        df: dict[str, int] = {}
        for tokens in pool_tokens:
            for token in tokens & query_tokens:
                df[token] = df.get(token, 0) + 1
        import math

        idf = {token: math.log((n + 1) / (df.get(token, 0) + 1)) + 1.0 for token in query_tokens}

        scored: list[tuple[float, object]] = []
        for chunk, tokens in zip(chunks, pool_tokens, strict=True):
            title = str(getattr(chunk, "title", "") or "")
            title_tokens = set(self._tokens(title))
            hit = 0.0
            for token in query_tokens & tokens:
                hit += idf[token] * (3.0 if token in title_tokens else 1.0)
            # 检索分数作为**次要**项（同分时才起作用），不让它盖过词元重合。
            scored.append((hit + 0.001 * float(getattr(chunk, "score", 0.0) or 0.0), chunk))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return scored

    def _allowed(self, department: str) -> bool:
        """投票出来的科室是否可输出。没配科室清单时不约束。"""
        names = self._names()
        if not names:
            return True
        return department in names

    # ------------------------------------------------------------------ 公开
    def rank(self, question: str, chunks: list[RetrievedChunk] | list[object]) -> list[DepartmentAdvice]:
        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}

        def add(department: str, weight: float, reason: str) -> None:
            if not self._allowed(department):
                return
            scores[department] = scores.get(department, 0.0) + weight
            reasons.setdefault(department, []).append(reason)

        # 规则 2/3/4：按卡片科室标签投票（权重来自**精排**后的重合度，不再是原始检索分）。
        # `precision_k>0` 时才截断——默认不截断，理由见 DEFAULT_PRECISION_K 里那张实测表。
        refined = self._refine(question, list(chunks))
        if self.precision_k > 0:
            refined = refined[: self.precision_k]
        for precise, chunk in refined:
            body = str(getattr(chunk, "body", "") or "")
            title = str(getattr(chunk, "title", "") or "")
            departments = self._departments_of(title, body)
            if not departments or precise <= 0:
                continue
            share = precise / len(departments)  # 规则 4：多科室卡片按科室数均分
            for department in departments:
                add(department, share, f"卡片「{title or '?'}」建议")

        # 规则 1：原话直呼（对**已出现的科室名**加权，而不是凭空创建新科室）
        candidate_names = self._names() or sorted(scores)
        for name in candidate_names:
            for variant in self._variants(name):
                if variant and variant in question:
                    add(name, self._DIRECT_WEIGHT, f"原话提到「{variant}」")
                    break

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return []
        total = sum(s for _, s in ranked) or 1.0
        out: list[DepartmentAdvice] = []
        for name, score in ranked[: self._MAX_DEPARTMENTS]:
            # 置信度 = 相对总分的占比（**只用于排序，不是概率**）。
            out.append(
                DepartmentAdvice(
                    name=name,
                    confidence=round(score / total, 4),
                    reason="；".join(dict.fromkeys(reasons[name]))[:200],
                )
            )
        return out
