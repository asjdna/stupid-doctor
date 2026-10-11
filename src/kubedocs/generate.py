"""带引用回答与拒答。

## 这个模块的三条硬规则

1. **只依据检索到的上下文回答**，并且每个论断都要带 `[n]` 引用标记；
2. **上下文里没有就明说没有**——拒答是正确输出，不是失败。检索为空时**不调用 LLM**：
   既省成本，也更诚实（让模型"凭印象答"正是幻觉的来源）；
3. **引用标记必须可核对**：`[n]` 只能指向真实提供给模型的那 n 段上下文。
   指向不存在的 `[99]` 会被剥掉并记入诊断——**一个指向不存在来源的引用，
   比没有引用更糟**。

## 为什么把"未配置密钥"也当成一种正常输出

业务逻辑要能跑通，而"生成回答"是唯一需要外网密钥的一步。所以这里把它设计成
**有记录的降级**：检索结果与引用照常返回，只是 `text` 为空、`refused=True`
并给出原因。这样没有密钥的环境（CI、面试官本机）依然能完整跑通整条链路，
而不是看到一坨 traceback。

## 关于"逐字保留标识符"

系统提示里专门有一条：参数名、字段名、命令行、错误码必须**原样保留**。
这不是凑数的规则——LLM 很容易把 `--eviction-hard` 改写成"驱逐硬阈值"、
把 `topologySpreadConstraints` 缩写成"拓扑约束"。对技术文档问答来说，
**改写标识符等于答错**。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kubedocs.config import Settings
from kubedocs.retrieve.hybrid import HybridRetriever, SearchResult
from kubedocs.store.base import ChunkHit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个 Kubernetes 官方文档助手。严格遵循以下规则：

1. 只能依据我提供的【文档片段】回答，不得使用片段之外的知识。
2. 每个论断后面必须紧跟引用标记，形如 [1]、[2]（可多个，如 [1][3]），编号对应片段编号。
3. 如果这些片段不足以回答问题，就明确回答"提供的文档片段中没有说明该问题"，
   并说明你还需要哪方面的信息。**不要猜测，不要用常识补全**。
4. 参数名、字段名、命令行参数、错误码、API 版本等标识符必须**逐字保留**，
   不得翻译、改写、缩写或省略。
5. 回答用与提问相同的语言，简洁分点，不要复述全部原文。"""

_CONTEXT_HEADER_RE = re.compile(r"^\[[^\]]*\]\s*\n+")
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


# --------------------------------------------------------------------------- 数据结构
@dataclass(frozen=True)
class Citation:
    """回答里 [n] 与来源的对应关系。引用必须能点开。"""

    index: int
    chunk_id: str
    title: str
    breadcrumb: str
    source_url: str

    @property
    def label(self) -> str:
        return f"{self.title} › {self.breadcrumb}" if self.breadcrumb else self.title

    def render(self) -> str:
        return f"[{self.index}] {self.label} — {self.source_url}"


@dataclass
class Answer:
    question: str
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    degraded: list[str] = field(default_factory=list)
    retrieval: SearchResult | None = None

    def render(self) -> str:
        parts: list[str] = []
        if self.refused:
            parts.append(f"（未作答：{self.refusal_reason}）")
        if self.text:
            parts.append(self.text)
        if self.citations:
            parts.append("\n参考来源：")
            parts.extend("  " + c.render() for c in self.citations)
        if self.degraded:
            parts.append("")
            parts.extend(f"⚠ {d}" for d in self.degraded)
        return "\n".join(parts)

    def to_json(self) -> dict[str, object]:
        return {
            "question": self.question,
            "text": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6),
            },
            "citations": [
                {
                    "index": c.index,
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "breadcrumb": c.breadcrumb,
                    "source_url": c.source_url,
                }
                for c in self.citations
            ],
            "degraded": self.degraded,
        }


# --------------------------------------------------------------------------- 上下文组装
def context_text(body: str) -> str:
    """去掉块首的 `[标题 › 面包屑]` 上下文头。

    这行是给**嵌入模型**用的（提高区分度，见 chunker 的说明）；放进提示词时
    反而冗余——因为编号标签里已经写了同样的来源信息，重复会稀释注意力。
    """
    return _CONTEXT_HEADER_RE.sub("", body, count=1).strip()


def build_context(
    hits: Sequence[ChunkHit],
    *,
    max_chars: int = 8000,
    per_chunk_chars: int = 1200,
    max_per_doc: int = 2,
) -> tuple[str, list[Citation]]:
    """把检索结果编号成上下文，返回 (上下文文本, 实际收录的引用)。

    两个刻意的约束：

    - **超出 `max_chars` 的片段会被丢弃**，并且不会出现在引用列表里——
      只给模型看了一部分片段，却允许它引用全部编号，是引用错位的常见来源。
    - **同一文档最多收录 `max_per_doc` 段**。检索是按"块"打的，同一篇文档里
      相邻小节的向量往往高度相似，会一起涌进前几名（实测一次查询的前 4 段里
      有 3 段来自同一篇）。这些近重复片段既挤掉别的文档，也让上下文冗余。
      限流后引用编号仍然连续（按收录顺序重编）。
    """
    blocks: list[str] = []
    citations: list[Citation] = []
    per_doc: dict[str, int] = {}
    used = 0
    for hit in hits:
        if per_doc.get(hit.doc_id, 0) >= max_per_doc and citations:
            # citations 非空才允许跳过：第一段必须保留，
            # 否则"有结果"会退化成"没结果"（宁可上下文冗余，不可空手）
            continue
        label = f"{hit.title} › {hit.breadcrumb}" if hit.breadcrumb else hit.title
        body = context_text(hit.body)
        if len(body) > per_chunk_chars:
            body = body[:per_chunk_chars].rstrip() + " …"
        idx = len(citations) + 1
        block = f"[{idx}] {label}\n{body}"
        if used + len(block) > max_chars and citations:
            break
        blocks.append(block)
        citations.append(
            Citation(
                index=idx,
                chunk_id=hit.chunk_id,
                title=hit.title,
                breadcrumb=hit.breadcrumb,
                source_url=hit.source_url,
            )
        )
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        used += len(block)
    return "\n\n".join(blocks), citations


def validate_citations(text: str, n_available: int) -> tuple[str, list[str]]:
    """核对回答里的 `[n]` 是否都指向真实提供的片段。

    返回 (清理后的文本, 问题列表)。越界标记会被**剥掉**：
    留着等于给出一个打不开的引用，比没有引用更有误导性。
    """
    problems: list[str] = []
    bad: set[str] = set()

    def _sub(m: re.Match[str]) -> str:
        k = int(m.group(1))
        if 1 <= k <= n_available:
            return m.group(0)
        bad.add(m.group(0))
        return ""

    cleaned = _MARKER_RE.sub(_sub, text)
    if bad:
        problems.append(f"模型引用了未提供的片段编号 {sorted(bad)}，已剥除")
    if not _MARKER_RE.search(cleaned):
        problems.append("回答没有任何引用标记——无法核对来源")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), problems


# --------------------------------------------------------------------------- 预算
class BudgetExceeded(RuntimeError):
    """当日花费超限。拒绝调用，而不是悄悄烧钱。"""


UNPRICED_CALL_FALLBACK_USD = 0.25
"""litellm **算不出价**时，每次调用按这个金额保守计入当日花费。

## 为什么不能按 0 记

`completion_cost()` 对未登记在价格表里的模型会抛异常（实测 litellm 1.102.1：
`Exception: This model isn't mapped yet...`）。原先的实现把这个异常吞掉并**按 0 记**，
后果是**闸门永远拦不住**：花了多少钱在账本上永远是 0，`check()` 也就永远不触发。
"算不出价"与"这次调用免费"是两件完全不同的事，不能混成同一个数字。

## 0.25 这个数怎么来的

按本应用**单次调用的最坏情况**取上限，再留安全系数：

- 输入上限 = `max_context_chars` 8000 字符 ≈ 2800 token（中英混排的粗略折算）；
- 输出上限 = `llm_max_tokens` 800 token；
- 主流供应商里最贵的一档（claude-3-opus 级别的 $15/1M 输入 + $75/1M 输出）
  ≈ `2800×15e-6 + 800×75e-6` ≈ **$0.10**。

0.25 是它的 2.5 倍，所以按这个数计入时**欠计**的概率很低（宁可高估，不可漏计）。
默认日预算 $5 时，它等价于"最多约 20 次无法定价的调用"。

## 运营者可以调

`KUBEDOCS_LLM_UNPRICED_CALL_USD=0` 表示"我的模型确实免费，不要按上限计入"
（此时仍会计数，只是不计金额）。默认**不假设免费**，这是刻意的默认方向。

## 注意"真免费"与"算不出价"是分得开的

litellm 对本地免费模型（实测 `ollama/llama3`）会**正常返回 0.0**，那是"定价为 0"，
按 0 记是对的；只有"抛异常 / 返回 None"才走这条保守路径。两者不混。
"""

_SPEND_LOCKS: dict[str, threading.Lock] = {}
_SPEND_LOCKS_GUARD = threading.Lock()
_UNPRICED_SUFFIX = ":unpriced_calls"


def _lock_for(path: Path) -> threading.Lock:
    """取该花费文件**进程内**唯一的锁。

    为什么按路径注册而不是每个 `DailyBudget` 一把锁：同一个花费文件会被多处构造
    （`Answerer` 一份、`/chat` 的工具循环一份），各自一把锁等于没有锁。
    """
    key = str(path.resolve()) if path.is_absolute() else str(path)
    with _SPEND_LOCKS_GUARD:
        lock = _SPEND_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SPEND_LOCKS[key] = lock
        return lock


class DailyBudget:
    """按天累计 LLM 花费，落盘成一个小 JSON 文件（可读、可手改、可审计）。

    ## 并发边界（如实写清，别高估这个闸门）

    - **进程内是原子的**：读-改-写整段持 `threading.Lock`（按文件路径注册）。
      FastAPI 的同步端点跑在线程池里，所以"两个并发请求同时读同一个文件、
      各自加完再写回"这种丢更新是真实存在的场景，这把锁挡的就是它。
    - **跨进程不是原子的**：多 worker / 多副本各持自己的锁、各写各的，
      互相看不见 ⇒ 真实上限是 `副本数 × 日预算`。**没有修**，理由是取舍可核：
      · 跨进程锁要按平台分叉导入（`fcntl.flock` / `msvcrt.locking`），或用
        `O_EXCL` 锁文件——后者要处理崩溃残留，而"锁文件残留"的失败形态是
        **闸门永久拒绝一切调用**（账本坏了，花钱的路径跟着不可用），
        比它要防的"丢一次计费"严重得多；
      · 换 SQLite 事务确实能跨进程原子，但要求放弃这个账本"人可读、可手改、
        可审计"的设计，也放弃已落盘的 `llm_spend.json` 兼容性。
      当前部署形态是**单进程**（`uvicorn kubedocs.api.app:app`，没有 `--workers`，
      见 `Dockerfile`），所以这条限制现在不生效；真要多副本部署，必须换成共享账本。
    - **金额精度**：文件里按 6 位小数累积，单次舍入误差可忽略。
    """

    def __init__(
        self, path: Path, limit_usd: float, unpriced_usd: float = UNPRICED_CALL_FALLBACK_USD
    ) -> None:
        self.path = path
        self.limit = limit_usd
        self.unpriced_usd = unpriced_usd

    def _load(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        try:
            data: dict[str, float] = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 文件坏了不该让服务整体不可用；保守做法是当作"今天没花过"
            return {}
        out: dict[str, float] = {}
        for key, value in data.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                # 手改坏了某一项时只丢那一项，不要因为一行脏数据把整本账丢掉
                continue
        return out

    def spent_today(self) -> float:
        return float(self._load().get(self._today(), 0.0))

    def unpriced_calls_today(self) -> int:
        """今日"算不出价"的调用次数（其金额已按上限计入，这里只用于可见性）。"""
        return int(self._load().get(self._today() + _UNPRICED_SUFFIX, 0.0))

    def check(self, estimated_usd: float = 0.0) -> None:
        """到达上限即拒绝**新的**调用（不是等超了才拒）。

        边界写清楚：「已达上限」用 `>=`，「加上本次预估会超」用 `>`。
        否则 spent 恰好等于 limit 时会放行一次调用，实际花掉的钱就超过上限了。
        """
        if self.limit <= 0:
            return
        spent = self.spent_today()
        if spent >= self.limit or spent + estimated_usd > self.limit:
            unpriced = self.unpriced_calls_today()
            tail = (
                f"；今天另有 {unpriced} 次调用算不出价，已按每次 ${self.unpriced_usd:.2f} 的上限保守计入"
                if unpriced
                else ""
            )
            raise BudgetExceeded(
                f"今日 LLM 花费 ${spent:.4f} 已达上限 ${self.limit:.2f}"
                f"（本次预估还需 ${estimated_usd:.4f}）{tail}"
            )

    def record(self, usd: float, *, priced: bool = True) -> float:
        """记一次调用，返回今天新的累计值。

        `priced=False` 表示**litellm 算不出这次调用的价**。此时**不许按 0 记**：
        按 `unpriced_usd` 的上限保守计入，并单独计数（`unpriced_calls_today()`），
        同时打一条 warning——三处都可见，因为"悄无声息地免费"正是原先的缺陷。
        """
        charge = float(usd)
        today = self._today()
        with _lock_for(self.path):
            data = self._load()
            payload: dict[str, object] = dict(data)
            if not priced:
                charge = max(charge, self.unpriced_usd)
                key = today + _UNPRICED_SUFFIX
                count = int(data.get(key, 0.0)) + 1
                payload[key] = count
                logger.warning(
                    "LLM 调用无法定价（litellm 价格表里没有这个模型？）：已按每次 $%.2f "
                    "的上限计入当日预算，今天第 %d 次",
                    self.unpriced_usd,
                    count,
                )
            payload[today] = new_total = round(data.get(today, 0.0) + charge, 6)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return new_total

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- LLM 客户端
@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_priced: bool = True
    """`False` = **这次调用的价算不出来**（litellm 价格表里没有这个模型）。

    与 `cost_usd == 0.0` 是两件事：真免费的本模型（例如走 ollama）会正常返回
    0.0，那时 `cost_priced` 仍是 `True`。调用方据此决定是否让预算按保守上限计入。
    """


class LLMUnavailable(RuntimeError):
    """没有可用的模型或密钥。调用方应把它当作"有记录的降级"，而不是崩溃。"""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResult: ...


def required_api_key_env(model: str) -> str:
    """从 LiteLLM 的 `provider/model` 名字推出该读哪个环境变量。"""
    provider = model.split("/", 1)[0].lower() if "/" in model else "openai"
    special = {
        "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
        "bedrock": "AWS_ACCESS_KEY_ID",
        "azure": "AZURE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    return special.get(provider, f"{provider.upper()}_API_KEY")


def has_api_key(model: str) -> bool:
    return bool(os.environ.get(required_api_key_env(model), "").strip())


class LiteLLMClient:
    """LiteLLM 包装：多供应商路由/重试/成本统计都交给它，自研是重复劳动。

    `litellm` 是**懒导入**的：它导入耗时数秒，而 `--help`、单元测试、
    纯检索路径都不该付出这个代价。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self._litellm: object | None = None

    @property
    def litellm(self) -> object:
        if self._litellm is None:
            import litellm

            # 供应商抖动时自己重试会更糟：交给 LiteLLM 统一处理
            litellm.drop_params = True
            self._litellm = litellm
        return self._litellm

    def complete(self, system: str, user: str) -> LLMResult:
        if not has_api_key(self.model):
            raise LLMUnavailable(
                f"未配置 {required_api_key_env(self.model)}（模型 {self.model}），"
                "无法生成回答；检索结果仍然可用"
            )
        litellm = self.litellm
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "timeout": self.settings.llm_timeout,
            "num_retries": self.settings.llm_max_retries,
        }
        fallbacks = self.settings.llm_fallback_list
        if fallbacks:
            kwargs["fallbacks"] = fallbacks
        resp = litellm.completion(**kwargs)  # type: ignore[attr-defined]
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        cost, priced = 0.0, True
        try:
            raw_cost = litellm.completion_cost(completion_response=resp)  # type: ignore[attr-defined]
            # `None` = litellm 找不到这个模型的价，**不是"免费"**
            priced = raw_cost is not None
            cost = float(raw_cost) if priced else 0.0
        except Exception:  # noqa: BLE001 - 算不出价不该让请求失败，但**不许按 0 记**
            # 未知模型（实测抛 `This model isn't mapped yet`）走这里。
            # 具体金额由 `DailyBudget.record(priced=False)` 按保守上限计入并计数。
            cost, priced = 0.0, False
        return LLMResult(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=cost,
            cost_priced=priced,
        )


# --------------------------------------------------------------------------- 主流程
class Answerer:
    """检索 + 生成 + 引用核对 + 预算控制。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: Settings,
        budget: DailyBudget | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.budget = budget or DailyBudget(
            settings.index_path / "llm_spend.json",
            settings.llm_daily_budget_usd,
            unpriced_usd=settings.llm_unpriced_call_usd,
        )

    def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        res = self.retriever.search(question, top_k=top_k)
        hits = list(res.results)
        ans = Answer(question=question, model=self.settings.llm_model, retrieval=res)
        for msg in res.diagnostics.get("degraded") or []:  # type: ignore[attr-defined]
            ans.degraded.append(str(msg))

        if not hits:
            # 不给模型"自由发挥"的机会：检索为空就直接拒答，且不花钱
            ans.refused = True
            ans.refusal_reason = "检索结果为空，官方文档中没有找到相关内容"
            ans.citations = []
            return ans

        context, citations = build_context(
            [
                ChunkHit(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    ordinal=h.ordinal,
                    title=h.title,
                    breadcrumb=h.breadcrumb,
                    source_url=h.source_url,
                    body=h.body,
                )
                for h in hits
            ],
            max_chars=self.settings.max_context_chars,
            max_per_doc=self.settings.max_chunks_per_doc,
        )
        ans.citations = citations

        user = (
            f"【文档片段】\n{context}\n\n【问题】\n{question}\n\n请依据上面的片段作答，并带上 [编号] 引用。"
        )
        try:
            self.budget.check()
        except BudgetExceeded as e:
            ans.refused = True
            ans.refusal_reason = str(e)
            return ans

        try:
            result = self.llm.complete(SYSTEM_PROMPT, user)
        except LLMUnavailable as e:
            ans.refused = True
            ans.refusal_reason = str(e)
            ans.degraded.append("仅返回检索结果（生成回答不可用）")
            return ans
        except Exception as e:  # noqa: BLE001
            # 供应商侧的失败（密钥无效 401、限流 429、超时、服务不可用）不该让整个请求崩掉：
            # 检索结果本身是有价值的，用户应当拿到它 + 一句明确的失败原因。
            # 但**必须把异常类型记进诊断**，否则就成了静默吞异常。
            ans.refused = True
            ans.refusal_reason = f"调用大模型失败：{type(e).__name__}: {e}"
            ans.degraded.append("仅返回检索结果（生成回答失败）")
            return ans

        ans.text, problems = validate_citations(result.text, len(citations))
        ans.prompt_tokens = result.prompt_tokens
        ans.completion_tokens = result.completion_tokens
        ans.cost_usd = result.cost_usd
        ans.degraded.extend(problems)
        # 定价为 0 的调用不必写文件（没有账要记）；但**算不出价**的必须记——
        # 否则"未知 = 免费"这条缺陷就从这个判断里溜回来了。
        if result.cost_usd or not result.cost_priced:
            self.budget.record(result.cost_usd, priced=result.cost_priced)
        if not result.cost_priced:
            ans.degraded.append(
                "这次调用**算不出价**（litellm 价格表里没有这个模型）：已按保守上限计入"
                "当日预算，请勿把它当成免费调用"
            )
        if not ans.text:
            ans.refused = True
            ans.refusal_reason = "模型返回空内容"
        return ans


def build_answerer(retriever: HybridRetriever, settings: Settings) -> Answerer:
    return Answerer(retriever, LiteLLMClient(settings), settings)
