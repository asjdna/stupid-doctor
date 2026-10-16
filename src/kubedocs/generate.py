"""带引用回答与拒答。

三条硬规则：

1. **只依据检索到的上下文回答**，并且每个论断都要带 `[n]` 引用标记；
2. **上下文里没有就明说没有**——拒答是正确输出，不是失败。检索为空时**不调用
   LLM**：既省成本，也避免模型"凭印象答"带来的幻觉；
3. **引用标记必须可核对**：`[n]` 只能指向真实提供给模型的那 n 段上下文。指向不
   存在编号的标记会被剥掉并记入诊断——指向不存在来源的引用比没有引用更糟。

"未配置密钥"也是一种正常输出（**有记录的降级**）：检索结果与引用照常返回，只是
`text` 为空、`refused=True` 并给出原因，没有密钥的环境（离线机器等）依然能跑通整条链路。

系统提示里专门要求参数名、字段名、命令行、错误码**逐字保留**：LLM 很容易把
`--eviction-hard` 改写成"驱逐硬阈值"、把 `topologySpreadConstraints` 缩写，
对技术文档问答来说改写标识符等于答错。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kubedocs.config import Settings
from kubedocs.observability import observe_llm_call, span, span_attrs
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
# 引用标记 `[n]`。位数**不设上限**：判据只是"编号是否 ≤ 提供的段数"，
# 写成 `\d{1,3}` 会漏掉 `[1234]`，让它既不被剥除、也不触发"没有引用"的告警。
_MARKER_RE = re.compile(r"\[(\d+)\]")


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
def strip_context_header(body: str) -> str:
    """去掉块首的 `[标题 › 面包屑]` 上下文头。

    这一行是给**嵌入模型**提高区分度用的；放进提示词反而冗余——编号标签里已经写了
    同样的来源信息，重复会稀释注意力。
    """
    return _CONTEXT_HEADER_RE.sub("", body, count=1).strip()


def build_context(
    hits: Sequence[ChunkHit],
    *,
    max_chars: int = 8000,
    per_chunk_chars: int = 3000,
    max_per_doc: int = 2,
) -> tuple[str, list[Citation]]:
    """把检索结果编号成上下文，返回 (上下文文本, 实际收录的引用)。

    三条约束：

    - **喂给模型的是父块**（`hit.context_text`）：检索用子块命中（精确），
      进提示词用父块（整节，读得到前后文）。子块只有 300 字，单独塞给模型常常是
      一句被切断的话；父块是它所在的那个小节。
    - **超出 `max_chars` 的片段会被丢弃**，并且不出现在引用列表里——只给模型看了
      一部分片段却允许它引用全部编号，是引用错位的常见来源。
    - **同一文档最多收录 `max_per_doc` 段**。检索按块打分，同一篇文档里相邻小节的
      向量往往高度相似，会一起涌进前几名，既挤掉别的文档又让上下文冗余。限流后
      引用编号按收录顺序重编，仍然连续。
    - **同一个父块只收录一次**：一个小节被切成好几个子块，它们可能一起命中前几名，
      而那几段文字是**同一节**——重复贴三遍既浪费上下文，也会让模型以为"有三个来源
      都这么说"。重复的命中直接跳过，不占编号。

    `per_chunk_chars` 默认 3000（= 父块上限）：它必须是**父块**的预算，不是子块的。
    按子块的 300 字来截父块，等于把刚拼好的语境又切掉。
    """
    blocks: list[str] = []
    citations: list[Citation] = []
    per_doc: dict[str, int] = {}
    seen_parents: set[str] = set()
    used = 0
    for hit in hits:
        if hit.parent_id:
            if hit.parent_id in seen_parents:
                continue
            seen_parents.add(hit.parent_id)
        if per_doc.get(hit.doc_id, 0) >= max_per_doc and citations:
            # `citations` 非空才允许跳过：第一段必须保留，否则"有结果"会退化成"没结果"
            continue
        label = f"{hit.title} › {hit.breadcrumb}" if hit.breadcrumb else hit.title
        body = strip_context_header(hit.context_text)
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

`completion_cost()` 对价格表里没有的模型会抛异常 / 返回 `None`。"算不出价"与
"这次调用免费"是两件事：按 0 记等于预算闸门永远拦不住，所以取一个偏高的常数。

0.25 的来历：按单次调用最坏情况留安全系数——输入上限 `max_context_chars`
8000 字符 ≈ 2800 token，输出上限 `llm_max_tokens`（默认 2000），主流供应商最贵的
一档（$15/1M 输入 + $75/1M 输出）≈ $0.19，0.25 是它的 2.5 倍。默认日预算 $5 时，
约等于"最多 20 次无法定价的调用"。

`KUBEDOCS_LLM_UNPRICED_CALL_USD=0` 表示"我的模型确实免费，不要按上限计入"
（仍会计数，只是不计金额）。真免费与算不出价分得开：litellm 对本地免费模型会正常
返回 0.0，那按 0 记是对的，只有抛异常 / 返回 `None` 才走这条保守路径。
"""

_SPEND_LOCKS: dict[str, threading.Lock] = {}
_SPEND_LOCKS_GUARD = threading.Lock()
_UNPRICED_SUFFIX = ":unpriced_calls"


def _lock_for(path: Path) -> threading.Lock:
    """取该花费文件**进程内**唯一的锁。

    同一个花费文件会被多处构造（`Answerer` 一份、`/chat` 的工具循环一份），
    按路径注册才能让它们共用同一把锁。
    """
    key = str(path.resolve()) if path.is_absolute() else str(path)
    with _SPEND_LOCKS_GUARD:
        lock = _SPEND_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SPEND_LOCKS[key] = lock
        return lock


_SPEND_PENDING: dict[str, float] = {}
"""在途调用已预留的金额（按账本路径）。用来把"正在花但还没记上"的钱算进已用量。"""


def _pending_for(path: Path) -> float:
    key = str(path.resolve()) if path.is_absolute() else str(path)
    with _SPEND_LOCKS_GUARD:
        return _SPEND_PENDING.get(key, 0.0)


def _add_pending(path: Path, delta: float) -> None:
    key = str(path.resolve()) if path.is_absolute() else str(path)
    with _SPEND_LOCKS_GUARD:
        _SPEND_PENDING[key] = max(0.0, _SPEND_PENDING.get(key, 0.0) + delta)


def _atomic_replace(tmp: Path, dest: Path) -> None:
    """`os.replace` 的 Windows 友好版本。

    Windows 上若目标文件此刻正被其他句柄打开（另一个进程在读、编辑器或杀毒软件
    占用），替换会以 `PermissionError` 失败，POSIX 没有这个语义。所以这里重试几次；
    仍然失败则退回"直接重写"并记一条 warning——退路不是原子的，但比让一次记账
    直接抛异常更好。
    """
    for attempt in range(5):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            time.sleep(0.01 * (attempt + 1))
    logger.warning("原子替换 %s 失败（目标被占用？）：退回直接重写", dest)
    dest.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.unlink(missing_ok=True)


class DailyBudget:
    """按天累计 LLM 花费，落盘成一个小 JSON 文件（可读、可手改、可审计）。

    并发边界（别高估这个闸门）：

    - **进程内是原子的**：读-改-写整段持 `threading.Lock`（按文件路径注册），
      "读"也在锁内——只给写加锁不够。写入用 `tmp` + `os.replace` 原子替换，否则
      并发读者可能读到被截断的空文件，进而当作"今天没花过"而放行（fail-open）。
    - **`check()` 与 `record()` 之间的窗口要预留**：两个方法各是一个临界区，
      N 个并发请求会在 spent 还没变时全部通过 `check()`，然后各自 `record()`，
      上限被冲过 N 倍。调用方应当用 `reserve()`，它把"在途调用"按保守金额计入判断。
    - **跨进程不是原子的**：多 worker / 多副本各持自己的锁、各写各的，真实上限是
      `副本数 × 日预算`。当前部署形态是**单进程**（`uvicorn kubedocs.api.app:app`，
      没有 `--workers`），这条限制不生效；真要多副本部署必须换成共享账本。
    - **金额精度**：文件里按 6 位小数累积，单次舍入误差可忽略。
    """

    def __init__(
        self,
        path: Path,
        limit_usd: float,
        unpriced_usd: float = UNPRICED_CALL_FALLBACK_USD,
        reserve_usd: float | None = None,
    ) -> None:
        self.path = path
        self.limit = limit_usd
        self.unpriced_usd = unpriced_usd
        # 预留金额默认取"算不出价"的保守上限：预留时不知道真实花费，
        # 用偏高的常数比用 0 安全（用 0 预留等于没预留）。
        self.reserve_usd = unpriced_usd if reserve_usd is None else reserve_usd

    def _load(self) -> dict[str, float]:
        """读账本。**调用方必须已持 `_lock_for(self.path)`**（见类文档）。"""
        if not self.path.exists():
            return {}
        try:
            data: dict[str, float] = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 文件被外部改坏不该让服务整体不可用：保守地当作"今天没花过"，
            # 同时留下 warning 以便被发现。
            logger.warning("LLM 账本 %s 无法解析，本次按 0 计（请检查该文件）", self.path)
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
        with _lock_for(self.path):
            return float(self._load().get(self._today(), 0.0))

    def unpriced_calls_today(self) -> int:
        """今日"算不出价"的调用次数（其金额已按上限计入，这里只用于可见性）。"""
        with _lock_for(self.path):
            return int(self._load().get(self._today() + _UNPRICED_SUFFIX, 0.0))

    def check(self, estimated_usd: float = 0.0) -> None:
        """到达上限即拒绝**新的**调用（不是等超了才拒）。

        边界写清楚：「已达上限」用 `>=`，「加上本次预估会超」用 `>`。
        否则 spent 恰好等于 limit 时会放行一次调用，实际花掉的钱就超过上限了。
        判断里还包含**在途预留**，否则并发请求会一起冲过上限。
        """
        with _lock_for(self.path):
            self._check_locked(estimated_usd)

    def _check_locked(self, estimated_usd: float) -> None:
        if self.limit <= 0:
            return
        spent = float(self._load().get(self._today(), 0.0))
        pending = _pending_for(self.path)
        used = spent + pending
        if used >= self.limit or used + estimated_usd > self.limit:
            unpriced = int(self._load().get(self._today() + _UNPRICED_SUFFIX, 0.0))
            tail = (
                f"；今天另有 {unpriced} 次调用算不出价，已按每次 ${self.unpriced_usd:.2f} 的上限保守计入"
                if unpriced
                else ""
            )
            inflight = f"（含在途预留 ${pending:.4f}）" if pending else ""
            raise BudgetExceeded(
                f"今日 LLM 花费 ${spent:.4f}{inflight} 已达上限 ${self.limit:.2f}"
                f"（本次预估还需 ${estimated_usd:.4f}）{tail}"
            )

    @contextmanager
    def reserve(self, estimated_usd: float | None = None) -> Iterator[None]:
        """为一次模型调用**预留**额度：进入时检查并占住，退出时释放。

        `check()` 与 `record()` 是两个独立的临界区，并发请求会在 spent 还没变时
        全部通过 `check()`，随后各自记账，把上限冲过 N 倍。预留把"在途调用"也算作
        已用量，堵住这条路径。

        预留时不知道真实花费，所以按 `reserve_usd`（默认 = 算不出价的上限）保守占额。
        """
        amount = self.reserve_usd if estimated_usd is None else estimated_usd
        with _lock_for(self.path):
            self._check_locked(amount)
        _add_pending(self.path, amount)
        try:
            yield
        finally:
            _add_pending(self.path, -amount)

    def record(self, usd: float, *, priced: bool = True) -> float:
        """记一次调用，返回今天新的累计值。

        `priced=False` 表示**litellm 算不出这次调用的价**，此时不许按 0 记：
        按 `unpriced_usd` 的上限保守计入，并单独计数（`unpriced_calls_today()`）、
        打一条 warning——三处都可见，避免"悄无声息地免费"。
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
            # 原子替换：`write_text` 会先截断，并发读者可能读到空文件，
            # 而空文件被当作"今天没花过"，导致闸门 fail-open。
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _atomic_replace(tmp, self.path)
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

    与 `cost_usd == 0.0` 是两件事：真免费的本地模型（例如走 ollama）会正常返回
    0.0，那时 `cost_priced` 仍是 `True`。调用方据此决定是否让预算按保守上限计入。
    """


class LLMUnavailable(RuntimeError):
    """没有可用的模型或密钥。调用方应把它当作"有记录的降级"，而不是崩溃。"""


@dataclass
class ToolCallRequest:
    """模型要求调用某个工具（OpenAI / LiteLLM function calling 的形状）。"""

    id: str
    name: str
    arguments: str
    """**原始 JSON 字符串**，不在这里解析：解析失败要由调用方当"可预期失败"处理，
    而不是在这里抛出去把整轮对话打断。"""


@dataclass
class LLMToolTurn:
    """一次**带工具**的模型回复：可能要调工具，也可能是最终答复。

    `cost_usd` / `cost_priced` 与 `LLMResult` 同义——工具循环里的每一次调用
    **都要计费**，否则多轮工具调用会变成绕过日预算的口子。
    """

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_priced: bool = True


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


_PLACEHOLDER_MARKERS = (
    "your-",
    "your_",
    "placeholder",
    "changeme",
    "change-me",
    "example",
    "xxxx",
    "<",
    ">",
)
"""明显不是密钥的占位符片段（大小写不敏感）。

`has_api_key` 判"不像占位符"用这张词表，另一条判据是"必须全是 ASCII"。
词表保留是因为它能把话说得更具体（"看起来是占位符" vs "含非 ASCII"），
日志里更有指导性。
"""


def has_api_key(model: str) -> bool:
    """该模型所需的环境变量是否**真的**配了密钥。

    "配了"= 非空 **且** 不像占位符 **且**全是 ASCII。后两条缺一不可：把 `.env` 里
    的说明文字当成密钥，会让调用方发起一次注定失败的请求，而用户只看到底层异常。
    """
    env = required_api_key_env(model)
    value = os.environ.get(env, "").strip()
    if not value:
        return False
    low = value.lower()
    if any(marker in low for marker in _PLACEHOLDER_MARKERS):
        logger.warning("%s 看起来是占位符（长度 %d），按未配置处理", env, len(value))
        return False
    if not value.isascii():
        # 这一条不是洁癖：密钥最终要进 HTTP 头（`Authorization: ...`），非 ASCII
        # 字符在头部无法编码——配置里留一句中文说明，暴露出来的是 litellm 的
        # 编码异常，看不出真正原因。任何真密钥都是可打印 ASCII，不会误伤。
        bad = sum(1 for c in value if not c.isascii())
        logger.warning("%s 含 %d 个非 ASCII 字符，不可能是密钥，按未配置处理", env, bad)
        return False
    return True


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
        with span("llm.call", model=self.model, purpose="text") as current:
            resp = litellm.completion(**kwargs)  # type: ignore[attr-defined]
            usage = getattr(resp, "usage", None)
            # 属性必须在 span 结束**之前**写：结束后再写只会得到
            # "Setting attribute on ended span" 警告，值也进不去
            span_attrs(
                current,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )
        text = (resp.choices[0].message.content or "").strip()
        cost, priced = 0.0, True
        try:
            raw_cost = litellm.completion_cost(completion_response=resp)  # type: ignore[attr-defined]
            # `None` = litellm 找不到这个模型的价，**不是"免费"**
            priced = raw_cost is not None
            cost = float(raw_cost) if priced else 0.0
        except Exception:  # noqa: BLE001 - 算不出价不该让请求失败，但**不许按 0 记**
            # 未知模型走这里；具体金额由 `DailyBudget.record(priced=False)`
            # 按保守上限计入并计数。
            cost, priced = 0.0, False
        return LLMResult(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=cost,
            cost_priced=priced,
        )

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMToolTurn:
        """带 function calling 的一次调用（工具循环用）。

        `complete()` 是**纯文本**契约，被 CLI / eval / 问诊三处共用；把 tools 混进去
        会让那三处都要处理"可能没有文本"的情况，所以单开一个方法。代价是这里重复了
        一段计费代码——下面的价格计算与 `complete()` 逐行一致，改动时必须同步。
        """
        if not has_api_key(self.model):
            raise LLMUnavailable(
                f"未配置 {required_api_key_env(self.model)}（模型 {self.model}），"
                "无法生成回答；检索结果仍然可用"
            )
        litellm = self.litellm
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "timeout": self.settings.llm_timeout,
            "num_retries": self.settings.llm_max_retries,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        fallbacks = self.settings.llm_fallback_list
        if fallbacks:
            kwargs["fallbacks"] = fallbacks
        with span("llm.call", model=self.model, purpose="tools") as current:
            resp = litellm.completion(**kwargs)  # type: ignore[attr-defined]
            resp_usage = getattr(resp, "usage", None)
            span_attrs(
                current,
                prompt_tokens=int(getattr(resp_usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(resp_usage, "completion_tokens", 0) or 0),
            )
        message = resp.choices[0].message
        calls: list[ToolCallRequest] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            fn = getattr(raw_call, "function", None)
            calls.append(
                ToolCallRequest(
                    id=str(getattr(raw_call, "id", "") or ""),
                    name=str(getattr(fn, "name", "") or ""),
                    arguments=str(getattr(fn, "arguments", "") or ""),
                )
            )
        usage = getattr(resp, "usage", None)
        cost, priced = 0.0, True
        try:
            raw_cost = litellm.completion_cost(completion_response=resp)  # type: ignore[attr-defined]
            priced = raw_cost is not None
            cost = float(raw_cost) if priced else 0.0
        except Exception:  # noqa: BLE001 - 与 complete() 同一口径，理由见那里
            cost, priced = 0.0, False
        return LLMToolTurn(
            content=str(getattr(message, "content", "") or "").strip(),
            tool_calls=calls,
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
            settings.llm_spend_path,
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
                    # 父块必须一起带上：`build_context` 用它当上下文（检索用子块、
                    # 生成用父块），漏了它这里就退回"只给模型 300 字的片段"。
                    parent_id=h.parent_id,
                    parent_text=h.parent_text,
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
            # 用 `reserve()` 而不是 `check()`：它把"这次在途调用"按保守金额算进已用量，
            # 否则并发请求会在 spent 还没变时一起通过检查、再各自记账，把上限冲过 N 倍。
            # 记账也放在预留区间内，让"预留 → 记账"之间没有新的空隙。
            with self.budget.reserve():
                result = self.llm.complete(SYSTEM_PROMPT, user)
                # 定价为 0 的调用不必写文件（没有账要记）；**算不出价**的必须记，
                # 否则"未知 = 免费"会从这个判断里溜回来。
                if result.cost_usd or not result.cost_priced:
                    self.budget.record(result.cost_usd, priced=result.cost_priced)
                # 指标与账本同步更新：账本是事实来源，指标是可被抓取的视图。
                observe_llm_call(
                    cost_usd=result.cost_usd,
                    priced=result.cost_priced,
                    budget_usd=self.settings.llm_daily_budget_usd,
                )
        except BudgetExceeded as e:
            ans.refused = True
            ans.refusal_reason = str(e)
            return ans
        except LLMUnavailable as e:
            ans.refused = True
            ans.refusal_reason = str(e)
            ans.degraded.append("仅返回检索结果（生成回答不可用）")
            return ans
        except Exception as e:  # noqa: BLE001
            # 供应商侧的失败（密钥无效 401、限流 429、超时、服务不可用）不该让整个请求崩掉：
            # 检索结果本身有价值，用户应当拿到它 + 一句明确的失败原因；但**异常类型
            # 必须记进诊断**，否则就成了静默吞异常。
            ans.refused = True
            ans.refusal_reason = f"调用大模型失败：{type(e).__name__}: {e}"
            ans.degraded.append("仅返回检索结果（生成回答失败）")
            return ans

        ans.text, problems = validate_citations(result.text, len(citations))
        ans.prompt_tokens = result.prompt_tokens
        ans.completion_tokens = result.completion_tokens
        ans.cost_usd = result.cost_usd
        ans.degraded.extend(problems)
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
