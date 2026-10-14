"""可替换的检索后端协议（方案 §4）。

## 为什么要抽这一层

产品要"**外接 RAG 系统**辅助模型做疾病分析"，而眼下没有真实系统可对接（只做功能测试）。
如果不先定协议，检索调用会长进业务代码的每个角落，将来接真实系统就是一次重写。
所以这里把检索收敛成**一个协议 + 两个实现**：

- `LocalIndexBackend`：包住现有 `HybridRetriever`（BM25 + 向量 + RRF），**零行为变化**；
- `HttpRagBackend`：**显式未实现**的桩。

## 为什么桩要抛异常而不是返回空列表

返回空列表会让"接错了后端"表现为"检索不到东西"——一个安静的错误，
排查时会被误判成语料问题。抛 `NotImplementedError` 是**大声**的：
没接就是没接。等真实 RAG 系统给出 API 形状再填，**不猜**接口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from kubedocs.retrieve.hybrid import HybridRetriever


class RetrievedChunk(BaseModel):
    """检索回来的一个片段。字段刻意与 `retrieve.hybrid.RetrievedChunk` 对齐（同名字段），
    这样适配器是一次纯拷贝，而不是一次"翻译"——翻译层是 bug 的温床。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    title: str
    breadcrumb: str = ""
    source_url: str = ""
    body: str = ""
    score: float = 0.0


class BackendHealth(BaseModel):
    """后端的健康状态。`ok=False` 时**必须**带上原因：静默不可用是本项目反复吃亏的地方。"""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    name: str
    detail: str = ""


class RetrievalBackend(Protocol):
    """检索后端。实现者只需保证语义，不需要继承任何基类。"""

    name: str

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        """返回按相关度降序的片段。空列表 = 确实没有命中，**不是**"后端坏了"。

        后端坏了要抛异常（或让 `health()` 报 `ok=False`）——把"没有"和"坏掉"混成
        同一个返回值，是让降级路径无法被测试的常见原因。
        """
        ...

    def health(self) -> BackendHealth:
        """探活。用于启动自检与 `/healthz` 级别的可观测。"""
        ...


class LocalIndexBackend:
    """默认实现：包住仓库内的混合检索器。

    **不做任何加工**：不截断、不重排、不改分数。这一层只负责"把仓库内的检索
    变成协议形状"，任何质量调整都该发生在检索器里（那里有消融测试守着）。
    """

    name = "local-index"

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        result = self._retriever.search(query, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                title=c.title,
                breadcrumb=c.breadcrumb,
                source_url=c.source_url,
                body=c.body,
                score=c.score,
            )
            for c in result.results
        ]

    def health(self) -> BackendHealth:
        """本地索引的"健康"就是"能不能检索一次"。这里只做形状检查，不发起真实查询：
        探活不该触发一次嵌入计算（那要几十毫秒到几十秒，见 README 性能基准）。
        """
        ok = self._retriever is not None
        return BackendHealth(
            ok=ok,
            name=self.name,
            detail="" if ok else "检索器未构造（索引可能不可用）",
        )


class HttpRagBackend:
    """外接 RAG 系统的**占位实现**（方案 §4）。

    刻意不写猜测性的 HTTP 调用：不知道对方的路径、鉴权方式、返回字段就写代码，
    写出来的一定是错的，而且会被误当成"已对接"。真实系统到位后，
    在 `search()` 里填调用、在 `health()` 里填探活即可，协议与上层都不用动。
    """

    name = "http-rag"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        raise NotImplementedError(
            "外接 RAG 后端尚未实现：需要真实系统的接口形状（路径 / 鉴权 / 返回字段）。"
            "在这之前请使用 LocalIndexBackend；见 docs/design/0001-医疗辅助诊断产品方案.md §4。"
        )

    def health(self) -> BackendHealth:
        return BackendHealth(ok=False, name=self.name, detail=f"未实现（endpoint={self.endpoint}）")


def assert_is_backend(obj: Any) -> RetrievalBackend:
    """启动期形状检查。比 mypy 更晚、比运行到一半才发现更早。

    为什么需要它：后端是**注入**的，接错一个只实现了 `search` 没有 `health` 的对象，
    mypy 在动态装配处看不出来。启动时喊一声，比线上第一次探活失败便宜得多。
    """
    for attr in ("name", "search", "health"):
        if not hasattr(obj, attr):
            raise TypeError(f"{type(obj).__name__} 不满足 RetrievalBackend 协议：缺少 {attr}")
    return cast("RetrievalBackend", obj)
