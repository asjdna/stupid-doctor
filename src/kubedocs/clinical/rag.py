"""可替换的检索后端协议。

产品要"**外接 RAG 系统**辅助模型做疾病分析"，而眼下没有真实系统可对接。
先把检索收敛成**一个协议 + 一个默认实现**，检索调用就不会长进业务代码的每个角落：

- `LocalIndexBackend`：包住现有 `HybridRetriever`（BM25 + 向量 + RRF），**零行为变化**。

将来接真实系统时，照着 `LocalIndexBackend` 的形状再写一个实现即可。两条约定不能破：
**空列表 = 确实没有命中**（不是"后端坏了"），后端坏了要抛异常或让 `health()` 报
`ok=False`——把"没有"和"坏掉"混成同一个返回值，会让降级路径无法被测试。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from kubedocs.retrieve.hybrid import HybridRetriever


class RetrievedChunk(BaseModel):
    """检索回来的一个片段。字段刻意与 `retrieve.hybrid.RetrievedChunk` 同名，
    这样适配器是一次纯拷贝，而不是一次"翻译"——翻译层是 bug 的温床。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    title: str
    breadcrumb: str = ""
    source_url: str = ""
    body: str = ""
    """**命中的那一段**（子块）：精确、短。用它回答"命中的是哪一句"。"""

    parent_text: str = ""
    """命中块所属**父块**（整节）的正文，可能为空（索引建在父子切割之前）。

    与 `body` 是两种用途，别混：`body` 是命中位置，`parent_text` 是它所在的完整语境。
    只给 `body`，模型会拿着半句话答题；只给 `parent_text` 并在长度上截断，又把命中那句截掉了。
    """

    score: float = 0.0


class BackendHealth(BaseModel):
    """后端的健康状态。`ok=False` 时**必须**带上原因：静默不可用会让故障无法定位。"""

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
        同一个返回值，会让降级路径无法被测试。
        """
        ...

    def health(self) -> BackendHealth:
        """探活。用于启动自检与 `/healthz` 级别的可观测。"""
        ...


class LocalIndexBackend:
    """默认实现：包住仓库内的混合检索器。

    **不做任何加工**：不截断、不重排、不改分数。这一层只负责"把仓库内的检索
    变成协议形状"，任何质量调整都该发生在检索器里。
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
                # 用 `getattr`：这一层的输入是**鸭子类型**的检索器（测试替身、将来的
                # 外接 RAG 系统都可能没有父块概念），缺字段应当退化成"没有父块语境"，
                # 而不是让整条检索链 AttributeError。
                parent_text=getattr(c, "parent_text", ""),
                score=c.score,
            )
            for c in result.results
        ]

    def health(self) -> BackendHealth:
        """本地索引的"健康"就是"能不能检索一次"。这里只做形状检查，不发起真实查询：
        探活不该触发一次嵌入计算（那要几十毫秒到几十秒）。
        """
        ok = self._retriever is not None
        return BackendHealth(
            ok=ok,
            name=self.name,
            detail="" if ok else "检索器未构造（索引可能不可用）",
        )
