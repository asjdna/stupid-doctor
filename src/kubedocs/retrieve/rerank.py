"""精排（可选）：cross-encoder 对 RRF 的候选做二次打分。

## 它解决 RRF 的什么短板

RRF 只看名次，**完全不管块的内容**。所以"包含答案的那一块"和"只是词面相似
的一块"在 RRF 里可能并列。cross-encoder 把 (query, chunk) 拼成一对一起编码，
能做真正的交互式相关性判断——这也是"混合召回 + cross-encoder 精排"成为
工业标准三段式（recall → fuse → rerank）的原因。

## 为什么默认关闭

1. 它是**串行**的：每个候选一次前向，20 个候选比一次向量检索慢两个数量级；
2. 它需要额外模型（bge-reranker-base 约 1.1GB）；
3. 它的收益必须在评测集上量出来才作数——没测之前默认关，是这个项目的纪律。

失败一律降级为"保持 RRF 顺序并记录诊断"，绝不因为精排不可用而让整个查询失败：
精排是锦上添花，召回才是生命线。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """精排器接口：输入候选（已按 RRF 排序），输出重排后的候选。"""

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers cross-encoder 包装（懒加载）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: object | None = None

    @property
    def model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover - 取决于环境
                raise RuntimeError("缺少 sentence-transformers，无法使用精排") from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        raw = self.model.predict(pairs, batch_size=self.batch_size)  # type: ignore[attr-defined]
        return [float(x) for x in raw]


def apply_rerank(reranker: Reranker, query: str, texts: Sequence[str]) -> tuple[list[float], str | None]:
    """给候选打分；不可用时返回 (空列表, 原因)，由调用方保持原序。

    把"失败"表达成返回值而不是异常：调用点需要的是"降级并记录"，
    而不是把异常往上扔给用户。
    """
    try:
        return reranker.score(query, texts), None
    except Exception as e:  # noqa: BLE001 - 精排任何失败都不应影响召回结果
        return [], f"{type(e).__name__}: {e}"
