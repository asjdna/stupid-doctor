"""文本嵌入：把块正文与查询映射到同一向量空间。

提供两种实现，用途截然不同，别混：

| 实现 | 用途 | 说明 |
|---|---|---|
| `BgeEmbedder` | 真实检索 | 本地 sentence-transformers，首次运行需下载模型 |
| `HashingEmbedder` | 单测 / 离线降级 | 无依赖、确定性、**不代表真实语义质量** |

`HashingEmbedder` 存在的意义是让"切块→写入→FTS→融合→引用"这条**管道**能被
快速、离线、可重复地测;它只擅长字面重合,任何语义改写(同义、换语序)都会退化。
所以它绝不出现在评测结论里——评测必须跑 BgeEmbedder。
"""
from __future__ import annotations

import os
import time
import zlib
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from kubedocs.config import Settings
from kubedocs.store.base import Vector


class EmbedderError(RuntimeError):
    """嵌入不可用（模型缺失、维度不符等），调用方应给出可操作的提示。"""


class Embedder(Protocol):
    """嵌入器最小接口。`dim` 必须与索引配置一致，否则矩阵乘法会崩。"""

    dim: int
    name: str

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


class HashingEmbedder:
    """确定性哈希向量：词元签名散列到固定维度，sublinear TF + L2 归一化。

    只用标准库（zlib.crc32），不依赖 torch，也不下载任何东西。
    刻意用带符号散列：词元映射到 ±1，让无关词元互相抵消，缓解哈希碰撞带来的
    "假相似"（纯正值累加会让任意两段文本都有正相关）。
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in "".join(
            ch.lower() if ch.isalnum() else " " for ch in text).split() if w]
        # 加二元词元，保留一点词序信息（单测里"pod 限制"与"限制 pod"应当可区分）
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        return words + bigrams

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in self._tokens(text):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, cnt in counts.items():
                h = zlib.crc32(tok.encode("utf-8"))
                col = h % self.dim
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[i, col] += sign * (1.0 + np.log(cnt))
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class BgeEmbedder:
    """本地 sentence-transformers 嵌入（默认 BAAI/bge-small-en-v1.5，384 维）。

    模型**懒加载**：只在第一次 encode 时载入。这样切块、写库、CLI 的 `--help`
    等路径都不必付出几秒的加载代价。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", *,
                 batch_size: int = 64, dim: int = 384, query_instruction: str = "",
                 hf_endpoint: str = "", hf_offline: bool = False) -> None:
        self.name = model_name
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.query_instruction = query_instruction
        self.hf_endpoint = hf_endpoint
        self.hf_offline = hf_offline
        self._model: object | None = None

    @property
    def model(self) -> object:
        if self._model is None:
            if self.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
                # 必须在 huggingface_hub 被导入/读取配置**之前**设置
                os.environ["HF_ENDPOINT"] = self.hf_endpoint
            if self.hf_offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - 取决于环境
                raise EmbedderError(
                    "缺少 sentence-transformers，无法使用本地嵌入；"
                    "请安装依赖，或把 embedding_provider 设为 hashing 做离线演示"
                ) from e
            self._model = SentenceTransformer(self.name)
            got = self._model.get_embedding_dimension()
            if got is not None:
                self.dim = int(got)
        return self._model

    def encode(self, texts: Sequence[str], *, is_query: bool = False,
               show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = list(texts)
        if is_query and self.query_instruction:
            payload = [f"{self.query_instruction}{t}" for t in payload]
        vecs = self.model.encode(  # type: ignore[attr-defined]
            payload, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        out = np.asarray(vecs, dtype=np.float32)
        if out.ndim != 2 or out.shape[1] != self.dim:
            raise EmbedderError(f"模型输出形状 {out.shape} 与 dim={self.dim} 不符")
        return out

    def n_over_length(self, texts: Sequence[str]) -> int:
        """统计有多少条文本会被静默截断。

        这一步值得做：sentence-transformers 遇到超长文本是**默默截断**的，
        于是"块尾部内容检索不到"会变成一个没有任何日志的静默缺陷。
        """
        if not texts:
            return 0
        tok = self.model.tokenizer  # type: ignore[attr-defined]
        max_len = int(self.model.max_seq_length)  # type: ignore[attr-defined]
        enc = tok(list(texts), add_special_tokens=True, truncation=False, padding=False)
        return sum(1 for ids in enc["input_ids"] if len(ids) > max_len)


def build_embedder(settings: Settings) -> Embedder:
    """按配置构造嵌入器。"""
    provider = settings.embedding_provider
    dim = settings.embedding_dim
    if provider == "hashing":
        return HashingEmbedder(dim=dim)
    if provider == "bge":
        return BgeEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            dim=dim,
            query_instruction=settings.query_instruction,
            hf_endpoint=settings.hf_endpoint,
            hf_offline=settings.hf_offline,
        )
    raise EmbedderError(f"不支持的 embedding_provider={provider!r}（可选 bge / hashing / api）")


def embed_texts(embedder: Embedder, texts: Sequence[str], *, is_query: bool = False,
                batch_size: int = 256,
                progress: Callable[[int, int, float], None] | None = None) -> Vector:
    """分批编码并拼成矩阵。

    分批的意义不只是显存/内存：进度可见 + 单批失败可定位（一次 2 万条崩了
    没法知道崩在哪）。`progress(done, total, elapsed_s)` 让调用方能把
    "还要多久"报给用户——编码是整条链路最慢的一步，没有进度就等于不可观测。
    """
    chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        # 显式分支而不是 **kwargs：Embedder 协议只承诺 encode(texts, is_query=...)，
        # show_progress 是 BgeEmbedder 特有的，混进协议会污染抽象。
        if isinstance(embedder, BgeEmbedder):
            part = embedder.encode(batch, is_query=is_query, show_progress=False)
        else:
            part = embedder.encode(batch, is_query=is_query)
        chunks.append(np.asarray(part, dtype=np.float32))
        if progress is not None:
            progress(min(i + batch_size, len(texts)), len(texts), time.perf_counter() - t0)
    if not chunks:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.vstack(chunks)
