"""域包协议：agent 内核与"具体知识/业务"之间**唯一**的耦合点。

kubedocs 现在有两个完全不同的域：

| | `k8s_docs` | `clinic` |
|---|---|---|
| 语料 | Kubernetes 官方文档（英文） | 中文医疗知识卡片 |
| 归一化 | 原样（FTS5 的 porter 够用） | jieba 预分词 |
| 嵌入 | bge-small-en（384 维） | bge-small-zh-v1.5（512 维） |
| 能力 | 检索 / 问答 | 问诊 / 挂号 / 紧急 |

内核里出现任何一处 `if domain == "clinic"`，这套东西就退化成"两个项目塞进一个仓库"。
所以内核只认 `DomainPack` 这张协议，域特有的东西（语料、归一化、嵌入、工具、评测集）
一律由域包提供。

协议只保留内核**真正需要**的 6 个方法：每多一个方法，加第三个域就多一处要实现、
多一处会写错。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # 只用于类型标注，避免 domains → store/retrieval 的循环导入
    from collections.abc import Callable

    from kubedocs.config import Settings
    from kubedocs.embed import Embedder
    from kubedocs.ingest import IngestReport
    from kubedocs.retrieve.hybrid import HybridRetriever
    from kubedocs.store.base import Store


@runtime_checkable
class DomainPack(Protocol):
    """一个域需要向内核提供的一切。"""

    name: str
    """域标识；同时作为索引目录名（`<index_dir>/<name>/`）。"""

    description: str
    """一句话说明这个域能回答什么。会被注入路由提示，也会在 `/domains` 端点暴露。"""

    normalizer_name: str
    """该域使用的归一化器名（`identity` / `chinese`）。必须与域内语言一致。"""

    embedding_model: str
    """该域使用的嵌入模型标识。写进索引元信息，也是换域时必须一起换的东西。"""

    embedding_dim: int
    """该域嵌入向量的维度。**必须与 `embedding_model` 的实际输出一致**：
    英文 bge-small-en-v1.5 是 384 维，中文 bge-small-zh-v1.5 是 512 维。
    两者混用不会报错，只会让向量矩阵与查询向量乘不上——所以这里显式声明。"""

    def corpus_root(self, settings: Settings) -> Path:
        """该域的语料根目录。"""

    def build_index(
        self,
        store: Store,
        embedder: Embedder,
        settings: Settings,
        *,
        rebuild: bool = False,
        limit_files: int | None = None,
        batch_files: int = 60,
        progress: Callable[[str], None] | None = None,
    ) -> IngestReport:
        """把该域语料灌入索引。返回摄取报告（内核据此打印统计）。

        `batch_files` 的含义由域自己解释（k8s 域是"每批提交的文件数"，
        clinic 域是"每次编码的卡片数"）——内核只负责把它传下去。
        """

    def retriever(self, store: Store, embedder: Embedder, settings: Settings) -> HybridRetriever:
        """构造该域的检索器。域不同，检索器内部配置可以不同。"""

    def tools(self) -> list[object]:
        """该域暴露给 agent 的工具列表。

        返回 `ToolSpec`（见 `agent/tools.py`）。**必须是纯数据描述**，
        不在这里执行任何 IO——工具的执行由内核统一编排与计时。
        """

    def eval_sets(self) -> list[Path]:
        """该域声明的金标集路径（相对仓库根）。

        **它只被"自省"消费**：`cli domains` 的打印与 `/domains` 的响应
        （`DomainRegistry.describe()`）。内核的评测命令走的是显式路径参数，
        不读这个字段——所以别把"声明了金标集"当成"闸门会跑它"。
        """


class DomainRegistry:
    """按名字找域包。与 `SkillRegistry` 同一套路数：注册 + 按名解析 + 可列举。"""

    _packs: dict[str, DomainPack] = {}

    @classmethod
    def register(cls, pack: DomainPack) -> None:
        if pack.name in cls._packs:
            raise ValueError(f"域 {pack.name!r} 已注册，不能重复注册")
        cls._packs[pack.name] = pack

    @classmethod
    def get(cls, name: str) -> DomainPack:
        if name not in cls._packs:
            raise KeyError(
                f"未知的域 {name!r}；已注册：{sorted(cls._packs)}。"
                "（域包通过 `kubedocs.domains.load_all()` 注册）"
            )
        return cls._packs[name]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._packs)

    @classmethod
    def describe(cls) -> list[dict[str, object]]:
        return [
            {
                "name": p.name,
                "description": p.description,
                "normalizer": p.normalizer_name,
                "embedding_model": p.embedding_model,
                "embedding_dim": p.embedding_dim,
                "eval_sets": [str(x) for x in p.eval_sets()],
            }
            for p in (cls._packs[n] for n in cls.names())
        ]

    @classmethod
    def clear(cls) -> None:
        """仅供测试重置注册表。"""
        cls._packs.clear()
