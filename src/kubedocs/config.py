"""集中配置：环境变量 -> 类型化 Settings。

用 pydantic-settings 而不是散落的 `os.getenv`，为的是三件事：启动即校验（缺 key、
维度、URL 格式错都在启动时报，而不是第一次请求时炸）、可调项只有一个真相源（就是
本文件）、`.env` 与真实环境变量同源（容器里注入 env 即可，不必改代码）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    """仓库根目录（`src/kubedocs/config.py` -> 上溯 3 层）。

    默认数据路径锚定在代码位置上而不是进程 cwd，否则从别处调 CLI（或 pytest 换了
    rootdir）时 `data/...` 会解析到一个空目录，表现为"索引里明明有数据却搜不到"。

    上溯三层只在源码布局下成立。`pip install .` 会把包装进 site-packages，
    此时根目录落到 site-packages 内部，`data/` 指向一个不存在也写不进去的位置，
    服务仍能启动但 `n_chunks` 是 0、检索全部降级为"索引不可用"。
    容器里用 `KUBEDOCS_REPO_ROOT` 显式指向挂载点（如 `/app`）即可覆盖；
    源码布局下不设它也能正确推断。
    """
    override = os.environ.get("KUBEDOCS_REPO_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def env_file_path() -> Path:
    """`.env` 的位置，同样锚定仓库根而不是 cwd。"""
    return repo_root() / ".env"


def load_env_file(path: Path | None = None) -> list[str]:
    """把 `.env` 里的键导出到 `os.environ`，返回本次新增的键名。

    pydantic-settings 只把 `.env` 的值填进 `Settings` 对象，不会写进 `os.environ`；
    而 LiteLLM、huggingface_hub 等第三方库以及本项目自己的 `has_api_key()` 读的都是
    `os.environ`。少了这一步，`cp .env.example .env` 填好密钥后模型调用仍会报
    "未配置密钥"。

    `override=False`：真实环境变量优先于 `.env`，容器里注入的密钥不会被文件覆盖。
    """
    target = path or env_file_path()
    if not target.exists():
        return []
    from dotenv import dotenv_values

    added: list[str] = []
    for key, value in dotenv_values(target).items():
        # 空值等同于未设置：`.env.example` 里成片的 `KEY=` 不该把
        # "没有密钥"伪装成"有密钥"（否则 has_api_key 会误判为可用）
        if value is None or not value.strip() or key in os.environ:
            continue
        os.environ[key] = value
        added.append(key)
    return added


class Settings(BaseSettings):
    """全部运行期配置。字段名 -> 环境变量名由 env_prefix 统一成 KUBEDOCS_*。"""

    model_config = SettingsConfigDict(
        env_prefix="KUBEDOCS_",
        # 锚定仓库根：pydantic-settings 默认按 cwd 找 .env，
        # 从别处调用时会读不到配置（与本文件 repo_root 的理由相同）
        env_file=str(repo_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 应用 ----------
    log_level: str = "INFO"
    api_port: int = 8080

    # ---------- 存储 ----------
    # 同一 Store 协议下的三种实现（环境变量 KUBEDOCS_STORE，前缀由 env_prefix 自动加）：
    #   sqlite-vec : 默认。SQLite（FTS5 全文）+ sqlite-vec 向量索引，零外部进程/零镜像/零网络
    #   sqlite     : numpy 暴力内积 + .npy 侧车矩阵，作对照与回退
    #   pgvector   : Postgres + pgvector HNSW，需要 database_url 可达
    # 非法取值直接报错，绝不静默回退：静默回退会让"我明明切到 pgvector 了、
    # 为什么数据还是旧的"变成一个没有任何日志的谜。
    store: Literal["sqlite-vec", "sqlite", "pgvector"] = "sqlite-vec"
    database_url: str = "postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs"
    db_pool_min: int = 1
    db_pool_max: int = 10
    # pgvector HNSW 建索引参数：m 是图的出边数，ef_construction 是建图时的候选队列，
    # 两者都只影响召回率与建索引耗时，改完必须重建索引。
    pg_hnsw_m: int = Field(default=16, ge=2, le=100)
    pg_hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
    # 查询期候选队列：越大召回越高、越慢，只影响查询，不需要重建索引。
    # 默认 200 的由来：在 1.8 万块的真实索引上，ef=100 时 top-10 与精确检索的平均集合
    # 重合是 0.942，ef=200 升到 0.980，ef=400 只到 0.984 而延迟继续增长——收益已递减。
    pg_hnsw_ef_search: int = Field(default=200, ge=1, le=1000)
    # SQLite 页缓存上限（KiB，sqlite-vec 后端）。默认 128MB：1.8 万块 × 384 维的向量表
    # 约 29MB，而 SQLite 默认页缓存只有 2MB，KNN 会反复穿透页缓存读盘（p50 从百毫秒
    # 量级降到 12ms）。SQLite 按需分配页面，给小索引设大上限不会白占内存。
    sqlite_cache_kib: int = Field(default=131072, ge=2000)
    # 索引与语料的根目录；实际索引在 <index_dir>/<domain>/ 下，每个域一份、互不干扰。
    index_dir: str = "data/index"
    # 当前域。域包（domains/）自带语料、归一化器与评测集，内核只认这个名字。
    domain: str = "k8s_docs"

    # ---------- 文本归一化 ----------
    # identity=原样（英文，FTS5 的 porter+unicode61 够用）
    # chinese =jieba 预分词（中文没有空格，不分词 FTS5 会把整句当成一个词元）
    normalizer: Literal["identity", "chinese"] = "identity"

    # ---------- 嵌入 ----------
    # bge     : 本地 sentence-transformers（真实语义向量，首次运行需下载模型）
    # hashing : 无依赖的确定性哈希向量，仅供离线单测/降级演示，不代表真实检索质量
    # api     : 走远端 embedding 接口
    embedding_provider: Literal["bge", "hashing", "api"] = "bge"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    # bge-small-en-v1.5 官方说明 v1.5 起可省略 query 指令；保留开关以便做 A/B。
    query_instruction: str = ""
    # 模型下载源。留空 = 走 HuggingFace 官方；企业内网/受限网络下指向镜像站。
    # 做成本项目配置而不是让用户自己 export，是因为"模型能否下到"决定第一次运行能不能起来。
    hf_endpoint: str = ""
    # 模型已缓存时置 True，跳过一切联网校验。在无法访问 huggingface.co 的网络里，
    # 联网校验不是"慢"而是**挂死到超时**（载入一个已缓存模型可超过 10 分钟无返回）。
    hf_offline: bool = False

    # ---------- 会话状态 ----------
    # clinic 图的 checkpoint 落盘路径。**留空 = 内存态**，即会话状态只活在进程内、
    # 重启就丢。用路径而不是布尔开关：路径本身就是它需要的信息，一个字段表达一件事，
    # 也避免"开关 + 另一处写路径"两处状态不一致。
    clinic_state_db: str = ""

    # ---------- 检索 ----------
    top_k: int = Field(default=8, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    fts_candidate_k: int = Field(default=50, ge=1)
    dense_candidate_k: int = Field(default=50, ge=1)
    rerank_enable: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5
    max_context_chars: int = 8000
    # 组装上下文时同一文档最多收录几段：同篇相邻小节的向量高度相似，
    # 不限流会让一段答案被近重复片段挤满，上下文既冗余又少了别的来源。
    max_chunks_per_doc: int = Field(default=2, ge=1, le=10)

    # ---------- 切块 ----------
    chunk_max_chars: int = Field(default=1200, ge=200)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    chunk_min_chars: int = Field(default=80, ge=0)

    # ---------- 大模型 ----------
    llm_model: str = "deepseek/deepseek-chat"
    llm_fallbacks: str = ""
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, ge=1)
    # 临床主判要求模型回一个结构化契约 JSON（科室 + 病症排序 + 检查 + 用药 + 引用 +
    # 白话总结），中文很吃 token：额度 800 时 20 条里有 7 条被截断，而截断的表现是
    # "模型输出不符合契约"，会被误读成模型不听话。`max_tokens` 是上限而非目标，
    # 调高它只抬高最坏情况的上界，不增加日常成本。
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_daily_budget_usd: float = Field(default=5.0, ge=0.0)
    # litellm **算不出价**（价格表里没有这个模型）时，每次调用按这个金额保守计入
    # 当日预算。默认**不假设免费**：按 0 记会让预算闸门永远拦不住。
    # 真跑本地免费模型（例如 ollama）时可以置 0：那种情况下 litellm 会正常返回
    # 0.0（`cost_priced=True`），本来就不走这条保守路径，所以一般不需要改这里。
    llm_unpriced_call_usd: float = Field(default=0.25, ge=0.0)

    # ---------- 可观测 ----------
    # `log_level` 有真实分支（见 `observability.py`）：configure_logging() 设 root logger 级别。
    # /metrics 与它无关：Prometheus 指标始终可用。

    # ---------- 数据 ----------
    corpus_dir: str = "data/raw/k8s-docs"
    docs_url_prefix: str = "https://kubernetes.io/docs/"

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_lt_max(cls, v: int, info: ValidationInfo) -> int:
        # 重叠必须严格小于块长，否则切块无法前进
        mx = (info.data or {}).get("chunk_max_chars")
        if mx is not None and v >= mx:
            raise ValueError(f"chunk_overlap_chars({v}) 必须小于 chunk_max_chars({mx})")
        return v

    @property
    def llm_fallback_list(self) -> list[str]:
        """把逗号分隔的 fallback 模型名解析成列表（LiteLLM 的 fallbacks 需要 list）。"""
        return [m.strip() for m in self.llm_fallbacks.split(",") if m.strip()]

    # ---------- 路径解析（相对路径一律锚定仓库根，绝不锚定 cwd）----------
    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else repo_root() / p

    @property
    def corpus_path(self) -> Path:
        return self._resolve(self.corpus_dir)

    @property
    def index_root(self) -> Path:
        """索引**根**目录（不含域子目录）。迁移/体检脚本要的是它。"""
        return self._resolve(self.index_dir)

    @property
    def index_path(self) -> Path:
        """当前域的索引目录：`<index_dir>/<domain>/`。

        按域分目录而不是分表：两个域的语料、嵌入模型、维度、归一化方式都可能不同，
        共用一个库文件只会让"换域时忘了换表"这类错误无从发现。
        """
        return self.index_root / self.domain

    @property
    def llm_spend_path(self) -> Path:
        """LLM 花费账本的位置：**全局一份**，不在域目录下。

        预算是"这份部署今天还能花多少"的约束，与用哪个域的语料无关。按域分账会让
        匿名调用方只要在请求里切换 `domain`，就能把日上限翻成"域数 × 预算"；
        也会把 clinic 会话的花费记进应用默认域的目录，审计那个域的人看不到这笔钱。
        账本因此锚在 `index_root` 上，两个 LLM 路径（`generate.Answerer` 与
        `agent/clinic_react`）都从这里取。
        """
        return self.index_root / "llm_spend.json"

    @property
    def effective_embedding_model(self) -> str:
        """真正决定向量空间的那个标识，写进索引元信息并参与闸门比较。

        `hashing` 提供商并不会加载 `embedding_model` 里写的那个模型，所以这里换成
        `hashing-<dim>`。若索引只记配置名，一次 hashing 运行与一次真实模型运行会被
        记成同一个模型，闸门就会拿两种质量完全不同的运行互相比较。
        """
        if self.embedding_provider == "hashing":
            return f"hashing-{self.embedding_dim}"
        return self.embedding_model

    @property
    def sqlite_path(self) -> Path:
        return self.index_path / "kubedocs.db"

    @property
    def vector_path(self) -> Path:
        return self.index_path / "chunk_vectors.npy"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例（lru_cache），便于测试用 `get_settings.cache_clear()` 重置。"""
    return Settings()
