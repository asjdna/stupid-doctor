"""集中配置：环境变量 -> 类型化 Settings。

为什么用 pydantic-settings 而不是 `os.getenv` 分散读取：
1. 启动即校验（缺 key、维度写错、URL 格式错都在启动时报，而不是在第一次请求时炸）；
2. 单一真相源——想知道"这个服务有哪些可调项"只看这一个文件；
3. `.env` 与真实环境变量同源，容器里注入 env 即可，不需要改代码。
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

    存在的意义：默认数据路径必须**锚定在代码位置上**，不能依赖进程 cwd。
    否则从别处调 CLI（或 pytest 换了 rootdir）时，`data/...` 会解析到一个
    空目录，"索引里明明有数据却搜不到"这种事故排查起来极贵。
    """
    return Path(__file__).resolve().parents[2]


def env_file_path() -> Path:
    """`.env` 的位置，同样锚定仓库根而不是 cwd。"""
    return repo_root() / ".env"


def load_env_file(path: Path | None = None) -> list[str]:
    """把 `.env` 里的键**导出到 `os.environ`**，返回本次新增的键名。

    为什么必须显式做这一步（而不是只靠 pydantic-settings 的 `env_file`）：

    pydantic-settings 只把 `.env` 的值填进 `Settings` 对象，**不会写进 `os.environ`**。
    而第三方库（LiteLLM、huggingface_hub 等）以及本项目自己的 `has_api_key()`
    读的都是 `os.environ`。结果就是 `cp .env.example .env` 填好密钥之后，
    模型调用仍然报"未配置密钥"——README 描述的那套流程**静默失效**。

    这个缺陷正是"回答层从未用真实 LLM 跑过"掩盖住的：所有单测都打桩，
    打桩路径不经过密钥读取。

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
    env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    api_port: int = 8080

    # ---------- 存储 ----------
    # 同一 Store 协议下的四种实现（选型理由与实测数字见 docs/adr/0004 与 0005）：
    #   sqlite-vec : 默认。SQLite（FTS5 全文）+ sqlite-vec 向量索引，零外部进程/零镜像/零网络
    #   sqlite     : 旧实现（numpy 暴力内积 + .npy 侧车矩阵），保留作对照与回退
    #   pgvector   : 生产实现（Postgres + pgvector HNSW），需要 database_url 可达
    #   milvus     : 可选实现（SQLite FTS5 全文 + Milvus Lite 本地文件里的 HNSW），无需外部服务
    # 环境变量名是 KUBEDOCS_STORE（env_prefix 自动加前缀）。
    # 非法取值**直接报错**，绝不静默回退到某个默认实现：静默回退会让"我明明切到
    # pgvector 了为什么数据还是旧的"变成一个没有任何日志的谜。
    store: Literal["sqlite-vec", "sqlite", "pgvector", "milvus"] = "sqlite-vec"
    database_url: str = "postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs"
    redis_url: str = "redis://127.0.0.1:6379/0"
    db_pool_min: int = 1
    db_pool_max: int = 10
    # pgvector HNSW 建索引参数（理由见 ADR-0004：m 决定图的出边数，
    # ef_construction 决定建图时的候选队列，两者都只影响召回率与建索引耗时）。
    pg_hnsw_m: int = Field(default=16, ge=2, le=100)
    pg_hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
    # 查询期候选队列：越大召回越高、越慢。**只影响查询，不需要重建索引**。
    #
    # 默认 200 是被本机实测逼出来的（2026-10-09，18861 块真实索引、50 条金标 ×3 轮，
    # 同轮对照、原始数据在 `data/eval/store_bench/latency_pgvector_ef*.json`）：
    #   ef=100 → top-10 与 numpy 精确检索**逐位相同** 36/50，平均集合重合 0.9420，dense p50 4.29ms
    #   ef=200 → 43/50，0.9800，p50 5.79ms（多召回 7 条查询的代价是 +1.5ms）
    #   ef=400 → 44/50，0.9840，p50 7.24ms（再多 1 条、再贵 1.45ms，**收益已递减**）
    # 归因与"为什么 pgvector 的混合检索质量仍低于冻结基线"是两件事，见 ADR-0004 第九节：
    # 那条落差来自全文路（Postgres 核心没有 BM25），与 ef_search 无关。
    pg_hnsw_ef_search: int = Field(default=200, ge=1, le=1000)
    # SQLite 页缓存上限（KiB，sqlite-vec 后端）。
    #
    # 这个参数是被实测逼出来的，不是抄来的：18861×384 的向量表 29MB，而 SQLite 默认
    # 页缓存只有 2MB，于是每次 KNN 都在反复穿透页缓存读盘——实测 KNN p50
    # 默认 66~147ms，调到 128MB 后 **12ms**（同一份数据、同一台机器）。
    # 只影响本地默认后端；SQLite 按需分配页面，给小索引设大上限不会白占内存。
    sqlite_cache_kib: int = Field(default=131072, ge=2000)
    # Milvus Lite 数据文件的**位置**（本地文件模式：`MilvusClient("<path>.db")`，
    # 不需要任何外部服务/容器）。留空 = `<index_dir>/<domain>/milvus.db`，
    # 与 `sqlite_path` 完全同一套"按域分目录"的组织方式：两个域的语料、维度、
    # 归一化方式都可能不同，共用一个数据目录只会让"切域时忘了切集合"无从发现。
    # 相对路径锚定仓库根（与 index_dir 一致），绝对路径原样使用。
    # **必须以 .db 结尾**：milvus-lite 只认以 .db 结尾的本地数据目录或 unix/http/tcp
    # 形式的远端地址，别的一律在打开时抛 `uri ... is illegal`，所以本实现在构造时先自己
    # 挡一道并给人话提示。注意 `.db` 结尾的路径在 milvus-lite 里是**目录**（实测：里面有
    # LOCK / manifest.json / wal / parquet 段文件），且**同一目录只允许一个进程打开**。
    milvus_db: str = ""
    # Milvus 侧 HNSW 的建索引参数（理由见 ADR-0005）。M / efConstruction
    # **只能重建索引才能改**：本实现把实际用到的取值记进索引元信息，
    # 配置与索引不一致时**直接报错**——"你以为调了、其实索引还是老的"正是本项目禁止的静默降级。
    milvus_hnsw_m: int = Field(default=16, ge=2, le=2048)
    milvus_hnsw_ef_construction: int = Field(default=200, ge=8, le=4096)
    # 查询期候选队列（HNSW 的 `ef`）：越大召回越高、越慢，**不需要重建索引**。
    # 取值与实测召回见 ADR-0005 第五节。
    milvus_hnsw_ef_search: int = Field(default=128, ge=1, le=4096)
    # 度量。默认 COSINE，与 sqlite-vec（`distance_metric=cosine`）和 pgvector
    # （`vector_cosine_ops`）一致；向量已归一化时 IP 与 COSINE 严格同序，
    # 保留这个开关只是为了让"度量到底有没有影响"可以在真实数据上 A/B，
    # 而不是让它变成一个没人注意的隐性变量。
    milvus_metric: Literal["COSINE", "IP"] = "COSINE"
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
    # 为什么要做成本项目配置而不是让用户自己 export：模型能否下到，决定了
    # "第一次运行能不能起来"，属于部署配置的一部分，应当与其它配置同源可查。
    hf_endpoint: str = ""
    # 模型已缓存时置 True，跳过一切联网校验。
    # 不是微调：在无法访问 huggingface.co 的网络里，联网校验不是"慢"，而是
    # **挂死到超时**（实测载入一个已缓存模型 >10 分钟无返回）。
    hf_offline: bool = False

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
    llm_max_tokens: int = Field(default=800, ge=1)
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_daily_budget_usd: float = Field(default=5.0, ge=0.0)
    # litellm **算不出价**（价格表里没有这个模型）时，每次调用按这个金额保守计入
    # 当日预算。默认**不假设免费**：按 0 记会让预算闸门永远拦不住（缺陷记录见
    # generate.UNPRICED_CALL_FALLBACK_USD 的说明，0.25 的取值推导也在那里）。
    # 真跑本地免费模型（例如 ollama）时可以置 0：那种情况下 litellm 会正常返回
    # 0.0（`cost_priced=True`），本来就不走这条保守路径，所以一般不需要改这里。
    llm_unpriced_call_usd: float = Field(default=0.25, ge=0.0)

    # ---------- 可观测 ----------
    otel_enable: bool = True
    otel_exporter: Literal["console", "otlp", "none"] = "console"
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ---------- 数据 ----------
    corpus_dir: str = "data/raw/k8s-docs"
    docs_url_prefix: str = "https://kubernetes.io/docs/"

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_lt_max(cls, v: int, info: ValidationInfo) -> int:
        # 允许相等（等于重叠不合法会死循环），但必须严格小于 max
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
    def effective_embedding_model(self) -> str:
        """真正决定向量空间的那个标识，写进索引元信息并参与闸门比较。

        为什么不能直接用 `embedding_model`：`hashing` 提供商并不会加载
        `BAAI/bge-small-en-v1.5`，但配置里的名字仍是它。若索引只记配置名，
        一次 hashing 运行与一次真实模型运行会**记成同一个模型**，
        闸门就会拿两种完全不同质量的运行互相比较——这是静默的错误结论来源。
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

    @property
    def milvus_path(self) -> Path:
        """Milvus（milvus-lite 本地文件模式）的数据目录。

        默认与 `sqlite_path` 同目录：索引的"位置"按域组织，换域就换一整套文件，
        不会出现"切了域但集合名没切"这类事故。`KUBEDOCS_MILVUS_DB` 可覆盖。
        注意它虽然在路径上以 `.db` 结尾，在 milvus-lite 眼里是**目录**（见 `milvus_db` 的注释）。
        """
        if self.milvus_db:
            return self._resolve(self.milvus_db)
        return self.index_path / "milvus.db"

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例（lru_cache），便于测试用 `get_settings.cache_clear()` 重置。"""
    return Settings()
