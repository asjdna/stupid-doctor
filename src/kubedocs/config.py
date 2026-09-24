"""集中配置：环境变量 -> 类型化 Settings。

为什么用 pydantic-settings 而不是 `os.getenv` 分散读取：
1. 启动即校验（缺 key、维度写错、URL 格式错都在启动时报，而不是在第一次请求时炸）；
2. 单一真相源——想知道"这个服务有哪些可调项"只看这一个文件；
3. `.env` 与真实环境变量同源，容器里注入 env 即可，不需要改代码。
"""
from __future__ import annotations

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


class Settings(BaseSettings):
    """全部运行期配置。字段名 -> 环境变量名由 env_prefix 统一成 KUBEDOCS_*。"""

    model_config = SettingsConfigDict(
        env_prefix="KUBEDOCS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 应用 ----------
    env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    api_port: int = 8080

    # ---------- 存储 ----------
    # 本地/CI 默认用嵌入式 SQLite（零外部依赖）；生产可换成 Postgres+pgvector。
    # 两者实现同一个 Store 协议，切换只影响配置，不影响检索与生成代码。
    database_url: str = "postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs"
    redis_url: str = "redis://127.0.0.1:6379/0"
    db_pool_min: int = 1
    db_pool_max: int = 10
    index_dir: str = "data/index"

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
    def index_path(self) -> Path:
        return self._resolve(self.index_dir)

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
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例（lru_cache），便于测试用 `get_settings.cache_clear()` 重置。"""
    return Settings()
