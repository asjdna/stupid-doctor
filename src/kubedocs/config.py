"""集中配置：环境变量 -> 类型化 Settings。

为什么用 pydantic-settings 而不是 `os.getenv` 分散读取：
1. 启动即校验（缺 key、维度写错、URL 格式错都在启动时报，而不是在第一次请求时炸）；
2. 单一真相源——想知道"这个服务有哪些可调项"只看这一个文件；
3. `.env` 与真实环境变量同源，容器里注入 env 即可，不需要改代码。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    database_url: str = "postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs"
    redis_url: str = "redis://127.0.0.1:6379/0"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # ---------- 嵌入 ----------
    embedding_provider: Literal["local", "api"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ---------- 检索 ----------
    top_k: int = Field(default=8, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
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

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例（lru_cache），便于测试用 `get_settings.cache_clear()` 重置。"""
    return Settings()
