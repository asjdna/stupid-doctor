"""配置测试：路径锚定、有效嵌入模型标识、校验器。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kubedocs.config import Settings, env_file_path, load_env_file, repo_root


class TestPathAnchoring:
    """相对路径必须锚定仓库根，绝不锚定进程 cwd。

    这是上一代项目踩过的坑：包被移动一层后，`__file__` 相对路径静默指向空目录，
    表现为"库里明明有数据却搜不到"，排查成本极高。
    """

    def test_relative_paths_resolve_under_repo_root(self):
        s = Settings(index_dir="data/index", corpus_dir="data/raw/k8s-docs")
        assert s.index_path.is_absolute()
        assert s.index_path == repo_root() / "data" / "index"
        assert s.corpus_path == repo_root() / "data" / "raw" / "k8s-docs"
        assert s.sqlite_path.parent == s.index_path
        assert s.vector_path.parent == s.index_path

    def test_absolute_paths_are_respected(self, tmp_path: Path):
        s = Settings(index_dir=str(tmp_path / "x"), corpus_dir=str(tmp_path / "y"))
        assert s.index_path == tmp_path / "x"
        assert s.corpus_path == tmp_path / "y"

    def test_index_files_live_inside_index_dir(self, tmp_path: Path):
        s = Settings(index_dir=str(tmp_path))
        assert s.sqlite_path.name == "kubedocs.db"
        assert s.vector_path.name == "chunk_vectors.npy"


class TestEffectiveEmbeddingModel:
    """索引里记的必须是**真正决定向量空间**的标识。

    否则一次 hashing 运行与一次真实模型运行会记成同一个模型名，
    闸门就会拿两种完全不同质量的运行互相比——静默的错误结论。
    """

    def test_hashing_reports_its_own_identity(self):
        s = Settings(embedding_provider="hashing", embedding_dim=384,
                     embedding_model="BAAI/bge-small-en-v1.5")
        assert s.effective_embedding_model == "hashing-384"
        assert s.effective_embedding_model != s.embedding_model

    def test_bge_reports_the_model_name(self):
        s = Settings(embedding_provider="bge", embedding_model="BAAI/bge-small-en-v1.5")
        assert s.effective_embedding_model == "BAAI/bge-small-en-v1.5"

    def test_hashing_dim_changes_identity(self):
        a = Settings(embedding_provider="hashing", embedding_dim=64)
        b = Settings(embedding_provider="hashing", embedding_dim=128)
        assert a.effective_embedding_model != b.effective_embedding_model


class TestValidators:
    def test_overlap_must_be_less_than_max(self):
        with pytest.raises(ValueError, match="必须小于"):
            Settings(chunk_max_chars=500, chunk_overlap_chars=500)

    def test_overlap_equal_to_max_is_rejected(self):
        with pytest.raises(ValueError):
            Settings(chunk_max_chars=200, chunk_overlap_chars=200)

    def test_top_k_bounds(self):
        with pytest.raises(ValueError):
            Settings(top_k=0)
        with pytest.raises(ValueError):
            Settings(top_k=101)


def test_llm_fallback_list_parsing():
    assert Settings(llm_fallbacks="a/b, c/d ,").llm_fallback_list == ["a/b", "c/d"]
    assert Settings(llm_fallbacks="").llm_fallback_list == []


def test_is_prod_flag():
    assert Settings(env="prod").is_prod is True
    assert Settings(env="local").is_prod is False


class TestLoadEnvFile:
    """回归：`.env` 必须被**导出到 os.environ**，否则第三方库读不到密钥。

    pydantic-settings 只把 .env 填进 Settings 对象，不碰 os.environ；
    而 LiteLLM、huggingface_hub 以及本项目自己的 has_api_key() 都读 os.environ。
    结果：`cp .env.example .env` 填好密钥后，模型调用仍报"未配置密钥"，
    README 描述的流程静默失效。这个缺陷被"回答层从未跑真实 LLM"掩盖了很久。
    """

    def test_values_are_exported(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("# 注释\nDEEPSEEK_API_KEY=sk-test-123\n", encoding="utf-8")
        added = load_env_file(env)
        assert "DEEPSEEK_API_KEY" in added
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-test-123"

    def test_real_environment_wins_over_file(self, tmp_path, monkeypatch):
        """容器里注入的密钥不该被 .env 覆盖。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "from-real-env")
        env = tmp_path / ".env"
        env.write_text("DEEPSEEK_API_KEY=from-file\n", encoding="utf-8")
        added = load_env_file(env)
        assert "DEEPSEEK_API_KEY" not in added
        assert os.environ["DEEPSEEK_API_KEY"] == "from-real-env"

    def test_blank_value_is_not_exported(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SOME_TOKEN", raising=False)
        env = tmp_path / ".env"
        env.write_text("SOME_TOKEN=\n", encoding="utf-8")
        assert load_env_file(env) == []
        assert "SOME_TOKEN" not in os.environ

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env_file(tmp_path / "nope.env") == []

    def test_env_file_path_is_anchored_to_repo_root(self):
        """必须锚定仓库根：pydantic-settings 默认按 cwd 找 .env。"""
        assert env_file_path() == repo_root() / ".env"
        assert env_file_path().is_absolute()
