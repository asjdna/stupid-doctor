"""共享测试夹具：单测一律用离线的 `HashingEmbedder`，不加载真实模型。

真实模型的检索质量由 `cli eval` 在真实语料上评测，单测只验证**管道正确性**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubedocs.config import Settings  # noqa: E402
from kubedocs.embed import HashingEmbedder  # noqa: E402
from kubedocs.store.sqlite_store import SqliteStore  # noqa: E402

DIM = 32

# 供应商密钥环境变量：测试必须在任何情况下都拿不到它们
_PROVIDER_KEY_VARS = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_API_KEY",
    "KUBEDOCS_LLM_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试**绝不**调用真实模型，也绝不花钱。

    两道闸门：清掉所有供应商密钥，让测试只走"无密钥降级"分支；再阻断 `.env` 导入——
    `cli.main()` 会把仓库根的 `.env` 导进 os.environ，密钥因而绕过降级分支，
    本该断言"降级"的用例会真的去发请求。
    """
    for var in _PROVIDER_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("kubedocs.config.load_env_file", lambda *a, **k: [])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """索引与语料都指向 tmp_path，绝不碰真实 data/ 目录。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return Settings(
        index_dir=str(tmp_path / "index"),
        corpus_dir=str(corpus),
        embedding_provider="hashing",
        embedding_dim=DIM,
        embedding_model="hashing-test",
        docs_url_prefix="https://kubernetes.io/docs/",
        rerank_enable=False,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(
        db_path=tmp_path / "t.db", vector_path=tmp_path / "t.npy", dim=DIM, embedding_model="hashing-test"
    )
    yield s
    s.close()


def write_doc(corpus: Path, rel: str, title: str, body: str) -> Path:
    path = corpus / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n---\n\n{body}\n", encoding="utf-8")
    return path
