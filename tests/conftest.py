"""共享测试夹具。

## 为什么默认用 HashingEmbedder 而不是真实模型

单测要快、要离线、要确定性。加载 bge-small 每个测试会话要多花 20 秒以上，
还会因为模型版本变化导致结果抖动。真实模型的检索质量由 `cli eval` 在
真实语料上评测（那才是它的位置），单测只验证**管道正确性**。
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
    s = SqliteStore(db_path=tmp_path / "t.db", vector_path=tmp_path / "t.npy",
                    dim=DIM, embedding_model="hashing-test")
    yield s
    s.close()


def write_doc(corpus: Path, rel: str, title: str, body: str) -> Path:
    path = corpus / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n---\n\n{body}\n", encoding="utf-8")
    return path
