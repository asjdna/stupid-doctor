"""摄取流水线测试：增量、幂等、删除同步、跨篇复用向量。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kubedocs.config import Settings
from kubedocs.embed import HashingEmbedder
from kubedocs.ingest import _sanity_check, file_digest, ingest, iter_corpus_files
from kubedocs.store.base import ChunkInput
from kubedocs.store.sqlite_store import SqliteStore

BODY_A = (
    "## Configure limits\n\n"
    "You can specify how much CPU and memory each container needs. "
    "The scheduler uses these values to decide which node can accept the Pod.\n\n"
    "## Notes\n\nLimits are enforced by the kubelet and the container runtime.\n"
)
BODY_B = (
    "## Eviction\n\n"
    "When a node runs out of memory the kubelet starts evicting pods. "
    "The eviction order depends on the quality of service class of each pod.\n"
)


def _write(corpus: Path, rel: str, title: str, body: str) -> None:
    p = corpus / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\ntitle: {title}\n---\n\n{body}\n", encoding="utf-8")


def _ingest(store: SqliteStore, settings: Settings, embedder: HashingEmbedder, **kw):
    return ingest(store, embedder, settings, **kw)


def test_missing_corpus_raises(tmp_path, store, embedder):
    s = Settings(index_dir=str(tmp_path / "i"), corpus_dir=str(tmp_path / "nope"),
                 embedding_provider="hashing", embedding_dim=32)
    with pytest.raises(FileNotFoundError):
        _ingest(store, s, embedder)


def test_full_ingest_then_idempotent(settings, store, embedder):
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    _write(settings.corpus_path, "sub/b.md", "Eviction", BODY_B)

    r1 = _ingest(store, settings, embedder)
    assert r1.files_scanned == 2
    assert r1.files_changed == 2
    assert r1.chunks_written == store.stats().n_chunks
    assert r1.chunks_embedded > 0
    assert r1.chunks_reused == 0

    # 第二次：完全没变 -> 一个文件都不该重新处理（这才是增量的意义）
    r2 = _ingest(store, settings, embedder)
    assert (r2.files_changed, r2.chunks_written, r2.chunks_embedded) == (0, 0, 0)
    assert r2.files_unchanged == 2
    assert store.stats().n_chunks == r1.chunks_written


def test_only_changed_file_is_reprocessed(settings, store, embedder):
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    _write(settings.corpus_path, "b.md", "Eviction", BODY_B)
    first = _ingest(store, settings, embedder)
    hashes_before = store.existing_doc_hashes()

    _write(settings.corpus_path, "a.md", "Resource Management",
           BODY_A + "\n## Added later\n\nThis section is brand new and should be indexed.\n")
    r2 = _ingest(store, settings, embedder)

    assert r2.files_changed == 1
    assert r2.files_unchanged == 1
    # 未变的文档连哈希都该原样不动（说明整篇被跳过了）
    assert store.existing_doc_hashes()["b.md"] == hashes_before["b.md"]
    # 改动的那篇：新内容可检索，旧内容已不在（是替换，不是叠加）
    assert store.keyword_search("brand", 10)
    assert store.get_chunks(["b::0000"]), "未变更文档的块不该被动过"
    assert store.stats().n_docs == 2
    assert store.stats().n_chunks >= first.chunks_written - 1


def test_removed_file_is_pruned(settings, store, embedder):
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    b = settings.corpus_path / "b.md"
    _write(settings.corpus_path, "b.md", "Eviction", BODY_B)
    first = _ingest(store, settings, embedder)

    b.unlink()
    r2 = _ingest(store, settings, embedder)
    assert r2.files_removed == 1
    assert r2.chunks_removed > 0
    assert store.stats().n_docs == 1
    assert store.stats().n_chunks == first.chunks_written - r2.chunks_removed
    assert store.keyword_search("evicting", 10) == []
    assert store.verify_fts() == "ok"


def test_limit_files_does_not_prune_the_rest(settings, store, embedder):
    """`--limit-files` 只处理语料子集，绝不能把"没出现在子集里"当成"已删除"。

    这是本项目最危险的一个默认行为：一旦搞错，一次冒烟跑就会把整个索引删掉。
    """
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    _write(settings.corpus_path, "b.md", "Eviction", BODY_B)
    _ingest(store, settings, embedder)
    assert store.stats().n_docs == 2

    r2 = _ingest(store, settings, embedder, limit_files=1)
    assert r2.files_removed == 0
    assert r2.chunks_removed == 0
    assert store.stats().n_docs == 2
    assert any("limit_files" in n for n in r2.notes)


def test_identical_content_reuses_vectors_from_index(settings, store, embedder):
    """同内容的两篇文档，第二篇应直接复用第一篇的向量（真实语料里模板小节极多）。"""
    _write(settings.corpus_path, "a.md", "Same Title", BODY_A)
    _write(settings.corpus_path, "b.md", "Same Title", BODY_A)
    r1 = _ingest(store, settings, embedder)
    # 批内去重：两篇内容一致，只编码一次
    assert r1.chunks_written > r1.chunks_embedded
    assert r1.chunks_embedded == 1

    # 删掉 b 再放回来：此时 b 的块不在库里，但同哈希的向量还在 -> 命中库内复用
    (settings.corpus_path / "b.md").unlink()
    _ingest(store, settings, embedder)
    _write(settings.corpus_path, "b.md", "Same Title", BODY_A)
    r3 = _ingest(store, settings, embedder)
    assert r3.chunks_reused > 0
    assert r3.chunks_embedded == 0


def test_intra_batch_dedup_with_duplicate_in_the_middle(settings, store, embedder):
    """回归测试：批内重复块**不在批尾**时，向量下标必须仍然对得上。

    真实全量摄取踩到过这个越界：重复块在中间，导致"用 pending 的下标去索引
    压缩后的矩阵"错位并 IndexError。这里刻意让重复出现在中间和开头。
    """
    _write(settings.corpus_path, "dup1.md", "Same Title", BODY_A)
    _write(settings.corpus_path, "other.md", "Different Title", BODY_B)
    _write(settings.corpus_path, "dup2.md", "Same Title", BODY_A)

    r = _ingest(store, settings, embedder)
    assert r.chunks_written == store.stats().n_chunks
    assert r.chunks_embedded == 2          # dup1 与 dup2 同内容，只编码一次
    # 三篇文档的块都拿到向量（否则 dense_search 会漏）
    assert store.dense_search(np.ones(32, dtype=np.float32), 50) != []
    assert len(store.doc_texts(["dup1", "dup2", "other"])) == 3


def test_empty_document_is_registered_to_avoid_rework(settings, store, embedder):
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    (settings.corpus_path / "empty.md").write_text("---\ntitle: Empty\n---\n", encoding="utf-8")
    r1 = _ingest(store, settings, embedder)
    assert r1.files_empty == 1
    r2 = _ingest(store, settings, embedder)
    assert r2.files_changed == 0          # 空文档也算"处理过"，不该每次重跑


def test_rebuild_clears_previous_index(settings, store, embedder):
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    _write(settings.corpus_path, "b.md", "Eviction", BODY_B)
    _ingest(store, settings, embedder)
    _write(settings.corpus_path, "b.md", "Eviction", "## Only\n\nCompletely different now.\n")
    r = _ingest(store, settings, embedder, rebuild=True)
    assert r.files_changed == 2
    assert store.stats().n_docs == 2


def test_report_summary_contains_key_numbers(settings, store, embedder):
    _write(settings.corpus_path, "a.md", "Resource Management", BODY_A)
    r = _ingest(store, settings, embedder)
    s = r.summary()
    assert "扫描 1 文件" in s
    assert "编码" in s and "复用" in s


class TestSanityCheck:
    """写入前的最后一道关：零向量会让块在向量路永远排最后，且完全不报错。"""

    def test_zero_vector_rejected(self, embedder):
        chunk = ChunkInput(chunk_id="a::0000", doc_id="a", ordinal=0, title="a",
                           breadcrumb="s", source_url="u", body="x", content_hash="h",
                           n_chars=1)
        with pytest.raises(RuntimeError, match="零向量"):
            _sanity_check({"a::0000": np.zeros(32, dtype=np.float32)}, [chunk], embedder)

    def test_missing_vector_rejected(self, embedder):
        chunk = ChunkInput(chunk_id="a::0000", doc_id="a", ordinal=0, title="a",
                           breadcrumb="s", source_url="u", body="x", content_hash="h",
                           n_chars=1)
        with pytest.raises(RuntimeError, match="没有向量"):
            _sanity_check({}, [chunk], embedder)

    def test_unnormalised_vector_is_fixed_in_place(self, embedder):
        chunk = ChunkInput(chunk_id="a::0000", doc_id="a", ordinal=0, title="a",
                           breadcrumb="s", source_url="u", body="x", content_hash="h",
                           n_chars=1)
        vecs = {"a::0000": np.full(32, 0.5, dtype=np.float32)}
        _sanity_check(vecs, [chunk], embedder)
        assert float(np.linalg.norm(vecs["a::0000"])) == pytest.approx(1.0)


def test_file_digest_changes_with_content(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("hello", encoding="utf-8")
    h1 = file_digest(p)
    p.write_text("hello world", encoding="utf-8")
    assert file_digest(p) != h1


def test_iter_corpus_files_is_sorted_and_limited(tmp_path):
    for name in ("b.md", "a.md", "c.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("x", encoding="utf-8")
    files = iter_corpus_files(tmp_path)
    assert [f.name for f in files] == ["a.md", "b.md", "c.md"]
    assert [f.name for f in iter_corpus_files(tmp_path, 2)] == ["a.md", "b.md"]
