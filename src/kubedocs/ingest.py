"""摄取流水线：语料目录 -> 切块 -> 嵌入 -> 写入索引。

## 为什么"增量"不是优化而是正确性要求

全量重建每次都要重新编码 1.8 万块（本机约 1.2 分钟）。单看一次可以忍，
但它带来一个更糟的后果：**任何一次摄取都必须从头成功，中途失败等于没有索引**。
而线上语料是持续变更的（K8s 文档每周都在改），能"只处理变了的文件"才谈得上
把摄取放进流水线。所以这里做三个层次的跳过：

1. **文档级**：文件内容哈希没变 -> 整篇跳过（不解析、不切块、不编码）；
2. **块级**：块的 content_hash 在库里已有 -> 复用旧向量，不重新编码；
3. **删除同步**：语料里消失的文件，其文档与块一并从索引清除。

第 2 条覆盖的是真实存在的浪费：K8s API reference 有大量逐字相同的模板小节，
同一段文字在几十个文档里出现，编码一次就够。

## 分批提交而不是一把梭

每批 N 个文件就 `upsert + finalize` 一次。好处是内存有界、进度可见，
且中途失败时**已完成的部分是有效的**（下次接着跑，不会全部重来）。
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kubedocs.chunker import Chunk, chunk_document, doc_id_from_path
from kubedocs.config import Settings
from kubedocs.embed import BgeEmbedder, Embedder, embed_texts
from kubedocs.store.base import ChunkInput, DocInput, Vector
from kubedocs.store.sqlite_store import SqliteStore


def file_digest(path: Path) -> str:
    """文档级内容指纹。

    读的是原始字节而不是"切块后的文本"：front matter 的改动（例如 weight、
    description）虽然不进正文，但往往意味着文档结构变了，重切一次更安全。
    """
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class IngestReport:
    files_scanned: int = 0
    files_changed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_empty: int = 0
    chunks_written: int = 0
    chunks_removed: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    n_over_length: int = 0
    batches: int = 0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def embed_savings_pct(self) -> float:
        total = self.chunks_embedded + self.chunks_reused
        return 100.0 * self.chunks_reused / total if total else 0.0

    def summary(self) -> str:
        return (
            f"扫描 {self.files_scanned} 文件（变更 {self.files_changed} / 未变 "
            f"{self.files_unchanged} / 删除 {self.files_removed} / 空文档 {self.files_empty}）"
            f"；写入 {self.chunks_written} 块，删除 {self.chunks_removed} 块；"
            f"编码 {self.chunks_embedded} 块，复用 {self.chunks_reused} 块"
            f"（省 {self.embed_savings_pct:.1f}%）；超长被截断 {self.n_over_length} 块；"
            f"耗时 {self.elapsed_s:.1f}s"
        )


def iter_corpus_files(corpus_dir: Path, limit: int | None = None) -> list[Path]:
    """列出语料里的 markdown，按相对路径排序（排序保证多机结果一致）。"""
    files = sorted(p for p in corpus_dir.rglob("*.md") if p.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _chunks_to_inputs(chunks: Iterable[Chunk], doc_id: str) -> list[ChunkInput]:
    return [
        ChunkInput(
            chunk_id=c.chunk_id, doc_id=doc_id, ordinal=c.index, title=c.title,
            breadcrumb=" › ".join(c.breadcrumb), source_url=c.source_url,
            body=c.text, content_hash=c.content_hash, n_chars=len(c.text),
        )
        for c in chunks
    ]


def collect_plan(
    store: SqliteStore, files: list[Path], root: Path, settings: Settings,
    *, detect_removals: bool = True,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """第一遍：只读文件哈希，判断哪些文件需要重切、哪些文件已从语料消失。

    刻意与"切块 + 编码"分开：这一遍只碰磁盘 IO，不做任何昂贵计算，
    所以能快速给出"本次要干活多少"的预估。

    `detect_removals=False` 用在 `--limit-files` 冒烟跑上：此时 `files` 只是语料的
    一个**子集**，把"没出现在子集里"当成"已删除"会把整个索引删掉——
    这是本项目最危险的一个默认行为，所以由调用方显式关掉。
    """
    known = store.existing_doc_hashes()
    todo: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in files:
        rel = _rel_posix(path, root)
        seen.add(rel)
        digest = file_digest(path)
        if known.get(rel) == digest:
            continue
        todo.append((path, digest))
    removed = sorted(set(known) - seen) if detect_removals else []
    return todo, removed


def ingest(
    store: SqliteStore,
    embedder: Embedder,
    settings: Settings,
    *,
    rebuild: bool = False,
    limit_files: int | None = None,
    batch_files: int = 200,
    progress: Callable[[str], None] | None = None,
) -> IngestReport:
    """跑一次摄取。`progress` 收到的是人类可读的进度行。"""
    log = progress or (lambda _msg: None)
    report = IngestReport()
    t0 = time.perf_counter()
    root = settings.corpus_path

    if rebuild:
        log("重建模式：清空现有索引")
        store.rebuild_from_scratch()

    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在：{root}（先跑 scripts/fetch_corpus.py）")

    files = iter_corpus_files(root, limit_files)
    report.files_scanned = len(files)
    if limit_files is not None:
        report.notes.append(f"limit_files={limit_files}：只处理前 {len(files)} 个文件，"
                            "不是完整索引，且本次不做删除同步")

    todo, removed = collect_plan(store, files, root, settings,
                                 detect_removals=limit_files is None)
    report.files_changed = len(todo)
    report.files_unchanged = report.files_scanned - len(todo)
    report.files_removed = len(removed)
    log(f"计划：{len(todo)} 个文件需处理，{len(removed)} 个已从语料移除")

    if removed:
        report.chunks_removed += store.delete_docs(set(removed))

    for start in range(0, len(todo), batch_files):
        batch = todo[start:start + batch_files]
        docs: list[DocInput] = []
        chunk_inputs: list[ChunkInput] = []
        for path, digest in batch:
            rel = _rel_posix(path, root)
            doc_id = doc_id_from_path(rel)
            chunks = chunk_document(
                rel, path.read_text(encoding="utf-8", errors="replace"),
                max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap_chars,
                min_chars=settings.chunk_min_chars, url_prefix=settings.docs_url_prefix,
            )
            title = chunks[0].title if chunks else doc_id.rsplit("/", 1)[-1]
            url = chunks[0].source_url if chunks else ""
            docs.append(DocInput(doc_id=doc_id, rel_path=rel, title=title,
                                 source_url=url, content_hash=digest))
            if not chunks:
                report.files_empty += 1
            chunk_inputs.extend(_chunks_to_inputs(chunks, doc_id))

        vectors = _vectors_for(store, embedder, chunk_inputs, report, log)
        store.upsert_docs(docs, chunk_inputs, vectors)
        store.finalize()
        report.batches += 1
        report.chunks_written += len(chunk_inputs)
        log(f"  批 {report.batches}: 文件 {len(docs)}，块 {len(chunk_inputs)}，"
            f"累计编码 {report.chunks_embedded}")

    report.elapsed_s = time.perf_counter() - t0
    return report


def _vectors_for(
    store: SqliteStore,
    embedder: Embedder,
    chunk_inputs: list[ChunkInput],
    report: IngestReport,
    log: Callable[[str], None],
) -> dict[str, Vector]:
    """给这批块准备向量：先复用（库内 + 批内），剩下的才编码。"""
    by_hash: dict[str, str] = {}          # content_hash -> 批内首个 chunk_id
    for c in chunk_inputs:
        by_hash.setdefault(c.content_hash, c.chunk_id)

    reused = store.existing_hash_to_vector(set(by_hash))
    out: dict[str, Vector] = {}

    # 1) 库内已有同内容的向量
    for c in chunk_inputs:
        v = reused.get(c.content_hash)
        if v is not None:
            out[c.chunk_id] = v
    report.chunks_reused += len(out)

    # 2) 剩下的去重后编码（批内重复只编码一次，再广播给同内容的块）
    pending = [c for c in chunk_inputs if c.chunk_id not in out]
    first_of_hash: dict[str, str] = {}
    for c in pending:
        first_of_hash.setdefault(c.content_hash, c.chunk_id)
    # 注意：必须用**去重后的列表**做下标基准。用 enumerate(pending) 的下标去索引
    # 压缩后的矩阵，只要重复块不是批内最后一个，就会越界（真实语料上确实踩到过）。
    firsts = [c for c in pending if first_of_hash[c.content_hash] == c.chunk_id]
    texts = [c.body for c in firsts]

    if texts:
        if isinstance(embedder, BgeEmbedder):
            over = embedder.n_over_length(texts)
            if over:
                report.n_over_length += over
                log(f"  提示：{over} 块超过嵌入模型最大长度，尾部内容不会被编码")
        log(f"  编码 {len(texts)} 条（本批 {len(pending)} 块去重后）…")

        def _report(done: int, total: int, elapsed: float) -> None:
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_min = (total - done) / rate / 60 if rate > 0 else 0.0
            log(f"    {done}/{total}  {rate:5.1f} 条/s  已用 {elapsed/60:4.1f} 分  "
                f"本批预计还需 {eta_min:4.1f} 分")

        mat = embed_texts(embedder, texts, progress=_report)
        vec_of_hash = {c.content_hash: mat[i] for i, c in enumerate(firsts)}
        for c in pending:
            out[c.chunk_id] = vec_of_hash[c.content_hash]
        report.chunks_embedded += len(texts)

    _sanity_check(out, chunk_inputs, embedder)
    return out


def _sanity_check(vectors: dict[str, Vector], chunks: list[ChunkInput],
                  embedder: Embedder) -> None:
    """写入前的最后一道关：不放过"零向量"和"未归一化"两种静默退化。

    零向量会与所有查询的余弦相似度为 0，于是这些块在向量路永远排最后——
    表现为"明明库里有，就是搜不到"，且没有任何报错。
    """
    for c in chunks:
        v = vectors.get(c.chunk_id)
        if v is None:
            raise RuntimeError(f"块 {c.chunk_id} 没有向量")
        if v.shape[0] != embedder.dim:
            raise RuntimeError(f"块 {c.chunk_id} 向量维度 {v.shape[0]} != {embedder.dim}")
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise RuntimeError(f"块 {c.chunk_id} 得到零向量（正文可能为空或全是停用词）")
        if abs(norm - 1.0) > 1e-3:
            vectors[c.chunk_id] = (v / norm).astype(np.float32)
