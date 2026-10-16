"""摄取流水线：语料目录 -> 解析 -> 切块 -> 嵌入 -> 写入索引。

增量摄取不是为了快，而是正确性要求：全量重建意味着"任何一次摄取都必须从头成功，
中途失败等于没有索引"，而语料是持续变更的（K8s 文档每周都在改）。所以这里做三个
层次的跳过：

1. **文档级**：文件内容哈希没变 -> 整篇跳过（不解析、不切块、不编码）；
2. **块级**：块的 content_hash 库里已有 -> 复用旧向量，不重新编码。K8s API reference
   有大量逐字相同的模板小节，同一段文字在几十个文档里出现，编码一次就够；
3. **删除同步**：语料里消失的文件，其文档与块一并从索引清除。

写入按批进行：每批 N 个文件就 `upsert + finalize` 一次，于是内存有界、进度可见，
且中途失败时**已完成的部分是有效的**，下次接着跑即可。

## 解析层（多格式）

"读文件"这一步交给 `parse.py`（PDF / Office / HTML / CSV / TXT / MD 各有解析器），
本模块只消费 `ParsedDoc`。三条与现场演示直接相关的约束在这里落地：

- **编码不静默丢字**：文本类格式的编码探测结果计进报告（`utf-8` / `gb18030` …）；
- **扫描件必点名**：无文本层的 PDF 计 `files_no_text` 并在 notes 里写
  "该文件无文本层（疑似扫描件），已跳过"，**绝不建空块**（空块 = 搜不到但查不出原因）；
- **单文件失败不影响整批**：解析异常计 `files_failed` + 原因进 notes，本批继续。
  失败的文件**不登记**（`continue`，不写 DocInput）：它可能只是暂时读不了
  （文件被占用、网络盘掉线），登记了就会因为"哈希没变"而永远不再重试；
  而扫描件/空文档是文件的固有属性，登记它们避免每次白跑。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kubedocs.chunker import (
    Chunk,
    DocumentTree,
    Parent,
    chunk_blocks_tree,
    chunk_document_tree,
    doc_id_from_path,
)
from kubedocs.config import Settings
from kubedocs.embed import BgeEmbedder, Embedder, embed_texts
from kubedocs.parse import MAX_FILE_BYTES, NO_TEXT_LAYER_MSG, ParsedDoc, discover, discover_report, parse
from kubedocs.store.base import ChunkInput, DocInput, ParentInput, Store, Vector

# 报告里逐条列出的细节上限：语料里坏掉 5000 个文件时，报告不该变成 5000 行日志。
# 计数永远是准的（summary 里），只是不逐条列。
_MAX_DETAIL_NOTES = 50

_MAX_FILE_MB = MAX_FILE_BYTES / (1024 * 1024)


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
    files_no_text: int = 0  # 抽不出文本（扫描件 PDF / 空内容）—— 必须在报告里点名
    files_failed: int = 0  # 解析失败（文件 + 原因），本批继续
    files_oversize: int = 0  # 超过大小上限，发现阶段就跳过
    chunks_written: int = 0
    parents_written: int = 0
    """写入的**父块**数。与 `chunks_written` 是两个数：父块是"几段语境"，
    子块是"几个检索单位"。只报其中一个，就没法回答"父子切割到底有没有生效"。"""
    chunks_removed: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    n_over_length: int = 0
    batches: int = 0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    notes_dropped: int = 0  # 超过 `_MAX_DETAIL_NOTES` 后只计数，不逐条列

    @property
    def embed_savings_pct(self) -> float:
        total = self.chunks_embedded + self.chunks_reused
        return 100.0 * self.chunks_reused / total if total else 0.0

    def note(self, text: str) -> None:
        """记一条人可读细节（文件级原因）。超上限只计数。"""
        if len(self.notes) < _MAX_DETAIL_NOTES:
            self.notes.append(text)
        else:
            self.notes_dropped += 1

    def summary(self) -> str:
        return (
            f"扫描 {self.files_scanned} 文件（变更 {self.files_changed} / 未变 "
            f"{self.files_unchanged} / 删除 {self.files_removed} / 空文档 {self.files_empty}）"
            f"；写入 {self.chunks_written} 块（父块 {self.parents_written}），删除 {self.chunks_removed} 块；"
            f"编码 {self.chunks_embedded} 块，复用 {self.chunks_reused} 块"
            f"（省 {self.embed_savings_pct:.1f}%）；超长被截断 {self.n_over_length} 块；"
            f"无文本层跳过 {self.files_no_text} 个，超大跳过 {self.files_oversize} 个，"
            f"解析失败 {self.files_failed} 个；耗时 {self.elapsed_s:.1f}s"
        )


def iter_corpus_files(corpus_dir: Path, limit: int | None = None) -> list[Path]:
    """列出语料里**受支持**的文件，按相对路径排序（排序保证多机结果一致）。

    受支持的后缀是 `parse.py` 的解析器注册表（PDF / Office / HTML / CSV / TXT / MD），
    不再写死 `*.md`；隐藏目录与超大文件由 `parse.discover` 跳过。
    `ingest()` 走的是 `parse.discover_report`（同一个发现逻辑），因为它还要拿到
    "被跳过的超大文件"清单写进报告。
    """
    files = discover(corpus_dir)
    if limit is not None:
        files = files[:limit]
    return files


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _parse_file(path: Path) -> ParsedDoc:
    """解析一个文件。`parse()` 自己已经不抛异常，这里再兜一层是防止"兜底逻辑本身出错"。"""
    try:
        return parse(path)
    except Exception as exc:  # 单文件失败不许影响整批，原因原样上报
        return ParsedDoc(error=f"{type(exc).__name__}: {exc}")


def _tree_for(rel: str, doc: ParsedDoc, settings: Settings) -> DocumentTree:
    """解析结果 -> **父子两层**。分派规则只有一条：**块自带位置就走位置路径**。

    标题型格式（md/txt/html/docx）解析出来是一个"整篇正文"块（`location` 为空），
    标题层级在正文的 `#` 行里，交给 `chunk_document_tree` 派生父块（标题小节）与
    子块（小节内的段落装箱）—— html/docx 顺带复用同一套切块规则，不另写一套。

    容器型格式（PDF 页 / 表格块 / 幻灯片）的块自带 `p.7` / `Sheet1!A1:C20` / `slide 3`，
    走 `chunk_blocks_tree`，位置作为面包屑首段进入引用，且**父块不跨位置**。
    """
    if doc.error or not doc.blocks:
        return DocumentTree(doc_id=doc_id_from_path(rel), title=doc.title or "", source_url="")
    if all(block.location for block in doc.blocks):
        return chunk_blocks_tree(
            rel,
            doc.blocks,
            title=doc.title,
            parent_max_chars=settings.chunk_parent_max_chars,
            child_target_chars=settings.chunk_max_chars,
            child_overlap=settings.chunk_overlap_chars,
            min_chars=settings.chunk_min_chars,
            url_prefix=settings.docs_url_prefix,
        )
    text = "\n\n".join(block.text for block in doc.blocks)
    return chunk_document_tree(
        rel,
        text,
        parent_max_chars=settings.chunk_parent_max_chars,
        child_target_chars=settings.chunk_max_chars,
        child_overlap=settings.chunk_overlap_chars,
        min_chars=settings.chunk_min_chars,
        url_prefix=settings.docs_url_prefix,
    )


def _citation_breadcrumb(breadcrumb_parts: list[str], location: str) -> str:
    """块的引用面包屑：把物理位置并进去。

    为什么加字段之外还要这一步：`ChunkInput.breadcrumb` 是既有的引用通道
    （-> `ChunkHit.citation`）。容器型格式的位置已经拼在面包屑首段（见 `chunk_blocks`），
    这里只兜防"有位置但没进面包屑"的情况，md 等标题型格式因此**一字不变**。

    参数收成两个基本类型（而不是收 `Chunk`）：父块与子块都要用同一套写法，
    而它们不是同一个类；用鸭子类型硬套只会让这里出现 `type: ignore`。
    """
    breadcrumb = " › ".join(breadcrumb_parts)
    if location and not breadcrumb.startswith(location):
        return f"{location} › {breadcrumb}" if breadcrumb else location
    return breadcrumb


def _chunks_to_inputs(chunks: Iterable[Chunk], doc_id: str) -> list[ChunkInput]:
    return [
        ChunkInput(
            chunk_id=c.chunk_id,
            doc_id=doc_id,
            ordinal=c.index,
            title=c.title,
            breadcrumb=_citation_breadcrumb(c.breadcrumb, c.location),
            source_url=c.source_url,
            body=c.text,
            content_hash=c.content_hash,
            n_chars=len(c.text),
            parent_id=c.parent_id,
        )
        for c in chunks
    ]


def _parents_to_inputs(parents: Iterable[Parent], doc_id: str) -> list[ParentInput]:
    """父块 -> 存储契约。

    引用面包屑与子块走**同一个函数**：同一段文字在检索结果与模型上下文里的出处写法
    必须一致，否则排查时会对不上号。
    """
    return [
        ParentInput(
            parent_id=p.parent_id,
            doc_id=doc_id,
            ordinal=p.index,
            title=p.title,
            breadcrumb=_citation_breadcrumb(p.breadcrumb, p.location),
            source_url=p.source_url,
            body=p.text,
            n_chars=len(p.text),
            content_hash=p.content_hash,
            location=p.location,
            n_children=p.n_children,
        )
        for p in parents
    ]


def _record_outcome(
    report: IngestReport,
    rel: str,
    doc: ParsedDoc,
    chunks: list[Chunk],
    encodings: dict[str, int],
) -> None:
    """把"这个文件被怎么处理了"写进报告：无文本层 / 空内容 / 编码 / 逐条警告。"""
    if not chunks:
        if doc.scanned:
            report.files_no_text += 1
            report.note(f"{NO_TEXT_LAYER_MSG}：{rel}")
        else:
            report.files_empty += 1
            report.note(f"{doc.skip_reason or '解析后没有可索引的文本，已跳过'}：{rel}")
    for warning in doc.warnings:
        report.note(f"{warning}：{rel}")
    if doc.encoding:
        encodings[doc.encoding] = encodings.get(doc.encoding, 0) + 1


def collect_plan(
    store: Store,
    files: list[Path],
    root: Path,
    settings: Settings,
    *,
    detect_removals: bool = True,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """第一遍：只读文件哈希，判断哪些文件需要重切、哪些文件已从语料消失。

    刻意与"切块 + 编码"分开：这一遍只碰磁盘 IO，不做昂贵计算，所以能快速给出
    "本次要干多少活"的预估。

    `detect_removals=False` 用在 `--limit-files` 冒烟跑上：此时 `files` 只是语料的
    一个**子集**，把"没出现在子集里"当成"已删除"会把整个索引删掉，所以由调用方显式关掉。
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
    store: Store,
    embedder: Embedder,
    settings: Settings,
    *,
    rebuild: bool = False,
    limit_files: int | None = None,
    batch_files: int = 60,
    progress: Callable[[str], None] | None = None,
) -> IngestReport:
    """跑一次摄取。`progress` 收到的是人类可读的进度行。

    `batch_files` 默认取小值（60）而不是"一批干完"：批大小决定**中断时最多丢多少工作**。
    全量语料下每 250 个文件的编码要 8 分钟以上，其间一次崩溃就会丢掉全部；分批提交的
    额外代价只是多几次矩阵重写，可忽略。
    """
    log = progress or (lambda _msg: None)
    report = IngestReport()
    t0 = time.perf_counter()
    root = settings.corpus_path

    if rebuild:
        log("重建模式：清空现有索引")
        store.rebuild_from_scratch()

    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在：{root}（先跑 scripts/fetch_corpus.py）")

    # 只走一次目录树：`discover_report` 同时带回"被跳过的超大文件"，那些也必须在报告里
    # 点名（否则"我明明放进去了"无从解释）。
    discovery = discover_report(root)
    report.files_oversize = len(discovery.oversize)
    for rel, size in discovery.oversize:
        report.note(f"超过 {_MAX_FILE_MB:.0f} MB 上限，已跳过：{rel}（{size / 1048576:.1f} MB）")
    files = discovery.files if limit_files is None else discovery.files[:limit_files]
    report.files_scanned = len(files)
    if limit_files is not None:
        report.notes.append(
            f"limit_files={limit_files}：只处理前 {len(files)} 个文件，不是完整索引，且本次不做删除同步"
        )

    todo, removed = collect_plan(store, files, root, settings, detect_removals=limit_files is None)
    report.files_changed = len(todo)
    report.files_unchanged = report.files_scanned - len(todo)
    report.files_removed = len(removed)
    log(f"计划：{len(todo)} 个文件需处理，{len(removed)} 个已从语料移除")

    if removed:
        report.chunks_removed += store.delete_docs(set(removed))

    encodings: dict[str, int] = {}
    for start in range(0, len(todo), batch_files):
        batch = todo[start : start + batch_files]
        docs: list[DocInput] = []
        chunk_inputs: list[ChunkInput] = []
        parent_inputs: list[ParentInput] = []
        for path, digest in batch:
            rel = _rel_posix(path, root)
            doc_id = doc_id_from_path(rel)
            doc = _parse_file(path)
            if doc.error:
                # 不登记（不写 DocInput）：坏文件可能只是暂时读不了（被占用/网络盘掉线），
                # 登记了就会因"哈希没变"永远不再重试。原因逐条进报告。
                report.files_failed += 1
                report.note(f"解析失败：{rel} —— {doc.error}")
                continue
            tree = _tree_for(rel, doc, settings)
            chunks = tree.children
            title = chunks[0].title if chunks else (doc.title or doc_id.rsplit("/", 1)[-1])
            url = chunks[0].source_url if chunks else ""
            docs.append(
                DocInput(doc_id=doc_id, rel_path=rel, title=title, source_url=url, content_hash=digest)
            )
            chunk_inputs.extend(_chunks_to_inputs(chunks, doc_id))
            # 父块与子块一起写：只写子块的话，命中后拿不到完整语境，
            # 而 `parent_id` 会指向一行根本不存在的父块（比没有父块更难排查）。
            parent_inputs.extend(_parents_to_inputs(tree.parents, doc_id))
            _record_outcome(report, rel, doc, chunks, encodings)

        vectors = _vectors_for(store, embedder, chunk_inputs, report, log)
        store.upsert_docs(docs, chunk_inputs, vectors, parent_inputs)
        store.finalize()
        report.batches += 1
        report.chunks_written += len(chunk_inputs)
        report.parents_written += len(parent_inputs)
        log(
            f"  批 {report.batches}: 文件 {len(docs)}，块 {len(chunk_inputs)}，"
            f"父块 {len(parent_inputs)}，累计编码 {report.chunks_embedded}"
        )

    if encodings:
        detail = "、".join(f"{name}×{count}" for name, count in sorted(encodings.items()))
        report.notes.append(
            f"本次解析文件的文本编码：{detail} —— 按 utf-8-sig/utf-8 -> gb18030 顺序严格探测，"
            f"两种都失败时用替换字符更少的那个并统计 U+FFFD，**从不使用 errors=ignore 静默丢字**"
        )
    if report.notes_dropped:
        report.notes.append(f"另有 {report.notes_dropped} 条同类细节未逐条列出（上面的计数是准的）")

    if removed and report.batches == 0:
        # **纯删除的一轮**：上面的循环体一次都没进，`finalize()` 也就没被调用过。
        # 而 `delete_docs()` 只标记"脏"，`vector_row ↔ 矩阵行号`的对齐要由
        # `finalize()` 重建。漏掉它的后果是索引带着错位活下去：重启后要么静默
        # 返回错误内容，要么越界 IndexError。删除本身也必须压实映射。
        store.finalize()

    report.elapsed_s = time.perf_counter() - t0
    return report


def _vectors_for(
    store: Store,
    embedder: Embedder,
    chunk_inputs: list[ChunkInput],
    report: IngestReport,
    log: Callable[[str], None],
) -> dict[str, Vector]:
    """给这批块准备向量：先复用（库内 + 批内），剩下的才编码。"""
    by_hash: dict[str, str] = {}  # content_hash -> 批内首个 chunk_id
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
    # 压缩后的矩阵，只要重复块不是批内最后一个，就会越界。
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
            log(
                f"    {done}/{total}  {rate:5.1f} 条/s  已用 {elapsed / 60:4.1f} 分  "
                f"本批预计还需 {eta_min:4.1f} 分"
            )

        mat = embed_texts(embedder, texts, progress=_report)
        vec_of_hash = {c.content_hash: mat[i] for i, c in enumerate(firsts)}
        for c in pending:
            out[c.chunk_id] = vec_of_hash[c.content_hash]
        report.chunks_embedded += len(texts)

    _sanity_check(out, chunk_inputs, embedder)
    return out


def _sanity_check(vectors: dict[str, Vector], chunks: list[ChunkInput], embedder: Embedder) -> None:
    """写入前的最后一道关：不放过"零向量"和"未归一化"两种静默退化。

    零向量与所有查询的余弦相似度都是 0，这些块在向量路永远排最后，表现为
    "明明库里有，就是搜不到"，且没有任何报错。
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
