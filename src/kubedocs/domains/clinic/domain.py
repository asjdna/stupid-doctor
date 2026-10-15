"""clinic 域包：把医疗域接上检索与工具，并注册进平台。

| 环节 | 实现 |
|---|---|
| 语料 | `data/clinic/cards.json`（由 `scripts/clinic_build_cards.py` 从上游派生；不入库） |
| 索引 | 每张卡片 = 一篇文档 + 一个块；中文经 `chinese` 归一化器（jieba 预分词）进 FTS5 |
| 检索 | 复用 kubedocs 的混合检索内核（FTS5 BM25 + 向量 + RRF） |
| 工具 | `search_docs` / `citation_check`（`agent/rag_tools.py`） |
| 回答 | self-ask（分解 → 检索 → 合成，带引用重编号） |

## 稠密路是"可选的第二遍"

全量 31592 张卡片用 bge-small-zh 编码约需一小时，所以索引分两遍：

    python -m kubedocs.cli --domain clinic --provider hashing ingest   # 秒级，只有 BM25 有效
    python -m kubedocs.cli --domain clinic --hf-offline ingest         # 一小时后，稠密路也可用

**第一遍不是"假数据"**：FTS5 的中文 BM25 是真实有效的检索通道，
稠密路只是在它之上再加一路召回。别把降级当成完整能力。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from kubedocs.chunker import source_url_for
from kubedocs.config import Settings
from kubedocs.domains.clinic.pack import clinic_cards_path
from kubedocs.embed import Embedder, embed_texts
from kubedocs.ingest import IngestReport
from kubedocs.retrieve.hybrid import HybridRetriever
from kubedocs.store.base import ChunkInput, DocInput, Store

CARDS_PER_DOC = 1


def cards_to_inputs(
    cards: list[dict],
    *,
    limit: int | None = None,
) -> tuple[list[DocInput], list[ChunkInput]]:
    """卡片 -> 文档/块。

    每张卡片独立成篇（doc_id 含类别与键），这样引用能精确指到"哪张卡"，
    文档级评测也能按卡片判定。
    """
    docs: list[DocInput] = []
    chunks: list[ChunkInput] = []
    for i, card in enumerate(cards if limit is None else cards[:limit]):
        kind = str(card.get("kind", "card"))
        key = str(card.get("key", f"card{i}"))
        text = str(card.get("text", "")).strip()
        if not text:
            continue
        doc_id = f"{kind}/{key}"
        # 前缀里已经带了 `/clinic/`，路径里就**不要**再写一遍：
        # 否则引用 URL 会多出一层（`https://kubedocs.local/clinic/clinic/disease/xxx/`）。
        url = source_url_for(f"{kind}/{key}.md", "https://kubedocs.local/clinic/")
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        docs.append(
            DocInput(doc_id=doc_id, rel_path=f"{doc_id}.md", title=key, source_url=url, content_hash=digest)
        )
        chunks.append(
            ChunkInput(
                chunk_id=f"{doc_id}::0000",
                doc_id=doc_id,
                ordinal=0,
                title=key,
                breadcrumb=kind,
                source_url=url,
                body=text,
                content_hash=digest,
                n_chars=len(text),
            )
        )
    return docs, chunks


class ClinicPack:
    """clinic 域的 `DomainPack` 实现。"""

    name = "clinic"
    description = (
        "中文医疗问诊 / 挂号 / 紧急：症状导诊、科室建议、号源预约、急诊红线识别（规则轨 + 知识卡片检索轨）"
    )
    # 中文必须预分词，否则 FTS5 的 unicode61 只能按标点切分（见 normalize.py）
    normalizer_name = "chinese"
    # 中文语义向量必须用中文模型。注意维度不同（zh 是 512、en 是 384）——
    # 它必须随域一起切换，否则 clinic 索引里的向量矩阵与查询向量乘不上。
    embedding_model = "BAAI/bge-small-zh-v1.5"
    embedding_dim = 512

    def __init__(self, *, batch: int = 512) -> None:
        self.batch = batch

    def corpus_root(self, settings: Settings) -> Path:
        return clinic_cards_path()

    def load_cards(self) -> list[dict]:
        path = clinic_cards_path()
        if not path.exists():
            raise FileNotFoundError(
                f"clinic 语料不存在：{path}\n"
                "先跑 `python scripts/clinic_build_cards.py`（需要上游 build_knowledge.py 的产物）。\n"
                "注意：clinic 域的**规则轨**（红线/症状词典/挂号）不依赖它，"
                "没有语料时问诊与挂号仍可运行。"
            )
        with open(path, encoding="utf-8") as fh:
            data: list[dict] = json.load(fh)
        return data

    def build_index(
        self,
        store: Store,
        embedder: Embedder,
        settings: Settings,
        *,
        rebuild: bool = False,
        limit_files: int | None = None,
        batch_files: int = 512,
        progress: Callable[[str], None] | None = None,
    ) -> IngestReport:
        """把卡片灌入 clinic 域自己的索引。

        `limit_files` 在这里的含义是"只取前 N 张卡片"，便于冒烟与测试。
        `batch_files` 是每次编码的卡片数（默认 512，可用 `--batch-files` 调）。
        """
        log: Callable[[str], None] = progress if progress is not None else (lambda _m: None)
        report = IngestReport()
        if rebuild:
            store.rebuild_from_scratch()
        cards = self.load_cards()
        docs, chunks = cards_to_inputs(cards, limit=limit_files)
        report.files_scanned = len(cards) if limit_files is None else min(limit_files, len(cards))
        report.files_changed = len(docs)
        batch = max(1, int(batch_files))
        log(f"卡片 {len(cards)} 张 -> 文档 {len(docs)} 篇 / 块 {len(chunks)} 个（编码批 {batch}）")

        for start in range(0, len(chunks), batch):
            part_docs = {d.doc_id: d for d in docs}
            chunk_slice = chunks[start : start + batch]
            ids = {c.doc_id for c in chunk_slice}
            texts = [c.body for c in chunk_slice]
            log(f"  编码 {start + len(texts)}/{len(chunks)} …")
            mat = embed_texts(embedder, texts)
            vectors = {c.chunk_id: mat[i] for i, c in enumerate(chunk_slice)}
            store.upsert_docs([part_docs[i] for i in ids], chunk_slice, vectors)
            store.finalize()
            report.chunks_written += len(chunk_slice)
            report.chunks_embedded += len(chunk_slice)
        report.elapsed_s = 0.0
        report.notes.append(
            "clinic 索引完成。若用的是 hashing 占位向量，**只有 FTS5 中文 BM25 通道有效**；"
            "要启用稠密路需用 bge-small-zh 重跑一次全量编码。"
        )
        return report

    def retriever(self, store: Store, embedder: Embedder, settings: Settings) -> HybridRetriever:
        return HybridRetriever(store, embedder, settings)

    def tools(self) -> list[object]:
        from kubedocs.agent.rag_tools import make_citation_check_tool, make_search_tool

        # 检索器在运行时注入（域包只声明工具形状，不持有连接）
        return [make_search_tool(None, domain=self.name), make_citation_check_tool()]

    def eval_sets(self) -> list[Path]:
        """**刻意返回空**。

        这个字段的语义是"**`cli eval` 那套检索金标**"，而 clinic 的金标是另一种形状
        （`查询 -> 期望科室/期望红线判定`），由 `scripts/eval_clinic.py` 直接按路径读，
        **不走这个协议字段**。把一个形状不同的文件挂在这里，等于让 `domains` 输出一个
        "看起来能 `cli eval`、实则不行"的承诺——那正是本字段要防的事。
        """
        return []

    def triage_links(
        self, *, departments_only: bool = True, max_cards: int | None = None
    ) -> dict[str, list[str]]:
        """症状 -> 候选。供假设打分器使用。

        ## 默认**只取科室**，不取疾病

        一个症状（如"咳嗽"）在知识图谱里关联几十种疾病，而只关联两三个科室；
        打分是等权累加，把疾病一起放进候选会让科室被疾病数量淹没。两个理由让
        "只取科室"成为正确设计：

        1. **导诊的产出本来就是科室**（该挂哪个科），不是疾病诊断；
        2. 本域有明确边界：**不下诊断结论**。把疾病候选排到前面等于绕开这条边界。

        疾病侧信息仍保留在卡片正文里（检索轨会用它生成接地回答），只是不参与规则轨排序。
        """
        links: dict[str, list[str]] = {}
        cards = self.load_cards()
        for card in cards[:max_cards] if max_cards else cards:
            if card.get("kind") != "symptom":
                continue
            target = list(card.get("departments", []))
            if not departments_only:
                target += list(card.get("diseases", []))
            if target:
                links[str(card["key"])] = sorted(set(target))
        return links


pack = ClinicPack()
