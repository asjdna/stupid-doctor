"""clinic 域包与检索接线测试：卡片 -> 索引 -> 中文检索。

不加载真实模型：HashingEmbedder 只负责建索引，断言只覆盖 BM25 通道，
稠密通道的语义质量由离线全量编码负责——测试因此秒级完成且不联网。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kubedocs.config import Settings
from kubedocs.domains import DomainRegistry, load_all
from kubedocs.domains.clinic.domain import ClinicPack, cards_to_inputs
from kubedocs.embed import HashingEmbedder
from kubedocs.normalize import ChineseNormalizer
from kubedocs.store.sqlite_store import SqliteStore

CARDS = [
    {
        "kind": "symptom",
        "key": "咳嗽",
        "text": "咳嗽。可能相关的疾病：支气管炎、肺炎。建议就诊科室：呼吸内科。",
        "diseases": ["支气管炎"],
        "departments": ["呼吸内科"],
    },
    {
        "kind": "symptom",
        "key": "头痛",
        "text": "头痛。可能相关的疾病：偏头痛。建议就诊科室：神经内科。",
        "diseases": ["偏头痛"],
        "departments": ["神经内科"],
    },
    {
        "kind": "disease",
        "key": "支气管炎",
        "text": "支气管炎。建议就诊科室：呼吸内科。常用药物：阿莫西林。",
        "departments": ["呼吸内科"],
        "drugs": ["阿莫西林"],
    },
]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        index_dir=str(tmp_path / "index"), domain="clinic", normalizer="chinese", embedding_dim=32
    )


def _store(settings: Settings) -> SqliteStore:
    return SqliteStore(
        db_path=settings.sqlite_path,
        vector_path=settings.vector_path,
        dim=settings.embedding_dim,
        embedding_model="hashing-32",
        normalizer=ChineseNormalizer(),
        domain="clinic",
    )


# 卡片 -> 索引
class TestCardsToInputs:
    def test_each_card_becomes_one_doc_and_one_chunk(self):
        docs, chunks = cards_to_inputs(CARDS)
        assert len(docs) == len(chunks) == 3
        assert docs[0].doc_id == "symptom/咳嗽"
        assert chunks[0].chunk_id == "symptom/咳嗽::0000"

    def test_empty_text_cards_are_skipped(self):
        docs, _ = cards_to_inputs([{"kind": "symptom", "key": "空", "text": "  "}])
        assert docs == []

    def test_limit_keeps_the_prefix(self):
        docs, _ = cards_to_inputs(CARDS, limit=2)
        assert [d.doc_id for d in docs] == ["symptom/咳嗽", "symptom/头痛"]

    def test_citation_url_is_built(self):
        docs, _ = cards_to_inputs(CARDS)
        assert docs[0].source_url.startswith("https://kubedocs.local/clinic/")


class TestPackBuildIndex:
    def test_missing_corpus_gives_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "kubedocs.domains.clinic.domain.clinic_cards_path", lambda: tmp_path / "nope.json"
        )
        with pytest.raises(FileNotFoundError, match="clinic_build_cards"):
            ClinicPack().load_cards()

    def test_ingest_then_chinese_bm25_retrieval_works(self, tmp_path, monkeypatch):
        """中文查询必须命中中文卡片：这是归一化层存在的意义。"""
        cards_file = tmp_path / "cards.json"
        cards_file.write_text(json.dumps(CARDS, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("kubedocs.domains.clinic.domain.clinic_cards_path", lambda: cards_file)

        settings = _settings(tmp_path)
        store = _store(settings)
        try:
            pack = ClinicPack(batch=2)
            report = pack.build_index(store, HashingEmbedder(dim=32), settings)
            assert report.chunks_written == 3
            assert store.stats().n_chunks == 3
            assert store.stats().normalizer == "chinese-jieba"
            assert store.verify_fts() == "ok"
            # 中文 BM25：不预分词的话这两条都命中不了
            assert [c for c, _ in store.keyword_search("咳嗽", 5)] == ["symptom/咳嗽::0000"]
            assert store.keyword_search("神经内科", 5)
        finally:
            store.close()

    def test_triage_links_only_uses_symptom_cards(self, tmp_path, monkeypatch):
        cards_file = tmp_path / "cards.json"
        cards_file.write_text(json.dumps(CARDS, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("kubedocs.domains.clinic.domain.clinic_cards_path", lambda: cards_file)
        links = ClinicPack().triage_links()
        assert set(links) == {"咳嗽", "头痛"}  # 疾病卡没有"症状"这一侧
        assert "呼吸内科" in links["咳嗽"]


# 域注册
class TestClinicDomainRegistration:
    def setup_method(self):
        DomainRegistry.clear()
        load_all()

    def teardown_method(self):
        DomainRegistry.clear()
        load_all()

    def test_both_domains_are_registered(self):
        assert DomainRegistry.names() == ["clinic", "k8s_docs"]

    def test_clinic_uses_the_chinese_normalizer(self):
        assert DomainRegistry.get("clinic").normalizer_name == "chinese"
        assert DomainRegistry.get("k8s_docs").normalizer_name == "identity"

    def test_declared_eval_sets_actually_exist_on_disk(self):
        """声明的金标集必须真实存在，否则自省输出只是空头承诺。

        断言的是"凡声明了就必须存在"——clinic 刻意返回空列表，空列表合法，假路径不合法。
        """
        from kubedocs.config import repo_root

        declared = {name: DomainRegistry.get(name).eval_sets() for name in DomainRegistry.names()}
        assert declared["k8s_docs"], "k8s 域必须有金标集"
        assert declared["clinic"] == [], "clinic 金标集尚未建立，应显式为空而不是假路径"
        for name, paths in declared.items():
            for p in paths:
                assert (repo_root() / p).exists(), f"{name} 声明的 {p} 不存在"

    def test_two_domains_get_separate_indexes(self, tmp_path):
        a = Settings(index_dir=str(tmp_path), domain="k8s_docs")
        b = Settings(index_dir=str(tmp_path), domain="clinic")
        assert a.sqlite_path != b.sqlite_path
