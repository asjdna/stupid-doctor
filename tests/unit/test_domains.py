"""P0 地基测试：文本归一化、多域隔离、域注册表。

这三样是"双域平台"的地基，一旦其中任何一处静默失效，表现都是
"换个域之后搜不到东西"——没有报错，极难排查。所以每一条都固化进测试。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kubedocs.config import Settings
from kubedocs.domains import DomainRegistry, load_all
from kubedocs.domains.base import DomainPack
from kubedocs.normalize import (
    ChineseNormalizer,
    IdentityNormalizer,
    build_normalizer,
)
from kubedocs.store.base import ChunkInput, DocInput
from kubedocs.store.sqlite_store import SqliteStore, build_match_expr


class TestIdentityNormalizer:
    def test_passthrough(self):
        n = IdentityNormalizer()
        assert n.index_text("Pod CPU limit") == "Pod CPU limit"
        assert n.query_text("Pod CPU limit") == "Pod CPU limit"

    def test_name(self):
        assert IdentityNormalizer().name == "identity"


class TestChineseNormalizer:
    """中文不分词的话，FTS5 的 unicode61 会把整句当成**一个词元**，查询永远不命中。"""

    def test_segments_into_space_separated_tokens(self):
        out = ChineseNormalizer().index_text("我最近总是胸痛")
        assert " " in out
        assert "胸痛" in out.split()

    def test_query_and_index_agree(self):
        """入库与查询必须走同一套切分，否则两个词汇空间对不上。"""
        n = ChineseNormalizer()
        assert set(n.query_text("胸痛喘不上气").split()) & \
               set(n.index_text("我胸痛而且喘不上气").split())

    def test_negation_is_never_dropped(self):
        """否定词被当停用词丢掉，「不胸痛」就等同于「胸痛」——这是安全事故，不是检索问题。"""
        n = ChineseNormalizer()
        for text, marker in [("不胸痛", "不"), ("没发烧", "没"), ("无咳嗽", "无"), ("未呕吐", "未")]:
            tokens = n.index_text(text).split()
            assert any(marker in t for t in tokens), f"{text!r} 的否定标记被丢了：{tokens}"

    def test_particles_are_dropped(self):
        tokens = ChineseNormalizer().index_text("请问一下我的头很痛呀").split()
        assert "请问" not in tokens and "一下" not in tokens and "呀" not in tokens
        assert "头" in tokens or "头痛" in tokens

    def test_punctuation_is_dropped(self):
        tokens = ChineseNormalizer().index_text("胸痛，喘不上气。").split()
        assert all(any(ch.isalnum() for ch in t) for t in tokens)

    def test_empty_input(self):
        assert ChineseNormalizer().index_text("") == ""
        assert ChineseNormalizer().index_text("，。！") == ""


def test_build_normalizer_by_name():
    assert build_normalizer("identity").name == "identity"
    assert build_normalizer("chinese").name == "chinese-jieba"
    with pytest.raises(ValueError, match="未知的 normalizer"):
        build_normalizer("klingon")


class TestMatchExprSupportsChinese:
    """`[A-Za-z0-9_]+` 会把中文查询的词元全丢掉——这是个没有任何报错的静默失效。"""

    def test_chinese_tokens_are_extracted(self):
        expr = build_match_expr("胸痛 呼吸困难")
        assert '"胸痛"' in expr and '"呼吸困难"' in expr

    def test_chinese_still_quoted_and_safe(self):
        expr = build_match_expr("胸痛 -呼吸困难")
        assert "NOT" not in expr
        assert expr.count('"') % 2 == 0


class TestChineseFullTextRoundTrip:
    """端到端：中文入库 -> 中文查询能命中（这是 P0 真正要证明的事）。"""

    def _store(self, tmp_path: Path) -> SqliteStore:
        return SqliteStore(db_path=tmp_path / "zh.db", vector_path=tmp_path / "zh.npy",
                           dim=8, embedding_model="test",
                           normalizer=ChineseNormalizer(), domain="clinic")

    def _seed(self, store: SqliteStore) -> None:
        doc = DocInput(doc_id="c1", rel_path="c1.md", title="症状卡", source_url="u",
                       content_hash="h")
        chunk = ChunkInput(chunk_id="c1::0000", doc_id="c1", ordinal=0, title="症状卡",
                           breadcrumb="心内科", source_url="u",
                           body="胸痛：可能与心绞痛相关，建议尽快到心内科就诊。",
                           content_hash="h0", n_chars=30)
        store.upsert_docs([doc], [chunk], {"c1::0000": np.ones(8, dtype=np.float32)})
        store.finalize()

    def test_chinese_query_hits_chinese_chunk(self, tmp_path):
        store = self._store(tmp_path)
        try:
            self._seed(store)
            hits = store.keyword_search("胸痛", 5)
            assert [c for c, _ in hits] == ["c1::0000"]
        finally:
            store.close()

    def test_colloquial_query_still_hits(self, tmp_path):
        store = self._store(tmp_path)
        try:
            self._seed(store)
            assert store.keyword_search("我最近胸口痛", 5) == []   # 「胸口痛」不等于「胸痛」
            assert store.keyword_search("心内科挂哪个科", 5)       # 科室词能命中
        finally:
            store.close()

    def test_without_segmentation_word_level_query_misses(self, tmp_path):
        """反证：不预分词时，**词级**查询命中不了——这正是引入归一化层的原因。

        注意 unicode61 会在标点处切分，所以"胸痛：..."这类文本碰巧可用；
        真正失效的是两种：无标点的长串（整串成一个词元），以及有标点但切出的是
        **短语级**词元（"建议尽快就医"只有整串出现才命中）。这里两种都验证。
        """
        body = "患者主诉胸痛伴呼吸困难需警惕急性冠脉综合征。建议尽快就医。"
        doc = DocInput(doc_id="c2", rel_path="c2.md", title="症状卡", source_url="u",
                       content_hash="h2")
        chunk = ChunkInput(chunk_id="c2::0000", doc_id="c2", ordinal=0, title="症状卡",
                           breadcrumb="急诊", source_url="u", body=body,
                           content_hash="h2-0", n_chars=len(body))
        vec = {"c2::0000": np.ones(8, dtype=np.float32)}

        plain = SqliteStore(db_path=tmp_path / "p.db", vector_path=tmp_path / "p.npy",
                            dim=8, embedding_model="test")           # identity
        seg = SqliteStore(db_path=tmp_path / "g.db", vector_path=tmp_path / "g.npy",
                          dim=8, embedding_model="test",
                          normalizer=ChineseNormalizer(), domain="clinic")
        try:
            for s in (plain, seg):
                s.upsert_docs([doc], [chunk], vec)
                s.finalize()
            # 无标点长串里的词：identity 查不到，分词后能查到
            assert plain.keyword_search("急性冠脉综合征", 5) == []
            assert not plain.keyword_search("呼吸困难", 5)
            assert seg.keyword_search("呼吸困难", 5)
            assert seg.keyword_search("急性冠脉综合征", 5)
            # 短语级词元同理
            assert not plain.keyword_search("就医", 5)
            assert seg.keyword_search("就医", 5)
        finally:
            plain.close()
            seg.close()

    def test_normalizer_mismatch_is_refused(self, tmp_path):
        """归一化器换了 = 倒排词元换了，旧索引对新查询基本不可用，必须显式拒绝。"""
        s1 = self._store(tmp_path)
        try:
            self._seed(s1)
        finally:
            s1.close()
        from kubedocs.store.sqlite_store import StoreError
        with pytest.raises(StoreError, match="归一化器"):
            SqliteStore(db_path=tmp_path / "zh.db", vector_path=tmp_path / "zh.npy",
                        dim=8, embedding_model="test", normalizer=IdentityNormalizer())


class TestDomainIsolation:
    def test_each_domain_gets_its_own_index_files(self, tmp_path):
        a = Settings(index_dir=str(tmp_path), domain="k8s_docs")
        b = Settings(index_dir=str(tmp_path), domain="clinic")
        assert a.sqlite_path != b.sqlite_path
        assert a.vector_path != b.vector_path
        assert a.sqlite_path.parent != b.sqlite_path.parent

    def test_domain_is_recorded_in_meta(self, tmp_path):
        s = Settings(index_dir=str(tmp_path), domain="clinic", normalizer="chinese",
                     embedding_dim=8)
        store = SqliteStore(db_path=s.sqlite_path, vector_path=s.vector_path, dim=8,
                            embedding_model="t", normalizer=ChineseNormalizer(),
                            domain="clinic")
        try:
            st = store.stats()
            assert st.domain == "clinic"
            assert st.normalizer == "chinese-jieba"
        finally:
            store.close()


class TestDomainRegistry:
    def setup_method(self):
        DomainRegistry.clear()
        load_all()

    def teardown_method(self):
        DomainRegistry.clear()
        load_all()

    def test_builtin_k8s_domain_is_registered(self):
        assert "k8s_docs" in DomainRegistry.names()

    def test_load_all_is_idempotent(self):
        before = DomainRegistry.names()
        load_all()
        load_all()
        assert DomainRegistry.names() == before

    def test_get_unknown_domain_lists_available(self):
        with pytest.raises(KeyError, match="未知的域"):
            DomainRegistry.get("nope")

    def test_duplicate_registration_is_refused(self):
        pack = DomainRegistry.get("k8s_docs")
        with pytest.raises(ValueError, match="已注册"):
            DomainRegistry.register(pack)

    def test_k8s_pack_satisfies_the_protocol(self):
        pack = DomainRegistry.get("k8s_docs")
        assert isinstance(pack, DomainPack)
        assert pack.normalizer_name == "identity"
        assert pack.eval_sets(), "域必须声明金标集，否则评测无从跑起"

    def test_describe_is_json_shaped(self):
        items = DomainRegistry.describe()
        assert items and {"name", "description", "normalizer", "eval_sets"} <= set(items[0])
