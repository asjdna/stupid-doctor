"""RRF 融合的单元测试。

重点验证三件事：
1. 两路都命中同一文档时它必须排第一（融合的本意）；
2. **单路为空时不得稀释另一路**——这是实现里最容易写错、也最难在评测里发现的地方；
3. 同分时排序确定性（否则评测与闸门会随进程抖动）。
"""
import pytest

from kubedocs.retrieve.fusion import Fused, explanation, rrf_fuse, truncate_dedup


def test_两路共同命中排第一():
    bm25 = ["a", "b", "c"]
    dense = ["c", "d", "a"]
    out = rrf_fuse([("bm25", bm25), ("dense", dense)])
    assert out[0].key == "a", [f.key for f in out]      # a 在两路都在前二
    assert {f.key for f in out} == {"a", "b", "c", "d"}


def test_名次越前分越高():
    out = rrf_fuse([("bm25", ["first", "second", "third"])])
    scores = [f.score for f in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0].key == "first"


def test_单路为空不稀释另一路():
    """关键：空路不能被当成"全 0 参与融合"，否则另一路分数被整体压低。

    这里比较"只有一路"与"一路 + 一个空路"的分数——两者必须完全一致。
    """
    only = rrf_fuse([("bm25", ["a", "b", "c"])])
    with_empty = rrf_fuse([("bm25", ["a", "b", "c"]), ("dense", [])])
    assert [f.score for f in only] == [f.score for f in with_empty]
    assert [f.key for f in only] == [f.key for f in with_empty]


def test_k_越小越看重头部():
    lists = [("bm25", ["a", "b", "c", "d", "e"])]
    small = rrf_fuse(lists, k=1)
    large = rrf_fuse(lists, k=1000)
    # k 小时首尾分差更大
    assert (small[0].score - small[-1].score) > (large[0].score - large[-1].score)


def test_k_非法时报错():
    with pytest.raises(ValueError, match="k 必须"):
        rrf_fuse([("bm25", ["a"])], k=0)


def test_权重可调():
    lists = [("bm25", ["a", "b"]), ("dense", ["b", "a"])]
    equal = rrf_fuse(lists)
    weighted = rrf_fuse(lists, weights={"bm25": 2.0, "dense": 0.5})
    # 等权时两文档同分（对称）；加权后 bm25 的第一名更占优 -> a 严格第一
    assert equal[0].score == equal[1].score
    assert weighted[0].key == "a" and weighted[0].score > weighted[1].score


def test_同分时排序确定():
    """同分按 key 排序：否则评测与闸门会随进程/哈希种子抖动。"""
    a = rrf_fuse([("x", ["b", "a"])])
    b = rrf_fuse([("x", ["b", "a"])])
    assert [f.key for f in a] == [f.key for f in b]
    # 构造明确同分：两路互为反向
    out = rrf_fuse([("l1", ["p", "q"]), ("l2", ["q", "p"])])
    assert [f.key for f in out] == ["p", "q"], "同分时应按 key 字典序稳定排序"


def test_记录各路名次用于解释():
    out = rrf_fuse([("bm25", ["a", "b"]), ("dense", ["b", "a"])])
    a = next(f for f in out if f.key == "a")
    assert a.ranks == {"bm25": 1, "dense": 2}


def test_忽略_None_键():
    out = rrf_fuse([("bm25", ["a", None, "b"])])   # type: ignore[list-item]
    assert {f.key for f in out} == {"a", "b"}


def test_truncate_dedup_保序去重():
    assert truncate_dedup(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert truncate_dedup(["a", "b", "a"], limit=2) == ["a", "b"]


def test_explanation_可读():
    f = Fused(key="concepts/foo::0002", score=0.03251, ranks={"bm25": 3, "dense": 7})
    text = explanation(f, names={"bm25": "BM25", "dense": "向量"})
    assert "concepts/foo::0002" in text
    assert "BM25=3" in text and "向量=7" in text
