"""检索文本归一化：把"入库文本"与"查询文本"映射成同一个可索引形态。

中文必须先分词再入库。FTS5 的 `unicode61` 分词器按"非字母数字"切词，而**汉字在它
眼里全是字母数字**，所以中文的切分实际由标点决定：有标点时切出的是短语级词元
（"建议尽快就医"只有整串出现才命中，查「就医」或「心内科」都落空），没标点时整句
变成一个超长词元，任何词级查询都命中不了。jieba 预分词把文本切成空格分隔的词元后，
入库形态与查询形态才可比：

    index_text("我最近总是胸痛") -> "最近 胸痛"      （进 FTS 索引）
    query_text("胸痛喘不上气")   -> "胸痛 喘不上气"  （进 MATCH 表达式）
    body 仍存原文，用于展示与引用 —— 索引形态与展示形态必须分开。

停用词表里绝不能出现否定词：医疗文本里「不 / 没 / 无 / 未」承载**否定**信息，把它们
丢弃会让「不胸痛」与「胸痛」在索引里变成同一个东西，这是安全相关的问题，不只是检索
质量问题。所以中文停用词只收纯语气词与寒暄语。
"""

from __future__ import annotations

from typing import Protocol

# 中文停用词：只收纯语气助词与寒暄。
# ⚠️ 严禁加入 不/没/无/未/别/否 —— 它们是否定标记，丢弃会让「不胸痛」等同于「胸痛」。
_ZH_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "哦",
        "嗯",
        "之",
        "其",
        "一下",
        "请问",
        "你好",
        "您好",
        "谢谢",
        "请",
        "帮",
        "麻烦",
        "就是",
        "还有",
    }
)


class Normalizer(Protocol):
    """归一化器：同一种语言/域，入库与查询必须用**同一个**实现。

    两个入口分开命名是有意的：入库走 `index_text`、查询走 `query_text`。
    将来若引入"入库与查询不对称"的处理（例如查询侧加同义词扩展），
    拆开的签名能容纳它而不必改协议。
    """

    name: str

    def index_text(self, text: str) -> str: ...

    def query_text(self, query: str) -> str: ...


class IdentityNormalizer:
    """原样透传。英文语料用这个：FTS5 的 unicode61 + porter 已经够用。"""

    name = "identity"

    def index_text(self, text: str) -> str:
        return text

    def query_text(self, query: str) -> str:
        return query


class ChineseNormalizer:
    """jieba 预分词：中文入库/查询前都切成空格分隔的词元。

    jieba 首次调用要构建词典（约 1 秒并打印一行日志），所以**懒加载**——
    CLI 的 `--help`、英文域的检索都不该付这个代价。
    """

    name = "chinese-jieba"

    def __init__(self, *, drop_stopwords: bool = True) -> None:
        self.drop_stopwords = drop_stopwords
        self._jieba: object | None = None

    @property
    def jieba(self) -> object:
        if self._jieba is None:
            import jieba

            jieba.setLogLevel(60)  # 静音"Building prefix dict"日志
            self._jieba = jieba
        return self._jieba

    def _segment(self, text: str) -> str:
        tokens: list[str] = []
        for raw in self.jieba.lcut(text):  # type: ignore[attr-defined]
            token = raw.strip()
            if not token:
                continue
            if self.drop_stopwords and token in _ZH_STOPWORDS:
                continue
            # 纯空白/纯标点丢弃（jieba 会原样吐出标点）
            if not any(ch.isalnum() for ch in token):
                continue
            tokens.append(token)
        # 保留原始大小写与字符：FTS5 的 unicode61 自己会做大小写折叠，
        # 这里提前小写化反而会让"原样入库"与"可选大小写查询"不一致。
        return " ".join(tokens)

    def index_text(self, text: str) -> str:
        return self._segment(text)

    def query_text(self, query: str) -> str:
        return self._segment(query)


def build_normalizer(name: str) -> Normalizer:
    """按配置构造归一化器。"""
    if name in ("identity", "en", "english"):
        return IdentityNormalizer()
    if name in ("chinese", "zh", "chinese-jieba"):
        return ChineseNormalizer()
    raise ValueError(f"未知的 normalizer：{name!r}（可选 identity / chinese）")
