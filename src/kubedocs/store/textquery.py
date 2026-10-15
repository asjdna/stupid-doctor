"""查询文本 -> 各存储引擎的安全检索表达式。

三个后端共用同一套词元抽取规则，只在最后一步的表达式语法上分叉：
SQLite FTS5 用 `"pod" OR "limit"`（MATCH 语法），Postgres 用 `'pod' \\| 'limit'`
（`to_tsquery` 语法），sqlite-vec 走向量路、不使用文本表达式。

不能把用户输入原样拼进表达式：FTS5 的 MATCH 与 Postgres 的 tsquery 各有元语法
（`-` `*` `"` `(` `:` `AND/OR/NOT`；`&` `|` `!` `<->` `()` 等），原样拼接会让
用户改写查询语义，也是一个注入面——`pod -limit` 在 FTS5 里就是"含 pod 且不含
limit"。这里只抽取 `\\w+`（Python 的 `\\w` 含 CJK，中文查询同样抽得到词元；
用 `[A-Za-z0-9_]+` 会让中文查询一个词元都抽不到），逐个转义后按 **OR** 连接：
用 OR 而不是 AND 是为了召回，多词在文档里往往并不同现，相关性交给
BM25 / ts_rank 排序。
"""

from __future__ import annotations

import re

# 词元抽取：`\w` 在 Python 3 下含 CJK
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# 只清掉功能词。不做词形还原——那是 FTS5 的 porter / Postgres 的 english 配置的职责。
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "up",
        "use",
        "used",
        "using",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


def query_tokens(query: str) -> list[str]:
    """抽出查询词元：去停用词、去重（保序）、小写化。

    返回空列表表示"查询里没有任何可检索词元"——调用方应据此返回**空结果**，
    而不是"匹配全部"，以免返回一堆无关内容。
    """
    seen: dict[str, None] = {}
    for tok in TOKEN_RE.findall(query):
        low = tok.lower()
        if low in STOPWORDS:
            continue
        seen.setdefault(low, None)
    return list(seen)


def build_match_expr(query: str) -> str:
    """FTS5 的 MATCH 表达式：`"a" OR "b"`。词元集天然不含特殊字符，注入面关闭。"""
    tokens = query_tokens(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def build_tsquery_expr(query: str) -> str:
    """Postgres `to_tsquery` 表达式：`'a' | 'b'`。

    词元只含 `\\w`（字母/数字/下划线/CJK），单引号与反斜杠都不可能出现在其中，
    因此不需要额外转义；这里仍显式剔除一次，避免将来放宽词元规则时把注入面带回来。
    """
    tokens = [t.replace("'", "").replace("\\", "") for t in query_tokens(query)]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    return " | ".join(f"'{t}'" for t in tokens)
