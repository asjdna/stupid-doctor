"""查询文本 -> 各存储引擎的安全检索表达式。

三个后端共用同一套词元抽取规则，**只在最后一步表达式语法上分叉**：

| 后端 | 表达式形态 | 说明 |
|---|---|---|
| SQLite FTS5 | `"pod" OR "limit"` | MATCH 的短语语法 |
| Postgres | `'pod' \\| 'limit'` | `to_tsquery` 的语法 |
| sqlite-vec | 不使用 | 向量路不吃文本查询 |

## 为什么必须自己抽词元，而不是把用户输入原样拼进去

FTS5 的 MATCH 与 Postgres 的 tsquery 各有自己的**元语法**
（FTS5：`-` `*` `"` `(` `:` `AND/OR/NOT`；tsquery：`&` `|` `!` `<->` `:` `*` `()`）。
把用户输入直接拼进去不只是"可能报语法错导致 500"，而是一个**注入面**：
`pod -limit` 在 FTS5 里是"含 pod 且不含 limit"，语义被用户改写；
`!` 在 tsquery 里同理。

所以这里只抽取 `\\w+`（Python 的 `\\w` 含 CJK，中文分词后的词元同样能被抽出来。
用 `[A-Za-z0-9_]+` 会让中文查询一个词元都抽不到——一个必须避免的静默失效），
逐个转义后再按 **OR** 连接。

用 OR 而不是 AND 是为了**召回**：多词查询在文档里往往并不同现，相关性交给
BM25 / ts_rank 排序——包含全部词元的文档排名自然会高。
"""
from __future__ import annotations

import re

# 词元抽取：`\w` 在 Python 3 下含 CJK
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# 只清掉功能词。不做词形还原——那是 FTS5 的 porter / Postgres 的 english 配置的职责。
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "not", "of", "on",
    "or", "that", "the", "their", "then", "there", "these", "this", "to", "up", "use",
    "used", "using", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
})


def query_tokens(query: str) -> list[str]:
    """抽出查询词元：去停用词、去重（保序）、小写化。

    返回空列表表示"查询里没有任何可检索词元"——调用方应据此返回**空结果**
    而不是"匹配全部"：搜不到比搜出一堆无关内容更诚实。
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
