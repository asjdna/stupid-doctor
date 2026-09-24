"""Markdown 切块：把 Hugo 风格的 Kubernetes 文档切成可检索、可引用的块。

## 为什么不能"按固定长度硬切"

K8s 文档是 Hugo markdown，直接按 1000 字符切会踩三个坑（都会污染检索质量）：

1. **front matter 是元数据不是正文**：`---` 之间的 title/weight/description 混进块里，
   会让所有块都"长得像"，稀释判别力。
2. **shortcode 是渲染指令**：`{{< note >}}…{{< /note >}}`、`{{< glossary_tooltip >}}`。
   整段丢掉会**丢失正文**（提示框里常是"注意：这样做会导致 X"这种关键内容），
   所以必须**保留文字、剥掉标记**。
3. **代码块不能被切断**：K8s 文档里大量 YAML 清单，切到一半的 `kubectl apply -f -`
   示例既不可读也会让 LLM 生成错误命令。

所以本模块的策略是：**先按标题切成语义段（并带上标题面包屑），超长段再按段落切，
且永不切进代码围栏内部**。纯标准库实现，可离线单测、零依赖。

## id 与内容指纹分开

- `chunk_id = "<相对路径>::<序号>"` —— 稳定、可预期，供 upsert 定位；
- `content_hash` —— 文本的指纹，供摄取管线判断"是否真的变了"。
两者分开，才能做到「幂等重入 + 增量更新」，而不是每次全量重写。
"""
from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

# Hugo shortcode：{{< ... >}} / {{% ... %}}，含自闭合与块级
_SHORTCODE_RE = re.compile(r"\{\{[<%][^}]*?[>%]\}\}", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# HTML 处理：K8s 的 API reference 用 HTML 表格排版，占语料近六成。
# 表格**结构要保留成可读文本**（否则一堆 <td> 标记会污染 BM25 与嵌入），
# 但代码围栏内的 `<` 不能动（`if [[ $x < 5 ]]` 是合法代码）。
_BLOCK_TAGS = ("table", "thead", "tbody", "tr", "div", "section", "ul", "ol", "p", "pre")
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>]*)?/?>")
_CELL_SPLIT_RE = re.compile(r"</(?:td|th)>\s*", re.IGNORECASE)
_ROW_END_RE = re.compile(r"</tr>\s*", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_DROP_BLOCK_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def strip_html(body: str) -> str:
    """把 HTML 标记转成可读文本，**但跳过代码围栏**。

    规则：
    - `<script>/<style>` 整块删掉（纯噪声）；
    - 表格单元格 `</td>` → ` | `，行 `</tr>` → 换行（把表格变回可读的行文本）；
    - `<br>` → 换行；其余标签只删标记、**保留标签内文字**（`<code>x</code>` → `x`）；
    - HTML 实体解码（`&lt;` `&amp;` `&nbsp;` 等）。
    """
    out: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:                       # 代码内部原样保留
            out.append(line)
            continue
        s = line
        if "<" in s:                       # 快路径：没有尖括号就不必跑标签替换
            s = _DROP_BLOCK_RE.sub("", s)
            s = _BR_RE.sub("\n", s)
            s = _CELL_SPLIT_RE.sub(" | ", s)
            s = _ROW_END_RE.sub("\n", s)
            s = _TAG_RE.sub("", s)
        # 实体解码必须**无条件**执行：`A &lt; B` 这行没有尖括号，但同样需要解码
        out.append(html.unescape(s))
    return "\n".join(out)


def _safe_overlap_tail(block: str, overlap: int) -> str:
    """取块尾用于重叠，**并对齐到安全边界**。

    按字符硬切会出现 `"...rip between"`、`"delete deployment -l app=redis"` 这种
    从词中间、甚至从代码块中间开始的块（真实语料实测）。这里优先对齐到最近的换行，
    其次空格；都对不齐就放弃重叠（少一点上下文，好过给出半截代码）。
    """
    if overlap <= 0 or not block:
        return ""
    tail = block[-overlap:]
    for sep in ("\n", " "):
        idx = tail.find(sep)
        if idx != -1 and idx < len(tail) - 1:
            return tail[idx + 1:]
    return ""


@dataclass
class Chunk:
    """一个可检索块。metadata 全部保留，供引用与调试。"""

    chunk_id: str
    doc_path: str            # 相对语料根的路径，如 concepts/architecture/cgroups.md
    index: int               # 该文档内的块序号
    text: str
    breadcrumb: list[str] = field(default_factory=list)   # 标题路径，如 ["Cgroups", "限制"]
    title: str = ""          # 文档标题（front matter > 首个 h1 > 文件名）
    source_url: str = ""     # 可点击的官方文档地址
    content_hash: str = ""

    @property
    def breadcrumb_text(self) -> str:
        return " › ".join(self.breadcrumb)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """剥离并解析 YAML front matter（只做 `key: value` 的轻量解析，不引入 YAML 依赖）。"""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[m.end():]


def clean_body(body: str) -> str:
    """剥掉渲染标记但**保留其文字**，并规整空白。

    注意顺序：先删 HTML 注释（纯噪声），再处理 shortcode 的**标记部分**
    （`{{< note >}}`/`{{< /note >}}` 删掉，但**不删**它们包裹的正文）。
    """
    body = _HTML_COMMENT_RE.sub("", body)
    # shortcode 只删标记本身；被它包裹的正文（在标记之外）自然保留
    body = _SHORTCODE_RE.sub("", body)
    # HTML（API reference 大量使用）转成可读文本，但不动代码围栏
    body = strip_html(body)
    # 三个以上连续空行压成一个空行（保持段落边界，避免块内出现大段空白）
    body = _MULTI_BLANK_RE.sub("\n\n", body)
    return body.strip()


def strip_legacy_anchors(body: str) -> str:
    """去掉 Hugo 的 `{#anchor}` 标题后缀，避免混进面包屑。"""
    return re.sub(r"\s*\{#[A-Za-z0-9_-]+\}\s*$", "", body, flags=re.MULTILINE)


def _iter_lines_outside_fence(body: str) -> Iterator[tuple[str, bool]]:
    """产出 (行, 是否在代码围栏内)。用于"标题只在围栏外才算标题"。"""
    in_fence = False
    for line in body.splitlines():
        if _CODE_FENCE_RE.match(line):
            yield line, in_fence          # 围栏行本身按当前状态归属
            in_fence = not in_fence
            continue
        yield line, in_fence


def split_sections(body: str) -> list[tuple[list[str], str]]:
    """按标题切成语义段，返回 [(标题面包屑, 正文), ...]。

    面包屑按标题层级维护：遇到同级或更高级标题就回退，这样
    `## 概念 / ### 限制` 下的内容面包屑是 ["概念", "限制"]。

    **标题行本身保留在正文里**（不只是当元数据）：否则「注意事项」这类标题词在
    检索里完全搜不到，引用也缺上下文。实测就是这么发现丢标题的——单测断言
    "关键内容点必须都能在某个块里找到"，标题词找不到即失败。
    """
    sections: list[tuple[list[str], str]] = []
    breadcrumb: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(([t for _, t in breadcrumb], text))
        buf.clear()

    for line, in_fence in _iter_lines_outside_fence(body):
        m = None if in_fence else _HEADING_RE.match(line)
        if not m:
            buf.append(line)
            continue
        flush()
        level = len(m.group(1))
        title = m.group(2).strip()
        while breadcrumb and breadcrumb[-1][0] >= level:
            breadcrumb.pop()
        breadcrumb.append((level, title))
        buf.append(line)          # 标题行进正文，保证可检索、引用可读
    flush()
    return sections


def _fence_toggle(paragraph: str) -> bool:
    """该段是否含奇数个围栏标记（奇数 = 切换代码块状态）。"""
    n = sum(1 for line in paragraph.splitlines() if _CODE_FENCE_RE.match(line))
    return n % 2 == 1


def _atomic_units(text: str) -> list[str]:
    """把文本拆成**原子单元**：普通段落，或一整个代码块（含其内部空行）。

    代码块在 markdown 里可以跨空行，所以"按空行切段"会把一个 YAML 清单切碎。
    这里先跨段跟踪围栏状态，把完整代码块合成一个单元，后续按长度装箱时**不切它**。
    """
    units: list[str] = []
    code_buf: list[str] = []
    in_fence = False
    for para in re.split(r"\n\s*\n", text):
        if in_fence:
            code_buf.append(para)
            if _fence_toggle(para):                  # 围栏在此段闭合
                in_fence = False
                units.append("\n\n".join(code_buf))
                code_buf = []
            continue
        if _fence_toggle(para):
            in_fence = True
            code_buf = [para]
            continue
        units.append(para)
    if code_buf:                                     # 围栏未闭合（文档写坏了）→ 原样保留
        units.append("\n\n".join(code_buf))
    return [u for u in units if u.strip()]


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """超长段落再切：按原子单元装箱，带尾部重叠。

    单个原子单元（例如一段超长 YAML）本身就超过 max_chars 时**允许超出**：
    宁可块大一点，也不要出现半截 `kubectl apply` 示例。
    """
    if len(text) <= max_chars:
        return [text]

    blocks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            blocks.append("\n\n".join(cur).strip())
            tail = _safe_overlap_tail("\n\n".join(cur), overlap)
            cur = [tail] if tail else []
            cur_len = len(tail)

    for unit in _atomic_units(text):
        if cur and cur_len + len(unit) + 2 > max_chars:
            flush()
        cur.append(unit)
        cur_len += len(unit) + 2
    if cur:
        blocks.append("\n\n".join(cur).strip())
    return [b for b in blocks if b]


def doc_id_from_path(rel_path: str) -> str:
    """相对路径 -> 稳定的文档 id（去扩展名、统一分隔符）。"""
    p = rel_path.replace("\\", "/")
    return p[:-3] if p.endswith(".md") else p


def source_url_for(rel_path: str, url_prefix: str) -> str:
    """相对路径 -> 官方文档 URL。

    Hugo 约定：`concepts/foo/_index.md` 与 `concepts/foo.md` 都对应 `concepts/foo/`。
    引用必须能点开，否则"带引用"就是空话。
    """
    p = rel_path.replace("\\", "/")
    if p.endswith(".md"):
        p = p[:-3]
    if p == "_index":
        p = ""
    elif p.endswith("/_index"):
        p = p[: -len("/_index")]
    base = url_prefix.rstrip("/")
    p = p.strip("/")
    # 注意：`_index.md` 在根目录时 p 为空，不能再拼一个 "/" 出来（否则 docs//）
    return f"{base}/{p}/" if p else f"{base}/"


def chunk_document(
    rel_path: str,
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
    min_chars: int = 80,
    url_prefix: str = "https://kubernetes.io/docs/",
    add_context: bool = True,
) -> list[Chunk]:
    """把一个 markdown 文档切成块。空文档返回 []。

    `add_context`（默认开）：给每块正文前加上「文档标题 › 标题面包屑」一行。

    为什么必须有：K8s 的 API reference 大量使用模板化小节（每个资源的
    「Query Parameters」表格逐字相同）。**不带文档上下文时它们是同形块**——
    实测 18861 块里有 5794 块（30.7%）内容完全重复。加上上下文后：
    - 重复块大幅下降（不同资源的同名小节不再同形）；
    - 检索命中后能直接看出"这是哪个文档的哪一节"，引用可读性也更好。
    代价是块变长一点（约 +30 字符），可接受。
    """
    meta, body = parse_front_matter(text)
    body = clean_body(strip_legacy_anchors(body))
    if not body:
        return []

    title = meta.get("title") or ""
    if not title:
        for line in body.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                title = m.group(2).strip()
                break
    if not title:
        title = doc_id_from_path(rel_path).rsplit("/", 1)[-1].replace("-", " ").title()

    doc_id = doc_id_from_path(rel_path)
    url = source_url_for(rel_path, url_prefix)

    # 先按标题切段；无标题的长文档会得到一段超长文本，交给 _split_long_text
    raw: list[tuple[list[str], str]] = []
    for crumb, sec in split_sections(body):
        for piece in _split_long_text(sec, max_chars, overlap):
            raw.append((crumb, piece))

    # 合并过短的碎片到前一块（避免"只有一句标题"的块污染检索）
    merged: list[tuple[list[str], str]] = []
    for crumb, piece in raw:
        if merged and len(piece) < min_chars:
            prev_crumb, prev_text = merged[-1]
            merged[-1] = (prev_crumb, prev_text + "\n\n" + piece)
        else:
            merged.append((crumb, piece))

    out: list[Chunk] = []
    for i, (crumb, piece) in enumerate(merged):
        body_text = piece.strip()
        if not body_text:
            continue
        breadcrumb = crumb or [title]
        # 上下文行：标题已在面包屑里就不重复它
        ctx_parts = breadcrumb if breadcrumb and breadcrumb[0] == title else [title, *breadcrumb]
        ctx = " › ".join(dict.fromkeys(p for p in ctx_parts if p))
        final_text = f"[{ctx}]\n\n{body_text}" if add_context and ctx else body_text
        out.append(
            Chunk(
                chunk_id=f"{doc_id}::{i:04d}",
                doc_path=rel_path.replace("\\", "/"),
                index=i,
                text=final_text,
                breadcrumb=breadcrumb,
                title=title,
                source_url=url,
                content_hash=hashlib.blake2b(final_text.encode("utf-8"),
                                             digest_size=8).hexdigest(),
            )
        )
    return out
