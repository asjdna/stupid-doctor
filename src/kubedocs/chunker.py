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

## 多格式（PDF / Office / HTML）走的是同一个切块器

`parse.py` 负责把任意格式变成**带位置的块**，本模块负责切。两条切块路径：

- `chunk_document`：标题型格式（md/txt/html/docx）—— 正文里的 `#` 标题行决定面包屑；
- `chunk_blocks`：容器型格式（PDF 页 / 表格块 / 幻灯片）—— 块自带 `location`
  （`p.7`、`Sheet1!A1:C20`、`slide 3`），位置作为面包屑首段进入引用。

两者共用 `_split_long_text`（代码块不切断、重叠对齐安全边界）与 `_assemble`
（编号、上下文头、内容指纹），不存在"另一套切块规则"。
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

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
        if in_fence:  # 代码内部原样保留
            out.append(line)
            continue
        s = line
        if "<" in s:  # 快路径：没有尖括号就不必跑标签替换
            s = _DROP_BLOCK_RE.sub("", s)
            s = _BR_RE.sub("\n", s)
            s = _CELL_SPLIT_RE.sub(" | ", s)
            s = _ROW_END_RE.sub("\n", s)
            s = _TAG_RE.sub("", s)
        # 实体解码必须**无条件**执行：`A &lt; B` 这行没有尖括号，但同样需要解码
        out.append(html.unescape(s))
    return "\n".join(out)


def _fence_open_at(text: str, offset: int) -> bool:
    """`offset` 处是否处于代码围栏内部（按行统计围栏标记的奇偶）。

    只看 offset 之前的行：`offset` 正好落在某个围栏行的开头时，那一行还没生效，
    所以"从一个围栏行开头开始的块"是合法的（它带着完整的 ```）。
    """
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if pos >= offset:
            break
        if _CODE_FENCE_RE.match(line):
            in_fence = not in_fence
        pos += len(line)
    return in_fence


def _safe_overlap_tail(block: str, overlap: int) -> str:
    """取块尾用于重叠，**并对齐到安全边界**。

    按字符硬切会出现 `"...rip between"`、`"delete deployment -l app=redis"` 这种
    从词中间、甚至从代码块中间开始的块（真实语料实测）。这里优先对齐到最近的换行，
    其次空格；都对不齐就放弃重叠（少一点上下文，好过给出半截代码）。

    ## 为什么还要判"是否在代码围栏内部"

    只对齐换行/空格**不够**：围栏内的 YAML 每行都是合法断点，于是尾巴可能整段是代码，
    下一块就以半截 `verbs:` / `- apiGroups:` 开头（开头的 ``` 留在上一块里）。
    实测全量语料：**18861 块里有 581 块（3.08%）围栏标记数为奇数**；把 overlap 设为 0
    只剩 2 块（正是源文档自身围栏不配对的那 2 篇）——即 579 块是重叠尾段造出来的。

    这与本模块开头写的不变量直接冲突（"永不切进代码围栏内部"），
    所以这里把"围栏内"也当作不安全的断点：**宁可放弃重叠**。
    """
    if overlap <= 0 or not block:
        return ""
    tail = block[-overlap:]
    for sep in ("\n", " "):
        idx = tail.find(sep)
        if idx != -1 and idx < len(tail) - 1:
            start = len(block) - overlap + idx + 1
            if _fence_open_at(block, start):
                # 断点落在代码块内部：换下一个候选边界再试；都落在围栏里就放弃重叠。
                continue
            return block[start:]
    return ""


@dataclass
class Chunk:
    """一个可检索块（**子块**）。metadata 全部保留，供引用与调试。"""

    chunk_id: str
    index: int  # 该文档内的块序号
    text: str
    breadcrumb: list[str] = field(default_factory=list)  # 标题路径，如 ["Cgroups", "限制"]
    title: str = ""  # 文档标题（front matter > 首个 h1 > 文件名）
    source_url: str = ""  # 可点击的官方文档地址
    content_hash: str = ""
    # 物理位置：`p.7` / `Sheet1!A1:C20` / `slide 3`；标题型格式（md/html/docx）是标题路径。
    # 存储层的字段是既有的 `breadcrumb`（`ChunkInput`/`ChunkHit` 没有位置字段），
    # 所以容器型格式的位置已经作为面包屑首段拼好，检索结果里的引用直接可读：
    # `[手册.pdf › p.7]`、`[台账.xlsx › Sheet1!A1:C20]`。
    location: str = ""
    parent_id: str = ""
    """所属**父块**的 id（`<doc_id>::p<序号>`）。空串 = 这份索引是父子切割之前建的，
    检索端会退回用子块正文当上下文（见 `docs/ARCHITECTURE.md` 的检索一节）。"""


@dataclass
class Parent:
    """一个**父块**：一段完整的语义单元（一个标题小节，或小节被长度上限切开的一片）。

    父块**不进检索索引**——它不参与 BM25、也不算向量。它的唯一用途是：某条子块命中后，
    把这一整段交给模型当上下文。这样"检索用的单位"和"喂给模型的单位"可以各自最优：
    子块小 => 命中精确；父块完整 => 模型看得到前后文，不会拿着一句被切一半的话去答题。
    """

    parent_id: str
    index: int  # 该文档内的父块序号
    doc_id: str
    text: str
    title: str = ""
    breadcrumb: list[str] = field(default_factory=list)
    source_url: str = ""
    location: str = ""
    content_hash: str = ""
    n_children: int = 0
    """挂在这个父块下的子块数。读路径与体检用它回答"这段到底有没有被索引到"。"""


@dataclass
class DocumentTree:
    """一份文档切出来的**父子两层**结果。"""

    doc_id: str
    title: str
    source_url: str
    parents: list[Parent] = field(default_factory=list)
    children: list[Chunk] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.doc_id}：{len(self.parents)} 个父块 / {len(self.children)} 个子块"


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
    return meta, text[m.end() :]


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
            yield line, in_fence  # 围栏行本身按当前状态归属
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
        buf.append(line)  # 标题行进正文，保证可检索、引用可读
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
            if _fence_toggle(para):  # 围栏在此段闭合
                in_fence = False
                units.append("\n\n".join(code_buf))
                code_buf = []
            continue
        if _fence_toggle(para):
            in_fence = True
            code_buf = [para]
            continue
        units.append(para)
    if code_buf:  # 围栏未闭合（文档写坏了）→ 原样保留
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


# 可摄入的文档后缀。**单一真相源**：`parse.py` 的解析器注册表必须恰好覆盖它
# （tests/unit/test_parse.py 有一条断言钉住这个不变式），`doc_id_from_path` 与
# `source_url_for` 也从这里派生 id 与 URL。
# 为什么定义在本模块而不是 `parse.py`：`parse -> chunker` 是单向依赖
# （parse 复用本模块的 `clean_body` / `strip_html` / `_split_long_text`），
# 反向 import 会成环；而 doc_id 与 URL 的映射本来就是这个模块的职责。
DOC_SUFFIXES: tuple[str, ...] = (
    ".md",
    ".txt",
    ".csv",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
)

# 页面型后缀：Hugo 约定 URL 指向"目录"（`concepts/foo.md` -> `/concepts/foo/`）。
# 其余（pdf/docx/xlsx/pptx/txt/csv）是**文件**，URL 必须保留文件名才点得开。
_PAGE_SUFFIXES: tuple[str, ...] = (".md", ".html", ".htm")


def doc_id_from_path(rel_path: str) -> str:
    """相对路径 -> 稳定的文档 id（去掉受支持的扩展名、统一分隔符）。"""
    p = rel_path.replace("\\", "/")
    lower = p.lower()
    for suffix in DOC_SUFFIXES:
        if lower.endswith(suffix):
            return p[: -len(suffix)]
    return p


def source_url_for(rel_path: str, url_prefix: str) -> str:
    """相对路径 -> 可点击的地址。

    Hugo 约定：`concepts/foo/_index.md` 与 `concepts/foo.md` 都对应 `concepts/foo/`。
    引用必须能点开，否则"带引用"就是空话。

    非页面型（PDF/Office/CSV/TXT）是**文件**：URL 保留文件名指向文件本身 ——
    给 `手册.pdf` 编一个带尾斜杠的"目录"地址是打不开的。
    """
    p = rel_path.replace("\\", "/")
    base = url_prefix.rstrip("/")
    lower = p.lower()
    if not lower.endswith(_PAGE_SUFFIXES):
        return f"{base}/{p.lstrip('/')}"
    for suffix in _PAGE_SUFFIXES:
        if lower.endswith(suffix):
            p = p[: -len(suffix)]
            break
    if p == "_index":
        p = ""
    elif p.endswith("/_index"):
        p = p[: -len("/_index")]
    p = p.strip("/")
    # 注意：`_index.md` 在根目录时 p 为空，不能再拼一个 "/" 出来（否则 docs//）
    return f"{base}/{p}/" if p else f"{base}/"


@dataclass
class _Piece:
    """切块中间态：一段文字 + 它该带的位置与面包屑。

    **现在它是"父块候选"**：`chunk_document` 的一个标题小节、或 `chunk_blocks` 的一个
    位置块（页 / 表格块 / 幻灯片）。`_assemble_tree` 再把每个候选切成父子两层。
    """

    location: str
    crumb: list[str]
    text: str


# 父子切割的默认尺寸。三个数各自回答一个问题，不要互相推导：
#
# - `CHILD_TARGET_CHARS=300`：**检索单位**。小到能精确命中一个具体说法（一句参数说明、
#   一行报错含义），而不是整节"泛泛相关"。原来的 1200 字块会把一个小节里的四个话题
#   平均成一个向量，其中任何一句都不再突出。
# - `CHILD_OVERLAP_CHARS=60`：相邻子块的重叠，留给"一句话正好被切在边界上"。
#   比原来的 150 小：子块本身就短，重叠按比例取，太大等于 300 字里一半是重复。
# - `PARENT_MAX_CHARS=3000`：**上下文单位**的上限。小节超过它就按段落边界拆成多个父块；
#   拆是为了"任何一个父块都能整段放进模型上下文"，不为检索。
CHILD_TARGET_CHARS = 300
CHILD_OVERLAP_CHARS = 60
PARENT_MAX_CHARS = 3000


def _context_header(title: str, breadcrumb: list[str]) -> str:
    """块首的「文档标题 › 标题面包屑」上下文行；没有可用标题时返回空串。

    标题已在面包屑里就不重复它（本模块一直以来的规则）。
    """
    parts = breadcrumb if breadcrumb and breadcrumb[0] == title else [title, *breadcrumb]
    return " › ".join(dict.fromkeys(p for p in parts if p))


def _split_parent(text: str, parent_max_chars: int) -> list[str]:
    """把小节按**段落边界**拆进长度上限（单段超长时允许超出，绝不切进代码块）。

    overlap 传 0：父块是"完整语境"，两片之间再加一段重叠只会让模型读两遍同样的字。
    """
    if parent_max_chars <= 0 or len(text) <= parent_max_chars:
        return [text]
    return _split_long_text(text, parent_max_chars, 0)


def _assemble_tree(
    pieces: list[_Piece],
    *,
    doc_id: str,
    title: str,
    url: str,
    add_context: bool,
    child_target_chars: int = CHILD_TARGET_CHARS,
    child_overlap: int = CHILD_OVERLAP_CHARS,
    parent_max_chars: int = PARENT_MAX_CHARS,
    min_chars: int = 80,
) -> DocumentTree:
    """中间态（父块候选）-> 父子两层。

    ## 编号与 id

    - 父：`parent_id = "<doc_id>::p<序号>"`，按父块出现顺序，文档内唯一。
    - 子：`chunk_id = "<doc_id>::<序号>"` —— 与父子切割之前**同一个形状**，序号是子块在
      文档内的顺序。不改形状的理由：`chunk_id` 是增量摄取定位行的键，换形状等于让已有
      索引的行全部对不上，而它们本来就是同一批内容。

    ## 短块合并**只在同一个父块内**做

    比 `min_chars` 还短的子块（例如只剩一行标题）会并进同父块的前一个子块。
    **绝不跨父块合并**：那会把"第 8 页末尾的残句"接到"第 7 页"的子块尾巴上，
    而引用说的还是第 7 页——同一类错误在 `chunk_blocks` 里已经踩过一次（见那里的说明）。
    """
    tree = DocumentTree(doc_id=doc_id, title=title, source_url=url)
    child_index = 0
    for p_index, piece in enumerate(pieces):
        body = piece.text.strip()
        if not body:
            continue
        crumb = piece.crumb or [title]
        ctx = _context_header(title, crumb)
        parent_id = f"{doc_id}::p{p_index:04d}"
        # 子块按**纯正文**切，再各自加上下文头：先加头再切的话只有第一个子块带得到它，
        # 后面的子块在检索里就成了"没有出处的句子"。
        child_texts: list[str] = []
        for raw in _split_long_text(body, child_target_chars, child_overlap):
            if child_texts and len(raw) < min_chars:
                child_texts[-1] = f"{child_texts[-1]}\n\n{raw}"
            else:
                child_texts.append(raw)
        for raw in child_texts:
            text = f"[{ctx}]\n\n{raw}" if add_context and ctx else raw
            tree.children.append(
                Chunk(
                    chunk_id=f"{doc_id}::{child_index:04d}",
                    index=child_index,
                    text=text,
                    breadcrumb=crumb,
                    title=title,
                    source_url=url,
                    content_hash=hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(),
                    location=piece.location,
                    parent_id=parent_id,
                )
            )
            child_index += 1
        parent_text = f"[{ctx}]\n\n{body}" if add_context and ctx else body
        tree.parents.append(
            Parent(
                parent_id=parent_id,
                index=len(tree.parents),
                doc_id=doc_id,
                text=parent_text,
                title=title,
                breadcrumb=crumb,
                source_url=url,
                location=piece.location,
                content_hash=hashlib.blake2b(parent_text.encode("utf-8"), digest_size=8).hexdigest(),
                n_children=len(child_texts),
            )
        )
    return tree


def _sections_to_pieces(body: str, *, parent_max_chars: int) -> list[_Piece]:
    """`chunk_document_tree` 的中间态：按标题切小节，过长的小节再按段落拆。"""
    raw: list[_Piece] = []
    for crumb, sec in split_sections(body):
        for part in _split_parent(sec, parent_max_chars):
            raw.append(_Piece(location=" › ".join(crumb), crumb=list(crumb), text=part))
    return raw


def chunk_document_tree(
    rel_path: str,
    text: str,
    *,
    parent_max_chars: int = PARENT_MAX_CHARS,
    child_target_chars: int = CHILD_TARGET_CHARS,
    child_overlap: int = CHILD_OVERLAP_CHARS,
    min_chars: int = 80,
    url_prefix: str = "https://kubernetes.io/docs/",
    add_context: bool = True,
) -> DocumentTree:
    """把一份标题型文档（md/txt/html/docx）切成**父子两层**。

    父 = 一个标题小节（过长时按段落边界拆成数片，见 `PARENT_MAX_CHARS`）；
    子 = 小节内部的段落装箱，目标 `CHILD_TARGET_CHARS`。

    `chunk_document` 是本函数的薄包装（只回子块），两条路不会各切一套。
    """
    meta, body = parse_front_matter(text)
    body = clean_body(strip_legacy_anchors(body))
    if not body:
        return DocumentTree(doc_id=doc_id_from_path(rel_path), title="", source_url="")

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
    return _assemble_tree(
        _sections_to_pieces(body, parent_max_chars=parent_max_chars),
        doc_id=doc_id,
        title=title,
        url=source_url_for(rel_path, url_prefix),
        add_context=add_context,
        child_target_chars=child_target_chars,
        child_overlap=child_overlap,
        parent_max_chars=parent_max_chars,
        min_chars=min_chars,
    )


def chunk_document(
    rel_path: str,
    text: str,
    *,
    max_chars: int = CHILD_TARGET_CHARS,
    overlap: int = CHILD_OVERLAP_CHARS,
    min_chars: int = 80,
    url_prefix: str = "https://kubernetes.io/docs/",
    add_context: bool = True,
) -> list[Chunk]:
    """把一个标题型文档切成**子块**（父子两层里的检索层）。需要父块（喂给模型的那一层）
    就用 `chunk_document_tree`：它回 `DocumentTree`，`.children` 与本函数的返回值逐字段一致。

    `add_context`（默认开）：给每块正文前加上「文档标题 › 标题面包屑」一行。

    为什么必须有：K8s 的 API reference 大量使用模板化小节（每个资源的
    「Query Parameters」表格逐字相同）。**不带文档上下文时它们是同形块**——
    实测 18861 块里有 5794 块（30.7%）内容完全重复。加上上下文后：
    - 重复块大幅下降（不同资源的同名小节不再同形）；
    - 检索命中后能直接看出"这是哪个文档的哪一节"，引用可读性也更好。
    代价是块变长一点（约 +30 字符），可接受。
    """
    return chunk_document_tree(
        rel_path,
        text,
        child_target_chars=max_chars,
        child_overlap=overlap,
        min_chars=min_chars,
        url_prefix=url_prefix,
        add_context=add_context,
    ).children


class PositionedBlock(Protocol):
    """`chunk_blocks` 需要的块形状（`parse.ParsedBlock` 结构上满足它）。

    用 Protocol 而不 import `parse` 的 dataclass：`parse -> chunker` 是单向依赖，
    反向 import 会成环。
    """

    text: str
    location: str
    heading_path: list[str]


def chunk_blocks_tree(
    rel_path: str,
    blocks: Iterable[PositionedBlock],
    *,
    title: str = "",
    parent_max_chars: int = PARENT_MAX_CHARS,
    child_target_chars: int = CHILD_TARGET_CHARS,
    child_overlap: int = CHILD_OVERLAP_CHARS,
    min_chars: int = 80,
    url_prefix: str = "https://kubernetes.io/docs/",
    add_context: bool = True,
) -> DocumentTree:
    """按"带位置的块"切父子两层：PDF 页 / 表格块 / 幻灯片。

    与 `chunk_document_tree` 的差别只有一处，而且是位置语义要求的：
    **父块不跨位置**。一个输入块（第 7 页的正文、`Sheet1!A1:C20` 那段、第 3 张幻灯片）
    就是父块候选，超长时只在**它内部**按段落拆。跨位置合并会把下一页的文字挂到上一页的
    引用下，"引用可追溯"就成了假话。
    """
    doc_id = doc_id_from_path(rel_path)
    url = source_url_for(rel_path, url_prefix)
    doc_title = title or doc_id.rsplit("/", 1)[-1].replace("-", " ").title()

    pieces: list[_Piece] = []
    for block in blocks:
        text = clean_body(block.text)
        if not text.strip():
            continue
        location = block.location or " › ".join(block.heading_path)
        crumb = [location, *block.heading_path] if location else list(block.heading_path)
        for part in _split_parent(text, parent_max_chars):
            # **同一位置**的相邻片先并成一个父块候选：`parse.py` 会把一段连续正文拆成
            # 多个块（同一页的不同段落），它们的位置相同、本来就是一个上下文单位。
            # 位置不同则一律不并 —— 那会把下一页的字挂到上一页的引用下。
            if pieces and pieces[-1].location == location:
                prev = pieces[-1]
                pieces[-1] = _Piece(location=prev.location, crumb=prev.crumb, text=f"{prev.text}\n\n{part}")
            else:
                pieces.append(_Piece(location=location, crumb=crumb, text=part))

    return _assemble_tree(
        pieces,
        doc_id=doc_id,
        title=doc_title,
        url=url,
        add_context=add_context,
        child_target_chars=child_target_chars,
        child_overlap=child_overlap,
        parent_max_chars=parent_max_chars,
        min_chars=min_chars,
    )


def chunk_blocks(
    rel_path: str,
    blocks: Iterable[PositionedBlock],
    *,
    title: str = "",
    max_chars: int = CHILD_TARGET_CHARS,
    overlap: int = CHILD_OVERLAP_CHARS,
    min_chars: int = 80,
    url_prefix: str = "https://kubernetes.io/docs/",
    add_context: bool = True,
) -> list[Chunk]:
    """按"带位置的块"切**子块**（父子两层里的检索层）。

    与 `chunk_document` 的差别只有两处，且都是位置语义要求的：

    1. **块自带位置**：`location`（`p.7` / `Sheet1!A1:C20` / `slide 3`）作为面包屑
       首段进入引用，于是引用是 `[手册.pdf › p.7]` 而不是只说"在这篇文档里"；
    2. **不跨位置合并碎片**：超短块只在**同一位置内**合并。跨位置合并会把下一页/
       下一张幻灯片的文字挂到上一页的引用下，"引用可追溯"就成了假话。

    需要父块就用 `chunk_blocks_tree`（它回 `DocumentTree`，`.children` 与本函数返回值一致）。
    """
    return chunk_blocks_tree(
        rel_path,
        blocks,
        title=title,
        child_target_chars=max_chars,
        child_overlap=overlap,
        min_chars=min_chars,
        url_prefix=url_prefix,
        add_context=add_context,
    ).children
