#!/usr/bin/env python
"""文档锚点校验：`文件:行号` 引用必须**指着它自己声称的东西**。

## 为什么需要它（而不是"查文件存在 + 行号不越界"）

SECURITY.md 顶部写着"所有结论都标了代码出处（`文件:行`），可以逐条核对"，
但直到本脚本出现之前，全仓**没有任何东西**真的去核对：`scripts/` 下只有
`check_gold.py`（查标注集的路径与关键词）和 `gen_lock.py --check`（查锁文件闭包），
都不看文档引用。姊妹仓库 shopmate 的同类脚本也只校验"文件存在 + 行号不越界"，
**查不出"锚点指向了错误内容"**——而后者才是真实发生过的失败形态：

- `ruff format` 全量重排（`fcd7a55`，110 个文件）会让所有指向 `.py` 的行号整体位移。
  行号仍然"存在"、也不越界，只是指到了别的函数上：一条绿灯但骗人的引用。
- 引用还可能**本来就写错了**（指向的那几行从来没有包含它声称的东西）。
  越界检查对这种情况完全无感。

所以本脚本做的是**内容级**校验：把引用所在那句散文里反引号包住的标识符抽出来，
要求它出现在锚点范围（前后各放宽 `--window` 行）里。散文说"记 index_error"并标了
`api/app.py` 的某几行，而那几行里找不到 `index_error` ⇒ 失败。

## 为什么用"散文里的标识符"当判据

文档给不出比这更强的机器可读线索：行号会漂、文件名会漂，只有**符号名**是稳定的
（SECURITY.md 开头自己也是这么约定的："若行号对不上，请按符号名搜索——每一条引用里
都带了函数名 / 字段名 / 类名"）。本脚本把这个约定变成门禁。

为了不让"随便指哪里都能过"，抽取时做两件事：

1. **跳过文件名字面量**：形如 `src/kubedocs/api/app.py` 的反引号会命中 `kubedocs` 这种
   在任何文件里都出现的词，等于没校验；能解析成仓内文件的反引号一律不当标识符
   （`Dockerfile`、`pyproject.toml`、`domains/clinic/mock.py` 都属此类）。
2. **强候选优先**：带下划线 / 长度 ≥ 6 / 含大写的名字（`index_error`、`DailyBudget`、
   `_confirm_side_effect`）比 `chat`、`user` 这类普通词更能定位代码。只要句子里存在强
   候选，就必须由强候选命中；否则退回到整句的普通候选。

## 无法校验的引用怎么办

不许静默跳过。有些引用的散文里**没有任何源码标识符**（例如"自述尚未在本机实测"
只指向文件头的一句注释），机器无从自己判断对错。这类必须逐条登记在 `WHITELIST`
里并写明理由，运行时打印条数。白名单**不是**免死金牌：每条都要给出 `expect`——
锚点范围里必须出现的字符串，锚点一旦漂走它就会失败。**既校验不过、又不在白名单里
⇒ 非零退出。**
白名单的键是 `(文档名, 锚点文本)`，**不含"引用所在的行号"**：位置不是身份，
把行号放进键里只会让每次改文档都要手工重算（实测被这个绊过 4 次）。
每条豁免还必须**真的被扫描命中**——`tests/unit/test_doc_refs.py` 会为此失败，
所以陈旧条目不会静默留着。

用法：
    python scripts/check_doc_refs.py              # 全仓文档
    python scripts/check_doc_refs.py --verbose    # 逐条打印命中的标识符
    python scripts/check_doc_refs.py --whitelist  # 只列白名单

退出码：0 全部通过；1 有引用站不住。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 锚点窗口：行号本身常有 ±1 的排版误差（空行、续行位置），但"位移 25 行指到别的函数"
# 必须被抓到。2 行是这两者之间的分界（实测：本次重锚前的真实位移是 +2 ~ +25）。
DEFAULT_WINDOW = 2

# 扫描范围：仓库文档 + 源码/测试/脚本里的文档字符串（那里也有 `文件:行` 引用）。
DOC_GLOBS = (
    "*.md",
    "docs/**/*.md",
    ".github/**/*.md",
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
)

# 建"仓内文件索引"时跳过的目录：第三方与生成物，与文档引用无关。
PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "data/index",
        "data/raw",
        "data/derived",
    }
)

# 判定"这个反引号是文件名而不是符号名"用的后缀（能解析成仓内文件的也算）。
_FILE_SUFFIXES = (
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".txt",
    ".sql",
    ".cfg",
    ".ini",
    ".lock",
    ".example",
    ".html",
    ".js",
    ".sh",
)

# 反引号里的内容。引用只写单行，所以不需要跨行匹配。
_BACKTICK = re.compile(r"`([^`\n]+)`")

# 引用形状：`<可空的路径>:<行号>[, <行号>|-<区间>]...`
# 路径里不允许出现空格与冒号——`python:3.10-slim`、`http://host:3000/x` 因此天然被排除。
_REF_SHAPE = re.compile(r"^(?P<path>[^\s:]*):(?P<spec>\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)$")

# 标识符抽取：反引号里的 `extra="forbid"` / `pod -limit` / `_confirm_side_effect()`
# 都要能取出里面的名字。
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# markdown 块的开头：列表项、有序列表项、表格行、标题、引用块。
# 用它切"同一句散文"，避免把相邻条目的标识符算进来（那会让引用"指哪都能过"）。
_BLOCK_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||#{1,6}\s|>)")

# 句子终止符。`；` 也算：文档里分号两侧常是两个不同的论断，跨过去就会串味。
_TERMINATORS = "。！？；|"

# 太泛、或属于散文而非源码的名字：留在候选里只会让"随便指哪里都能过"。
_STOPWORDS = frozenset(
    {
        "true",
        "false",
        "none",
        "null",
        "with",
        "http",
        "https",
        "json",
        "yaml",
        "docs",
        "data",
        "test",
        "tests",
        "main",
        "init",
        "base",
        "self",
        "file",
        "path",
        "name",
        "list",
        "docker",
        "compose",
        "python",
        "redis",
        "postgres",
    }
)

# =====================================================================================
# 白名单：散文里没有源码标识符、机器无法自己判断对错的引用。
#
# 键是 (文档相对路径, 引用所在的散文行号, 反引号里的引用原文)。
# 行号取"引用所在的那一行文档"，**不是**目标文件的行号——重锚不会移动文档行，
# 所以条目在重锚后依然有效。文档被改写导致键不再匹配时，条目会失效，那条引用退回按
# "散文里的标识符"校验：抽不出标识符就变红。失败方向是响的，不会变成假绿。
# =====================================================================================


@dataclass(frozen=True)
class White:
    """一条白名单：为什么机器校验不了 + 仍然生效的内容断言。"""

    reason: str
    expect: str | None = None


WHITELIST: dict[tuple[str, str], White] = {
    # 只登记"整句里除锚点外没有任何源码标识符"的引用，而且每条都带 `expect`
    # （锚点漂走时它必须失败——白名单不是免死金牌）。
    #
    # ⚠️ 键是 `(文档名, 锚点文本)`，**刻意不含"引用所在的行号"**。
    # 原先键是 `(文档名, 行号, 锚点文本)`，后果是：只要在文档**上方**插一行，
    # 豁免就失效并报"锚点已漂移"，而真正防漂移的是 `expect`（它校验的是**目标文件**
    # 在锚点处的内容）。行号在键里只贡献"每次改文档都要手工重算"——
    # **本轮实测被这个机制拦下 4 次**，每次都是纯粹的重算开销，不是发现了真问题。
    # 位置不是身份：键里去掉行号后，"改写被豁免的那句话"仍然会让它失效（因为锚点文本变了）。
    # 安全性由 tests/unit/test_doc_refs.py 兜住：每条豁免都必须**真的被扫描命中**，
    # 陈旧条目会直接测试失败，不会静默留着。
    ("SECURITY.md", ".gitignore:53-54"): White(
        reason="这一句在说 clinic 卡片语料不入库，反引号里只有这个锚点",
        expect="clinic",
    ),
    ("SECURITY.md", "clinic_baseline.json:8"): White(
        reason="markdown 表格行：单元格里只有指标名与数字、以及这个锚点，没有源码符号",
        expect='"recall"',
    ),
    ("SECURITY.md", "clinic_baseline.json:9"): White(
        reason="同上（表格行，单元格里只有数字与锚点）",
        expect='"false_positive_rate"',
    ),
    ("SECURITY.md", "clinic_baseline.json:18"): White(
        reason="同上（表格行，单元格里只有数字与锚点）",
        expect='"accuracy"',
    ),
    ("SECURITY.md", "docker-compose.yml:123"): White(
        reason="这一句在说端口映射，反引号里只有端口号与这个锚点",
        expect="8080:8080",
    ),
}


@dataclass(frozen=True)
class Anchor:
    """一处引用。"""

    doc: str  # 引用所在的文档（仓内相对路径）
    line: int  # 引用所在的行号（1 基）
    text: str  # 反引号里的原文，例如 "api/app.py:129-145, 241-252"
    target: str  # 解析出的目标文件（仓内相对路径）
    ranges: tuple[tuple[int, int], ...]
    bare: bool  # 是否是无文件名的续写引用（冒号开头的那种，沿用同段上文的目标文件）
    unit: str  # 用来抽标识符的"同一句散文"


@dataclass
class Finding:
    """一条结论：通过 / 失败 / 白名单豁免。"""

    anchor: Anchor
    status: str  # ok / fail / white
    detail: str
    candidates: tuple[str, ...] = ()
    matched: str = ""


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    docs: int = 0

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "fail"]

    @property
    def whitelisted(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "white"]

    @property
    def verified(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "ok"]


def _parse_ranges(spec: str) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.append((int(lo), int(hi)))
        else:
            out.append((int(part), int(part)))
    return tuple(out)


def _format_ranges(ranges: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


def _ref_shape(token: str) -> re.Match[str] | None:
    return _REF_SHAPE.match(token.strip())


def _ends_sentence(line: str) -> bool:
    stripped = line.rstrip()
    return bool(stripped) and stripped[-1] in _TERMINATORS


def _starts_block(line: str) -> bool:
    return bool(_BLOCK_START.match(line))


def _logical_sentence(lines: list[str], index: int) -> str:
    """取引用所在的那"一句"散文（一句话被折成多行时算一句）。

    只取引用所在行是不够的：中文文档常把一句话折成两三行，标识符与锚点因此不在同一行
    （SECURITY.md 第 203-205 行就是）。取整段又太宽：相邻条目的标识符会被算进来，
    校验就退化成"指哪都能过"。所以向两侧扩展到**换行或换条目**为止。
    """
    start = index
    while start - 1 >= 0:
        prev = lines[start - 1]
        if not prev.strip() or _ends_sentence(prev) or _starts_block(lines[start]):
            break
        start -= 1
    end = index
    while end + 1 < len(lines):
        cur = lines[end]
        if not cur.strip() or _ends_sentence(cur) or _starts_block(lines[end + 1]):
            break
        end += 1
    return "\n".join(lines[start : end + 1])


def _is_file_token(token: str, index: list[str]) -> bool:
    """这个反引号写的是**文件**（而不是符号）吗？

    `src/kubedocs/api/app.py` 里能抽出 `kubedocs`——一个在几乎任何文件里都出现的词。
    把它当标识符等于没校验，所以能解析成仓内文件、或带源码后缀的反引号一律不抽。
    """
    stripped = token.strip()
    if stripped.endswith(_FILE_SUFFIXES):
        return True
    resolved, _ = _resolve_target(stripped, index)
    return resolved is not None


def _is_strong(ident: str) -> bool:
    """强候选：带下划线 / 够长 / 含大写。比 `chat`、`user` 这类普通词更能定位代码。"""
    if "_" in ident or len(ident) >= 6:
        return True
    return any(ch.isupper() for ch in ident)


def _candidates(unit: str, ref_text: str, index: list[str]) -> tuple[str, ...]:
    """从散文里抽候选标识符，去掉这个引用自己（引用不该拿自己验证自己）。"""
    found: list[str] = []
    for token in _BACKTICK.findall(unit):
        if _ref_shape(token) or _is_file_token(token, index):
            continue
        for ident in _IDENT.findall(token):
            if ident in _STOPWORDS or len(ident) < 3:
                continue
            if ident not in found:
                found.append(ident)
    ref_idents = set(_IDENT.findall(ref_text))
    return tuple(name for name in found if name not in ref_idents)


def _walk_files(root: Path) -> list[str]:
    """仓内全部文件（相对路径，posix 风格），供目标文件解析用。

    用 `os.walk` 而不是 `rglob` + 事后过滤：`.venv` 里有几万个文件，先走进去再丢掉
    会让每次校验都白扫一遍（本检查器在 CI 与单测里都会被调用）。
    """
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath).relative_to(root).as_posix()
        keep: list[str] = []
        for name in dirnames:
            child = name if here == "." else f"{here}/{name}"
            if name in PRUNED_DIRS or child in PRUNED_DIRS:
                continue
            keep.append(name)
        dirnames[:] = sorted(keep)
        for name in sorted(filenames):
            out.append(name if here == "." else f"{here}/{name}")
    return out


def _doc_files(root: Path) -> list[str]:
    seen: dict[str, None] = {}
    for pattern in DOC_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                seen[path.relative_to(root).as_posix()] = None
    return list(seen)


def _resolve_target(name: str, index: list[str]) -> tuple[str | None, str]:
    """把引用里的文件名解析成仓内路径。

    文档里的写法是缩写（`api/app.py`、`generate.py`），所以先试"相对仓库根"，
    再试"按路径后缀唯一匹配"。**歧义不猜**：两个文件同名就报错，让人把引用写全。
    """
    if name in index:
        return name, ""
    suffix = "/" + name.lstrip("/")
    hits = [item for item in index if item.endswith(suffix)]
    if len(hits) == 1:
        return hits[0], ""
    if not hits:
        return None, f"目标文件不存在：{name}"
    return None, f"目标文件有歧义（{name}）：{', '.join(sorted(hits)[:4])}"


# 纯文件引用：反引号里的 `xxx.md`，或 markdown 链接 `[文字](xxx.md)`。
#
# ## 为什么要单独查它（这是本检查器此前的**整类盲区**）
#
# 上面的 `文件:行号` 校验要求"路径 + 冒号 + 数字"，于是**只写文件名的引用
# 完全不进入校验**——README/ADR 里那种指向另一份 ADR 的引用正是这一类。
# 实测后果：`docs/adr/0005-<名字>.md` 并不存在，而 12 处代码引用它，
# 门禁一路绿灯（该缺口当时由 ADR-0004 自己记成"文档债"，现已补上该文件）。
# 一个"引用了不存在的文件"却没人报警的文档体系，不能算被校验过。
#
# 只查 `.md`：其它后缀（`.py` / `.json` / `.yml`）在文档里常被当作**名字**提及，
# 一律当引用会大面积误报；`.md` 才是这条纪律真正需要覆盖的那一类。
#
# ## 只查"看起来是本仓库路径"的 `.md`（精度优先）
#
# 第一版规则太宽，实测一次就抓出 11 处，其中**只有 1 处是真的**（其余是示例路径、
# 跨仓路径、以及本文件自己的 docstring 示例）。宽规则会让这个门禁迅速变成噪声，
# 而噪声门禁最后一定会被无视——那比没有门禁更糟。所以收紧成四条：
#
# 1. 含 `/`（裸文件名跳过：文档里 `_index.md` / `xxx.md` 这种"举个例子"太常见）；
# 2. 不以 `/` 开头（`/content/en/docs/...` 那种是**上游语料**里的路径，不是本仓的）；
# 3. 不含 `...` / `*` / `{}` / `<>`（省略号与通配是"示意"，不是引用）；
# 4. **首段是已知的仓内顶层目录**（`docs/` / `src/` / `scripts/` / `tests/` /
#    `.github/` / `docker/`）。这样 `concepts/foo.md`（k8s 文档里的示意路径）与
#    `data/raw/OpenCMKG/README.md`（**另一个仓库**里的文件、且 `data/` 本就被 gitignore）
#    都不会被误判。
#
# 代价写清楚：一个**裸文件名**形式的悬空引用（例如正文写 `README.md` 而它被删了）
# 查不出来。这是刻意换来的精度，不是遗漏。
_FILE_REF_RE = re.compile(r"(?<![\w./-])([A-Za-z0-9_./\-\u4e00-\u9fff]+\.md)")
_MD_LINK_RE = re.compile(r"\]\(([^)\s]+?\.md)(?:#[^)\s]*)?\)")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_CHECKED_ROOTS = ("docs/", "src/", "scripts/", "tests/", ".github/", "docker/")


def _is_checkable_ref(name: str) -> bool:
    """这个 `.md` 记号是否值得去查存在性（规则见上面那段注释）。"""
    if name.startswith("/") or "/" not in name:
        return False
    if "..." in name or any(ch in name for ch in "*{}<>"):
        return False
    return name.startswith(_CHECKED_ROOTS)


def collect_file_refs(doc: str, text: str) -> list[tuple[int, str, bool]]:
    """收集 `(行号, 路径, 是否 markdown 链接)`。

    跳过代码围栏内部：那里常常是"示例里的文件名"，不是本仓库的引用。
    """
    out: list[tuple[int, str, bool]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _MD_LINK_RE.finditer(line):
            out.append((lineno, m.group(1), True))
        for token in _BACKTICK.findall(line):
            if _ref_shape(token) is not None:
                continue  # 已经由 `文件:行号` 那条路校验
            if token.startswith(("http://", "https://")):
                continue
            found = _FILE_REF_RE.search(token)
            if found:
                out.append((lineno, found.group(1), False))
    return out


def _resolve_file_ref(raw: str, doc: str, index: list[str]) -> str | None:
    """把文档里的 `.md` 路径解析成仓内路径；解析不出返回 None。

    依次试：相对仓库根、相对**该文档所在目录**（markdown 链接的常见写法）、
    以及"按路径后缀唯一匹配"。
    """
    # 注意用"去前缀"而不是 `lstrip("./")`：后者是**字符集**剥离，
    # 会把 `.github/...` 开头的点也吃掉（实测踩过：`.github/PULL_REQUEST_TEMPLATE.md`
    # 被解析成 `github/...` 然后报"不存在"）。
    name = raw
    while name.startswith("./"):
        name = name[2:]
    name = name.lstrip("/")
    if name in index:
        return name
    doc_dir = str(Path(doc).parent).replace("\\", "/")
    if doc_dir != ".":
        joined = f"{doc_dir}/{name}"
        if joined in index:
            return joined
    target, _ = _resolve_target(name, index)
    return target


def check_file_refs(root: Path, docs: list[str], index: list[str]) -> list[str]:
    """校验全部纯 `.md` 引用指向的文件存在。返回失败描述列表。"""
    failures: list[str] = []
    for doc in docs:
        path = root / doc
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, raw, is_link in collect_file_refs(doc, text):
            name = raw[2:] if raw.startswith("./") else raw
            if not _is_checkable_ref(name):
                continue
            if _resolve_file_ref(raw, doc, index) is not None:
                continue
            kind = "markdown 链接" if is_link else "反引号引用"
            failures.append(f"{doc}:{lineno} {kind} `{raw}`：仓内不存在这个文件")
    return failures


def collect_anchors(doc: str, text: str, index: list[str]) -> list[Anchor]:
    """从一份文档里收集全部引用。无文件名的续写引用沿用同段上文的目标文件。"""
    lines = text.splitlines()
    anchors: list[Anchor] = []
    carry: str | None = None
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            carry = None  # 空行 = 新段落，续写引用不再沿用上文
            continue
        for token in _BACKTICK.findall(line):
            match = _ref_shape(token)
            if match is None:
                continue
            raw_path = match.group("path").strip()
            ranges = _parse_ranges(match.group("spec"))
            unit = _logical_sentence(lines, lineno - 1)
            if raw_path:
                target, _ = _resolve_target(raw_path, index)
                if target:
                    carry = target
                anchors.append(
                    Anchor(
                        doc=doc,
                        line=lineno,
                        text=token.strip(),
                        target=target or raw_path,
                        ranges=ranges,
                        bare=False,
                        unit=unit,
                    )
                )
            else:
                anchors.append(
                    Anchor(
                        doc=doc,
                        line=lineno,
                        text=token.strip(),
                        target=carry or "",
                        ranges=ranges,
                        bare=True,
                        unit=unit,
                    )
                )
    return anchors


def _range_text(lines: list[str], lo: int, hi: int, window: int) -> str:
    start = max(0, lo - 1 - window)
    end = min(len(lines), hi + window)
    return "\n".join(lines[start:end])


def check_paths(
    root: Path,
    docs: list[str],
    *,
    whitelist: dict[tuple[str, str], White] | None = None,
    window: int = DEFAULT_WINDOW,
) -> Report:
    """校验 `root` 下这些文档（仓内相对路径）里的全部引用。"""
    table = WHITELIST if whitelist is None else whitelist
    index = _walk_files(root)
    report = Report(docs=len(docs))
    cache: dict[str, list[str]] = {}

    for doc in docs:
        doc_path = root / doc
        if not doc_path.is_file():
            continue
        text = doc_path.read_text(encoding="utf-8")
        for anchor in collect_anchors(doc, text, index):
            report.findings.append(_check_one(root, anchor, table, window, cache, index))
    return report


def _check_one(
    root: Path,
    anchor: Anchor,
    whitelist: dict[tuple[str, str], White],
    window: int,
    cache: dict[str, list[str]],
    index: list[str],
) -> Finding:
    where = f"{anchor.doc}:{anchor.line} `{anchor.text}`"

    if not anchor.target:
        return Finding(anchor, "fail", f"{where} 无归属目标文件（续写引用 `:行` 之前没有出现过文件名）")

    target, why = _resolve_target(anchor.target, index)
    if target is None:
        return Finding(anchor, "fail", f"{where} {why}")

    if target not in cache:
        cache[target] = (root / target).read_text(encoding="utf-8").splitlines()
    target_lines = cache[target]
    n_lines = len(target_lines)

    for lo, hi in anchor.ranges:
        if lo < 1 or hi < lo or hi > n_lines:
            return Finding(
                anchor,
                "fail",
                f"{where} 行号越界：{target} 只有 {n_lines} 行，引用的是 {lo}-{hi}",
            )

    detail = f"{where} -> {target}:{_format_ranges(anchor.ranges)}"
    white = whitelist.get((anchor.doc, anchor.text))

    # 白名单优先：这些引用的判据由人在表里写死（`expect`），机器不再自己抽标识符。
    # `expect` 是**强断言**——锚点漂走时它必须失败，否则白名单就成了免死金牌。
    if white is not None:
        if white.expect is not None:
            blob = "\n".join(_range_text(target_lines, lo, hi, window) for lo, hi in anchor.ranges)
            if white.expect not in blob:
                return Finding(
                    anchor,
                    "fail",
                    f"{detail} 白名单钉住的 {white.expect!r} 不在锚点范围内（锚点已漂移）",
                )
        note = f"白名单：{white.reason}"
        if white.expect is None:
            note += "（**无内容断言**，只能人工复核）"
        return Finding(anchor, "white", f"{detail} {note}", matched=white.expect or "")

    candidates = _candidates(anchor.unit, anchor.text, index)
    # 长的名字更能定位代码，也更能说明"命中在哪里"：报 `NEXTAUTH_SECRET` 比报 `kubedocs` 有用。
    strong = tuple(sorted((name for name in candidates if _is_strong(name)), key=len, reverse=True))
    wanted = strong or tuple(sorted(candidates, key=len, reverse=True))

    if not wanted:
        return Finding(anchor, "fail", f"{detail} 散文里没有可校验的标识符（白名单未登记）")

    for lo, hi in anchor.ranges:
        blob = _range_text(target_lines, lo, hi, window)
        for name in wanted:
            if name in blob:
                return Finding(anchor, "ok", detail, candidates=candidates, matched=name)

    kind = "强候选" if strong else "候选"
    return Finding(
        anchor,
        "fail",
        f"{detail} 锚点范围内找不到散文里的任何{kind}（{' / '.join(wanted)}）",
        candidates=candidates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="文档锚点校验（内容级）")
    parser.add_argument("--root", default=str(REPO_ROOT), help="仓库根（默认脚本所在仓库）")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="锚点前后放宽的行数")
    parser.add_argument("--verbose", action="store_true", help="逐条打印命中的标识符")
    parser.add_argument("--whitelist", action="store_true", help="只列出白名单")
    args = parser.parse_args(argv)

    if args.whitelist:
        print(f"白名单 {len(WHITELIST)} 条：")
        # 不打印行号：键里已经没有它，而打印一个"会漂移的位置"只会误导。
        # 想知道它现在在哪一行，看正常输出里的 `SECURITY.md:<实际行号>`。
        for (doc, text), white in sorted(WHITELIST.items()):
            expect = f"须含 {white.expect!r}" if white.expect else "**无内容断言**"
            print(f"  - {doc} `{text}` [{expect}]：{white.reason}")
        return 0

    root = Path(args.root).resolve()
    docs = _doc_files(root)
    report = check_paths(root, docs, window=args.window)
    # 第二类校验：纯 `.md` 文件引用（`文件:行号` 那条路看不见它们，见 collect_file_refs 的说明）。
    file_failures = check_file_refs(root, docs, _walk_files(root))

    print(f"扫描文档   : {report.docs} 个文件（*.md + 源码/测试/脚本里的文档字符串）")
    print(f"引用处数   : {len(report.findings)}")
    print(f"内容核对通过 : {len(report.verified)}")
    print(f"白名单     : {len(report.whitelisted)} 条（每条在脚本内写明理由）")
    for finding in report.whitelisted:
        print(f"  · {finding.detail}")
    for finding in report.failures:
        print(f"  ✗ {finding.detail}")
    for failure in file_failures:
        print(f"  ✗ {failure}")

    if args.verbose:
        print("\n逐条：")
        for finding in report.findings:
            mark = {"ok": "OK", "fail": "✗", "white": "白"}[finding.status]
            extra = f"命中 {finding.matched!r}" if finding.matched else ""
            print(f"  [{mark}] {finding.detail}  {extra}")

    if report.failures or file_failures:
        print(
            f"\n结果：不通过（{len(report.failures)} 处行号锚点未被内容校验挡住，"
            f"{len(file_failures)} 处引用的文件不存在）"
        )
        return 1
    print("\n结果：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
