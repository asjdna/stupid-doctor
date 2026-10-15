#!/usr/bin/env python
"""文档锚点门禁：`文件:行号` 引用必须指着它自己声称的东西，纯 `.md` 文件引用必须存在。

校验两类引用：

1. **行号锚点（内容级）**：抽出引用所在那句散文里反引号包住的标识符，要求它出现在锚点范围
   （前后各放宽 `--window` 行）里：散文说"记 index_error"并标了某几行、那几行里找不到
   `index_error` ⇒ 失败。行号"存在且不越界"并不代表指对了地方。
2. **纯 `.md` 文件引用**：反引号里或 markdown 链接里的 `.md` 路径必须真的存在。单列一条是
   因为第 1 类要求"路径 + 冒号 + 数字"，看不见只写文件名的引用。

判据是散文里的标识符：行号会漂、文件名会漂，只有**符号名**稳定，这是文档能给出的最强机器
可读线索。抽取时**跳过文件名字面量**（形如 `src/kubedocs/api/app.py` 的反引号会命中
`kubedocs` 这种处处都有的词，等于没校验），并让**强候选优先**：带下划线 / 长度 ≥ 6 / 含大写
的名字（`index_error`、`DailyBudget`）比 `chat`、`user` 这类普通词更能定位代码，句子里有强
候选就必须由强候选命中，否则退回整句的普通候选。

无法校验的引用不许静默跳过：散文里**没有任何源码标识符**的引用（例如"自述尚未在本机实测"只
指向文件头的一句注释）机器无从判断对错，必须逐条登记在 `WHITELIST` 里并写明理由。键是
`(文档名, 锚点文本)`，**刻意不含引用所在的行号**（位置不是身份），且每条豁免必须**真的被扫描
命中**，否则单测失败、陈旧条目不会静默留着。白名单**不是**免死金牌：每条都要给出 `expect`
（锚点范围内必须出现的字符串），锚点一旦漂走它就会失败；**既校验不过又未登记 ⇒ 非零退出**。

用法：
    python scripts/check_doc_refs.py              # 全仓文档
    python scripts/check_doc_refs.py --root DIR   # 指定仓库根（默认脚本所在仓库）
    python scripts/check_doc_refs.py --window 4   # 锚点前后放宽的行数（默认 2）
    python scripts/check_doc_refs.py --verbose    # 逐条打印状态与命中的标识符
    python scripts/check_doc_refs.py --whitelist  # 只列白名单，不扫描

输出：`扫描文档` 是参与扫描的文件数（*.md 与源码/测试/脚本里的文档字符串），`引用处数` 是
抽到的引用总数，`内容核对通过` 是标识符在锚点范围内命中的条数，`白名单` 是被豁免的条数（随后
逐条列理由）。带 `✗` 的行是一处失败：行号锚点失败＝标识符在锚点范围内找不到，`.md` 引用失败
＝该文件在仓内不存在。

退出码：0 全部通过；1 有引用站不住（行号锚点没通过内容校验，或引用的文件不存在）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 锚点窗口：容下 ±1 的排版误差（空行、续行位置），但"位移若干行指到别的函数"必须被抓到。
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

# 引用形状：`<可空路径>:<行号>[, <行号>|-<区间>]...`；路径不许有空格与冒号，
# `python:3.10-slim`、`http://host:3000/x` 因此天然被排除。
_REF_SHAPE = re.compile(r"^(?P<path>[^\s:]*):(?P<spec>\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)$")

# 标识符抽取：反引号里的 `extra="forbid"`、`_confirm_side_effect()` 也要能取出名字。
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# markdown 块的开头：列表项、有序列表项、表格行、标题、引用块。用它切"同一句散文"，
# 避免把相邻条目的标识符算进来（那会让引用"指哪都能过"）。
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
# 键是 (文档相对路径, 反引号里的引用原文)，**刻意不含引用所在的行号**——位置不是身份。
# =====================================================================================


@dataclass(frozen=True)
class White:
    """一条白名单：为什么机器校验不了 + 仍然生效的内容断言。"""

    reason: str
    expect: str | None = None


WHITELIST: dict[tuple[str, str], White] = {
    # 只登记"整句里除锚点外没有任何源码标识符"的引用，且每条都要给 `expect`：
    # 白名单不是免死金牌，锚点漂走时必须失败。
    # 目前为空：仓内引用的同一句散文里都有可核对的源码标识符，不需要豁免。
    # 新增一条前先确认机器真的判断不了——能靠标识符核对的引用不该进这张表。
}


@dataclass(frozen=True)
class Anchor:
    """一处引用。"""

    doc: str  # 引用所在的文档（仓内相对路径）
    line: int  # 引用所在的行号（1 基）
    text: str  # 反引号里的引用原文（形如 路径:行号）
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

    只取所在行不够，取整段又太宽（相邻条目的标识符会被算进来），所以向两侧扩到换行或换条目。
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
    """这个反引号写的是**文件**（而不是符号）吗？能解析成仓内文件、或带源码后缀的
    反引号一律不抽：`kubedocs` 这种处处都出现的词当标识符等于没校验。
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

    用 `os.walk` 下降时剪枝，而非 `rglob` 事后过滤：`.venv` 里几万个文件不该白扫。
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
    """把引用里的文件名（文档里常是缩写，如 `api/app.py`）解析成仓内路径。

    先试"相对仓库根"，再试"按路径后缀唯一匹配"。**歧义不猜**：同名就报错，让人把引用写全。
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


# 纯文件引用：反引号里的 `xxx.md`，或 markdown 链接 `[文字](xxx.md)`，只查目标文件是否存在。
# 单列这一类，是因为上面 `文件:行号` 那条路要求"路径 + 冒号 + 数字"，看不见只写文件名的引用。
# 只查 `.md`（其它后缀在文档里常被当作**名字**提及，一律当引用会大面积误报），且只查"看起来是
# 本仓库路径"的（精度优先：宽规则会让门禁变成噪声而终被无视）。四条规则缺一不可：
# 1. 含 `/`（裸文件名跳过：`_index.md` 这种"举个例子"太常见）；
# 2. 不以 `/` 开头（`/content/en/docs/...` 那种是**上游语料**里的路径，不是本仓的）；
# 3. 不含 `...` / `*` / `{}` / `<>`（省略号与通配是"示意"，不是引用）；
# 4. **首段是已知的仓内顶层目录**（`docs/` / `src/` / `scripts/` / `tests/` / `.github/` / `docker/`）。
# 代价：**裸文件名**形式的悬空引用（正文写 `README.md` 而它被删了）查不出来——刻意换来的精度。
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
    """收集 `(行号, 路径, 是否 markdown 链接)`，跳过代码围栏内部（那里常是示例文件名）。"""
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

    依次试：相对仓库根、相对该文档所在目录（markdown 链接的常见写法）、按路径后缀唯一匹配。
    """
    # 用"反复去掉 ./ 前缀"而不是 `lstrip("./")`：后者是**字符集**剥离，会把 `.github/...`
    # 开头的点也吃掉，解析出 `github/...` 这种不存在的路径。
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

    # 白名单优先：判据由人在表里写死（`expect`），锚点漂走时必须失败，否则就成了免死金牌。
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
        # 不打印行号：键里已经没有它，打印一个"会漂移的位置"只会误导。
        for (doc, text), white in sorted(WHITELIST.items()):
            expect = f"须含 {white.expect!r}" if white.expect else "**无内容断言**"
            print(f"  - {doc} `{text}` [{expect}]：{white.reason}")
        return 0

    root = Path(args.root).resolve()
    docs = _doc_files(root)
    report = check_paths(root, docs, window=args.window)
    # 第二类校验：纯 `.md` 文件引用（`文件:行号` 那条路看不见它们，规则见 _FILE_REF_RE 上方）。
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
