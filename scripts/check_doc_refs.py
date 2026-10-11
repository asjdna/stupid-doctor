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


WHITELIST: dict[tuple[str, int, str], White] = {
    ("SECURITY.md", 8, "Dockerfile:2"): White(
        reason="第 6-9 行是引用块，反引号里除这个锚点外只有文件名（`Dockerfile` / "
        "`.github/workflows/ci.yml`），没有可抽的符号名",
        expect="尚未在本机实测",
    ),
    ("SECURITY.md", 8, "docker-compose.yml:3"): White(
        reason="与上一条同一句（第 6-9 行），同样只有文件名",
        expect="尚未在本机实测",
    ),
    ("SECURITY.md", 22, ".gitignore:53-54"): White(
        reason="第 21-22 行在说 clinic 卡片语料不入库，反引号里只有这个锚点",
        expect="clinic",
    ),
    ("SECURITY.md", 152, "clinic_baseline.json:6-7"): White(
        reason="markdown 表格行：单元格里只有数字与这个锚点，没有符号名",
        expect='"recall"',
    ),
    ("SECURITY.md", 153, "clinic_baseline.json:7"): White(
        reason="同上（表格行，单元格里只有数字与锚点）",
        expect='"false_positive_rate"',
    ),
    ("SECURITY.md", 154, "clinic_baseline.json:16-17"): White(
        reason="同上（表格行，单元格里只有数字与锚点）",
        expect='"accuracy"',
    ),
    ("SECURITY.md", 203, "generate.py:534-536"): White(
        reason="第 203-205 行那句只说『片段与工具返回结果直接拼进提示词』，括号里就是两个锚点，"
        "没有任何符号名",
        expect="【文档片段】",
    ),
    ("SECURITY.md", 204, "tool_loop.py:322-327"): White(
        reason="与上一条同一句（第 203-205 行），同样没有符号名",
        expect="_tool_message",
    ),
    ("SECURITY.md", 212, "tool_loop.py:373-378"): White(
        reason="第 212-213 行只说『工具参数逐个走 pydantic 校验』，`pydantic` 不是本仓符号",
        expect="model_validate",
    ),
    ("SECURITY.md", 213, "tool_loop.py:397-405"): White(
        reason="与上一条同一句（第 212-213 行）：『命中注册表 / 回灌可用列表』没有反引号符号名",
        expect="_lookup",
    ),
    ("SECURITY.md", 225, "docs/adr/0001-技术选型.md:19"): White(
        reason="这一句的反引号符号是包名 `ragas` / `deepeval`，它们是『全仓没有任何 import』的宾语，"
        "不是锚点目标的符号；拿它们去 ADR 第 19 行里找是本检查器的误报。锚点目标是那张表里的"
        "『评测…Ragas 对照』行（『这两个包只被文档提到过』的唯一文档出处）",
        expect="Ragas",
    ),
    ("SECURITY.md", 241, "generate.py:193-214"): White(
        reason="表格行：正文唯一的反引号是举例用的越界标记 `[99]`，不是符号名",
        expect="validate_citations",
    ),
    ("SECURITY.md", 242, "generate.py:509-514"): White(
        reason="表格行：正文没有反引号符号名（『打桩 LLM』不是符号）",
        expect="if not hits",
    ),
    ("SECURITY.md", 244, "tool_loop.py:250-256"): White(
        reason="表格行：正文（『按名索引 / 重名报错』）没有反引号符号名",
        expect="registry",
    ),
    ("SECURITY.md", 245, "tool_loop.py:409-423"): White(
        reason="表格行：正文（『默认拒绝执行』）没有反引号符号名",
        expect="_confirm_side_effect",
    ),
    ("SECURITY.md", 284, ":54-55, 68-69, 87-88"): White(
        reason="第 284-285 行的反引号只有端口号（`5432` / `6379` / `8080`，抽不出标识符）与这个锚点",
        expect="6379",
    ),
    ("SECURITY.md", 307, "Dockerfile:2"): White(
        reason="第 306-308 行是『未验证事项』条目，反引号里只有两个文件名 + 两个锚点",
        expect="尚未在本机实测",
    ),
    ("SECURITY.md", 307, "docker-compose.yml:3"): White(
        reason="与上一条同一句（第 306-308 行），同样只有文件名",
        expect="尚未在本机实测",
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
    whitelist: dict[tuple[str, int, str], White] | None = None,
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
    whitelist: dict[tuple[str, int, str], White],
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
    white = whitelist.get((anchor.doc, anchor.line, anchor.text))

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
        for (doc, line, text), white in sorted(WHITELIST.items()):
            expect = f"须含 {white.expect!r}" if white.expect else "**无内容断言**"
            print(f"  - {doc}:{line} `{text}` [{expect}]：{white.reason}")
        return 0

    root = Path(args.root).resolve()
    report = check_paths(root, _doc_files(root), window=args.window)

    print(f"扫描文档   : {report.docs} 个文件（*.md + 源码/测试/脚本里的文档字符串）")
    print(f"引用处数   : {len(report.findings)}")
    print(f"内容核对通过 : {len(report.verified)}")
    print(f"白名单     : {len(report.whitelisted)} 条（每条在脚本内写明理由）")
    for finding in report.whitelisted:
        print(f"  · {finding.detail}")
    for finding in report.failures:
        print(f"  ✗ {finding.detail}")

    if args.verbose:
        print("\n逐条：")
        for finding in report.findings:
            mark = {"ok": "OK", "fail": "✗", "white": "白"}[finding.status]
            extra = f"命中 {finding.matched!r}" if finding.matched else ""
            print(f"  [{mark}] {finding.detail}  {extra}")

    if report.failures:
        print(f"\n结果：不通过（{len(report.failures)} 处引用没有被内容校验挡住）")
        return 1
    print("\n结果：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
