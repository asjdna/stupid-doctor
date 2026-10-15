"""把检索与引用核对暴露成 `ToolSpec`：工具是**声明 + 执行体**，执行由内核编排。

schema 与执行体放在同一个 `ToolSpec` 里，好处是"声明的东西就是能跑的东西"；但执行体
必须薄——只做一次检索、把结果压成模型能读的 dict，不做编排、不做多轮循环、不写状态。
那些是图的事：工具一旦自己编排，内核就失去了统一的计时、重试与副作用控制。

正文按字符截断并留下 `…`（上限见 `BODY_LIMIT`）：一个块的正文可能上千字符，5 条命中
就能把模型上下文里的预算吃干净；让模型知道"后面还有，但被截了"比拿到半句话更安全。

引用核对单独成一个只读工具：生成容易写出打不开的 `[99]`，核对要能独立复算，
评测与人工才能在不重新生成的前提下检查同一份答案。规则与 `generate.validate_citations`
同源，但那一步会剥掉越界标记、会改文本，这里**只报告、不修改**。

与域解耦：`make_search_tool` 只按鸭子类型用 `retriever.search(query, top_k=...)`，
模块顶层不 import 任何具体检索器实现或 `generate`（那条依赖链会拖进配置、LiteLLM、
嵌入模型），否则内核就会绑定某一个域的存储选型。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from kubedocs.agent.tools import SearchDocsParams, ToolError, ToolSpec

BODY_LIMIT = 600
"""单条命中的正文最多交给模型的字符数（构造工具时可用 `body_limit` 覆盖）。"""

_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
"""引用标记的形状与 `generate` 保持一致：1~3 位数字包在方括号里。"""


class CitationCheckParams(BaseModel):
    """核对一份答案里的 `[n]` 引用。"""

    text: str = Field(min_length=1, max_length=20000, description="待核对的答案全文")
    n_available: int = Field(ge=0, le=100, description="实际提供给模型的片段条数")


def _text(value: object) -> str:
    """把属性安全地取成 str；`None` 变空串而不是 `"None"`，避免假数据混进结果。"""
    return "" if value is None else str(value)


def _coerce_search_params(params: object) -> SearchDocsParams:
    """把传进来的参数统一成 `SearchDocsParams`；不合法一律转成 `ToolError`。

    正常情况下内核已经用 schema 校验过一遍，这里是第二道闸门：模型调用参数是外部输入，
    一旦原始 JSON 被直接递进来，也要给出可读的失败原因，而不是 `AttributeError` 的 traceback。
    """
    if isinstance(params, SearchDocsParams):
        return params
    try:
        return SearchDocsParams.model_validate(params)
    except ValidationError as e:
        raise ToolError(f"search_docs 参数不合法：{e}") from e


def _hit_payload(hit: object, *, body_limit: int) -> dict[str, Any]:
    """一条命中压成纯 dict：只留模型确实用得上的字段，并截断正文。"""
    body = _text(getattr(hit, "body", ""))
    return {
        "chunk_id": _text(getattr(hit, "chunk_id", "")),
        "doc_id": _text(getattr(hit, "doc_id", "")),
        "title": _text(getattr(hit, "title", "")),
        "breadcrumb": _text(getattr(hit, "breadcrumb", "")),
        "source_url": _text(getattr(hit, "source_url", "")),
        "body": f"{body[:body_limit]} …" if len(body) > body_limit else body,
    }


def make_search_tool(
    retriever: object, *, domain: str = "k8s_docs", top_k: int = 5, body_limit: int = BODY_LIMIT
) -> ToolSpec:
    """构造 `search_docs` 工具。

    `top_k` 是**内核给出的检索预算上限**：模型可以在 schema 允许的 1~20 里提要求，
    但实际取 `min(请求值, top_k)`。工具是预算的所有者，不能因为模型一句
    `top_k=20` 就把上下文撑爆。
    """
    if top_k < 1:
        raise ValueError(f"top_k 必须 >= 1（收到 {top_k}）：0 条命中的检索没有意义")

    def handler(params: SearchDocsParams) -> dict[str, Any]:
        """跑一次检索，返回 `{"query", "n", "hits"}`。

        失败一律转成 `ToolError`（可预期失败 -> 一句人话），不抛裸异常：
        检索器炸掉或参数不合法都不该让整张图 traceback。
        """
        spec = _coerce_search_params(params)
        query = spec.query.strip()
        if not query:
            raise ToolError("search_docs 的 query 只有空白字符：没有可检索的内容")

        k = min(spec.top_k, top_k)
        search = getattr(retriever, "search", None)
        if not callable(search):
            raise ToolError(
                f"检索器 {type(retriever).__name__} 没有可调用的 search 方法："
                "无法执行检索（域包的 retriever 配置有误）"
            )
        try:
            result = search(query, top_k=k)
        except ToolError:
            raise
        except Exception as e:  # noqa: BLE001 - 检索失败是可预期失败，转成人话更合适
            raise ToolError(f"检索失败（{type(e).__name__}: {e}）：{query!r}") from e

        raw_hits = getattr(result, "results", None)
        if raw_hits is None:
            raise ToolError(
                f"检索器返回了没有 results 字段的对象（{type(result).__name__}）：无法把结果交给模型"
            )
        hits = [_hit_payload(h, body_limit=body_limit) for h in raw_hits]
        payload: dict[str, Any] = {"query": query, "n": len(hits), "hits": hits}
        return payload

    description = (
        "在文档语料里做混合检索（全文 + 向量 + 融合），返回按相关度排序的片段："
        "每条含 chunk_id/doc_id/title/breadcrumb/source_url/body（body 已截断）。"
        f"最多返回 {top_k} 条。用它取证；没有命中的话题不要凭记忆作答。"
    )
    when_not = (
        "不要用它：① 寒暄、意图澄清、追问（“你好”“这个呢”）——先把问题问清楚；"
        "② 询问集群的实时状态（现在有几个 pod、当前副本数），静态文档语料里没有这种答案；"
        "③ 已经检索过的同一句话——先读已有结果，重复检索只会烧预算；"
        "④ 需要写操作（创建/删除资源）时，检索不提供任何执行能力。"
    )
    return ToolSpec(
        name="search_docs",
        description=description,
        params=SearchDocsParams,
        domain=domain,
        when_not=when_not,
        side_effect=False,
        handler=handler,
    )


def make_citation_check_tool() -> ToolSpec:
    """构造 `check_citations` 工具：只核对 `[n]`，不改写答案。

    返回 `{"ok": bool, "problems": [...]}`。两类问题会被指出：
    指向不存在片段的编号（打不开的引用比没有引用更误导），
    以及一个引用都没有（无从核对来源）。
    """

    def handler(params: CitationCheckParams) -> dict[str, Any]:
        """核对引用编号，返回 `{"ok", "problems"}`。"""
        if not params.text.strip():
            raise ToolError("check_citations 的 text 只有空白字符：没有可核对的答案")
        text, n_available = params.text, params.n_available

        problems: list[str] = []
        bad = sorted(
            {int(m.group(1)) for m in _MARKER_RE.finditer(text) if not 1 <= int(m.group(1)) <= n_available}
        )
        if bad:
            labels = [f"[{i}]" for i in bad]
            problems.append(f"引用了未提供的片段编号 {labels}")
        if n_available <= 0:
            problems.append("没有提供任何片段，答案里不应该出现引用")
        elif not _MARKER_RE.search(text):
            problems.append("答案没有任何引用标记，无法核对来源")

        payload: dict[str, Any] = {"ok": not problems, "problems": problems}
        return payload

    description = (
        "核对答案里的 [n] 引用是否都指向真实提供过的片段，返回 {ok, problems}。"
        "problems 为空即通过；越界编号和“一个引用都没有”都会被指出。"
        "只读：它不改写答案，只给结论。"
    )
    when_not = (
        "不要用它：① 还没有答案草稿时；② 片段条数为 0 时（没有可引用的来源，"
        "调用只会得到“无引用”这个必然结论）；③ 想让它判断论断是否**真的**被片段支持——"
        "它只核对编号是否存在，事实性判断要另做（人工或评测器）；"
        "④ 想让它修正答案，请改用生成流程，它不改文本。"
    )
    return ToolSpec(
        name="check_citations",
        description=description,
        params=CitationCheckParams,
        when_not=when_not,
        side_effect=False,
        handler=handler,
    )
