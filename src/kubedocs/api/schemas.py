"""HTTP 层的数据契约（pydantic v2）：进来的形状与出去的形状。

这里只放数据契约：没有 IO、没有业务判断、没有对 `Settings` 的依赖。之所以与路由分开，
是因为请求体校验与响应形状是**对外接口**，改路由时顺手改契约是接口不稳定的常见来源。

请求体是**外部输入**，而检索与生成都要花时间/花钱，所以每个字段都设了上限：

- `q ≤ 300`：更长的"查询"已经不是查询，会被原样送进嵌入模型与 FTS5；
- `message ≤ 2000`：单轮对话文本的上限（再长应该走工单，不是聊天）；
- `top_k ∈ [1, 20]`：`top_k=100000` 会让融合与取块把内存吃满；
- `extra="forbid"`：未知字段直接 422——静默丢弃会把"我明明传了 top_k 却没生效"
  变成一个查不出来的问题。

越界一律 422（由 pydantic/FastAPI 给出），**不在这里手工判**。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # 只为类型标注：运行期不需要 import 检索/生成层的实现
    from kubedocs.retrieve.hybrid import RetrievedChunk

BODY_LIMIT = 600
"""`/search` 返回的单条正文上限（字符）。

与 `agent/rag_tools.BODY_LIMIT` 保持一致：同一个项目里"给模型的片段"与"给调用方的
片段"截到同一长度，省得两边对不上时不知道以谁为准。
"""


def _clip(text: str, limit: int) -> str:
    """截断到 `limit` 字符（含省略号，保证总长**不超**上限）。"""
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- 响应形状
class Hit(BaseModel):
    """一条检索命中。字段与 `RetrievedChunk` 对齐，正文已截断。"""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    doc_id: str
    title: str
    breadcrumb: str
    source_url: str
    body: str
    score: float = 0.0
    ranks: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, *, body_limit: int = BODY_LIMIT) -> Hit:
        """由检索结果构造响应条目，正文截断到 `body_limit`。

        一条命中可能上千字符，`top_k=20` 时响应体轻松过百 KB，而调用方（前端/探针）
        多数时候只需要判断"这条讲的是不是我要的"。`source_url` **不截断**——它才是
        正文被截之后仍能追溯原文的保证。
        """
        return cls.model_validate(chunk).model_copy(update={"body": _clip(chunk.body, body_limit)})


class SearchResponse(BaseModel):
    """`/search` 的响应：命中 + 诊断 + 一句说明。"""

    query: str
    hits: list[Hit]
    diagnostics: dict[str, Any]
    note: str = ""
    """空索引 / 无命中时的人话说明；正常命中时为空串。

    单独一个字段而不是塞进 `diagnostics`：调用方最常见的问题就是"为什么没结果"，
    它值得一个稳定、不用解析嵌套结构的答案。
    """


class CitationOut(BaseModel):
    """回答里 `[n]` 与来源的对应关系。引用必须能点开。"""

    model_config = ConfigDict(from_attributes=True)

    index: int
    chunk_id: str
    title: str
    breadcrumb: str
    source_url: str
    label: str


class AskResponse(BaseModel):
    """`/ask` 的响应：回答 + 引用 + 降级说明。

    `degraded` 非空**不是错误**：没有 LLM 密钥、索引为空、模型调用失败都会走到这里，
    而检索结果照常返回。把"降级"做成一个正常字段，比让它伪装成空回答更容易被发现。
    """

    answer: str = ""
    citations: list[CitationOut] = Field(default_factory=list)
    n_hits: int = 0
    degraded: str = ""


class HealthResponse(BaseModel):
    """存活探针。字段刻意保持极少：探针要的是"活着吗、哪个域、有多少块"（外加两个后端读数）。

    **存储降级不算不存活**（所以它不进 `ok`）：配置的是 pgvector、数据库不可达时会降级为
    本地 sqlite-vec，服务照常可用。但这件事必须看得见，所以单开一个字段而不是塞进 `ok`——
    探针报 red 会触发重启，而重启解决不了"Postgres 没起"。
    """

    ok: bool
    domain: str
    n_chunks: int
    store: str = ""
    """实际生效的后端名（降级时与配置值不同）。"""
    store_fallback: str = ""
    """非空表示发生了降级，内容是原因。空串 = 用的就是配置里的后端。"""


class ToolsResponse(BaseModel):
    """`/tools` 的自省输出。"""

    tools: list[dict[str, Any]]


# --------------------------------------------------------------------------- 请求形状
class SearchRequest(BaseModel):
    """`/search` 的请求体。"""

    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=300, description="自然语言查询")
    top_k: int = Field(default=8, ge=1, le=20, description="返回条数，1~20")
    domain: str = Field(default="", max_length=32, description="域；留空表示用服务当前配置的域")

    @field_validator("q")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """只有空白的查询必须拒绝：它注定无结果，却会消耗一次嵌入与一次全文检索。"""
        if not value.strip():
            raise ValueError("q 不能只有空白字符")
        return value.strip()


class AskRequest(SearchRequest):
    """`/ask` 的请求体：与 `/search` 同体。

    两个端点先检索、后生成，参数没有任何理由不同；共用一套 schema 也让
    "换个端点试试"时不必重新学一遍字段。
    """


class ChatRequest(BaseModel):
    """`/chat` 的请求体（域固定为 clinic）。"""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000, description="用户这一轮说的话")
    thread_id: str = Field(
        default="", max_length=64, description="会话 id；留空则由服务端生成并在首个 SSE 事件里返回"
    )
    patient_id: str = Field(default="anon", max_length=64, description="患者标识（号源系统用）")

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """空白消息拒绝：问诊图会把空串当成"用户什么都没说"，只会白跑一轮。"""
        if not value.strip():
            raise ValueError("message 不能只有空白字符")
        return value.strip()
