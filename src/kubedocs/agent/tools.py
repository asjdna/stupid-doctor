"""工具协议：agent 能调用的能力必须**先声明 schema，再执行**。

## 为什么不用字符串路由

参照项目里多智能体的路由是 `if "travel_advisor" in content`——让模型的**自由文本**
决定走哪条分支。它有两个硬伤：模型的措辞一变就失配；且没有 schema 约束，
模型可以调出一个不存在的工具或漏传参数。

这里所有能力都先声明成 `ToolSpec`（名字 + 说明 + Pydantic 参数模型），
路由与调用都走**结构化输出 / function calling**：模型只能在这些名字里选，
参数由 schema 校验。**执行与计时由内核统一做**，工具自身不做 IO 编排。

## 为什么工具描述要写"什么时候不要用"

只写"这个工具做什么"会让模型在相似的场景里乱调（例如问诊中途去调挂号）。
所以 `ToolSpec.when_not` 是必填的——它直接进提示词，是约束模型的主要手段。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ToolError(Exception):
    """工具执行的可预期失败（参数不合法、号源已满等）。

    与编程错误区分：这类错误应当被转成一句人话回复，而不是 traceback。
    """


class EmptyParams(BaseModel):
    """无参数工具的参数模型（显式给出，避免 schema 里出现 null）。"""


class SearchDocsParams(BaseModel):
    """检索文档。"""

    query: str = Field(min_length=1, max_length=300, description="检索用的自然语言查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数")


class ListDepartmentsParams(BaseModel):
    """列出可挂号科室。"""

    contains: str = Field(default="", max_length=32, description="只保留名字里含该字样的科室，留空返回全部")


class GetAvailabilityParams(BaseModel):
    """查询某科室的号源。"""

    department: str = Field(min_length=1, max_length=32, description="科室名")
    date: str = Field(default="", description="YYYY-MM-DD，留空表示今天")


class BookParams(BaseModel):
    """取号。"""

    department: str = Field(min_length=1, max_length=32, description="科室名")
    date: str = Field(default="", description="YYYY-MM-DD，留空表示今天")
    time_slot: str = Field(default="", description="上午 / 下午，留空自动选第一个有余号的")


class AskUserParams(BaseModel):
    """向用户追问（人在环路：图会在此挂起）。"""

    question: str = Field(min_length=1, max_length=500, description="要问用户的问题")
    options: list[str] = Field(
        default_factory=list, max_length=6, description="可选项；给出选项能显著降低用户的回答成本"
    )


@dataclass(frozen=True)
class ToolSpec:
    """一个可被 agent 调用的能力。**纯数据描述**，不含 IO。"""

    name: str
    description: str
    params: type[BaseModel]
    domain: str = "k8s_docs"
    when_not: str = ""
    """什么时候**不该**用这个工具。必填——它比 `description` 更能约束模型。"""

    side_effect: bool = False
    """是否有副作用（挂号有、检索没有）。有副作用的工具在图上需要额外确认。"""

    handler: Callable[..., Any] | None = field(default=None, compare=False)
    """执行体。`None` 表示只声明不实现（用于评测与自省）。"""

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI/LiteLLM 的 function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}\n不适用：{self.when_not}",
                "parameters": self.params.model_json_schema(),
            },
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "when_not": self.when_not,
            "domain": self.domain,
            "side_effect": self.side_effect,
        }


class ToolRegistry:
    """工具注册表：注册 / 按名取用 / 列举 / 转 function calling schema。

    与 domain 包同一套路数（自描述 + 按名解析 + 可列举）：加一个能力 = 注册一条，
    内核与提示词都不用改。
    """

    _tools: dict[str, ToolSpec] = {}

    @classmethod
    def register(cls, spec: ToolSpec) -> None:
        if spec.name in cls._tools:
            raise ValueError(f"工具 {spec.name!r} 已注册，不能重复注册")
        if not spec.when_not:
            raise ValueError(f"工具 {spec.name!r} 必须声明 when_not：只写'做什么'会让模型在相近场景里乱调")
        cls._tools[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> ToolSpec:
        if name not in cls._tools:
            raise ToolError(f"没有名为 {name!r} 的工具；已注册：{sorted(cls._tools)}")
        return cls._tools[name]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._tools)

    @classmethod
    def for_domain(cls, domain: str) -> list[ToolSpec]:
        return [cls._tools[n] for n in cls.names() if cls._tools[n].domain == domain]

    @classmethod
    def schemas(cls, domain: str | None = None) -> list[dict[str, Any]]:
        specs = [cls._tools[n] for n in cls.names()] if domain is None else cls.for_domain(domain)
        return [s.schema() for s in specs]

    @classmethod
    def describe(cls) -> list[dict[str, Any]]:
        return [cls._tools[n].describe() for n in cls.names()]

    @classmethod
    def clear(cls) -> None:
        """仅供测试重置。"""
        cls._tools.clear()
