"""结构化路由：让模型**填表**，而不是让它的自由文本来决定分支。

候选域由代码给出（`build_candidates` 从域包的 `name`/`description` 构造），模型只能在这份
清单里选；回答必须落进 `RouteDecision`（pydantic 模型：域名、置信度、理由都有类型与范围
约束），实现方负责把它绑到 `with_structured_output(RouteDecision)`。

出口**永远**是一个合法域：非法值、调用异常、没配模型，一律回退 `default`（confidence=0.0）
并把原因写进 `reason`。子串匹配式的自由文本路由会在措辞变化时失配、允许模型编出不存在的
域、事后无从审计；而路由选错一个域，后面的检索、工具、提示词、引用全都白做，
所以这里宁可带记录地降级。内核不硬编码域知识，加域不用改本模块。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

_REASON_MAX = 200
"""`reason` 的字符上限；兜底路径也必须守这个界，否则降级会因超长而抛校验错。"""


class RouteDecision(BaseModel):
    """一次路由决定。**只用它做分支**，绝不去解析模型的自由文本。"""

    domain: str = Field(description="选中的域名，必须是候选之一")
    confidence: float = Field(ge=0.0, le=1.0, description="对本次选择的自评置信度")
    reason: str = Field(default="", max_length=_REASON_MAX, description="一句话理由，用于审计与排查")


class Router(Protocol):
    """结构化路由的 LLM 接口。实现方负责把它绑到 with_structured_output。"""

    def decide(self, query: str, candidates: Sequence[dict[str, Any]]) -> RouteDecision: ...


def _text(value: object) -> str:
    """把属性安全地取成 str：`None` 变空串而不是 `"None"`。

    后者是个**真值**，会让"没有 name 的域包"混进候选清单，表现为一个永远选不中的候选。
    """
    return "" if value is None else str(value)


def build_candidates(domains: Sequence[object]) -> list[dict[str, Any]]:
    """从域包构造候选描述（模型能看到的唯一一份"可选域"清单）。

    只取 `name` 与 `description`，并同时接受 `DomainRegistry.describe()` 那种 `dict` 形状。
    缺少 `name` 的域包视为**编程错误**并抛 `ValueError`：静默跳过会让注册失败的域以
    "总是选不中"的形式失踪。加第三个域时这里一个字都不用改。
    """
    candidates: list[dict[str, Any]] = []
    for domain in domains:
        if isinstance(domain, Mapping):
            name = _text(domain.get("name"))
            description = _text(domain.get("description"))
        else:
            name = _text(getattr(domain, "name", ""))
            description = _text(getattr(domain, "description", ""))
        if not name:
            raise ValueError(
                f"域包 {domain!r} 没有可用的 name，无法作为路由候选：这是注册环节的编程错误，不该由路由兜底"
            )
        candidates.append({"name": name, "description": description})
    return candidates


def _make(domain: str, confidence: float, reason: str) -> RouteDecision:
    """统一出口：理由截断到 schema 上限。

    兜底路径存在的意义是"不崩"，若它自己因理由超长触发 pydantic 校验错，
    就把降级变成了崩溃——那是最不该出错的地方出错。
    """
    return RouteDecision(domain=domain, confidence=confidence, reason=reason[:_REASON_MAX])


def _degrade(default: str, names: Sequence[str], why: str) -> RouteDecision:
    """降级到 `default`，置信度固定 0.0。

    0.0 是给调用方的信号：这次选择**没有模型背书**，可据此决定是否追问用户或记指标。
    若 `default` 自己也不在候选里，额外点出来——那是配置错误，但比"整个请求失败"轻，
    所以照常返回并让人看得见。
    """
    if default not in names:
        why += f"；且默认域 {default!r} 不在候选 {list(names)} 里（配置错误，请修）"
    return _make(default, 0.0, why)


def route(
    query: str, *, llm: Router | None, candidates: Sequence[dict[str, Any]], default: str
) -> RouteDecision:
    """返回最终决定。规则按顺序生效：

    1. 没有任何候选 -> 抛 `ValueError`：这是"域包没注册"的编程错误，
       让模型或默认值把它糊过去只会把错误推到更远的地方；
    2. 只有一个候选 -> 直接返回它，confidence=1.0，**不走 LLM**：此时路由没有信息量，
       调用一次模型只是多花钱、多一次把唯一合法值答错的机会；
    3. `llm` 为 `None` 或抛异常 -> 返回 `default`，confidence=0.0，`reason` 写明降级原因；
    4. 模型给出的 domain 不在候选里 -> **拒绝并回退 default**。模型编域名是常见失败模式，
       静默接受它会让后面全员崩掉。
    """
    if not candidates:
        raise ValueError("路由候选为空：没有任何域包注册，这是编程错误，不该由模型或默认值兜底")

    names = [_text(c.get("name")) for c in candidates]

    if len(names) == 1:
        return _make(names[0], 1.0, "只有一个候选域，直接选中，未调用模型")

    if llm is None:
        return _degrade(default, names, "未配置路由模型，回退默认域")

    try:
        decision = llm.decide(query, candidates)
        # 这三行留在 try 里：替身对象字段缺失/类型不对同样属于"路由不可用"，
        # 应当降级而不是把异常抛给上层。
        chosen = _text(decision.domain)
        confidence = min(1.0, max(0.0, float(decision.confidence)))
        reason = _text(decision.reason)
    except Exception as e:  # noqa: BLE001 - 路由失败必须降级，不能让整轮对话崩在这里
        return _degrade(
            default, names, f"路由模型调用失败（{type(e).__name__}: {e}），回退默认域 {default!r}"
        )

    if chosen not in names:
        # 精确匹配，不做大小写折叠或模糊匹配：一个"差一点"的域名恰恰说明模型没按
        # schema 选，猜它想选哪个比直接回退更危险。
        return _degrade(
            default,
            names,
            f"模型给出了不在候选里的域 {chosen!r}，已拒绝并回退默认域 {default!r}（合法候选：{names}）",
        )

    return _make(chosen, confidence, reason or f"模型选择 {chosen}")
