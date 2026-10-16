"""**工具循环**：由大模型自己决定调用哪些模块，而不是代码写死调用顺序。

底层约束是**大模型**：它负责判断病情、决定是否转急诊，也决定**用哪些工具取证**。
所以代码只提供**工具清单 + 执行器**（`agent.tools.ToolRegistry` 是唯一真相源），
模型自己决定调哪个、调几次。代码保留的只有三件**非医学**的事：
轮次上限、工具白名单、以及把工具返回的证据编号累积起来以便事后校验引用。

## 边界

- **证据编号由本模块分配**（`K1`、`K2`…），模型引用它们。所以"模型可以不检索"
  **不会**让"建议必须有依据"失效：不检索就没有编号可用，也就给不出检查/用药建议。
- **有副作用的工具默认不给模型**（挂号会占号源）。需要时由调用方显式
  `allow_side_effects=True`，并在执行前拿到用户确认。
- **轮次上限是硬的**：模型不能靠"再查一次"把预算烧穿，到顶就要求它用现有证据作答。
- **失败不吞**：工具抛错会作为 `tool` 消息回给模型（它能看到"这次检索失败了"并换策略），
  同时记进 `ToolLoopResult.calls` 供排查。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem
from kubedocs.clinical.intake import IntakeState

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
"""工具循环的最大轮次（一轮 = 一次模型回复 + 执行它要求的全部工具调用）。

一次问诊典型只需要 1~3 次检索（症状、可能的检查、可能的用药），4 轮给了"再查一次"
的余地，又不会让一次对话的成本无上界。**这是硬上限**：到顶时会在最后一条消息里
明确要求模型用现有证据作答。
"""


class ToolTurnClient(Protocol):
    """带工具的一次调用。实现见 `generate.LiteLLMClient.complete_with_tools`。"""

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...


@dataclass
class ToolCallRecord:
    """一次工具调用的留档（参数、结果、成败）。"""

    name: str
    arguments: str
    ok: bool
    summary: str
    """给排查看的简短结果，不是给模型的原文。"""


@dataclass
class ToolLoopResult:
    final_text: str
    calls: list[ToolCallRecord] = field(default_factory=list)
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    degraded: str = ""
    rounds: int = 0


def tool_schemas(*, domain: str = "clinic", allow_side_effects: bool = False) -> list[dict[str, Any]]:
    """取该域可给模型用的工具 schema。

    **有副作用的工具默认不暴露**：模型一旦能直接取号，"确认"那套语义就得同时交给它，
    而确认是流程责任、不该由模型自觉。与 `agent/tools.py` 的 `ToolSpec.side_effect` 同一套约定。
    """
    from kubedocs.agent.tools import ToolRegistry

    out: list[dict[str, Any]] = []
    for spec in ToolRegistry.for_domain(domain):
        if spec.side_effect and not allow_side_effects:
            continue
        out.append(spec.schema())
    return out


def describe_tools(*, domain: str = "clinic", allow_side_effects: bool = False) -> str:
    """把工具清单渲染成提示词里的一段**文字**（模型的第二次机会）。

    为什么既有 schema 又要文字：function calling 的 `description` 常常被供应商截断或忽略，
    而"什么时候**不该**用"这类约束（`when_not`）恰恰最需要被读到。
    这段文字写进 system 消息，schema 走 `tools=`，两条路都通。
    """
    from kubedocs.agent.tools import ToolRegistry

    lines: list[str] = []
    for spec in ToolRegistry.for_domain(domain):
        if spec.side_effect and not allow_side_effects:
            continue
        lines.append(f"- `{spec.name}`：{spec.description}")
        if spec.when_not:
            lines.append(f"  不适用：{spec.when_not}")
    if not lines:
        return ""
    return (
        "## 你可以调用的工具（**由你自己决定调哪个、调几次**）\n\n"
        "系统不会替你检索。需要依据时**你要自己调工具**——不调就没有依据，"
        "也就不能给检查或用药建议（见 R3）。\n\n" + "\n".join(lines)
    )


class ClinicalToolLoop:
    """跑工具循环，并把工具返回的证据累积成 `EvidenceBundle`。"""

    def __init__(
        self,
        client: ToolTurnClient,
        *,
        domain: str = "clinic",
        max_rounds: int = MAX_TOOL_ROUNDS,
        allow_side_effects: bool = False,
        max_evidence: int = 20,
    ) -> None:
        self._client = client
        self._domain = domain
        self._max_rounds = max(1, max_rounds)
        self._allow_side_effects = allow_side_effects
        self._max_evidence = max_evidence

    # ------------------------------------------------------------------ 执行
    def _execute(self, name: str, arguments: str) -> tuple[dict[str, Any], ToolCallRecord]:
        """执行一次工具调用。**把失败转成模型看得懂的一句话**，不让它打断对话。"""
        from kubedocs.agent.tools import ToolError, ToolRegistry

        try:
            spec = ToolRegistry.get(name)
        except (KeyError, ToolError):
            # `ToolRegistry.get()` 抛的是 `ToolError`（不是 `KeyError`）。
            # 只捕获 KeyError 时，"模型点了一个不存在的工具"会把整轮问诊炸掉——
            # 而模型偶尔写错工具名是常态，不是异常。
            msg = f"没有名为 {name!r} 的工具。可用工具见系统提示里的清单。"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)
        if spec.side_effect and not self._allow_side_effects:
            msg = f"工具 {name!r} 有副作用，当前会话未授权调用。"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)
        if spec.handler is None:
            msg = f"工具 {name!r} 只声明了能力、没有执行体（用于评测）。"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)

        try:
            payload = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(payload, dict):
                raise ValueError("参数必须是一个 JSON 对象")
        except (ValueError, json.JSONDecodeError) as e:
            msg = f"参数不是合法 JSON 对象（{e}）：{arguments[:200]!r}"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)

        try:
            params = spec.params.model_validate(payload)
        except Exception as e:  # noqa: BLE001 - 参数不合 schema 是模型常犯的错，回给它让它改
            msg = f"参数不符合 {name} 的 schema：{e}"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)

        try:
            result = spec.handler(params)
        except ToolError as e:
            msg = f"工具执行失败：{e}"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)
        except Exception as e:  # noqa: BLE001 - 工具炸掉不该让整轮问诊挂掉，但必须留痕
            logger.warning("工具 %s 抛出未预期异常：%s: %s", name, type(e).__name__, e)
            msg = f"工具内部错误（{type(e).__name__}）：{e}"
            return {"error": msg}, ToolCallRecord(name, arguments, False, msg)

        if not isinstance(result, dict):
            result = {"result": result}
        return result, ToolCallRecord(name, arguments, True, _summarize(result))

    def _collect_evidence(self, bundle_items: list[EvidenceItem], result: dict[str, Any]) -> None:
        """把 `search_docs` 返回的命中累积成证据（编号 `K1`、`K2`…）。

        **编号由本模块分配**，不信任工具返回的 chunk_id 形状：契约校验要的是
        "模型引用的编号真的存在"，编号必须是单一来源。
        """
        hits = result.get("hits")
        if not isinstance(hits, list):
            return
        for hit in hits:
            if len(bundle_items) >= self._max_evidence:
                return
            if not isinstance(hit, dict):
                continue
            n = len(bundle_items) + 1
            bundle_items.append(
                EvidenceItem(
                    id=f"K{n}",
                    kind="knowledge",
                    text=str(hit.get("body") or hit.get("title") or "").strip(),
                    score=float(hit.get("score") or 0.0),
                    source_url=str(hit.get("source_url") or ""),
                )
            )

    # ------------------------------------------------------------------ 主流程
    def run(
        self,
        *,
        system: str,
        question: str,
        intake: IntakeState,
        schemas: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ToolLoopResult:
        """跑工具循环。

        `history` 是**本会话之前的对话**（`[{"role","content"}, ...]`，由调用方持有）。
        必须有它：产品要求"科室/用药作为回答之后的可选扩展项，等用户来问"——
        患者追问"那该挂哪个科"时，模型若看不到自己上一轮说过什么，
        就会把整个判断重跑一遍，**同一个病例可能问出另一个科室**。
        """
        tools = (
            schemas
            if schemas is not None
            else tool_schemas(domain=self._domain, allow_side_effects=self._allow_side_effects)
        )
        catalogue = describe_tools(domain=self._domain, allow_side_effects=self._allow_side_effects)
        system_full = f"{system}\n\n---\n\n{catalogue}" if catalogue else system
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_full}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": f"{intake.render()}\n\n【患者这一轮说的话】\n{question}"})
        items: list[EvidenceItem] = []
        calls: list[ToolCallRecord] = []

        for round_no in range(1, self._max_rounds + 1):
            turn = self._client.complete_with_tools(messages, tools)
            pending = list(getattr(turn, "tool_calls", []) or [])
            content = str(getattr(turn, "content", "") or "")
            if not pending:
                # 没有再要工具 ⇒ 这就是最终答复
                return ToolLoopResult(
                    final_text=content,
                    calls=calls,
                    evidence=EvidenceBundle(items=items),
                    rounds=round_no,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": c.id or f"call_{round_no}_{i}",
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for i, c in enumerate(pending)
                    ],
                }
            )
            for i, call in enumerate(pending):
                result, record = self._execute(call.name, call.arguments)
                calls.append(record)
                self._collect_evidence(items, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{round_no}_{i}",
                        "content": json.dumps(result, ensure_ascii=False)[:6000],
                    }
                )

        # 轮次用尽：明确要求它用现有证据作答，**不静默截断**
        messages.append(
            {
                "role": "user",
                "content": (
                    f"已达到工具调用上限（{self._max_rounds} 轮）。"
                    "请立刻基于**已经拿到的依据**输出最终 JSON 结论；"
                    "没有依据支撑的检查或用药建议就不要给。"
                ),
            }
        )
        turn = self._client.complete_with_tools(messages, [])
        return ToolLoopResult(
            final_text=str(getattr(turn, "content", "") or ""),
            calls=calls,
            evidence=EvidenceBundle(items=items),
            rounds=self._max_rounds + 1,
        )


def _summarize(result: dict[str, Any]) -> str:
    """给排查看的一句话结果摘要（不是喂给模型的原文）。"""
    if "error" in result:
        return f"error: {result['error']}"
    hits = result.get("hits")
    if isinstance(hits, list):
        return f"hits={len(hits)}"
    return ",".join(f"{k}={v}" for k, v in list(result.items())[:4] if not isinstance(v, (list, dict)))
