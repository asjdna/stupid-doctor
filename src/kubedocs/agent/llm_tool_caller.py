"""`ToolCaller` 的真实实现：把 LiteLLM 的 function calling 接到工具循环上。

`tool_loop.py` 只定义 `ToolCaller` 协议，本模块把注册好的工具以 `tools=` 交给真实模型，
并把回包解析成循环需要的形式。

调用方必须知道的契约：

1. `arguments` 是模型给出的 **JSON 字符串**，这里 `json.loads` 成 dict；解析失败不抛异常，
   而是以 `{"__parse_error__": ...}` 作为参数交给循环回灌，让模型自己改；
2. `tool_calls` 可能是 `None`（只是普通回答），此时返回空列表，调用方据此区分
   "要调工具"与"直接作答"；
3. 返回值里的 `cost_priced=False` 表示**这次算不出价**，不是 0 元：工具循环会把它交给预算
   按保守上限计入。模型真定价 0（如本地模型）仍然是 `cost_priced=True`。
"""

from __future__ import annotations

import json
from typing import Any

from kubedocs.config import Settings
from kubedocs.observability import span, span_attrs


class LiteLLMToolCaller:
    """用 LiteLLM 调模型，返回 `{"content", "tool_calls"}`。

    `tools` 为空时**不带 tools 参数**：有些供应商在 `tools=[]` 时会报错，
    而这个实现同时要服务"纯对话"（无工具）场景。

    返回值额外多了两个键：`cost_usd` 与 `cost_priced`（含义见模块文档第 3 条）。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self.calls = 0
        self.cost = 0.0
        self.unpriced_calls = 0
        """其中**算不出价**的次数（金额已由预算按保守上限计入，这里只做可见性）。"""
        self._litellm: Any = None

    @property
    def litellm(self) -> Any:
        if self._litellm is None:
            import litellm

            litellm.drop_params = True
            self._litellm = litellm
        return self._litellm

    def choose(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "timeout": self.settings.llm_timeout,
            "num_retries": self.settings.llm_max_retries,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        with span("llm.call", model=self.model, purpose="choose") as current:
            resp = self.litellm.completion(**kwargs)
            usage = getattr(resp, "usage", None)
            # 属性在 span 结束前写；结束后写只会得到 warning，值进不去
            span_attrs(
                current,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )
        self.calls += 1
        cost, priced = 0.0, True
        try:
            raw_cost = self.litellm.completion_cost(completion_response=resp)
            # `None` = litellm 价格表里查不到这个模型，**不是"免费"**
            priced = raw_cost is not None
            cost = float(raw_cost) if priced else 0.0
        except Exception:  # noqa: BLE001 - 未知模型算不出价：不阻断主流程，但**不报成 0**
            # 这里只负责把"没算出来"如实标出，金额由 tool_loop -> DailyBudget 按保守上限计入
            # （不能当成免费）。
            cost, priced = 0.0, False
            self.unpriced_calls += 1
        self.cost += cost

        msg = resp.choices[0].message
        out: dict[str, Any] = {
            "content": str(getattr(msg, "content", "") or ""),
            "tool_calls": [],
            "cost_usd": cost,
            "cost_priced": priced,
        }
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw = getattr(fn, "arguments", "") or ""
            args: dict[str, Any] = {}
            parse_error = ""
            try:
                parsed = json.loads(raw) if raw else {}
                args = parsed if isinstance(parsed, dict) else {}
                if not isinstance(parsed, dict):
                    parse_error = f"参数不是 JSON 对象：{type(parsed).__name__}"
            except json.JSONDecodeError as e:
                parse_error = f"参数不是合法 JSON：{e}"
            if parse_error:
                # 不抛异常：让工具循环把它当"这一步失败"回灌给模型，模型能自己改
                args = {"__parse_error__": parse_error}
            out["tool_calls"].append({"name": str(getattr(fn, "name", "")), "arguments": args})
        return out
