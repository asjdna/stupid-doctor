"""`ToolCaller` 的真实实现：把 LiteLLM 的 function calling 接到工具循环上。

## 为什么必须有这个文件

`tool_loop.py` 只定义了 `ToolCaller` **协议**，没有任何实现——
也就是说工具循环写好了、工具也注册好了，但**没有东西把 `tools=` 交给真实模型**。
整条链路仍然差最后一段。这个文件就是那一段。

## 三个容易出错的地方

1. **`arguments` 是 JSON 字符串**，不是 dict。模型返回的是字符串，
   必须 `json.loads`；解析失败不能让循环崩，要把它当成"参数非法"交给循环处理
   （`arguments` 留空 dict，错误写进内容里由循环回灌）。
2. **`tool_calls` 可能是 `None`**（只是普通回答）。必须区分"要调工具"与"直接作答"。
3. **成本拿不到不该阻断，但也绝不能报成 0 就完事**：未知模型算不出价时，
   这里返回 `cost_priced=False`，由工具循环交给预算按**保守上限**计入。
   曾经的写法是 `except: self.cost = self.cost`（一个**字面意义上的空操作**），
   等于把这次调用悄悄记成免费——预算闸门因此永远拦不住（见 SECURITY.md 的 S-2）。
   注意"真免费"（本地模型的定价就是 0）与"算不出价"是两件事，前者仍然是
   `cost_priced=True`：实测 litellm 1.102.1 对 `ollama/llama3` 正常返回 `0.0`。
"""

from __future__ import annotations

import json
from typing import Any

from kubedocs.config import Settings


class LiteLLMToolCaller:
    """用 LiteLLM 调模型，返回 `{"content", "tool_calls"}`。

    `tools` 为空时**不带 tools 参数**：有些供应商在 `tools=[]` 时会报错，
    而这个实现同时要服务"纯对话"（无工具）场景。

    返回值额外多了两个键：`cost_usd` 与 `cost_priced`（见模块文档第 3 条）。
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
        resp = self.litellm.completion(**kwargs)
        self.calls += 1
        cost, priced = 0.0, True
        try:
            raw_cost = self.litellm.completion_cost(completion_response=resp)
            # `None` = litellm 价格表里查不到这个模型，**不是"免费"**
            priced = raw_cost is not None
            cost = float(raw_cost) if priced else 0.0
        except Exception:  # noqa: BLE001 - 未知模型算不出价：不阻断主流程，但**不报成 0**
            # 实测 litellm 1.102.1 对未登记模型抛
            # `Exception: This model isn't mapped yet`。这里只负责把"没算出来"如实标出来，
            # 金额由 tool_loop -> DailyBudget 按保守上限计入（不能当成免费）。
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
