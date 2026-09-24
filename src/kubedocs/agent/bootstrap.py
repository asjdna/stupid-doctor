"""工具装配：把域的工具**注册进 ToolRegistry**，让运行期真的用得上。

## 为什么需要这个文件

`ToolRegistry` 写好了、`ToolSpec` 写好了、`rag_tools`/`clinic/tools` 也写好了，
但**从来没有地方调用过 `register`** —— 结果是整条工具链路是死的（全仓库没有任何
地方执行过一次工具）。这个文件就是那根缺失的接线。

## 幂等

`register_domain_tools` 可以被反复调用（API 每次启动、CLI 每次执行都会走到），
所以先清掉该域的旧注册再重新注册——因为工具是**闭包捕获了具体 service/retriever 实例**
的，重新装配必须替换掉旧实例，否则会拿到以前那个已经关闭的号源库。
"""
from __future__ import annotations

import dataclasses
from typing import Any

from kubedocs.agent.tools import ToolRegistry, ToolSpec

# 记录"每个域上一次实际注册了哪些工具名"。
# 不能靠 `for_domain(domain)` 来清理：像 `check_citations` 这种**跨域通用**工具的
# `domain` 字段是 k8s_docs，按域过滤会漏掉它，于是第二次装配直接撞"已注册"报错。
_REGISTERED: dict[str, list[str]] = {}


def unregister_domain(domain: str) -> int:
    """清掉某域上一次注册的工具，返回清掉的数量。"""
    names = _REGISTERED.pop(domain, [])
    for n in names:
        ToolRegistry._tools.pop(n, None)  # noqa: SLF001 - 注册表未提供 unregister，成对维护
    return len(names)


def register_domain_tools(domain: str, *, service: Any = None,
                          retriever: Any = None) -> list[ToolSpec]:
    """装配并注册某个域的工具，返回注册结果。**幂等**。

    - `clinic`：号源三件套（列科室 / 查号源 / 取号）+ 检索与引用核对
    - `k8s_docs`：检索与引用核对

    刻意**不在这里做 IO**：号源库与检索器由调用方构造并传入，本函数只做注册。
    这样测试可以传入替身，也避免"导入即打开数据库"这种隐蔽副作用。
    """
    from kubedocs.agent.rag_tools import (
        make_citation_check_tool,
        make_search_tool,
    )

    unregister_domain(domain)
    tools: list[ToolSpec] = []
    if domain == "clinic":
        if service is None:
            raise ValueError("装配 clinic 域工具需要传入 BookingService")
        from kubedocs.domains.clinic.tools import clinic_tools

        tools.extend(clinic_tools(service))
    if retriever is not None:
        tools.append(make_search_tool(retriever, domain=domain))
    # 引用核对是跨域通用工具，但注册时要按当前域归属，
    # 否则 `/tools?domain=clinic` 看不到它、按域清理也会漏掉它
    tools.append(dataclasses.replace(make_citation_check_tool(), domain=domain))
    for spec in tools:
        ToolRegistry.register(spec)
    _REGISTERED[domain] = [s.name for s in tools]
    return tools


def registered_for(domain: str) -> list[dict[str, Any]]:
    """自省：某域当前注册了哪些工具（供 `/tools` 端点与调试用）。"""
    return [s.describe() for s in ToolRegistry.for_domain(domain)]
