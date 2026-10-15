"""工具装配：把域的工具注册进 `ToolRegistry`，供运行期的循环调用。

`register_domain_tools` 是唯一装配入口：`clinic` 域得到号源三件套 + 检索 + 引用核对，
`k8s_docs` 域得到检索 + 引用核对。工具是闭包捕获具体 service/retriever 实例的，
所以重复装配会先清掉该域的旧注册再重新注册（**幂等**），避免拿到已关闭的号源库。
本模块只做注册、不做 IO：号源库与检索器由调用方构造后传入，便于测试替身。

扩展方式：新增一个域 = 实现域包并在 `kubedocs.domains.load_all` 里注册，
再在 `register_domain_tools` 里加一条分支。
"""

from __future__ import annotations

import dataclasses
from typing import Any

from kubedocs.agent.tools import ToolRegistry, ToolSpec

# 记录每个域上一次实际注册了哪些工具名，供反注册使用。
# 不能按 `for_domain(domain)` 反查：像 `check_citations` 这类跨域通用工具的 `domain`
# 字段是 k8s_docs，按域过滤会漏掉它，重复装配时会撞"已注册"。
_REGISTERED: dict[str, list[str]] = {}


def unregister_domain(domain: str) -> int:
    """清掉某域上一次注册的工具，返回清掉的数量。"""
    names = _REGISTERED.pop(domain, [])
    for n in names:
        ToolRegistry._tools.pop(n, None)  # noqa: SLF001 - 注册表未提供 unregister，成对维护
    return len(names)


def register_domain_tools(domain: str, *, service: Any = None, retriever: Any = None) -> list[ToolSpec]:
    """装配并注册某个域的工具，返回注册结果。**幂等**。

    - `clinic`：号源三件套（列科室 / 查号源 / 取号）+ 检索与引用核对；`service` 必填
    - `k8s_docs`：检索与引用核对（`retriever` 为 `None` 时只有引用核对）

    只做注册、不做 IO：号源库与检索器由调用方构造后传入，便于测试替身，
    也避免"导入即打开数据库"这种隐蔽副作用。
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
    # 引用核对是跨域通用工具，但注册时按当前域归属，
    # 否则 `/tools?domain=clinic` 看不到它、按域清理也会漏掉它
    tools.append(dataclasses.replace(make_citation_check_tool(), domain=domain))
    for spec in tools:
        ToolRegistry.register(spec)
    _REGISTERED[domain] = [s.name for s in tools]
    return tools


def registered_for(domain: str) -> list[dict[str, Any]]:
    """自省：某域当前注册了哪些工具（供 `/tools` 端点与调试用）。"""
    return [s.describe() for s in ToolRegistry.for_domain(domain)]
