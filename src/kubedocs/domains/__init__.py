"""域包层：agent 内核与具体知识/业务之间的边界。

内核只认 `DomainPack` 协议；域特有的语料、归一化器、嵌入模型、工具与评测集
都由各域包自报。加第三个域 = 新增一个模块 + 在 `load_all()` 里注册，
不需要改内核的一行代码。
"""
from __future__ import annotations

from kubedocs.domains.base import DomainPack, DomainRegistry

__all__ = ["DomainPack", "DomainRegistry", "load_all"]


def load_all() -> list[str]:
    """注册所有内置域包，返回已注册的域名。**幂等**（重复调用不会报错）。"""
    from kubedocs.domains.k8s_docs import pack as k8s_pack

    known = DomainRegistry.names()
    if k8s_pack.name not in known:
        DomainRegistry.register(k8s_pack)
    return DomainRegistry.names()
