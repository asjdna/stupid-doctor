"""HTTP 服务层（FastAPI）：把领域能力暴露成一组端点。

这里刻意不在模块级 import `kubedocs.api.app`：该模块在导入时就构造应用实例
（`app = create_app()`，供 `uvicorn kubedocs.api.app:app` 使用），而 `create_app()`
会**打开 SQLite 索引**。若在此 import，任何 `import kubedocs.api`（包括只想拿类型
标注的调用方与测试收集阶段）都会顺带打开一份索引库。

所以用 PEP 562 的模块级 `__getattr__` 做**惰性再导出**：
`from kubedocs.api import create_app` 照常可用，代价只在真正用到时才付。
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """惰性暴露 `create_app`。"""
    if name == "create_app":
        from kubedocs.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
