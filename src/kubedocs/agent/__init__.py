"""Agent 内核：图、工具、会话状态与推理。

与 `kubedocs.domains` 的分工：域提供知识与业务（语料、词典、号源、红线），
内核提供控制流（LangGraph 图、工具协议、会话状态、持久化、中断、推理）。
内核里不出现 `if domain == ...`，加一个新域不必改内核代码。
"""

from __future__ import annotations

from kubedocs.agent.state import (
    CitedAnswer,
    ClinicState,
    EvidenceSource,
    RedLine,
    SymptomEvidence,
    empty_state,
)

__all__ = [
    "CitedAnswer",
    "ClinicState",
    "EvidenceSource",
    "RedLine",
    "SymptomEvidence",
    "empty_state",
]
