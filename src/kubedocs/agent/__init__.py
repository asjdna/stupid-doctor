"""Agent 内核：图、工具、状态、推理。

与域（`kubedocs.domains`）的分工：域提供**知识与业务**（语料、词典、号源、红线），
内核提供**控制流**（图、工具协议、状态、持久化、中断、推理）。内核里不出现
任何 `if domain == ...`。
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
