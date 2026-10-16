"""医疗辅助诊断产品层。

**平台是底座，本层是产品。**

| 模块 | 职责 |
|---|---|
| `contracts` | 模型输出 / 前端渲染 / 评测断言三方共用的唯一形状 |
| `rag` | 可替换的检索后端 |
| `triage` | 检索式分诊（目标 ≥80%） |
| `evidence` | 证据装配：红线（证据，非闸门）/ 分诊 / RAG |
| `intake` | 结构化采问槽位 + 下一问 |
| `prompts` | 系统提示词（版本化、被单测锁死） |
| `reasoning` | 大模型主判 → 契约化输出（解析 / 校验 / 降级） |
| `pipeline` | 编排：采问 → 证据 → 工具循环 → 契约化应答 |

一句话记住边界：**平台提供检索、图、工具、评测、可观测；判断与交付物由本层定义。**
"""

from __future__ import annotations

from kubedocs.clinical.contracts import (
    ClinicalAssessment,
    DepartmentAdvice,
    DifferentialItem,
    ExamAdvice,
    MedicationAdvice,
    validate_against,
)
from kubedocs.clinical.evidence import EvidenceBundle, EvidenceItem, build_evidence
from kubedocs.clinical.pipeline import ClinicalPipeline, ClinicalTurn
from kubedocs.clinical.prompts import prompt_version, system_prompt
from kubedocs.clinical.rag import (
    BackendHealth,
    LocalIndexBackend,
    RetrievalBackend,
    RetrievedChunk,
)
from kubedocs.clinical.reasoning import ClinicalReasoner, LiteLLMAdapter
from kubedocs.clinical.triage import RetrievalTriage

__all__ = [
    "BackendHealth",
    "ClinicalAssessment",
    "ClinicalPipeline",
    "ClinicalReasoner",
    "ClinicalTurn",
    "DepartmentAdvice",
    "DifferentialItem",
    "EvidenceBundle",
    "EvidenceItem",
    "ExamAdvice",
    "LiteLLMAdapter",
    "LocalIndexBackend",
    "MedicationAdvice",
    "RetrievalBackend",
    "RetrievalTriage",
    "RetrievedChunk",
    "build_evidence",
    "prompt_version",
    "system_prompt",
    "validate_against",
]
