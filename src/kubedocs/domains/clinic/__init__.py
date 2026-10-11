"""clinic 域：医疗问诊 / 挂号 / 紧急（中文）。

## 这个域的三条流程

| 流程 | 判定方式 | 特点 |
|---|---|---|
| **紧急** | 确定性红线 + 三维门控（`redline` + `polarity`） | 零延迟、可审计；图上一条 LLM 不可绕过的强制边 |
| **问诊** | 规则字典（症状/科室/意图）+ 可选医疗 RAG 接地 | 双轨：规则轨必需，检索轨是增强 |
| **挂号** | `AppointmentProvider` 契约 + SQLite 号源库 | 并发不超卖由数据库约束保证 |

## 为什么规则轨是必需的、检索轨是可选的

医疗语料（OpenCMKG 派生的知识卡片）标注**「仅用于学术研究，不得用于商用」**，
因此本仓库**不携带**该语料（详见 docs/adr/0003）。这带来一个设计约束：
**没有语料时，三条流程必须仍能完整跑通**。所以：

- 症状识别、科室归一化、意图判定、急诊红线 —— 全部由 `data/` 下约 271 行数据字典驱动，**必需**；
- 知识卡片检索 —— 提供"为什么这样建议"的接地依据与引用，**可选增强**。

这样仓库在任何环境下都能演示核心流程，而受限语料只是加分项。

## 移植来源

`polarity.py` 与 `data/` 下的六个字典自 clinic 域原样移植（原 medical-agent 项目），
零外部导入、判定逻辑未改，仅调整了文档中的路径引用。`provider` / `sqlite_provider` /
`booking` 保留原有契约与数据库约束，按 kubedocs 的代码约定重写。
"""

from __future__ import annotations

__all__ = [
    "BookingService",
    "MockAppointmentProvider",
    "RiskGuard",
    "SqliteAppointmentProvider",
]
