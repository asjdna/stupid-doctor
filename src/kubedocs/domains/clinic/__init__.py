"""clinic 域：医疗问诊 / 挂号 / 紧急（中文）。

## 这个域的三条流程

| 流程 | 判定方式 | 特点 |
|---|---|---|
| **紧急** | **图路径**：确定性红线 + 三维门控（`redline` + `polarity`），图上一条 LLM 不可绕过的强制边<br>**临床管线（`/consult`）**：模型按病症 + 严重程度判 `red_flags`，`redline` 已退出判断链 | 两条路径**取舍相反**，详见 `redline.py` 顶部那张表；不要混着看 |
| **问诊** | 规则字典（症状/科室/意图）+ 可选医疗 RAG 接地 | 双轨：规则轨必需，检索轨是增强 |
| **挂号** | `AppointmentProvider` 契约 + SQLite 号源库 | 并发不超卖由数据库约束保证 |

## 为什么规则轨是必需的、检索轨是可选的

医疗语料（OpenCMKG 派生的知识卡片）标注**「仅用于学术研究，不得用于商用」**。
**`data/clinic/` 的知识卡片与 `data/raw/` 的原始语料都不入库**
（`.gitignore` 排除；`git ls-files data/clinic data/raw` 输出为空，可自行核对）。
**但要说准确**：`data/eval/prototype/` 下**确实入库了几个原型派生的评测产物**，
其中 `disease_to_department.json` 是 OpenCMKG 关系的**直接导出**（8,819 条）。
**公开分发（例如推到 GitHub）等于分发这些派生数据，需自行确认上游许可** ——
见 `data/eval/prototype/README.md` 的"许可"一节，那里列清了本目录到底带了什么。

这带来一个设计约束：**没有语料时，三条流程必须仍能完整跑通**。所以：

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
