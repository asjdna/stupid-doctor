# Changelog

本文件记录 kubedocs 的重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

**关于版本与日期，先把话说清楚**：仓库**没有 tag、没有 remote、也没有发布过任何制品**，
所以下面的 `[0.1.0]` 标注为 **未发布（no tag yet）**——这个版本号来自 `pyproject.toml` 的
`version = "0.1.0"`，不是一次发布。**不写版本日期**，因为写任何一个日期都会是编的。
条目里的短哈希是真实提交，`git show <哈希>` 可以逐条核对。

---

## [0.1.0] - 未发布（no tag yet）

首个版本的全部内容，由 22 个提交归纳（提交日期区间 2026-09-24 ~ 2026-09-25，
取自 `git log` 的 committer date）。**没有删除项**——历史里没有任何功能或文件被移除。

### Added

- 项目骨架：`pyproject.toml`（ruff / mypy / pytest 配置 + CLI 入口）、语料获取脚本
  `scripts/fetch_corpus.py`、切块器与真实语料体检 `scripts/chunk_stats.py`（`dc5dc00`）
- 工程脚手架：`README.md`、`docs/adr/`、`docker-compose.yml`、`Dockerfile`、
  `.github/workflows/ci.yml`、`Makefile`（`62f4c18`）
- **本机零容器跑通「摄取 → 混合检索 → 评测闸门」**：本地索引（SQLite + FTS5 + numpy 向量）、
  增量/幂等摄取、混合检索 + RRF 融合 + 可选精排、评测 harness / 指标 / 基线闸门
  （`00da3aa`）
- 回答层：带引用回答、引用越界核对、检索为空即拒答、按日累计的 LLM 预算上限
  （`d50bc33`）
- 全量语料跑通（1720 篇 / 18,861 块）与**首次真实评测**：hybrid hit@10 = 0.84、闸门通过；
  新增逐条复查未命中的 `scripts/diagnose_misses.py`、冻结基线 `data/eval/retrieval_baseline.json`
  （`9493193`）
- **P0 双域平台地基**：域协议（`domains/base.py`）、多域索引（`<index_dir>/<domain>/`）、
  中文归一化（`normalize.py`）、索引布局迁移脚本 `scripts/migrate_index_layout.py`（`6232223`）
- **P1 医疗域资产迁移**：三维门控（polarity / antonym / noise 词表）、急诊红线
  （`redline.py`）、号源库（`booking.py` / `sqlite_provider.py`）、知识卡片构建脚本
  `scripts/clinic_build_cards.py`（`13356e3`）
- **P2 agent 内核**：图（`kernel.py`）、工具协议（`tools.py`）、状态与持久化
  （`state.py`）、中断/续跑、self-ask 推理（`selfask.py`）、假设-检验追问
  （`hypothesis.py`）、检索路由（`routing.py`）（`a3b77cf`）
- P3：clinic 域包接上检索与工具，并注册进双域平台（`4be16c2`）
- P4：clinic 域评测（`evaluation/clinic.py`）+ 红线硬闸门（召回容差为 0）；评测发现
  一个**数据级根因**（`e6a1bad`）
- P5：A/B 对照（self-ask vs 单轮 RAG）——**结论是"没有证据支持 self-ask"**，
  并暴露其闸门失效（`02b343c`）
- P7：带依赖的多跳（`agent/multihop.py`）+ 两轮筛选 + 逐子目标标注 —— **多跳反而更差**
  （诚实负面结论）（`547bbc4`）
- P8：用例工厂 `scripts/build_multihop_cases.py`，样本扩到 46 条 —— 多跳被
  **有统计意义地**证否，且依赖机制从未触发（`f3c34a6`）
- 工具循环接线 + 服务化 + 前端演示：FastAPI 服务（`api/app.py` / `api/schemas.py`）、
  LLM 工具调用实现 `agent/llm_tool_caller.py`、端到端冒烟脚本 `scripts/smoke_api.py`
  （端到端 15/15 通过）（`525aa6d`）
- LLM 驱动的 clinic 路径 `ClinicReactRuntime`：红线前置（确定性、LLM 不可绕过）+
  工具接线 + 挂号确认策略（`3185a19`）

### Changed

- 回答上下文去冗余：**同一文档最多收录 2 段**（`config.max_chunks_per_doc` 默认 2）（`76471b8`）
- `/chat` 从确定性图切到 **LLM 驱动的工具循环**（模型自己选工具、自己决定步数），
  并留下真实运行记录 `scripts/demo_clinic_agent.py`（`be399a4`）

### Fixed

- 健壮性与可用性收尾：LLM 调用失败改为**降级**（检索结果照常返回而不是整个请求崩）、
  嵌入批大小默认收紧、修正 CLI 全局参数位置（`f60a8f7`）
- **`.env` 里的密钥到不了模型调用**（pydantic-settings 不写 `os.environ`，而
  LiteLLM 与 `has_api_key()` 读的正是它）——README 描述的那套流程此前是**静默失效**的；
  同时修掉"测试可能真实花钱"的两个缺陷（`65537e0`）
- 修 A/B 脚本的 ruff 告警（`S110` / `C416` / `SIM105`）：成本统计失败改为显式保留原值，
  而不是空 `pass`（`b11f2c0`）
- 修正 P5 对 self-ask 的错误判断，并在复合问题上做了一次**真正隔离"分解"**的 A/B
  （结论仍是负面）（`3fcd4c3`）
- **工具协议接线**：clinic 号源工具 + `bootstrap.register_domain_tools` 幂等注册
  ——在此之前**全仓库从未调用过 `register`**，整条工具链路是死的；同一次提交补上
  `agent/tool_loop.py` 与对话前端 `data/web/index.html`（`1090c71`）

### Removed

- 无。22 个提交里没有任何对外行为或文件被删除（`git log --diff-filter=D` 为空）。

---

## 怎么读这个 changelog

- 每条后面的短哈希指向**一个真实提交**，不是一组合并叙述：`git show 1090c71`。
- 负面结论（self-ask 无用、多跳更差）**照实写进来**，不粉饰成"优化了检索"。
  它们是这个项目的资产：说明假设被证否过，而不是没试过。
- 想看每个数字是怎么来的，看 `README.md` 的「实测数字」「评测：怎么保证分数不是自欺」
  两节，以及 `docs/检索与评测口径.md`。
- 尚未完成的部分（LangGraph agent 落地、FastAPI `/metrics`、pgvector 生产实现、
  Docker 全栈实测）**不在本文件里**——还没做的事不是"变更"。它们记在 README 的 Roadmap。
