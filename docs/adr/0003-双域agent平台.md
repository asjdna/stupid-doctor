# ADR-0003：双域 Agent 平台（含原生系统对接检测与资源策略）

- 状态：已接受（P0 已落地）
- 日期：2026-09-24
- 相关：ADR-0001（技术选型）、ADR-0002（先跑通业务逻辑）

## 一、背景与目标

kubedocs 原本是单一域（Kubernetes 英文文档检索问答）。目标是在**同一个仓库**里再加一个
医疗域（问诊 / 挂号 / 紧急），并且这次必须是**真 agent 架构**（图 + 工具 + 持久化 + 中断），
而不是像 medical-agent 那样把流程写死成一串 if-else。

同时要回答两个现实问题：

1. **能不能和原生系统（medical-agent）在线对接**，而不是把它的东西再抄一遍？
2. **怎么避免无意义的资源浪费**（尤其是那 74.9 MB 的知识语料）？

## 二、决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | **域无关的内核 + 域包（`DomainPack` 协议）** | 内核里出现任何一处 `if domain == "clinic"`，这套东西就退化成"两个项目塞进一个仓库"。内核只认协议。 |
| D2 | **每域一份独立索引**（`<index_dir>/<domain>/`） | 两域的语言、嵌入模型、维度、归一化方式都不同，共用一个库文件只会让"换域忘了换表"这类错误无从发现。 |
| D3 | **索引形态与展示形态分离**（`chunks.body` / `chunks.body_normalized`） | 中文必须先 jieba 预分词才可能被 FTS5 正确切词；但引用与展示必须用原文。 |
| D4 | **不移植 medical-agent 的 `orchestrator` 与 `llm_gateway`**，只借它的**能力注册模式** | 见第四节。 |
| D5 | **不做运行时在线对接**，但保留对接缝（`AppointmentProvider` 契约） | 见第三节。 |
| D6 | **医疗语料不入库**，只在本地按需重建 | 见第五节（许可 + 体积）。 |

## 三、原生系统对接检测（结论：技术上可行，但**不采用**运行时对接）

对 `medical-agent` 做了三项检测：

### 检测 1：它的 HTTP 接口完整可用 ✅

`app/main.py` 是唯一装配点，挂载 4 个路由组。**注意一个坑**：`app.routes` 里看到的是
`_IncludedRouter` 对象，它们不暴露 `.path`，直接枚举会误判为"只有 4 条默认路由"。
必须读 `app.openapi()["paths"]` 才看得到真实接口面：

```
/            /skills      /register        /login
/chat（SSE 真流式）        /kb/search（知识库检索）
/admin/{users,stats,audit,metrics,schedules,appointments,records,compress,llm/stats}
/admin/rag/{search,freshness,refresh,refresh/log}
```

⇒ 只要起一个进程，kubedocs 完全可以 `POST /chat` 或 `GET /kb/search` 复用它。

### 检测 2：语料全部可从版本库内的输入重建 ✅

重建链路（脚本齐全，输入均已入库）：

```
data/raw/OpenCMKG/（20 MB，已入库）
   └─ scripts/build_knowledge.py ─▶ data/knowledge/*.json（派生，未入库）
        └─ scripts/build_cards.py ─▶ cards.json（8.4 MB）
             └─ rag_ingest.py ─▶ dense_vectors.npy（39.4 MB）
```

`medical-agent` 自己的 `.gitignore` 写得很清楚：`data/knowledge/` **刻意不入库**，
因为它是由上面这些输入构建出来的派生语料。

### 检测 3：许可限制 ⚠️（这条决定了资源策略）

`data/raw/OpenCMKG/README.md` 原文：

> 仅用于学术研究，不得用于商用。如认为侵犯了您的版权，请联系删除

其上游是 QAKG（GitHub）、OwnThink、CHIP2021 三份数据。`medical-agent` 的 `.gitignore`
里也留了警示：「把 data/raw/OpenCMKG 纳入版本库前请确认许可」。

### 结论：不采用运行时对接

| 方案 | 判定 | 原因 |
|---|---|---|
| kubedocs 通过 HTTP 调 medical-agent 的 `/chat`、`/kb/search` | ❌ | 会让 kubedocs 的医疗域退化成"前端壳"：agent 架构、工具调用、状态机都在别人进程里，**无法演示本项目要展示的东西**；且评测要跨进程，破坏"同一套尺子"的前提 |
| 复制 `cards.json` + `dense_vectors.npy` 进 kubedocs | ❌ | 48 MB 派生数据，**几分钟就能重建**；且继承了上游的学术许可限制 |
| 复制 `OpenCMKG` 原始三元组进 kubedocs | ❌ | 20 MB，且**公开作品仓库不该携带「仅限学术研究」的数据** |
| **借鉴它的 Provider 契约与领域数据，在 kubedocs 内独立实现** | ✅ | 契约（`AppointmentProvider`）本就为"换真实系统"而设计；领域数据字典仅 14 KB |

## 四、为什么不移植它的"agent 架构"

核实结论：**medical-agent 没有 LLM agent，它有"编排架构"**。

| Agent 架构要素 | medical-agent |
|---|---|
| 能力组织 / 自省 | ✅ `SkillRegistry` 注册 9 个能力，`Orchestrator.__init__` 按名字解析，`/skills` 可自省 |
| 会话状态 / 长期记忆 | ✅ `Session` 标志位 + 近期消息与摘要注入 |
| 决策 / 路由 | ⚠️ **有，但是规则与统计**：`Agent._dispatch` 用关键词表；`intent/fusion` 三路加权置信度；`_should_use_cot` 是阈值判断 |
| 降级链 / 预算 / 追踪 | ✅ 三级降级闭环、`BudgetExceeded`、`llm_calls.jsonl` |
| 循环 / 自主迭代、工具协议、LLM 决定下一步 | ❌ 0 处 |
| 图 / 状态机 | ❌ 0 处（`langgraph` 在 `app/`、`tests/`、`scripts/` 中引用数为 0） |
| 可中断 / 可续跑 | ⚠️ `pending_emergency` 等是**状态标志位**（下一轮走不同分支），不是可挂起恢复的中断 |

`orchestrator._handle_core` 是写死的线性领域流程：
`pending_emergency 分流 → risk.check 红线 → transformer.transform → extractor.extract → 冲突处理 → …`

⇒ **把它搬进 kubedocs 等于把医疗分诊流水线搬进 K8s 文档问答**：既依赖 `domain/polarity`、
`risk_guard`、`symptom_dict` 等医疗语义跑不通，也让 kubedocs 多出 400 行读不懂的逻辑。

**借的是模式，不是代码**：`DomainRegistry` / `SkillRegistry` 的"自描述 + 按名解析 + 可列举"
思路照搬，用 kubedocs 自己的契约重写；`llm_gateway` 不借（与 LiteLLM 层重叠，会造成两套网关、
两个成本口径）。

## 五、资源策略（避免无意义浪费）

| 资源 | 处置 | 理由 |
|---|---|---|
| `dense_vectors.npy`（39.4 MB） | **不复制**，需要时重建（约 2 分钟） | 纯派生；且模型已在本地缓存，重建成本极低 |
| `cards.json`（8.4 MB） | **不复制**，由 kubedocs 侧脚本重建 | 同上；重建保证卡片格式与 kubedocs 的引用策略一致 |
| `OpenCMKG`（20 MB） | **不入库**，本地按需放置 | 学术许可限制（见检测 3） |
| 领域数据字典（14 KB） | 迁移并改写 | 体量可忽略，且是规则轨的必需底座 |
| 挂号契约 / 号源 schema / 红线逻辑 | 迁移并改写 | 契约本就为复用而设计；逻辑仅数百行 |
| medical-agent 的代码模块 | **不 import** | 跨仓库耦合会让两个交付物的依赖图纠缠 |

**降级设计（关键）**：clinic 域的**规则轨（14 KB 数据字典 + 红线）是必需，检索轨是可选增强**。
没有医疗语料时，问诊 / 挂号 / 紧急三条流程依然能完整跑通——所以仓库在任何环境下都能
演示核心流程，而受限的 RAG 语料只是一个加分项。这同时把许可问题的影响降到了零。

## 六、P0 的技术决策与实测

### 6.1 中文必须预分词（但理由要比"整句变成一个词元"更准确）

`unicode61` 按"非字母数字"切分，而汉字在它眼里全是字母数字——**所以中文的切分实际由标点决定**：

```
有标点："胸痛：可能与心绞痛相关，建议尽快就医"
        -> ["胸痛", "可能与心绞痛相关", "建议尽快就医"]    短语级，勉强可用
无标点："患者主诉胸痛伴呼吸困难需警惕急性冠脉综合征"
        -> 一个超长词元，任何词级查询都命中不了
```

即使有标点，切出的是**短语级**词元：「建议尽快就医」只有整串出现才命中，
用户查「就医」「心内科」都落空。实测（同一份文本、两种归一化器）：

| 查询 | `identity`（不预分词） | `chinese`（jieba） |
|---|---|---|
| 呼吸困难 | ❌ 命中不到 | ✅ 命中 |
| 急性冠脉综合征 | ❌ | ✅ |
| 就医 | ❌ | ✅ |

> 这条修正在写测试时才发现：我最初的说明写的是"整句会变成一个词元"，
> 而实际上**有标点的句子是能被部分切开的**。文档已按实测改正——
> 一个不准确的"为什么"比没有"为什么"更危险。

### 6.2 顺带修掉一个一直存在的性能缺陷（2.1× / 2.4×）

迁移后复查延迟，发现 k8s 域 hybrid p50 从 129 ms 涨到 163 ms。逐段测量后定位到
**不是** schema 迁移引入的，而是 `keyword_search` 一直在 FTS 上直接 JOIN `chunks`：

| 阶段 | 耗时 |
|---|---|
| 纯 MATCH（9062 行命中） | 0.04 ms |
| MATCH + `ORDER BY bm25` | 6.60 ms |
| MATCH + bm25 + **JOIN chunks** | **57.50 ms** |

JOIN 让执行计划变成"先把 9000 多行全连上再排序"。改为**两步取**（先取前 N 个 rowid，
再按 rowid 查 `chunk_id`）后：

| 模式 | 改前 p50 / p95 | 改后 p50 / p95 |
|---|---|---|
| hybrid | 162.7 / 244.9 ms | **60.7 / 75.2 ms** |
| bm25-only | 121.1 / 207.3 ms | **15.4 / 27.1 ms** |
| dense-only | 43.0 / 45.4 ms | 47.3 / 55.9 ms（未受影响） |

相对最初基线（128.8 / 179.9 ms）是 **2.1× / 2.4× 提速，且检索指标逐位不变**。

### 6.3 schema v1 → v2 迁移**不需要重新嵌入**

v2 的唯一结构变化是全文索引从 `body` 改为 `body_normalized`。英文域归一化是原样透传，
所以迁移只需 `ADD COLUMN` + `UPDATE = body` + 重建 FTS 倒排索引（FTS5 的 `'rebuild'` 指令），
**完全不碰 `chunk_vectors.npy`**——省掉了 34 分钟的嵌入。

实测：1720 篇 / 18861 块迁移后 FTS 一致性 `ok`，评测 hit@10 仍为 **0.8400**（逐位相同）。

## 七、代价与边界

| 边界 | 说明 |
|---|---|
| 医疗 RAG 需要本地语料 | 仓库不携带受限语料；无语料时退化为规则轨（核心流程仍可用） |
| 不做真实 HIS / 短信对接 | 只保留 `AppointmentProvider` 与通知接口，实现留给接入方 |
| 两个域各自持有索引 | 磁盘占用翻倍（可接受；换来实现简单、隔离彻底） |
| 中文检索质量尚未与 medical-agent 对照 | P1 起用它的 `rag_gold`（120 条）做对照；**不达标就如实记录并考虑移植 `rank_bm25`** |

## 八、明确不做

- ❌ 不 import medical-agent 的任何模块
- ❌ 不移植 `orchestrator` / `llm_gateway`
- ❌ 不复制受限语料与派生向量
- ❌ 不改动 medical-agent（毕设交付物须保持可独立运行）
- ❌ 不下诊断结论、不推荐具体用药剂量
