# kubedocs

> 面向 **Kubernetes 官方文档**的检索问答服务：混合检索（BM25 + 向量）+ RRF 融合 + 可选重排，
> 回答**必须带可点击引用**，查不到就明说。**一条 `docker compose up` 起全栈**。

这是一个**工业级工程作品**（非研究原型）：重点不在"我实现了一个算法"，而在
**选型判断、可观测、评测驱动、成本与延迟约束、失败模式设计**。

---

## 状态（诚实标注）

| 部分 | 状态 |
|---|---|
| 语料获取（1720 篇官方 markdown / 16.2 MB） | ✅ 已验证 |
| 切块器 + 真实语料体检 | ✅ 已验证（35 单测 + 语料体检数字见下） |
| RRF 多路融合 | ✅ 已验证（单测） |
| Postgres/pgvector 灌库、混合检索、API、Agent、评测闸门 | 🚧 进行中 |
| `docker compose up` 全栈 | ⚠️ **文件已写，但本机尚未安装 Docker，未实测** |

> 本项目坚持一件事：**没跑过的就不写"已完成"**。上面标 🚧/⚠️ 的部分在验证后会更新。

---

## 为什么做这个（以及它和"从零手写"路线的区别）

我此前的项目（医疗预问诊 Agent）把检索、网关、认证、评测都自己写了一遍。那个项目适合讲
**领域深耕**（中文临床语境判定没有现成实现，只能自研）。但工业场景里更重要的判断是
**"什么该用成熟件、什么必须自研"**：

| 部件 | 本项目选择 | 理由 |
|---|---|---|
| 检索存储 | **Postgres + pgvector** | 一库同时承担向量、全文检索、业务数据；省一套运维 |
| 多路融合 | **RRF**（自实现，20 行） | 免标定、跨语料稳定，与 OpenSearch 内置策略同源 |
| LLM 网关 | **LiteLLM** | 多供应商路由/重试/回退/成本统计，自研是重复劳动 |
| Agent 编排 | **LangGraph** | 耐久状态 + 中断恢复，不必自己写状态机与持久化 |
| 嵌入 | 本地 BGE（可切 API） | 医疗/企业语料常不允许出境；本地也可离线复现 |
| 可观测 | **OpenTelemetry + Prometheus**（+ Langfuse 可选） | 标准协议，不锁供应商 |
| 评测 | **自建金标闸门 + Ragas 行业口径** | 闸门纪律（可失败、拒绝跨集比较）是框架给不了的 |

设计决策的完整记录见 [`docs/adr/`](docs/adr/)。

---

## 架构

```
                     ┌──────────────────────── 一条命令起全栈 ────────────────────────┐
   客户端 ──HTTP/SSE──▶  api (FastAPI)  ──▶  agent (LangGraph)  ──▶  LiteLLM ──▶ LLM 供应商
                          │                      │
                          │                      ├─▶ retrieve:  BM25(Postgres FTS) ┐
                          │                      │              向量(pgvector)     ├─ RRF 融合 ─ 重排
                          │                      │                                 ┘
                          ▼                      ▼
                     Redis (缓存/限流)      Postgres 16 + pgvector (chunks / 向量 / 业务)
                          │
                          └─▶ OpenTelemetry ─▶ Prometheus /(可选)Langfuse
```

**检索链路**：`query → 查询改写(可选) → 两路并行召回 → RRF 融合 → 重排(可选) → 组装上下文 → 带引用回答`
每一段都带 span，可在 trace 里看到"这句话是从哪个文档哪一节来的"。

---

## 快速开始

```bash
git clone <repo> && cd kubedocs
cp .env.example .env          # 只需填 LLM 供应商的 key；不填也能跑检索与评测

docker compose up -d --build  # db(pgvector) + redis + api
docker compose exec api python scripts/fetch_corpus.py   # 拉官方文档（1720 篇）
docker compose exec api kubedocs ingest                  # 切块 + 嵌入 + 入库
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/ask -H 'content-type: application/json' \
     -d '{"q":"How do I set a Pod CPU limit?"}' | jq
```

本地开发（不用容器）：

```bash
python -m venv --system-site-packages .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                    # 单元测试（不需要 DB）
pytest -m integration        # 需要 Postgres/Redis（无 Docker 时自动跳过）
ruff check . && mypy src scripts
```

---

## 实测数字

### 语料与切块（`python scripts/chunk_stats.py`，1720 篇官方文档）

| 项 | 值 |
|---|---|
| 文件 / 体积 | 1720 篇 markdown / 16.2 MB（与 GitHub 树接口预测**完全一致**） |
| 块数 | 18861 |
| 块长 中位 / p95 | 764 / 1621 字符 |
| 含代码块的块 | 16.3% |
| 非法引用 URL | **0** |
| 内容逐字重复块 | **1.12%**（见下） |
| 切块吞吐 | ~1085 文件/s |

**体检驱动的三次修正**（都在真实语料上量化过）：

| 问题 | 修法 | 效果 |
|---|---|---|
| API reference 的 HTML 表格原样入库（占语料 59%） | `strip_html`：剥标记留文字、表格转可读行、**不碰代码围栏** | p95 块长 2125 → 1540 |
| 重叠按字符硬切，产生 `...rip between` 这类半截块 | 重叠对齐到词/行边界，对不齐就放弃重叠 | 消除词中断块 |
| 模板化小节跨文档同形 → **30.7% 逐字重复块** | 块首加「文档标题 › 面包屑」上下文头 | 重复 **30.7% → 1.12%** |

### 环境约束（实测，决定了取数方式）

| 通道 | 结果 |
|---|---|
| `git clone github.com` | ❌ 连不上（21s 超时） |
| raw.githubusercontent / jsDelivr / statically / githack | ❌ 全部连接被拒 |
| GitHub API | ✅ 可用但 **限流 60/h**（取 1720 个文件必然撞墙） |
| codeload tarball | ✅ 335 MB / 2.1 分钟 |
| huggingface.co | ❌ 超时 → 走 **hf-mirror.com** ✅ |

⇒ 唯一可行路径是 tarball 流式解包，见 [`scripts/fetch_corpus.py`](scripts/fetch_corpus.py)。

---

## 工程质量

- **闸门全部阻塞**：`ruff`（含 S/B/SIM 规则）、`mypy`（`disallow_untyped_defs`）、`pytest`
- **分层的测试**：`tests/unit`（零依赖、秒级）／`pytest -m integration`（需 DB，无 Docker 自动跳过）
- **确定性**：融合结果同分按 key 排序——否则评测与闸门会随进程抖动
- **可观测**：每个阶段一个 span；`/metrics` 暴露 Prometheus 指标
- **失败模式**：查不到就拒答并说明；供应商不可用走 fallback；预算超限**拒绝**而不是悄悄烧钱

---

## Roadmap

- [x] 骨架 + 语料 + 切块器 + RRF（里程碑 1）
- [ ] Postgres/pgvector schema + 幂等/增量摄取（里程碑 2）
- [ ] 混合检索 + 重排 + 带引用回答 + `/ask` SSE（里程碑 3）
- [ ] 金标集 + 检索/引用指标闸门 + Ragas 对照（里程碑 4）
- [ ] LangGraph Agent（检索不足时重试/改查）+ OTel/Langfuse 全链路（里程碑 5）
- [ ] k6/Locust 压测 + P95/成本 SLO 写入 CI（里程碑 6）
- [ ] Docker 全栈实测 + 截图/演示（待本机装 Docker）
