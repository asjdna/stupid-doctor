# kubedocs

> **A Rough Architecture for a Medical Agent** —— 一个双域检索问答与 Agent 平台：
> `clinic`（医疗问诊 / 挂号 / 急诊识别，中文）与 `k8s_docs`（Kubernetes 官方文档，英文）。
> 检索走混合召回（BM25 + 向量 + RRF 融合）+ 可选重排；回答必须带可点击引用，查不到就明说。
> **本机零容器即可跑通全链路。**

## 这个项目解决什么

它面对的是"**用户说不清自己的问题**"这一类场景。以医疗为例，患者会说"这两天老咳嗽，还有点低烧，痰是黄的"——
零散、口语、可能用错词。系统要做的是：

1. 判断**紧急程度**，识别需要立刻就医的情况；
2. 给出**可能的方向排序**，并说清每个方向为什么值得考虑；
3. 给出**建议做哪些检查**、**可以考虑哪类药**（只讲类别，不给剂量）；
4. 建议**该挂哪个科**，必要时走挂号流程；
5. 全程用**患者听得懂的一段话**讲出来，而不是丢一张表格。

`k8s_docs` 域是同一套底座的第二个应用：面向技术文档的检索问答，回答带引用、可点击核对。

## 核心思路

**检索必须有据，判断交给模型，安全由契约兜底。**

- **检索式证据**：所有医学断言都要能追到本地知识卡片。检索不到就不给建议，不用模型的通用知识补足。
- **模型主导判断**：模型自己决定调哪些工具取证（检索、引用核对……），自己按"病症 + 严重程度"
  判断是否属于急重症，并输出结构化的判断依据。
- **契约兜住矛盾**：输出由 pydantic 契约约束。例如"判出了急重症却给常规建议"会被判定为违规；
  没有依据的检查/用药建议会被直接丢弃，而不是照发给用户。
- **对话是最终产物**：结构化结论用于后端留档与校验，用户看到的是模型组织好的那段话。

## 快速开始

### 本机（无需 Docker）

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 1) 拉语料（一次就够，1720 篇 / 16.2 MB）
python scripts/fetch_corpus.py

# 2) 建索引（首次全量约 35 分钟，嵌入占绝大部分）
python -m kubedocs.cli --hf-endpoint https://hf-mirror.com ingest

# 3) 检索
python -m kubedocs.cli search "how do I stop a container using too much memory" --explain

# 4) 带引用回答（需要 LLM 密钥；没有密钥时返回检索结果并说明原因，不会崩）
python -m kubedocs.cli ask "怎么限制一个 Pod 的 CPU？" --show-context

# 5) 索引体检（含 FTS 一致性自检）
python -m kubedocs.cli stats --verify
```

### 医疗问诊（`clinic` 域）

```bash
# 卡片语料与科室补全表（见下方「知识依据」）
python scripts/clinic_build_cards.py
python scripts/clinic_derive_departments.py --triples <OpenCMKG/triples.txt>

# 建索引（首次全量约 1 小时，CPU 编码）
python -m kubedocs.cli --domain clinic ingest

# 起服务，打开交互式问诊页
python -m kubedocs.cli --domain clinic serve --port 8082
#   → http://127.0.0.1:8082/consult
```

### Docker（全栈：Postgres/pgvector + API）

```bash
export APT_MIRROR=mirrors.aliyun.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 可选：装 CPU 版 torch，镜像从 ~10 GB 降到 ~2.3 GB
export TORCH_CPU_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
export TORCH_CPU_VERSION=2.9.1+cpu

docker compose up -d          # → http://127.0.0.1:8080
```

### Windows 一键启动

双击 `start.cmd` 即可：默认以 **clinic 域**在 **8000** 起服务（被占用就往后找），必要时先建索引，
然后自动打开**问诊页** `http://127.0.0.1:8000/consult`；本机已经有健康的实例时直接附着上去，
不再另起一个。换域用 `start.cmd --domain k8s_docs`（那时打开的是控制台页）。

## 页面与端点

| 路径 | 用途 |
|---|---|
| `/consult` | **交互式问诊**：多轮采问 + 对话式建议；急诊时顶部出现醒目提示 |
| `/console` | 控制台：索引体检、`/metrics` 摘要、每路检索名次、降级原因 |
| `/` | 演示页：检索 / 工具清单 / 问诊的最小示例 |

```
GET  /healthz  /metrics  /domains  /tools
POST /search   /ask      /chat
GET  /         /console
GET  /consult  POST /consult  POST /consult/reset  GET /consult/status  GET /consult/encounters
```

`/search` 与 `/ask` 共用请求体：`{"q": "...", "top_k": 8}`。
`GET /consult/encounters?patient_id=...&limit=20` 读回该患者的最近就诊记录
（主诉 / 时间 / 结论摘要 / 对话），是"就诊记录真的落库了"的查证入口；
`POST /consult` 的 `patient_id` 只是记录归属，**不是认证**（见 [`SECURITY.md`](SECURITY.md)）。

### MCP（Claude Desktop / Cursor 等外部客户端）

```bash
python -m kubedocs.cli mcp --domain k8s_docs                 # 长驻：stdio，一条命令即可用
python -m kubedocs.cli mcp --domain clinic --allow-side-effects
```

两个域都暴露**真实工具**：`k8s_docs` → `search_docs` + `check_citations`；`clinic` 再加
`list_departments` / `get_availability` / `book_appointment`。工具定义与 HTTP `/tools`、
Agent 工具循环**同源**（`ToolRegistry`），不另写一份。

- **副作用默认不外露**：取号会真实占号，默认不在清单里；开 `--allow-side-effects` 后
  调用时仍必须传 `confirm: true`，否则拒绝执行。
- **启动不加载嵌入模型**：列工具不碰索引，第一次真调 `search_docs` 才装配 store + 检索器；
  索引没建时返回可读的 `isError`（提示先跑 `ingest`），而不是把服务打挂。
- 域决定嵌入模型与维度（clinic 中文 512 维 / k8s_docs 英文 384 维），不要跨域共用索引。

客户端配置（stdio，`command` 指向本仓库 venv 里的 python）：

```json
{"mcpServers": {"kubedocs": {
  "command": "C:/path/to/kubedocs/.venv/Scripts/python.exe",
  "args": ["-m", "kubedocs", "mcp", "--domain", "k8s_docs"]}}}
```

## 架构

```
                       ┌──────────────── 域包（各自一个语料与检索配置）────────────────┐
   DomainRegistry  ────┤  k8s_docs：bge-small-en-v1.5（384 维）· data/raw/k8s-docs      │
                       │  clinic  ：bge-small-zh-v1.5（512 维）· data/clinic/cards.json │
                       └───────────────────────────────────────────────────────────────┘
                                              │
        检索：BM25 + 向量 + RRF(k=60) ──► 可选重排（bge-reranker-base，默认关）
                                              │
        存储：sqlite-vec（默认）· pgvector（容器）· sqlite（回退）
```

**分层**

| 层 | 目录 | 职责 |
|---|---|---|
| 存储 | `store/` | 三种后端统一接口；文档 / 块 / 全文索引 / 向量索引 |
| 检索 | `retrieve/` | 混合召回、RRF 融合、重排 |
| 摄取 | `ingest.py` | 切块、编码、分批提交、删除同步 |
| 域 | `domains/` | 每个域自带语料、嵌入模型、检索器配置与工具 |
| Agent | `agent/` | 工具注册表、ReAct 循环、图编排（LangGraph） |
| 医疗产品层 | `clinical/` | 契约、分诊、证据、采问、提示词、工具编排、挂号 |
| API | `api/` | FastAPI 端点与静态页面 |
| MCP | `mcp/` | 把同一份工具注册表经 MCP 协议（stdio）暴露给外部客户端 |
| 观测 | `observability.py` | Prometheus 指标 |

### 一次问诊的流程

```
患者发言
  │
  ├─ 补全/整理已采集信息（时长、部位、伴随症状…）
  │
  ├─ 工具编排：模型自行决定调用哪些工具取证
  │     search_docs（检索知识卡片）· check_citations（核对引用）…
  │     证据编号 K1、K2… 由程序分配，模型引用它们
  │
  ├─ 模型输出结构化结论（契约化）
  │     urgency · red_flags（病症 + 严重程度 + 依据）
  │     departments · differentials · exams · medications · patient_summary
  │
  ├─ 契约校验与执行
  │     · 判出急重症却给常规档 → 记为违规
  │     · 没有依据的检查/用药 → 丢弃
  │
  └─ 返回给用户：`patient_summary` 那段话 +（急重症时）急诊提示
```

### 检索式分诊

科室建议由**两阶段检索**得出，并叠加一张科室补全表：

1. **粗排**：混合检索取候选卡片池（默认 50 条）；
2. **精排**：用 jieba 词元的 IDF 加权重合度重排；
3. **投票**：按卡片标注的科室投票，多科室卡片按科室数均分票权；
4. **科室补全表**：语料里有相当一部分卡片的科室标注缺失，用图谱关系与别名把它补回来，
   补回的标签带**来源与置信度**，可抽检、可回退。

### 挂号

挂号在 `/chat` 这条链上是一个工具：解析科室 → 查号源 → 请求确认 → 占号 → 出凭据。
带副作用的工具默认不暴露给模型，需要显式开启；产生副作用前必须由调用方确认。

## 知识依据在哪

检索到的**知识卡片就是检查/用药建议的唯一来源**。它们体积较大且受上游许可约束，**不在仓库里**：

| 语料 | 怎么拿到 | 落在 |
|---|---|---|
| k8s 官方文档（1720 篇 / 16.2 MB） | `python scripts/fetch_corpus.py` | `data/raw/k8s-docs/` |
| clinic 知识卡片（31,592 张 / 8.4 MB） | `python scripts/clinic_build_cards.py` | `data/clinic/cards.json` |
| clinic 科室补全表（4,727 条） | `python scripts/clinic_derive_departments.py --triples <triples.txt>` | `data/clinic/department_map.json` |

索引同样是派生物，可随时重建（本机用 sqlite-vec，容器用 pgvector）：

```bash
python -m kubedocs.cli --domain clinic ingest                          # 本机
docker compose exec api python -m kubedocs.cli --domain clinic ingest   # 容器
```

> 容器把宿主的 `./data` 挂进 `/app/data`，所以卡片文件与索引都在容器可见的同一数据目录里：
> `cards.json` 是卡片本体，Postgres 的 `clinic` schema 是它的向量索引。两者都不在镜像内。
>
> 换了嵌入模型后索引维度会变，必须重建。维度是向量列的**类型**，`ingest --rebuild` 只清空数据、
> 改不了它；正确做法是丢掉那个 schema（索引是派生物）：
>
> ```bash
> docker exec kubedocs-db-1 psql -U kubedocs -d kubedocs -c "DROP SCHEMA IF EXISTS clinic CASCADE;"
> docker compose exec api python -m kubedocs.cli --domain clinic ingest
> ```

许可与数据来源见 [NOTICE.md](NOTICE.md)。

## 评测与质量保障

```bash
python scripts/check_gold.py                 # 标注集自检：引用的文档与关键词必须真实存在
python -m kubedocs.cli eval --write-baseline # 首次冻结基线
python -m kubedocs.cli eval --gate           # 之后每次回归检查（指标退化即失败）
```

| 维度 | 测法 | 说明 |
|---|---|---|
| 检索 | hit@k、MRR，与**冻结基线**比对 | `eval --gate`；基线绑定语料指纹与嵌入模型，不可比时**拒绝比较**，而不是给个假通过 |
| 分诊 | 500 例分层用例（严重度 × 语言风格） | `scripts/eval_prototype_patients.py`，离线免费 |
| 急诊识别 | 带严重度标注的用例 | `scripts/eval_emergency.py` |
| 问诊质量 | 带结构化答案的医案 | `scripts/eval_medical_cot.py` |

分诊与急诊评测依赖本地语料，在有语料的机器上运行；检索闸门与全量单元测试由
`scripts/ci_local.py` 覆盖。

## 实测数字（本机，i7-13700H / 16 GB / CPU）

| 项 | 数值 |
|---|---|
| 索引规模 | k8s_docs 18,822 块 · clinic 31,592 张卡片 |
| 首次全量摄取 | k8s 文档约 35 分钟（嵌入占绝大部分）· clinic 31,592 张卡片约 12 分钟 |
| `/healthz` 延迟 | p50 1.6 ms（带 2 秒统计缓存）· 约 580 req/s |
| 首次检索冷启动 | 约 69 秒（加载嵌入模型），之后 0.1 秒级 |
| 单次问诊 | 通常 2~8 次模型调用，取决于模型自己决定调多少工具 |

## 配置

配置来自环境变量或 `.env`（见 `.env.example`）。常用的几个：

| 变量 | 默认 | 说明 |
|---|---|---|
| `KUBEDOCS_DOMAIN` | `k8s_docs` | 当前域 |
| `KUBEDOCS_STORE` | `sqlite-vec` | 向量后端：`sqlite-vec` / `pgvector` / `sqlite` |
| `KUBEDOCS_DATABASE_URL` | — | 用 pgvector 时必填 |
| `KUBEDOCS_LLM_MODEL` | `deepseek/deepseek-chat` | 经 LiteLLM 路由，可换供应商 |
| `KUBEDOCS_LLM_DAILY_BUDGET_USD` | `2.0` | 日花费上限，账本落在 `data/index/llm_spend.json` |
| `KUBEDOCS_EMBEDDING_PROVIDER` | `bge` | `hashing` 用于不下载模型的冒烟测试 |
| `KUBEDOCS_HF_ENDPOINT` | — | 拉模型时的镜像端点（受限网络下有用） |

就诊记录（患者 / 就诊 / 消息）不再有单独的库配置：它落在 **Postgres 的 `clinical` schema**
（连接串就是 `KUBEDOCS_DATABASE_URL`）。Postgres 不可达时 `/consult` 照常可用，只是那一轮
不留档（会记一条 warning）。

## 目录结构

```
src/kubedocs/
  config.py  cli.py  ingest.py  embed.py  normalize.py  generate.py
  store/        # sqlite · sqlite-vec · pgvector
  retrieve/     # 混合检索、RRF、重排
  domains/      # k8s_docs · clinic（各自带语料与配置）
  agent/        # 工具注册表、ReAct 循环、图编排
  clinical/     # 契约 · 分诊 · 证据 · 采问 · 提示词 · 工具编排
  api/          # FastAPI 端点与页面
  mcp/          # MCP server（stdio）：把工具注册表暴露给外部 MCP 客户端
  evaluation/   # 评测闸门与基线
docs/           # 架构、评测、设计决策
scripts/        # 摄取、评测、运维脚本（各脚本内 --help 有说明）
tests/          # 单元测试
```

## 扩展方式

- **加一个域**：新增一个域包（语料路径、嵌入模型、检索器、工具），在 `domains.load_all()` 注册一行。
- **加一个工具**：在 `agent/tools.py` 的 `ToolRegistry` 注册一个 `ToolSpec`（含"什么时候不该用"）。
  同一个注册表同时供给 HTTP 端点、Agent 循环与 MCP server（`kubedocs mcp`），不写第二份定义。
- **加一步编排**：需要在服务端固定顺序里插入一步（采问、取证、判定、落库）时，改
  `clinical/orchestration.py` 的工具循环，而不是在提示词里堆约束。
- **换检索/存储后端**：实现 `Store` 或重排接口，在配置里切换即可。

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

代码以 [MIT 许可证](LICENSE) 发布。`data/eval/prototype/` 下是第三方派生数据，
不在 MIT 覆盖范围内，仅用于研究与评测 —— 完整清单与边界见 [NOTICE.md](NOTICE.md)。
