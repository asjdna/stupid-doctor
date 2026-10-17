# kubedocs

> **A Rough Architecture for a Medical Agent** —— 一个双域检索问答与 Agent 平台：
> `clinic`（医疗问诊 / 挂号 / 急诊识别，中文）与 `k8s_docs`（Kubernetes 官方文档，英文）。
> 检索走混合召回（BM25 + 向量 + RRF 融合）+ 可选重排；回答必须带可点击引用，查不到就明说。
> 默认索引后端是 Postgres + pgvector；没有 Postgres 时降级为本地 `sqlite-vec`，**本机零容器也能跑通全链路**。

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

**检索必须有据，判断交给模型，约束层核对它的每一处输出。**

- **检索式证据**：所有医学断言都要能追到本地知识卡片。检索不到就不给建议，不用模型的通用知识补足。
- **模型主导判断**：模型自己决定调哪些工具取证（检索、引用核对……），自己按"病症 + 严重程度"
  判断是否属于急重症，并输出结构化的判断依据。
- **规则资产只提供线索**：急诊红线词表、口语症状词表、症状→科室票数这些确定性资产被装成一份
  「参考标准」面板（`FactSet`），每条都写着"这是线索，不是结论"，**一条结论都不下**。
  `/chat` 把它拼进系统提示交给模型；`/consult` 的模型看到的是硬性条款与工具目录、依据靠它
  自己调工具取回，同一份面板在那里由约束层拿去核对输出（详见
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 第七节）。
- **约束层核对输出**：`clinical/verify.py` 拿**同一份事实**校验模型的每一处输出。动作幻觉
  （服务端没挂上号却说"已为您挂上"）、编造凭据号、检索为空却给医学内容属于 `block`——
  带原因重生成一次，仍违规则那段散文**不上屏**，改由服务端拼一段只含事实的回答；
  编造依据编号、科室未接地属于 `strip`——丢掉那一条；未解释的红线线索、服务端没给过的金额
  属于 `warn`——只留痕。
- **无判断者就没有结论**：没有大模型时不做任何兜底，只回一句话（见下）。
- **对话是最终产物**：结构化结论用于后端留档与校验，用户看到的是模型组织好的那段话。

## 没有大模型时会怎样

判断权全部在大模型，于是"模型在不在"是一条**架构级状态**：没有判断者就没有结论。
本仓库没有降级层——不拿检索结果冒充回答，也不让规则轨顶替结论。文案只有两句，
唯一定义在 `src/kubedocs/llm_status.py`：

| 情形 | 回什么 | 该做什么 |
|---|---|---|
| 压根没配密钥 | 「未连接大模型，请连接大模型后重试。」 | 去配密钥 |
| 配了但这次调用没成功（超时 / 限流 / 输出不可解析 / 预算用尽） | 「大模型调用失败，请稍后重试。」 | 重试；运维查网络与额度 |

两句刻意不合并：把"连了但失败"说成"未连接大模型"，会让运维去翻一个其实配好的密钥。

- `/consult` 没密钥时**不检索、不装配面板、不进工具循环**，只回那句话；
- `/chat` 的 `message` 事件就是那句话（`status="no_llm"`），不发 `tool` 事件，也不回退到确定性图；
- `/ask` 在**检索之前**就返回那句话，`citations=[]`、`n_hits=0`；
- CLI `ask` 把这句话打到 stderr，退出码 **3**（前置条件不满足，不是参数错）。

**不需要大模型的入口一律不受影响**：`/search`（与 CLI `search`）、`ingest`、`stats`、
`/healthz`、`/metrics`、`domains`、`tools`，以及 MCP 工具——MCP 用的是客户端自己的模型，
服务端不替它调模型。

## 快速开始

默认向量后端是 **`pgvector`**（Postgres + pgvector）：索引与全文检索在同一个库里，
跨重启、跨副本共享。数据库不可达时会**降级**为本地 `sqlite-vec`（零外部进程），
降级在三处可见：

- `GET /healthz` 的 `store`（实际生效的后端）与 `store_fallback`（降级原因）；
- `kubedocs stats` 打印的「存储后端 X（配置 Y）」，降级时多一行 `⚠ 已降级：…`；
- 服务日志里那条 `WARNING`（`store.from_settings` 打的）。

**降级有代价**：两个后端的索引存在不同位置，降级期间建的索引在 Postgres 恢复后是**看不见**的；
要回到 pgvector，先起数据库再重跑 `ingest`。

### 本机（无需 Docker）

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 0) 默认后端需要一个 Postgres。本机没有就跳过这一步：第 2 步会自动降级为本地 sqlite-vec
#    （降级原因在 stats / /healthz 里能看到）。要起一个库：docker compose up -d db
export KUBEDOCS_DATABASE_URL=postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs

# 1) 拉语料（一次就够；当前 1722 篇 / 16.2 MB，篇数随上游 main 分支漂移）
python scripts/fetch_corpus.py

# 2) 建索引（首次全量约 35 分钟，嵌入占绝大部分）
python -m kubedocs.cli --hf-endpoint https://hf-mirror.com ingest

# 3) 检索
python -m kubedocs.cli search "how do I stop a container using too much memory" --explain

# 4) 带引用回答（需要 LLM 密钥；没密钥时只回一句「未连接大模型」并以退出码 3 结束，不做检索兜底）
python -m kubedocs.cli ask "怎么限制一个 Pod 的 CPU？" --show-context

# 5) 索引体检（含 FTS 一致性自检；会打印实际生效的后端）
python -m kubedocs.cli stats --verify
```

### 灌自己的资料（多格式）

`ingest` 认的后缀：`.md` / `.txt` / `.csv` / `.html` / `.htm` / `.pdf` / `.docx` / `.xlsx` / `.pptx`。
把文件放进当前域的语料目录（`k8s_docs` 域默认 `data/raw/k8s-docs`，可用 `--corpus-dir` 或
`KUBEDOCS_CORPUS_DIR` 换）再跑一次摄取即可：

```bash
python -m kubedocs.cli --corpus-dir data/raw/mine ingest
```

- **位置跟着引用走**：PDF 是 `p.7`、表格是 `Sheet1!A1:C20`、幻灯片是 `slide 3`、
  标题型格式（md / txt / html / docx）是标题路径。
- **扫描件 PDF 会被点名跳过**（抽不出文本层，不做 OCR），绝不建空块——"搜不到"必须能查出原因。
- **单文件失败不影响整批**：不支持的格式、文件损坏、缺解析库都变成报告里一条可读原因。
- **不静默丢字**：文本类格式按 `utf-8-sig` → `utf-8` → `gb18030` 严格探测，
  都不行时用替换解码并把替换字符数写进报告。

### 医疗问诊（`clinic` 域）

```bash
# 卡片语料与科室补全表（见下方「知识依据」）
#   `--source` 必填（或设 KUBEDOCS_CLINIC_SOURCE）：它指向跑过上游 `scripts/build_knowledge.py`
#   的产物目录；不给就以退出码 3 说明缺什么，不会拿空目录编出一份卡片集。
python scripts/clinic_build_cards.py --source <上游产物目录>
python scripts/clinic_derive_departments.py --triples <OpenCMKG/triples.txt>

# 建索引（首次全量约 1 小时，CPU 编码）
python -m kubedocs.cli --domain clinic ingest

# 号源排班（只对 Postgres 号源有意义；serve 起来的进程首次用到号源时也会幂等补一次）
python -m kubedocs.cli schedule generate

# 起服务，打开交互式问诊页
python -m kubedocs.cli --domain clinic serve --port 8082
#   → http://127.0.0.1:8082/consult
```

挂号演示（需要配置 LLM 密钥）：在问诊页说"我想挂呼吸内科"，助手只会给你一张**待确认**卡片
（号源没被占用），点卡片上的「确认挂号」才真的取号；取到的号出现在「我的挂号记录」卡片里，
点「退号」可以退掉。

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

双击 `start.cmd` 即可：默认以 **clinic 域**在 **8000** 起服务（被占用就往后找），
然后自动打开**问诊页** `http://127.0.0.1:8000/consult`；本机已经有健康的实例时直接附着上去，
不再另起一个。索引缺失**不会阻止启动**，脚本只打印建索引的命令。
换域用 `start.cmd --domain k8s_docs`（那时打开的是控制台页）。

## 页面与端点

| 路径 | 用途 |
|---|---|
| `/consult` | **交互式问诊**：多轮采问 + 对话式建议；急诊时顶部出现醒目提示 |
| `/console` | 控制台：索引体检、`/metrics` 摘要、每路检索名次、检索降级原因 |
| `/` | 演示页：检索 / 工具清单 / 问诊的最小示例 |

```
GET  /healthz  /metrics  /domains  /tools
POST /search   /ask      /chat
GET  /         /console
GET  /docs     /redoc    /openapi.json      # FastAPI 自带（serve / start.cmd 会打印 /docs）
GET  /consult  POST /consult  POST /consult/reset  GET /consult/status
GET  /consult/encounters  GET /consult/appointments
POST /consult/booking/confirm  POST /consult/booking/cancel
```

`/search` 与 `/ask` 共用请求体：`{"q": "...", "top_k": 8}`。
`GET /consult/encounters?patient_id=...&limit=20` 读回该患者的最近就诊记录
（主诉 / 时间 / 结论摘要 / 对话 / 这次就诊挂出的号），是"就诊记录真的落库了"的查证入口；
`GET /consult/appointments?patient_id=...` 是患者本人的挂号记录。
所有 `patient_id` 都只是归属，**不是认证**（见 [`SECURITY.md`](SECURITY.md)）。

**问诊响应里的两个留档字段**（`POST /consult`）：`reference` 是这一轮的事实集（面板形状：
急诊线索含三维明细、症状归一、科室候选与票数、知识依据编号、业务事实与装配问题），
`contract_violations` 是这一轮被约束层拦下的东西（`kind` / `severity` / `detail`）。
两者**必须成对看**：只有同时知道"它手上有什么依据"和"它哪里越界"，才能判断这次是依据不够，
还是模型没守规矩。`/chat` 的 `message` 事件带**同一对**字段：`reference` 是那份真的拼进它
系统提示的面板，`contract_violations` 是散文路径被事实校验拦下的违规——包括重生成后回正的
那些（回正不等于没发生过）。

**安全与不确定性字段**（`POST /consult`）：`safety` 是第三层安全门这一轮的留档
（`fired` = 触发的不变量明细、`ids` = 不变量 id、`followup` = 服务端补的兜底追问）；
`confidence` 是模型对结论的自报置信度（0~1，**缺省 `null`**，不进任何阈值，只用于行为约束
与校准统计）；`information_gaps` 是「还缺什么信息」，`followup_questions` 是「要怎么问」。
`red_flags[]` 每条带 `angle`（判断角度）/ `likelihood`（`ruled_out` 到 `very_likely` 的
不确定性档位）/ `discriminator`（能区分「要急救 / 不用急救」的那一个关键问题）/
`ruled_out_by`（判为已排除时的依据）。`/chat` 的 SSE `message` 事件带同一份 `safety`。

**挂号的两个写端点**：`POST /consult/booking/confirm` 与 `POST /consult/booking/cancel`。
确认请求**只带 `proposal_id`**（外加会话与就诊人标识）：科室、日期、时段、挂号费都取自服务端
存的那张待确认单，前端与模型都改不动。业务失败（单子过期 / 号被占满 / 停诊）返回 **409 + 一句
人话**，参数不合法由 pydantic 拦成 422。

### MCP（Claude Desktop / Cursor 等外部客户端）

```bash
python -m kubedocs.cli mcp --domain k8s_docs                 # 长驻：stdio，一条命令即可用
python -m kubedocs.cli mcp --domain clinic --allow-side-effects --patient-id <就诊人标识>
```

两个域都暴露**真实工具**：`k8s_docs` → `search_docs` + `check_citations`；`clinic` 再加
`list_departments` / `get_availability` / `booking_window` / `my_appointments`。工具定义与
HTTP `/tools`、Agent 工具循环**同源**（`ToolRegistry`），不另写一份。

- **副作用默认不外露**：`book_appointment` 会真实占号，默认不在清单里；开
  `--allow-side-effects` 后调用时仍必须传 `confirm: true`，否则拒绝执行。
- **MCP 这一侧没有提议工具**：`propose_booking` 需要一份进程内的待确认单存储，MCP 装配不带它，
  所以号源那几个工具都是只读的；确认与退号不在 MCP 上，走 HTTP 的
  `/consult/booking/confirm` 与 `/consult/booking/cancel`。
- **`--patient-id` 决定"这个会话代表谁"**：`my_appointments` 与取号都按它记账（默认 `anon`）。
- **列工具不碰索引**：握手与 `tools/list` 都不碰索引、不加载嵌入模型。检索器由**后台线程预热**
  （默认开：第一次 `search_docs` 不必现付一分钟的模型加载；`--no-warmup` 关掉它，退回"第一次
  真调才装配"）。索引没建时返回可读的 `isError`（提示先跑 `ingest`），而不是把服务打挂。
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
        存储：pgvector（默认）· sqlite-vec（离线回退）· sqlite（最小依赖 / 对照）
```

**分层**

| 层 | 目录 | 职责 |
|---|---|---|
| 存储 | `store/` | 三种后端统一接口；文档 / 父块 / 块 / 全文索引 / 向量索引 |
| 检索 | `retrieve/` | 混合召回、RRF 融合、重排 |
| 解析 | `parse.py` | PDF / Office / HTML / CSV / TXT / MD → 带位置的块 |
| 摄取 | `ingest.py` | 切块（父子两层）、编码、分批提交、删除同步 |
| 域 | `domains/` | 每个域自带语料、嵌入模型、检索器配置与工具 |
| Agent | `agent/` | 工具注册表、ReAct 循环、确定性图（对照臂，不在请求路径上） |
| 医疗产品层 | `clinical/` | 唯一装配入口（`runtime.py`）· 证据层（参考标准面板）· 提示词 · 约束层（事实校验）· 安全门 · 采问 · 就诊记录 |
| API | `api/` | FastAPI 端点与静态页面 |
| MCP | `mcp/` | 把同一份工具注册表经 MCP 协议（stdio）暴露给外部客户端 |
| 观测 | `observability.py` | Prometheus 指标 |

**一块内容存两层**：进索引的**子块**（300 字符级，管命中精确）和喂给模型的**父块**
（一个标题小节，管语境完整）。命中后取 `context_text`：有父块给父块，没有就退回子块正文。
`body`（命中的那一段）与父块是两种用途，`search_docs` 与证据装配都会一起给出。

### 一次问诊的流程

三层各做一件事：**证据层给线索，判断层下结论，约束层核对结论**。

```
患者发言
  │
  ├─ 补全/整理已采集信息（时长、部位、伴随症状…）
  │
  ├─ 装配「参考标准」面板：急诊线索（关键词命中 + 断言/经验者/时间性三维取值与明细）
  │      · 症状归一（原话词 → 标准词 → 语料键）· 科室候选（票数 + 来源）
  │      · 知识依据 · 业务事实（含"本轮服务端没有任何已确认的挂号/退号动作"）
  │      每条都标着"这是线索，不是结论"；`/chat` 把它拼进系统提示，`/consult` 用于校验与留档
  │
  ├─ 工具编排：模型自行决定调用哪些工具取证
  │     search_docs（检索知识卡片）· check_citations（核对引用）…
  │     证据编号 K1、K2… 由程序分配，模型引用它们
  │
  ├─ 模型输出结构化结论（契约化）——**判断只由它做**
  │     intent（需求类型 medical/booking/admin）· urgency
  │     red_flags（病症 + 严重程度 + 判断角度 + 不确定性档位 + 依据）
  │     departments · differentials · exams · medications · patient_summary
  │     confidence（自报置信度）· information_gaps · followup_questions
  │
  ├─ 约束层校验：拿同一份事实集核对它的每一处输出
  │     block  ⇒ 带违规原因重生成一次；仍违规 ⇒ 那段散文不上屏，换成只含事实的回答
  │                （事实回答里保留服务端已知的科室与号源，`action="answer"`——这是降级，不是故障）
  │     strip  ⇒ 丢掉那一条（编造依据的条目、未接地的科室、事务性需求里多余的医学内容）
  │     warn   ⇒ 只留痕（进响应里的 contract_violations）
  │
  ├─ 执行契约：没有依据的检查/用药 → 丢弃（**在校验之后**，否则"检索为空却给医学内容"
  │     会因为字段已被清空而查不出来，而散文里那句话照样上屏）
  │
  ├─ 第三层安全门（纯代码，最后一道）：三条不变量
  │     判出急症却没有可执行的急救动作 ⇒ **补**一句固定指引（唯一「加内容」的一条）
  │     触发的不变量进审计日志，并随响应的 `safety` 留档
  │
  └─ 返回给用户：`patient_summary` 那段话 +（急重症时）急诊提示
```

面板**装配两次、递两次**：`/consult` 在跑循环**之前**先装配一份**线索版**
（`render(defer_retrieval=True)`：规则线索齐全，知识依据留空并写明"要不要检索由你自己
决定"）交给模型，循环跑完后再装配完整版用于校验与留档——那时才知道模型自己检索到了什么，
面板里的知识依据就是它取回的那些片段。`/chat` 一次装配，同一份既递给模型也用于校验。
上图按"要判断就得先有线索"的语义顺序画。

### 检索式分诊

科室建议由**两阶段检索**得出，并叠加一张科室补全表。
它是**给模型的线索之一**（进参考标准面板的"科室候选"，带票数与来源），不是系统给出的结论——
最终挂哪个科由模型判断：

1. **粗排**：混合检索取候选卡片池（产品路径 `/consult` 取 `top_k=10`；
   离线评测脚本 `eval_prototype_patients.py --coarse-k` 的默认值是 50）；
2. **精排**：用 jieba 词元的 IDF 加权重合度重排；
3. **投票**：按卡片标注的科室投票，多科室卡片按科室数均分票权；
4. **科室补全表**：语料里有相当一部分卡片的科室标注缺失，用图谱关系与别名把它补回来，
   补回的标签带**来源与置信度**，可抽检、可回退。

### 挂号：模型提议、患者点确认、服务端执行

号源只到科室（科室 × 日期 × 上午/下午 × 排队号），不做医生维度。默认落 Postgres 的
`booking` schema；数据库不可达时退回进程内号源（`/consult/status` 的 `booking_backend` 会说出来）。

| 角色 | 能做什么 |
|---|---|
| 模型 | 四个只读工具：`list_departments` / `get_availability` / `booking_window` / `my_appointments`，外加只读的 `propose_booking`——它只产出一张**待确认单**，号源一个都不动 |
| 患者 | 在卡片上点「确认挂号」才真的取号；点「退号」才真的退号 |
| 服务端 | `POST /consult/booking/confirm` 与 `POST /consult/booking/cancel` 是仅有的两个写入口，参数取自服务端存的那张单子 |

- **确认请求只带 `proposal_id`**：前端与模型都改不动科室、日期、时段与挂号费。
- **待确认单一次性、15 分钟过期、进程内**：双击确认、网络重试、刷新页面重放都只会有一张生效；
  重启进程后重新说一句"帮我挂号"即可。
- **`book_appointment` 仍在册**，但标着 `side_effect=True`，工具循环默认不交给模型。
- `/chat` 与 `/consult` 是同一套流程：都由模型提议、由前端卡片上的按钮触发同一个确认端点。
- **排班**：每科室每天上午/下午各 10 个号、今天起 7 天，`kubedocs schedule generate` 幂等补齐
  （绝不覆盖已有余量，所以放号窗口会随时间向前滚动）。停诊 / 开诊 / 改号量分别是
  `kubedocs schedule close|open|capacity`，看窗口是 `kubedocs schedule show`。
- **挂号费**的真相源是 `domains/clinic/data/departments.py`（呼吸内科 35 元，其余默认 30 元），
  号源库里的 `fee` 只是它的快照。

## 知识依据在哪

检索到的**知识卡片就是检查/用药建议的唯一来源**。它们体积较大且受上游许可约束，**不在仓库里**：

| 语料 | 怎么拿到 | 落在 |
|---|---|---|
| k8s 官方文档（1722 篇 / 16.2 MB） | `python scripts/fetch_corpus.py` | `data/raw/k8s-docs/` |
| clinic 知识卡片（31,592 张 / 8.2 MB） | `python scripts/clinic_build_cards.py --source <上游产物目录>` | `data/clinic/cards.json` |
| clinic 科室补全表（4,727 条） | `python scripts/clinic_derive_departments.py --triples <triples.txt>` | `data/clinic/department_map.json` |

### 想跑通就得先有语料？——用你自己的文档也行

上面三样**都不随仓库分发**（本仓库没有它们的再分发权，边界见 [NOTICE.md](NOTICE.md)），
索引同样不入库。所以**新克隆的仓库检索是空的**：此时系统按设计**不给医学内容**
（`no_medical_advice_without_evidence` 会把那段散文换成只含事实的回答），
**不会**拿模型的通用知识顶替检索结果。这不是判断力下降，是依据没了。

换成你自己的文档同样能把整条链路跑通（检索 / 引用核对 / 约束层 / 安全门全都在）：

```bash
# 任意 Markdown / PDF / Office：放进一个目录，用 k8s_docs 域摄取
python -m kubedocs.cli --corpus-dir <你的文档目录> ingest
python -m kubedocs.cli search "你的问题"

# clinic 域换成自己的卡片表：写 data/clinic/cards.json 再摄取
python -m kubedocs.cli --domain clinic ingest
```

卡片最小形状（`cards.json` 是数组，每张卡至少要这三个键）：

```json
{"kind": "symptom", "key": "my_cough",
 "text": "症状：夜间干咳。若伴低热超过三天，建议就诊呼吸内科。",
 "diseases": ["my_cough"], "departments": ["呼吸内科"]}
```

`kind` / `key` / `text` 决定索引与引用，`diseases` / `departments` 供分诊投票。

⚠ **两个实测出来的坑，换语料前必须知道**：

1. **`clinic` 域的语料路径是写死的**：`clinic_cards_path()` 固定 `data/clinic/cards.json`，
   **没有环境变量可换**。想并存多套卡片就换文件，或自己加一个域。
2. **语料太小比没有语料更危险**：安全门判的是**"有没有依据"**，不是**"依据相不相关"**。
   用一份只有 3 张卡片的索引去问卡片完全没涵盖的问题（实测问「脚踝扭伤两周、石膏什么时候能拆」），
   检索仍会命中那 3 张卡，模型于是"有依据"地用自己的通用知识答了骨科问题——
   `safety` 不触发、`contract_violations` 为 0。所以自带语料要**覆盖你真正会问的范围**，
   读回答时也要核对 `reference.evidence_ids` 指向的卡片是否切题。

口径与机制见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 的「语料按需装配」。

索引同样是派生物，可随时重建。**默认后端 `pgvector` 把索引放进 Postgres**：

```bash
python -m kubedocs.cli --domain clinic ingest                          # 按配置的后端写
docker compose exec api python -m kubedocs.cli --domain clinic ingest   # 容器（KUBEDOCS_STORE=pgvector）
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
>
> 改了切块参数（`KUBEDOCS_CHUNK_MAX_CHARS` / `KUBEDOCS_CHUNK_OVERLAP_CHARS` /
> `KUBEDOCS_CHUNK_PARENT_MAX_CHARS`）同样必须重建索引：块边界与块数是**摄取时**定下来的，
> 只改配置不重跑，索引里还是旧的那一套。

许可与数据来源见 [NOTICE.md](NOTICE.md)。

## 安全怎么保证

安全不来自某一次评测分数，而来自**三层防护 + 临床治理**：

| 层 | 在哪 | 负责什么 |
|---|---|---|
| L1 语义理解 | 大模型（`clinical/prompts/system.md` 的 R1） | **多角度**识别危险信号——症状性质 / 组合共现 / 时间过程 / 严重度 / 高危人群 / 情境诱因；每条写明判断角度、不确定性档位与依据 |
| L2 主动问诊 | 大模型 + 输出契约 | 没被排除的危险信号必须问清能区分「要急救 / 不用急救」的那一个关键问题；模型没给追问时，由服务端补一句确定性的追问 |
| L3 确定性安全门 | 服务端纯代码（`clinical/safety.py`） | 三条不变量，模型说什么都不例外 |

L3 的三条不变量是：**没有检索到资料就不给医学内容**、**不得声称服务端没执行过的挂号动作**、
**判出急症必须给出可执行的急救动作**（判据只认 120 / 急诊 / 救护车 / 立刻就医这类**去向**；
缺了就补一句固定指引，是唯一一条「加内容」的不变量）。前两条删内容、第三条补内容，都不做
医学判断——凡是需要医学判断的（这条信号算不算危险、这个科室对不对）留在 L1 / L2 并留档。

三层是**纵深而不是备份**：L1 的漏报由 L2 的追问补上，两者都失手时由 L3 兜住「至少给对动作」，
任何一层被单独拿掉，另外两层都补不上它的洞。每次不变量触发都追加一行审计记录
（`data/index/safety_audit.jsonl`，只记不变量 / 动作 / `trace_id`，**不记患者原话**），
`GET /consult/status` 的 `safety_audit_recent` 能读回来。

**改动要留痕**：安全资产（提示词 / 危险信号词表 / 症状词表 / 科室表 / 金标集）的清单、
哈希与治理缺口落在 `data/governance/safety_assets.json`，由 `scripts/check_governance.py`
在 CI 里核对——资产改动要同步哈希与版本理由、不变量与代码注册表一一对应、每条不变量在
`tests/` 里有覆盖，对不上就红。分层的完整说明见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，
治理机制与已知缺口见 [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md)。

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
| clinic 规则臂（`rule`） | 红线召回 / 挂号意图 / 分诊（**确定性、离线、免费**） | `scripts/eval_clinic.py --arm rule`；**线索质量**的稳定读数——红线词表漏一条会直接反映在召回读数上（但**不会**让闸门变红，见下一行） |
| clinic 工具循环臂（`llm` / `tool`） | **产品路径**：`pipeline.advance(原话)`，模型自己决定调不调检索 | `scripts/eval_clinic.py --arm llm`；需要索引 + 密钥，会花钱；每题一次真实模型调用 |
| clinic 单发对照臂（`reasoner`） | `ClinicalReasoner` 单发判断，不检索（便宜） | 用来对照「有工具 / 没工具」的差别，不是产品路径 |
| clinic 闸门（`--gate`） | 只卡一件事：**这次运行能不能作为证据** | 未评测 / 有题目没拿到模型判断 ⇒ 不通过；指标、参考目标与**安全不变量的触发**都只记录（硬信号由长期监听报给运维） |

分诊与急诊评测依赖本地语料，在有语料的机器上运行；检索闸门与全量单元测试由
`scripts/ci_local.py` 覆盖（`quality` job **要求 `KUBEDOCS_DATABASE_URL` 可达**，
连不上就直接失败，不会静默 skip 出一片假绿）。**`ci_local.py` 的 `gate` job 跑的是检索闸门**
（`eval --top-k 10 --gate`，域走默认的 `k8s_docs`），与 clinic 域无关；clinic 那条闸门要单独跑：

```bash
python scripts/eval_clinic.py --arm both --gate   # 规则臂 + 工具循环臂（产品路径）
```

`--arm rule` 只跑确定性的规则臂（不花钱、可离线复现），它同样只记录指标——规则臂是
线索提供者，拿它的分数当门限是用错尺子。`--arm reasoner` 换一条不检索的单发判断路线，
用来看「有工具 / 没工具」的差别；`--arm both`（默认）是规则臂 + 工具循环臂。
两个臂的指标都**只记录**：`--gate` 只在**这次运行不能作为证据**时失败（一条可评测用例都没有，
或有题目没拿到模型判断）。**安全不变量的触发也是记录**——它说明"模型这一轮需要被纠正"
（每条都已被安全门处置：换成只含事实的回答、或补上急救指引），硬信号由长期监听按 `alert`
报给运维并让监听的退出码变 1。评测的退出码回答"这次测成了吗"，监听的退出码回答"系统还正常吗"。

**装配只有一处**：工具循环臂与 `/consult` 都走 `src/kubedocs/clinical/runtime.py` 的
`build_clinical_assembly`（`open_domain_index` 负责按域开索引、造检索器）——同一次工具注册、
同一个面板建造器（`domains/clinic/reference.py` 的 `make_reference_builder` 与
`booking_facts`）、同一本日预算账、同一套安全门。评测侧自己再写一遍装配，就会得到一个
"产品不会走的形状"，而分数看起来仍然合理。评测侧的差异只有三处，都是刻意的：号源用
`MockAppointmentProvider`（不碰真实号源库）、不落就诊记录、不写审计文件——评测会在短时间
里造出大量假会话，混进真库只会污染治理读数。

> **治理档案里的一次读数**（工具循环臂，31 条用例、共 69 次模型调用，约 $0.035）：
>
> | 臂 | 危险信号召回 | 误报率 | 分诊 top3 |
> |---|---|---|---|
> | 规则臂（`rule`） | 100%（TP=9 FN=0） | 0%（FP=0 TN=17） | 100%（5/5） |
> | 工具循环臂（`tool_loop`） | 100%（TP=9 FN=0） | 23.53%（FP=4 TN=13） | 100%（5/5） |
>
> 参考目标是 **召回 100% / 误报率 ≤20% / 分诊 ≥60%**（`reporting_targets()` 导出，与实测值
> 并排打印并进档案）。这些数字与目标的差距**只进记录，不决定退出码**。工具循环臂的"报了危险
> 信号"判据是**判成急症，或还有未排除的危险信号**——把看到过但判断不该上报的信号写成
> `ruled_out`（并写明依据）是留痕，不算误报。剩下 4 条误报要分开看：`n02`「我父亲昨天说心口疼」、
> `n03`「我家孩子喘不上气，要不要去医院」、`n04`「朋友说他突然看不清了」是**金标口径**
> （患者本人急症）与模型行为（对话里有人有危险就报）的分歧；`s02`「这两天有点头痛，睡不好」
> 则是**良性主诉被判成了危险信号，算模型判错**。要改的是提示词与用例口径，不是门限。
>
> 同一轮里工具循环臂的**确定性安全不变量 0 条**、置信度校准 `n=31` Brier `0.20`（只报告，
> 不上门限；分箱见 [`docs/EVALUATION.md`](docs/EVALUATION.md)）。**不同轮次会波动**：有一轮
> 触发 2 条不变量，**每一条都已被服务端处置过**（把那段散文换成只含事实的回答 / 补上急救
> 指引），所以它说明的是"模型这一轮需要被纠正"，**不是**"患者看到了不该看的内容"。
>
> **没有密钥时判断臂标为「未评测」，`--gate` 同样不通过**——有题目没拿到模型判断属于
> 「这次运行不能作为证据」，未评测不等于通过，否则删掉密钥就成了让闸门变绿的手段。
>
> **`--archive` 把整份报告写成治理档案**（`eval_clinic.py` 的默认值是**空**——不传就什么都不写；
> 默认落点 `data/eval/clinic_archive/` 属于 `monitor_clinic.py`。文件名带 UTC 时间戳，永不覆盖），
> **同时追加一行趋势记录**到 `data/eval/clinic_trend.jsonl`（落点可用
> `--trend-path` 改）：趋势与异常靠档案与趋势记录比。工具循环臂的分诊 100% 是 5 条用例上的
> 结果，样本很小，不要当作临床准确率（口径见 [`docs/EVALUATION.md`](docs/EVALUATION.md)）。

### 长期监听

单次评测回答"现在怎么样"，趋势回答"是不是在变差"：

```bash
python scripts/monitor_clinic.py --status                  # 只看历史（不花钱、不联网）
python scripts/monitor_clinic.py --run                     # 跑一轮：评测 → 留档 → 趋势 → 提醒
python scripts/monitor_clinic.py --follow --interval 3600  # 常驻：每小时一轮（容器边车 / cron 都行）
```

`--run` 与 `--follow` **复用 `scripts/eval_clinic.py` 的 `main()`**（同一份装配、同一套闸门、
同一份报告与档案）：监听自己再写一遍评测，就会得到一份"和产品不一样"的趋势。

**它不新增门限**——指标不该承担安全责任，所以监听只把这一轮与最近 `--window`（默认 5）轮的
**实际区间**摆在一起，指出哪一项变了：

| 级别 | 判据 | 影响退出码 |
|---|---|---|
| `alert` | 触发了确定性安全不变量 / 这一轮没跑成（未评测）/ 有题目没拿到判断 | **是**（退出码 1） |
| `note` | 召回下降、误报上升、分诊下降、单题花费上升、提示词或金标集变化 | 默认否（`--strict` 才硬拦） |

`note` 的判据是"低于最近几轮的最低值 / 高于最高值"，不与人拍的阈值比；工具循环臂与单发臂
**分开比**（数量级不同：一个检索、一个不检索）。`--status` 永远退出 0（它只读历史）。

趋势记录是 `data/eval/clinic_trend.jsonl`（`src/kubedocs/evaluation/monitor.py` 定义每行一条
扁平 JSON，追加、永不覆盖），与治理档案同目录；记录里**不记患者原话**，与安全审计同一条纪律。
运行期还有一条不用跑评测的监听：Prometheus 计数
`kubedocs_safety_invariant_fired_total{invariant,domain}`——审计日志回答"发生了什么"，计数
回答"最近是不是变多了"，两者出自同一个收口点（`clinical/safety.py` 的 `append_audit`）。

## 实测数字（本机，i7-13700H / 16 GB / CPU）

| 项 | 数值 |
|---|---|
| 索引规模 | clinic 31,592 张卡片（每张卡片 = 一父块 + 一子块）；k8s_docs 的父子块数由切块参数与语料共同决定 |
| 首次全量摄取 | k8s 文档约 35 分钟（嵌入占绝大部分）· clinic 31,592 张卡片约 1 小时（CPU 编码，与 `domains/clinic/domain.py` 的口径一致） |
| `/healthz` 延迟 | p50 1.6 ms（带 2 秒统计缓存）· 约 580 req/s |
| 首次检索冷启动 | 约 69 秒（加载嵌入模型），之后 0.1 秒级 |
| 单次问诊 | 工具调用轮次硬上限 4 轮（`MAX_TOOL_ROUNDS`）；轮次用尽后还会再发一次"用现有依据作答"的请求，所以单轮最多 **5 次模型调用**；被约束层拦下时会带原因再走一轮 |

## 配置

配置来自环境变量或 `.env`（见 `.env.example`）。常用的几个：

| 变量 | 默认 | 说明 |
|---|---|---|
| `KUBEDOCS_DOMAIN` | `k8s_docs` | 当前域 |
| `KUBEDOCS_STORE` | `pgvector` | 向量后端：`pgvector` / `sqlite-vec` / `sqlite`；数据库不可达时降级为 `sqlite-vec` |
| `KUBEDOCS_DATABASE_URL` | `postgresql://kubedocs:kubedocs@127.0.0.1:5432/kubedocs` | pgvector 索引、就诊记录与号源都读它 |
| `KUBEDOCS_CHUNK_MAX_CHARS` | `300` | 子块（检索单位）目标长度；改了要重建索引 |
| `KUBEDOCS_CHUNK_OVERLAP_CHARS` | `60` | 相邻子块的重叠（必须小于上一项）；改了要重建索引 |
| `KUBEDOCS_CHUNK_MIN_CHARS` | `80` | 短于此的子块并入同一父块内的前一块；改了要重建索引 |
| `KUBEDOCS_CHUNK_PARENT_MAX_CHARS` | `3000` | 父块（标题小节）上限，超过就按段落拆；改了要重建索引 |
| `KUBEDOCS_MAX_CHUNKS_PER_DOC` | `2` | 组装上下文时同一文档最多收录几段（防止近重复片段挤满上下文） |
| `KUBEDOCS_LLM_MODEL` | `deepseek/deepseek-chat` | 经 LiteLLM 路由，可换供应商 |
| `KUBEDOCS_LLM_FALLBACKS` | — | 逗号分隔的备用模型，主模型失败时依次回退 |
| `KUBEDOCS_LLM_DAILY_BUDGET_USD` | `5.0` | 日花费上限，账本落在 `data/index/llm_spend.json` |
| `KUBEDOCS_EMBEDDING_PROVIDER` | `bge` | `hashing` 用于不下载模型的冒烟测试 |
| `KUBEDOCS_HF_ENDPOINT` | —（代码默认空串） | 拉模型时的镜像端点（受限网络下有用）。注意 `.env.example` 里默认给了 `https://hf-mirror.com` |

就诊记录（患者 / 就诊 / 消息 / 挂号关联）与号源（`booking` schema）也没有单独的库配置：
它们都落在 **Postgres**（连接串就是 `KUBEDOCS_DATABASE_URL`），schema 分别是 `clinical` 与
`booking`。Postgres 不可达时 `/consult` 照常可用：就诊记录降级为不落库（记一条 warning），
号源退回进程内（`/consult/status` 的 `booking_backend` 会说出来）。

**记忆放在哪**（三类，只有最后一类刻意不落库）：

| 记忆 | 内容 | 落在 | 重启后 |
|---|---|---|---|
| **会话状态 / 实体记忆** | 采问槽位（主诉 / 病程 / 伴随症状 / 既往史 / 用药 / 过敏，含"问过几次"）+ 最近 6 轮对话上文 | **Postgres `clinical.sessions`**（一行一会话，可覆盖）；进程内另留一份缓存 | **还在**（`/consult/status` 的 `session_store` 会写明） |
| **长期记忆 / 病历** | 患者、每轮就诊（主诉 + 结构化结论 JSON）、消息、就诊↔挂号关联 | **Postgres `clinical`**（`patients` / `encounters` / `messages` / `bookings`） | **还在**（备份 / PITR 走数据库） |
| 待确认挂号单 | 科室 / 日期 / 时段 / 费用 / 余量 | **进程内**（一次性、15 分钟过期） | 丢了——它是**提议**，不是记录 |

记录库不可达时三层一起降级：会话与就诊记录都只留在进程内（`/consult/status` 会显示
`session_store: in-process` 与 `persistence_backend: NoopPersistence`），问诊本身不受影响。
`/consult/reset` 会**同时**清内存与库里那一行——只清内存的话，下一次请求会把旧槽位捞回来。

## 目录结构

```
src/kubedocs/
  config.py  cli.py  ingest.py  parse.py  embed.py  normalize.py  generate.py
  store/        # pgvector · sqlite-vec · sqlite
  retrieve/     # 混合检索、RRF、重排
  domains/      # k8s_docs · clinic（各自带语料、配置与工具；clinic 带号源与挂号）
  agent/        # 工具注册表、ReAct 循环、确定性图（对照臂，不在请求路径上）
  clinical/     # 唯一装配入口 runtime.py · 证据层（参考标准面板）· 提示词 · 约束层（事实校验）· 安全门 · 采问 · 就诊记录
  api/          # FastAPI 端点与页面
  mcp/          # MCP server（stdio）：把工具注册表暴露给外部 MCP 客户端
  evaluation/   # 评测记录、参考目标与基线 · 长期监听（monitor.py）
docs/           # 架构、评测、治理、设计决策
scripts/        # 摄取、评测、监听、运维脚本（各脚本内 --help 有说明）
tests/          # 单元测试
```

## 扩展方式

- **加一个域**：新增一个域包（语料路径、嵌入模型、检索器、工具），在 `domains.load_all()` 注册一行。
- **加一个工具**：在 `agent/tools.py` 的 `ToolRegistry` 注册一个 `ToolSpec`（含"什么时候不该用"）。
  同一个注册表同时供给 HTTP 端点、Agent 循环与 MCP server（`kubedocs mcp`），不写第二份定义。
- **加一种可摄取格式**：在 `parse.py` 的解析器注册表加一条，并同步 `chunker.DOC_SUFFIXES`
  （有测试钉住"发现得了就必须解析得了"这个不变式）。
- **加一步编排**：需要在服务端固定顺序里插入一步（采问、取证、判定、落库）时，改
  `clinical/orchestration.py` 的工具循环，而不是在提示词里堆约束。
- **调约束层判据**：改 `clinical/verify.py`；严重度决定处置（`block` 换回答、`strip` 丢一条、
  `warn` 只留痕），改判据时要想清楚它该落在哪一级。
- **换检索/存储后端**：实现 `Store` 协议或重排接口，在配置里切换即可。
- **调切块尺寸**：改 `KUBEDOCS_CHUNK_MAX_CHARS` / `KUBEDOCS_CHUNK_PARENT_MAX_CHARS` 后
  **重建索引**——块边界与块数是摄取时定下来的。

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

代码以 [MIT 许可证](LICENSE) 发布。`data/eval/prototype/` 下是第三方派生数据，
不在 MIT 覆盖范围内，仅用于研究与评测 —— 完整清单与边界见 [NOTICE.md](NOTICE.md)。
