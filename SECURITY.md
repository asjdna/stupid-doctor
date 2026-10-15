# 安全策略

本文件写**这个仓库当前真实的安全状态**：哪些攻击面已经封过、哪些还开着、部署者必须自己补什么。
所有结论都标了**代码出处**（`文件:行`），可以逐条核对；没有核对过的写"未验证"。

> 本项目是一个**尚未上线的工程作品**：仓库无 remote、CI 从未在真实 runner 上跑过
> （`.github/workflows/ci.yml` 存在但未执行）。**容器栈已在 2026-10-13 实测**：
> `docker compose up -d` 起 db + api、健康检查通过、`/healthz` 报 18811 块、
> `/metrics` 与 `/search`、`/chat` 的 SSE 红线都验过（见 README「Docker 全栈实测」）；
> 那次实测抓到 3 个缺陷（容器内 `repo_root` 解析错、`HF_HOME` 非 root 不可读、
> 依赖层与源码层耦合）**都已修**。
> 仍请按"未经过生产验证的代码"对待它。

> **本次修订说明**：下面所有 `文件:行` 锚点已按当前 `main` 重新核对（本轮改动集中在
> 红线判定、挂号确认与归属、预算账本原子性、评测闸门字段校验、clinic 域接线、
> 观测接线）。若行号仍然对不上（文件又被改过），请**按符号名搜索**——每一条引用里
> 都带了函数名 / 字段名 / 类名，那才是稳定的锚点。

---

## 1. 威胁模型与适用范围

**在范围内**：`src/kubedocs/**` 里的 HTTP 服务（`api/`）、agent 工具循环（`agent/`）、
检索（`retrieve/`）、带引用回答（`generate.py`）、评测闸门（`evaluation/`）。

**不在范围内**：

- 上游 Kubernetes 官方文档的**内容**本身（由 `scripts/fetch_corpus.py` 拉取，非本项目产出）；
- clinic 域的知识卡片语料（第三方派生数据集，上游标注"仅用于学术研究"，
  见 `.gitignore:53-54`，**不入库**）；
- 基础镜像与第三方依赖自身的漏洞（`Dockerfile` 用的是 `python:3.10-slim`、
  数据库用 `paradedb/paradedb:0.26.1-pg16`（PG16 + pgvector + pg_search））。

**信任边界**：本项目**没有**认证层，因此当前的真实边界是**网络位置**——
能连到 8080 端口的人就是"合法调用方"。见缺口 S-1。

---

## 2. 已知缺口（按"能造成多大后果"排序）

### S-1 全部 HTTP 端点无认证、无鉴权（严重，未修复）

`src/kubedocs/api/app.py` 的 9 个端点全部没有认证：`GET /healthz`、`GET /metrics`、
`GET /domains`、`GET /tools`、`POST /search`、`POST /ask`、`POST /chat`、`GET /`、
`GET /console`（`api/app.py:649-763`）。

代码可核的"没有"：全仓 `src/` 下搜不到 `CORSMiddleware` / `TrustedHostMiddleware`，
也搜不到任何 `Depends(...)` 形式的凭据校验（唯一的中间件是观测用的
`_observe_requests`，它只计数、不鉴权）。`Settings` 里也没有任何
"API key / token / 允许来源"字段（`config.py` 的 `Settings` 字段定义）。

后果（按端点）：

| 端点 | 匿名调用者能做什么 |
|---|---|
| `POST /ask` | 消耗部署者的 LLM 额度（真实计费调用） |
| `POST /chat` | 走 LLM 工具循环；且可触达**有副作用**的 `book_appointment` |
| `POST /search` | 消耗 CPU（嵌入 + 全文检索），无频率上限 |
| `GET /metrics` | 泄露内部指标（端点分布、成本、预算上限） |
| `GET /tools`、`GET /domains` | 泄露内部工具与域的清单（自省信息） |

**当前缓解到什么程度**：`/chat` 的挂号工具用进程内替身号源
（`MockAppointmentProvider`，`api/app.py:654-656`），所以现在被占用的只是内存里的假号源，
不是真实 HIS。换成真实号源库后这条立刻变成"任何人都能替别人取号"。

**部署者必须做的**：把服务放在反向代理 / API 网关之后，由网关做认证与授权；
或者在 `create_app()` 里自己加一层 `Depends` 校验。本仓库**不提供**开箱可用的认证。

### S-2 `/chat` 路径的 LLM 成本上限（**已修复**，残留一条多副本限制）

修复前：仓库里**有**成本闸门，但它只覆盖一条路径：

- `generate.DailyBudget` 在 `Answerer.answer()` 里 `check()`，这条路径 = CLI `ask` 与
  `POST /ask`；
- `POST /chat` 走 `agent/clinic_react.py` → `agent/llm_tool_caller.py`：
  `LiteLLMToolCaller` 只**累加** `self.cost`，**从不与 `llm_daily_budget_usd` 比较**；
  而 `ClinicReactRuntime.turn()` 每一轮都新建一个 caller，累加值随对象一起丢弃。
  ⇒ `/chat` 的花费**没有日上限**。

**现在的形态（逐条对应修法）**：

- **闸门接进工具循环**：`run_tool_loop(..., budget=...)` 在**每次调用模型之前**取额度、
  **调用之后** `budget.record(...)`（`tool_loop.py` 的 `SpendGuard` 协议 +
  `tool_loop.py:103` 的 `_budget_slot`）。超限时那一次模型调用**根本不发生**，
  异常**原样上抛**（不落进循环里那个"LLM 是外部服务"的 `except`，
  也不变成一条 note——`/chat` 的响应里看不到 notes）。
  预算对象由 `ClinicReactRuntime` 持有（长寿命）；构造点在 `clinic_react.py:136`，
  用的是 `DailyBudget`，与 CLI `ask` / `POST /ask` 同一本账。
- **一本账**：账本路径是 `config.py:306` 的 `llm_spend_path`——**全局一份**，
  不在域目录下。此前写成 `<index_dir>/<domain>/llm_spend.json`，后果是
  "切换 domain 就能把上限翻成 N 份"，而且 clinic 会话的花费被记进 **k8s_docs** 目录
  （审计 `data/index/clinic/` 的人看不到）。磁盘上曾同时存在
  `data/index/llm_spend.json` 与 `data/index/k8s_docs/llm_spend.json` 两本账，
  已合并求和。
- **算不出价不许按 0 记**：`completion_cost()` 对未登记模型会抛异常
  （实测 litellm 1.102.1：`Exception: This model isn't mapped yet`），
  原先把它吞掉并按 0 记 ⇒ 账本永远是 0 ⇒ 闸门永远不触发。
  现在 `LLMResult.cost_priced` / `ToolCaller` 返回的 `cost_priced=False` 明确表达
  "这次算不出价"，`DailyBudget.record(priced=False)` 按
  `generate.py:228` 的 `UNPRICED_CALL_FALLBACK_USD`（默认 **$0.25**）**保守计入**，
  并单独计数（`unpriced_calls_today()`）、打一条 warning，
  `/chat` 还在 SSE 的 `message` 事件里回 `unpriced_llm_calls`。
  "真免费"（本地模型定价就是 0）与"算不出价"是两件事，前者仍是 `cost_priced=True`
  （实测 `ollama/llama3` → `0.0`）。可用 `KUBEDOCS_LLM_UNPRICED_CALL_USD=0` 关掉罚金
  （仍计数）。
- **读写原子性（这条此前写错了，现在才真的成立）**：账本的**读也在锁内**，
  写入是 **`tmp` + 原子替换**（`generate.py:297` 的 `_atomic_replace`）。
  此前 `record()` 用 `write_text`（先截断再写）且 `spent_today()` 无锁，
  并发读者会读到空文件 ⇒ `JSONDecodeError` ⇒ 当作"今天没花过" ⇒ **闸门放行**
  （fail-open，已实测复现）。修复后实测：3 写 3 读并发 900 次记账，
  读者读到空文件 **0** 次、异常 **0** 次。注意 Windows 上 `os.replace`
  在目标被其他句柄打开时会失败，所以那里有重试 + 退路。
- **`check()` 与 `record()` 之间的窗口**：`generate.py:422` 的 `reserve()`
  把"在途调用"按保守金额计入本次判断。实测：上限 $1.0、每次预留 $0.3 时，
  10 个并发请求只放行 **3** 个（修复前会全部放行再各自记账，冲过上限）。
- **跨进程仍未原子**，如实留作限制：多 worker / 多副本时真实上限仍是
  `副本数 × 日预算`。不修的取舍写在 `DailyBudget` 的 docstring 里
  （跨进程锁要按平台分叉导入，或引入"崩溃后锁文件残留 = 闸门永久拒绝一切调用"这种
  **比丢一次计费严重得多**的失败形态）。当前部署形态是单进程
  （`uvicorn kubedocs.api.app:app`，无 `--workers`）。

**仍然没解决的部分（别高估这个闸门）**：`/chat` 依然**没有速率限制**（见 S-3），
所以"日上限"只限制总花费，不限制一个匿名调用者把预算在几分钟内烧完。

### S-3 无速率限制（未修复）；`/chat` 会话状态的属主与上界（**已修复**）

全仓没有任何限流/并发上限代码（令牌桶、信号量、队列都不存在）。
这条**没修**，也不在本次范围内（认证/限流是下一批）。

`/chat` 的会话历史挂在进程内存里，键是**客户端提供的 `thread_id`**。
修复前它有两个性质，现在各自被封掉：

- ~~**没有属主校验**~~（**已修复**）：`thread_id` **首见即绑定 `patient_id`**
  （`api/threads.py` 的 `ThreadStore.claim`），换一个 `patient_id` 用同一个
  `thread_id` ⇒ **409** + 可读原因；**不静默当新会话、也不静默接到别人的历史上**。
  属主判定放在**端点里**（`api/app.py` 的 `chat()`），响应头还没发出，
  所以能给真正的 4xx，因此工具循环路径与确定性图路径受同一道检查。
  `owner_of` / 拒绝理由都**不回显属主是谁**（否则拒绝本身就成了一次身份探测）。
- ~~**没有上界**~~（**已修复**）：现在是 LRU 有界表
  （`MAX_THREADS=256` 个会话 × 每会话 8 条消息）。选"条数上界"而不是 TTL 的理由：
  要防的是内存无界增长，条数界是**确定性**的，TTL 不是。

**诚实边界（这条绑定不是认证）**：属主 = `patient_id`，而它的默认值是 `"anon"`
（`api/schemas.py`）。所以所有**不传** `patient_id` 的匿名客户端共享同一个属主桶，
它们之间仍然只靠 `thread_id` 的随机性（32 位十六进制）隔离。
这条绑定挡的是**误撞 / 复用别人的 thread_id**，挡不住有意的身份伪造。

**一条仍然部分成立的"无界"性质，如实记下**：无 LLM 密钥时 `/chat` 回退到确定性图，
那份会话状态按 `thread_id` 累积。属主校验对它有效（校验在端点里），
但**淘汰对它无效**（那是一份独立的 checkpointer 表）。
更强的一条：**LRU 淘汰会连同 owner 一起丢掉**（`api/threads.py` 的 `_evict_locked`），
而 checkpointer 里那份会话状态仍在——淘汰之后重新 `claim` 同一个 `thread_id` 会被接受，
于是属主校验不再保护那份**仍被保留**的病情状态。触发条件较窄（需要先发生 LRU 换出，
且服务端生成的 thread_id 是 32 位十六进制），但它是真实存在的窗口。

**2026-10-14 起这条的性质变了一半**（借用参考项目的"外部记忆"做法）：新增
`KUBEDOCS_CLINIC_STATE_DB`，配了它就让图状态落 SQLite，**进程重启后可续跑**。
因此上面那个窗口：
· **默认（留空）仍是内存态**，与以前逐字相同，窗口不变；
· **配了之后窗口变大**——那份病情状态不再随进程退出而消失，而是留在盘上，
  而 LRU 淘汰与属主表都还是内存里的。⇒ **配了它就必须同时处理"谁能读到那个文件"**：
  文件权限、以及多机构隔离（后者仍未实现，见 `docs/design/0001-医疗辅助诊断产品方案.md`）。
  它存的是**会话状态**，不是病历表；真正的患者数据模型仍是占位（`clinical/persistence.py`）。

**输入体积本身是有界的**（这部分做得对）：`q ≤ 300`、`top_k ∈ [1,20]`、
`message ≤ 2000`、`thread_id ≤ 64`、`extra="forbid"`（`api/schemas.py`）。
所以原先暴涨的是**键的数量**，不是单个请求的体积；现在键的数量也有上界了。

### S-4 分诊（症状 → 科室）指标实测为 0（质量缺口，医疗域特有）

这不是"被攻击"，而是**在医疗场景下会被当成安全问题**的能力缺口：系统给出错误科室
等于给出错误指引。实测数字（`data/eval/clinic_baseline.json`，已入库）：

| 指标 | 值 | 出处 |
|---|---|---|
| 红线（急诊识别）召回 | **1.0**（tp 9 / fn 0） | `clinic_baseline.json:8` |
| 红线误报率 | 0.0（fp 0 / tn 17） | `clinic_baseline.json:9` |
| 挂号意图准确率 | 1.0（6/6） | `clinic_baseline.json:18` |
| **分诊准确率** | **0.0（hit 0 / total 5，`evaluated=true`）** | `clinic_baseline.json:25` |

分诊 5 条用例的真实返回：4 条**一个候选都没给出**，1 条给出了 3 个错误科室
（`t04 期望 皮肤科，实际 ['急诊科','感染科','肛肠科']`）——后者不是"没答"，
而是**答错且看起来很有把握**，这比空结果更危险。

根因是**可核的数据级问题**，不是调参问题：知识卡片 `data/clinic/cards.json` 共
**31,592** 张（symptom 12,577 / disease 14,726 / drug 4,289），而症状词典
`src/kubedocs/domains/clinic/data/symptom_dict.py` 与之**两套词汇不重合**
（证据键覆盖率 1/6）。`data/clinic/` 被 `.gitignore:53-54` 排除、仓库不携带，
但**本机存在**——所以这条红不是"缺语料"，而是"两套词汇对不上"。

评测器对这件事的处理是**诚实**的：缺语料时 `evaluate_triage()` 返回
`evaluated=False`（`evaluation/clinic.py`），而不是把"答不出来"计成 0 分。

**第二件事（已修复，行为反转了，别按旧文档推理）**：闸门现在**看**分诊，
而且**看红线有没有被评过**：

- 分诊**未评测**（`evaluated=False`，或评测了却没有一条可评测用例）⇒ **闸门失败**；
- 分诊准确率 **< 60%**（`DEFAULT_TRIAGE_MIN_ACCURACY`）⇒ **闸门失败**；
- ~~红线召回在分母为 0 时返回 1.0~~ ⇒ 现在 `evaluation/clinic.py:118` 的
  `redline_recall` 在分母为 0 时返回 **`None`**，`gate_failures()` 把"红线未评测"
  判为失败（`evaluation/clinic.py:296`）。修复前删掉全部 9 条
  `expect_red_line=true` 用例，输出是「红线召回 **100.00%** (TP=0 FN=0)」——
  一个"没有检验任何红线"的运行拿到了安全通过。**空集不得满足安全闸门。**
- clinic 基线现在**绑定金标指纹**（`evaluation/clinic.py:174` 的
  `clinic_gold_sha256`，写入见 `scripts/eval_clinic.py:179`）。此前只存指标，
  换掉/删掉用例与"变好了"在闸门眼里长得一样。

**⇒ 所以 `python scripts/eval_clinic.py --gate` 现在**（修复后实测）
**退出码 2、打印「闸门：不通过」**。**这个红是预期的、也是对的**：
基线里本来就记着 `triage.accuracy = 0.0`。

**基线不许为了让闸门变绿而改**。本次给基线**增加了 `gold_sha256` 与 `n_cases`
两个绑定字段，没有改动任何指标值**（这属于"把尺子钉住"，不是"把红灯涂成绿灯"）。

**红线的 1.0 也不要高估**：那是自建规则表在自建用例上的成绩，规则与用例同一作者、
同一时期（`data/eval/clinic_gold.jsonl` 26 条红线用例），**不是独立临床验证**。

### S-5 提示词注入与工具滥用（中等，部分缓解）

- 检索到的片段与工具返回结果都被**直接拼进提示词**（`generate.py` 的
  `build_context` 调用点、`tool_loop.py` 的 `_tool_message`）。语料来自
  kubernetes.io（受信度较高），但 clinic 卡片是第三方派生数据。
- `/chat` 的控制流由模型决定。副作用工具的执行闸门是
  `tool_loop.py:460` 的 `_confirm_side_effect()`：**默认拒绝**，
  只有 `confirm` 回调点头才放行。而 `/chat` 传入的回调是
  `(lambda _call: True) if user_asked_to_book else None`（`clinic_react.py:216`）——
  也就是说，只要用户这一轮的原文里出现挂号意图，闸门对该轮所有副作用工具全开。
  **这道闸门防的是"模型自作主张"，不防"调用方未经授权"**（见 S-1）。
- **挂号意图判定已能识别否定**（`domains/clinic/pack.py` 的 `wants_booking`）：
  修复前 `"挂号" in "我不想挂号"` 为真，用户在**明确否认**时也会被占号
  （图路径实测出过票）。同时图路径现在**必须先确认**才下单
  （`agent/kernel.py:193` 的 `node_confirm` + `kernel.py:238` 的 `_after_confirm`）：
  关键词命中只证明"提到了挂号"，不证明"同意占用号源"。
- **号源归属不再是模型能编的**：`patient_id` 由调用方经
  `run_tool_loop(context=...)` 注入（`agent/tool_loop.py:489` 的 `_invoke`），
  **不进工具 schema**（`BookParams` 只有 department/date/time_slot），
  所以模型无法伪造"这是谁"。修复前它压根传不到执行体，
  **所有号都记在 `"anon"` 名下**（实测确认）。
- 已缓解的部分：
  - 工具参数逐个走 pydantic 校验（`model_validate`，`tool_loop.py:425`）；
  - 工具名必须命中注册表（`_lookup`，`tool_loop.py:448`）；
  - 重名直接报错而不是静默覆盖（`registry`，`tool_loop.py:289-290`）；
  - 工具执行体（号源三件套 / 检索）不接受任意 URL 或文件路径，
    因此**没有 SSRF 或任意文件读取面**；
  - 模型输出只当文本用，**没有** `eval` / `exec` / shell 调用
    （全仓无 `subprocess` / `shell=True` / `pickle` / `yaml.load` 使用）。

### S-6 依赖与供应链（未验证）

- `requirements.lock.txt` 是**本机环境**的闭包快照（由 `scripts/gen_lock.py` 生成），
  它**不含哈希**，也不做版本求解。**CI 三个 job 都跑 `pip install -e ".[dev]"`，
  从不安装这个锁文件**，所以 CI 拉的是"当时的最新版本"而不是锁定的集合——
  复现性由 `make lock` + `gen_lock.py --check` 本地保证，CI 不消费它。
- 锁文件生成时 **`eval` optional 组（`ragas` / `deepeval`）未安装，因此未纳入**
  （header 里有明写）。这两个包在当前代码里**没有任何 import**，
  属于"声明了但没用上"。
- 本次**删除了 `redis` 依赖**（以及 `types-redis`）：全仓零处使用它，
  与它配套的 `Settings.redis_url`、compose 的 redis 服务一并删除。
  `prometheus-client` 与 `opentelemetry-*` 则从"声明了没人 import"变成**真实使用**
  （见 `observability.py`）。
- 无法在离线环境验证"每个依赖都能从 PyPI 获取"。CI 里**没有**依赖漏洞扫描
  （无 `pip-audit`，Dependabot 告警未启用）。唯一沾边的是 `ruff` 的 `S` 规则集
  （flake8-bandit 子集）在阻塞门禁里（`pyproject.toml` 的 `[tool.ruff.lint] select`），
  它抓的是代码模式，不是依赖 CVE。

---

## 3. 已经封闭的攻击面（不是漏洞；列出来是为了说明哪些面被专门处理过）

| 面 | 现状 | 代码出处 |
|---|---|---|
| FTS5 `MATCH` 表达式注入 | 只抽取词元并逐个双引号短语化，用户输入 `pod -limit` / `"` 不再改变语法、不再 500 | `store/textquery.py` 的 `build_match_expr`（含对应单测） |
| 引用越界（模型编 `[99]`） | 越界标记被**剥除**并记入诊断；一个引用都没有也会被标记。正则已放宽到 `\d+`——此前 `\d{1,3}` 漏掉 `[1234]`，让它**静默留下** | `generate.py:63` 的 `_MARKER_RE`、`generate.py:199` 的 `validate_citations` |
| 检索为空却调 LLM | 直接拒答，**不调用模型**（判据是 `if not hits`；单测断言打桩 LLM 一次都没被调用） | `generate.py:738` |
| **急诊红线每一轮都判** | 此前只在**首轮** `question` 上判，续答轮被直接送回 `observe` ⇒ 用户在回答追问时才说出"胸痛"永远不会转急诊。现在 resume 轮也先过红线 | `agent/kernel.py:116` 的 `node_redline`、`kernel.py:243` 的 `_after_ask` |
| **挂号必须显式确认** | 关键词命中不等于同意占用号源；图路径新增 `confirm` 节点（真 `interrupt` 挂起），未确认绝不进 `book` | `agent/kernel.py:193`、`kernel.py:238`、`agent/state.py:127` 的 `booking_confirmed` |
| **号源归属来自调用方** | `patient_id` 经 `context=` 注入执行体，**不在工具 schema 里**（模型无法伪造） | `agent/tool_loop.py:489`、`agent/clinic_react.py:226` |
| 存储层故障降级 | 向量路不可用时（嵌入**或**存储抛错）退化为纯全文并记录原因，不再把 `StoreError` 直接抛给用户 | `retrieve/hybrid.py:131` |
| **向量与块行错位不可静默存在** | `_reload_index` 校验 `vector_row` 连续性与矩阵行数，不一致就大声失败（成因通常是"纯删除的一轮 ingest 没调 `finalize()`"） | `store/sqlite_store.py:281`、`ingest.py:222` |
| **评测闸门缺字段 = 拒绝比较** | 基线缺 `top_k`/`n_chunks`/`corpus_sha`/`metrics`/`gold_sha256`/`embedding_model`/`schema` 任一即拒绝，不再"缺了就跳过校验"（那正是 `top_k` 假绿洞的成因） | `evaluation/harness.py:219-227` |
| **跨语料比较被精确拦住** | `corpus_sha`（`(rel_path, n_chunks)` 的 sha256，两个后端都按 `ORDER BY rel_path` 迭代 ⇒ 确定性）不等即拒绝，逃生口是 `--allow-corpus-drift`（放行但留警告）。此前它**只被计算与落盘、从未被比较**，实际把关的只是 2% 的块数容差 ⇒ 容差内的语料变化被**静默**放过（实测踩过：+0.058%） | `evaluation/harness.py:268-278`、`harness.py:461` 的 `_corpus_fingerprint` |
| 请求体越界 / 未知字段 | pydantic 校验，越界 422、`extra="forbid"` | `api/schemas.py` |
| 工具同名冲突 | 注册表（`registry`）按名索引，重名直接报错而不是静默覆盖 | `agent/tool_loop.py:289-290` |
| 副作用工具默认行为 | **默认拒绝执行**，不是默认执行后再回滚（见 `_confirm_side_effect`） | `agent/tool_loop.py:460` |
| 空索引 / 索引损坏 | 不返回 500、不泄露 traceback：记 `index_error` 并返回 200 + 说明 | `api/app.py:143-153` |
| 容器内权限 | api 以非 root（`USER app`，uid 10001）运行 | `Dockerfile:147` |
| **观测开关不再是假开关** | 日志级别真正生效（`configure_logging`）；OTel 真正建 provider（`setup_tracing`）；`/metrics` 真实存在（`metrics_payload`）。此前这三个字段全仓无人读取 | `observability.py:77`、`observability.py:100`、`observability.py:148` |

---

## 4. 部署者最低要求（照做，否则上面的缺口会直接变成事故）

1. **不要把它直接暴露到公网。** 先补认证（S-1），再考虑任何形式的对外服务。
2. **密钥必须由环境显式提供**，且只放在 `.env` 或容器环境变量里：
   - 密钥读取路径是 `os.environ`（`generate.py:529` 的 `required_api_key_env()` /
     `generate.py:565` 的 `has_api_key()`）；`.env` 由 `config.load_env_file()` 导入
     `os.environ`，**真实环境变量优先、`.env` 不覆盖已有值**
     （`config.py:67` 的 `override=False`）。
   - `.env` 里**留空的键等于未设置**（`config.py:78`）：这防止 `.env.example` 里成片的
     `KEY=` 把"没有密钥"伪装成"有密钥"。
   - **占位/说明文本不算密钥**（`generate.py:541` 的 `_PLACEHOLDER_MARKERS` +
     `generate.py:565` 的非 ASCII 判据）。两道判据缺一不可，因为**本机 `.env` 里
     真实存在的那段值两道都踩到了**：它是一句中文说明（38 字符 / 78 字节，
     20 个非 ASCII 字符），一个占位符关键词都不含——而 `has_api_key()` 原先只判
     "非空"，于是返回 True，LiteLLM 把这段值放进 HTTP 头时抛
     `'ascii' codec can't encode characters in position 9-24`。用户看到的是
     "模型报错"，指不到"你 `.env` 里写的是一句中文"，**比直接降级更坏**。
     任何真密钥都是可打印 ASCII（要进 `Authorization` 头），所以第二条判据不会误伤。
3. **`.env` 永不入库。** 它已被 `.gitignore` 覆盖；`docker-compose.yml:103` 用
   `env_file: .env` 把密钥注入 api 容器——它不会被打印，
   但会出现在 `docker inspect` 的输出里，共享宿主机的场合要注意。
4. **钱包护栏**：`KUBEDOCS_LLM_DAILY_BUDGET_USD` 覆盖 `/ask`、CLI `ask` **和
   `/chat` 的工具循环**，且是**一本全局账**（`config.py:306` 的 `llm_spend_path`）。
   但仍有两条要你自己加：
   - **速率限制不在本项目里**（S-3），日上限只限制总额、不限制烧得多快；
   - **多副本下上限会变成 `副本数 × 预算`**（S-2 的残留限制）。
   `KUBEDOCS_LLM_UNPRICED_CALL_USD` 控制"模型算不出价"时每次的保守计入金额
   （默认 0.25 美元；置 0 = 不计金额但**仍然计数**）。
5. **离线 / 在线的差别**（这决定了"没有密钥能不能用"）：
   - **完全离线可用**：单测、检索、评测、闸门、索引体检。嵌入用
     `KUBEDOCS_EMBEDDING_PROVIDER=hashing`（确定性哈希，仅供管道自检，
     **不代表检索质量**）或已缓存的 `bge` + `KUBEDOCS_HF_OFFLINE=true`。
   - **受限网络**：`huggingface.co` 不可达；用 `KUBEDOCS_HF_ENDPOINT=https://hf-mirror.com`
     拉模型，之后置 `KUBEDOCS_HF_OFFLINE=true`（模型已缓存时联网校验会**挂死** >10 分钟）。
   - **只有"生成回答"需要外网**：没有密钥时 `/ask` 返回检索结果 + 降级说明、
     `/chat` 回退到确定性图（红线与挂号仍可用），都不会崩
     （`api/app.py:334` 的 `has_api_key` 分支）。
6. **数据合规**：`data/clinic/` 的知识卡片不入库、且上游标注"仅用于学术研究"。
   要用在真实问诊场景，**先自己解决该数据集的授权**，本仓库不代为授权。
7. **公网部署前至少替换**：`docker-compose.yml:62` 的 Postgres 口令
   （`kubedocs/kubedocs`）。
   compose 把 Postgres 的 `5432` 与 api 的 `8080` 映射到宿主机
   （端口映射见 `docker-compose.yml:123`）。
   本次已移除 redis 服务与它的 `6379` 映射、以及未使用的 langfuse 服务
   （连同 `NEXTAUTH_SECRET` / `SALT` 占位值）——它们此前是"没人连的容器"。

---

## 5. 怎么上报

请使用 **GitHub Private vulnerability reporting**（仓库页 → `Security` →
`Advisories` → `Report a vulnerability`）。这是本仓库唯一的接收渠道，
同时也是**私密**渠道：请**不要**在公开 issue / PR 里贴出可利用细节。

请在报告里给出：受影响的文件与版本（commit 短哈希）、复现步骤、
你观察到的实际影响、以及你希望的处理方式。如果 7 天内没有回音，可以再开一个
**不含技术细节**的公开 issue 提醒一下。

本仓库**没有**安全邮箱列表（也不打算编一个出来）：Private reporting 是平台能力，
不依赖任何个人联系方式。

---

## 6. 未验证事项（诚实清单）

- **容器镜像的"从零重建"在本机没有跑完**：`docker compose up -d` 与容器内的行为已实测
  （见 README「Docker 全栈实测」），但那次验证是在**现有镜像里挂载当前源码**跑的
  ——因为重建镜像需要的外网大文件下载在本机会**反复停滞**（同一构建里 tuna 能把 553 MB
  的 torch 拉完，另一条 184 MB 的 wheel 却卡死 6 分钟零字节）。
  所以：`Dockerfile` / `docker-compose.yml` 的**修复已就位并被上述验证覆盖**，
  但"干净环境一次构建成功"这条**没有**在本机跑通。
- **基础镜像仍是浮动 tag**：`python:3.10-slim` 实测解析到 Debian 13（trixie），
  与 compose 里"版本都钉死"的纪律不一致。没修（换 tag 会影响运行时依赖），如实记下。
- **镜像里的依赖不等于锁文件**：Docker 构建走 `pip install .`，**不消费
  `requirements.lock.txt`**，实测装到的 `langgraph` / `litellm` / `torch` 都比锁文件新。
- CI（`.github/workflows/ci.yml`）**从未在真实 runner 上执行过**（仓库无 remote）。
- 红线规则只有 26 条自建用例，**没有**外部/临床验证。
- 分诊链路的分数**未测到可信水平**：`data/clinic/cards.json` 存在（31,592 张卡），
  且 clinic 索引已按中文模型全量建成（31,592 块，见 README），
  但 `triage.accuracy=0.0` 的根因是**两套词汇不重合**，
  所以"有了索引分诊就会好"不成立——这条仍然待做。
- **分诊闸门现在是红的（预期）**：`scripts/eval_clinic.py --gate` 退出码 2。
  这条红**没有被"修绿"**。
- **多副本下的 LLM 花费上限未验证**：跨进程账本仍未原子（S-2），
  当前部署形态是单进程所以不生效，但**没有实测过多副本**。
- **pgvector 的集成测试已重跑**（2026-10-13 起，Docker 可达时）：`test_pgvector_store.py`
  **15 条** + `test_store_contract.py` 的 pgvector 参数 **18 条**，合起来
  **33 passed / 0 skipped**（命令：`pytest tests/unit/test_pgvector_store.py tests/unit/test_store_contract.py -k "pgvector or not (sqlite or milvus)"`）。
  此处原先写的是"**90 passed / 0 skipped**"——那个数字与任何一次可复现的命令都对不上，
  已更正。注意这 18 条在库不可达时**不是 skip，而是从参数化里消失**，
  所以"0 skipped"本身不足以证明它们跑过，要看收集到的条数。
  同一天把当前本地索引重灌进 pgvector：全文/向量自检都 ok、抽样 200 块向量
  **逐位一致**、字符总数两边相同，`cli --store pgvector eval --top-k 10 --gate`
  **退出码 0**（hybrid hit@10 0.8600 ≥ 冻结基线 0.8400）。
- **多副本 / 容器化下的 LLM 花费上限仍未实测**（与上一条的多进程限制同源）。
- **上游语料没有钉版本**（新增，2026-10-14 实测暴露）：`scripts/fetch_corpus.py` 从
  codeload 取 `main` 的 tarball，**不记录 commit / 不校验摘要**。于是"照文档重跑一遍"
  会得到**不同**的语料：实测 2026-10-14 重拉后文档 1720→1722、块数 18811→18822
  （`corpus_sha` `f6be8d65…`→`f9ffa225…`）。后果与缓解：
  · 评测的**可比性**已由 `corpus_sha` 精确把关（不等即拒绝，`--allow-corpus-drift` 才放行），
    所以这条**不会**再静默污染指标比较；
  · 但"复现某一次的分数"需要那份语料本身，而它没有可寻址的标识——**这一点仍未解决**。
    想彻底解决就得钉上游 commit 或把语料摘要写进仓库并在抓取时校验。
- **PG / 语料之外，评测链路已无"静默"环节**：闸门缺字段拒绝、跨语料指纹拒绝、
  超预算拒绝、向量与块行错位大声失败（详见第 3 节）——**唯一剩下的静默面就是上面这条语料来源**。
- 依赖的 PyPI 可达性与 CVE 状态**未测**。
- **负载测试做了一半**（2026-10-14 更新）：`scripts/bench_api.py` 已在**单机单进程**上
  量过 `/search` 与 `/healthz` 的并发 × 延迟分位 × 吞吐 × 错误率（数字见 README「性能基准」），
  过程中还抓到并修掉一个真实缺陷（`/healthz` 每请求做一次 17.3 ms 全表 `LIKE` 扫描：
  吞吐 19.0 → 581 req/s，p50 54.6 → 1.6 ms）。**但仍然只做了一半**：
  · 只在一台 16 GB 开发机上，且同时跑着 Docker Desktop（数字不可迁移）；
  · **没有长稳压**——内存泄漏 / GC / 句柄数 / 连接池耗尽都没观测；
  · **没有多副本压测**——而多副本正好会踩到 S-2 的单进程预算账本，两件事必须一起做；
  · **没有压 LLM 路径**（`/ask`、`/chat` 要真密钥且真花钱）。
- **渗透测试与模糊测试仍然从未做过**（这一条没有变）。
