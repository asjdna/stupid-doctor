# 安全策略

本文件写**这个仓库当前真实的安全状态**：哪些攻击面已经封过、哪些还开着、部署者必须自己补什么。
所有结论都标了**代码出处**（`文件:行`），可以逐条核对；没有核对过的写"未验证"。

> 本项目是一个**尚未上线的工程作品**：仓库无 remote、CI 从未在真实 runner 上跑过
> （`.github/workflows/ci.yml` 存在但未执行）、`Dockerfile` 与 `docker-compose.yml`
> 自述"尚未在本机实测"（`Dockerfile:2`、`docker-compose.yml:3`）。
> 请按"未经过生产验证的代码"对待它。

---

## 1. 威胁模型与适用范围

**在范围内**：`src/kubedocs/**` 里的 HTTP 服务（`api/`）、agent 工具循环（`agent/`）、
检索（`retrieve/`）、带引用回答（`generate.py`）、评测闸门（`evaluation/`）。

**不在范围内**：

- 上游 Kubernetes 官方文档的**内容**本身（由 `scripts/fetch_corpus.py` 拉取，非本项目产出）；
- clinic 域的知识卡片语料（第三方派生数据集，上游标注"仅用于学术研究"，
  见 `.gitignore:53-54`，**不入库**）；
- 基础镜像与第三方依赖自身的漏洞（`Dockerfile` 用的是 `python:3.10-slim`、
  `pgvector/pgvector:pg16`、`redis:7.4-alpine`）。

**信任边界**：本项目**没有**认证层，因此当前的真实边界是**网络位置**——
能连到 8080 端口的人就是"合法调用方"。见缺口 S-1。

> 引用约定：出处里的 `文件:行` 对应写这份文档时的 `main`。若行号对不上（文件被改过），
> 请**按符号名搜索**——每一条引用里都带了函数名 / 字段名 / 类名，那才是稳定的锚点。

---

## 2. 已知缺口（按"能造成多大后果"排序）

### S-1 全部 HTTP 端点无认证、无鉴权（严重，未修复）

`src/kubedocs/api/app.py` 的 7 个端点全部没有认证：`GET /healthz`、`GET /domains`、
`GET /tools`、`POST /search`、`POST /ask`、`POST /chat`、`GET /`（`api/app.py:602-696`）。

代码可核的"没有"：全仓 `src/` 下搜不到 `add_middleware` / `@app.middleware` /
`CORSMiddleware` / `TrustedHostMiddleware`，也搜不到任何 `Depends(...)` 形式的
凭据校验（`fastapi.Depends` 在 `src/` 中零出现）。`Settings` 里也没有任何
"API key / token / 允许来源"字段（`config.py` 的 `Settings` 字段定义）。

后果（按端点）：

| 端点 | 匿名调用者能做什么 |
|---|---|
| `POST /ask` | 消耗部署者的 LLM 额度（真实计费调用） |
| `POST /chat` | 走 LLM 工具循环；且可触达**有副作用**的 `book_appointment` |
| `POST /search` | 消耗 CPU（嵌入 + 全文检索），无频率上限 |
| `GET /tools`、`GET /domains` | 泄露内部工具与域的清单（自省信息） |

**当前缓解到什么程度**：`/chat` 的挂号工具用进程内替身号源
（`MockAppointmentProvider`，`api/app.py:317-320`、`domains/clinic/mock.py`），
所以现在被占用的只是内存里的假号源，不是真实 HIS。换成真实号源库后这条立刻变成
"任何人都能替别人取号"。

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

- **闸门接进工具循环**：`run_tool_loop(..., budget=...)` 在**每次调用模型之前**
  `budget.check()`、**调用之后** `budget.record(...)`（`tool_loop.py` 的
  `SpendGuard` 协议）。超限时那一次模型调用**根本不发生**，
  异常**原样上抛**（不落进循环里那个"LLM 是外部服务"的 `except`，
  也不变成一条 note——`/chat` 的响应里看不到 notes）。
  预算对象由 `ClinicReactRuntime` 持有（长寿命），
  与 CLI `ask` / `POST /ask` **共用同一个** `llm_spend.json`：两条路径算同一个钱包。
- **算不出价不许按 0 记**：`completion_cost()` 对未登记模型会抛异常
  （实测 litellm 1.102.1：`Exception: This model isn't mapped yet`），
  原先把它吞掉并按 0 记 ⇒ 账本永远是 0 ⇒ 闸门永远不触发。
  现在 `LLMResult.cost_priced` / `ToolCaller` 返回的 `cost_priced=False` 明确表达
  "这次算不出价"，`DailyBudget.record(priced=False)` 按
  `UNPRICED_CALL_FALLBACK_USD`（默认 **$0.25**，取值推导见该常量）**保守计入**，
  并单独计数（`unpriced_calls_today()` / `DailyBudget` 文件里的 `<日期>:unpriced_calls`）、
  打一条 warning，`/chat` 还在 SSE 的 `message` 事件里回 `unpriced_llm_calls`。
  "真免费"（本地模型定价就是 0）与"算不出价"是两件事，前者仍是 `cost_priced=True`
  （实测 `ollama/llama3` → `0.0`）。可用 `KUBEDOCS_LLM_UNPRICED_CALL_USD=0` 关掉罚金
  （仍计数）。
- **无锁读改写**：**进程内已原子**（`threading.Lock`，按文件路径注册——同一个账本被
  `Answerer` 与工具循环各构造一次，各自一把锁等于没锁）。
  **跨进程仍未原子**，如实留作限制：多 worker / 多副本时真实上限仍是
  `副本数 × 日预算`。不修的取舍写在 `DailyBudget` 的 docstring 里
  （跨进程锁要按平台分叉导入，或引入"崩溃后锁文件残留 = 闸门永久拒绝一切调用"这种
  **比丢一次计费严重得多**的失败形态；换 SQLite 则要放弃账本"人可读、可手改、可审计"）。
  当前部署形态是单进程（`uvicorn kubedocs.api.app:app`，无 `--workers`）。

**仍然没解决的部分（别高估这个闸门）**：`/chat` 依然**没有速率限制**（见 S-3），
所以"日上限"只限制总花费，不限制一个匿名调用者把预算在几分钟内烧完。
真正的配额要等认证/限流那一批，或在网关层做。

### S-3 无速率限制（未修复）；`/chat` 会话状态的属主与上界（**已修复**）

全仓没有任何限流/并发上限代码（`slowapi`、令牌桶、信号量、队列都不存在）。
这条**没修**，也不在本次范围内（认证/限流是下一批）。

`/chat` 的会话历史挂在进程内存里，键是**客户端提供的 `thread_id`**。
修复前它有两个性质，现在各自被封掉：

- ~~**没有属主校验**~~（**已修复**）：`api/app.py` 原先直接拿 `thread_id` 当
  `app.state.clinic_history` 的键，于是猜到/复用别人的 `thread_id` 就能续聊别人的会话
  （历史里是别人的主诉、科室、甚至挂号动作）。
  现在 `thread_id` **首见即绑定 `patient_id`**（`api/threads.py` 的 `ThreadStore.claim`），
  换一个 `patient_id` 用同一个 `thread_id` ⇒ **409** + 可读原因；
  **不静默当新会话、也不静默接到别人的历史上**。属主判定放在**端点里**
  （`api/app.py` 的 `chat()`，响应头还没发出，所以能给真正的 4xx），
  因此工具循环路径与确定性图路径受同一道检查。
  `owner_of` / 拒绝理由都**不回显属主是谁**（否则拒绝本身就成了一次身份探测）。
- ~~**没有上界**~~（**已修复**）：现在是 LRU 有界表
  （`MAX_THREADS=256` 个会话 × 每会话 8 条消息，逐字沿用原先的 `[-8:]` 口径）。
  选"条数上界"而不是 TTL 的理由：要防的是内存无界增长，条数界是**确定性**的，
  TTL 不是（一阵突发流量能在任何 TTL 到期前塞进任意多个线程）。
  见 `api/threads.py`。

**诚实边界（这条绑定不是认证）**：属主 = `patient_id`，而它的默认值是 `"anon"`
（`api/schemas.py`）。所以所有**不传** `patient_id` 的匿名客户端共享同一个属主桶，
它们之间仍然只靠 `thread_id` 的随机性（32 位十六进制）隔离。
这条绑定挡的是**误撞 / 复用别人的 thread_id**（以及"后端把两个患者悄悄合成一个会话"），
挡不住有意的身份伪造——真正的"这是谁"必须由网关层解决（见 S-1）。

同一条"无界"性质在**图路径**上仍然部分成立，如实记下：无 LLM 密钥时 `/chat` 回退到
确定性图，而 `_get_clinic_agent()` 构造 agent 时**没有传 checkpointer**，
于是 `kernel.py` 的默认分支生效（`InMemorySaver()`）——那份状态同样只在内存里、
同样按 `thread_id` 累积。**属主校验对它有效**（校验在端点里），
但**淘汰对它无效**（那是一份独立的 checkpointer 表，本次没有动它）。

**输入体积本身是有界的**（这部分做得对，如实记录）：`q ≤ 300`、`top_k ∈ [1,20]`、
`message ≤ 2000`、`thread_id ≤ 64`、`extra="forbid"`（`api/schemas.py:125-168`）。
所以原先暴涨的是**键的数量**，不是单个请求的体积；现在键的数量也有上界了。

### S-4 分诊（症状 → 科室）指标实测为 0（质量缺口，医疗域特有）

这不是"被攻击"，而是**在医疗场景下会被当成安全问题**的能力缺口：系统给出错误科室
等于给出错误指引。实测数字（`data/eval/clinic_baseline.json`，已入库）：

| 指标 | 值 | 出处 |
|---|---|---|
| 红线（急诊识别）召回 | **1.0**（tp 9 / fn 0） | `clinic_baseline.json:6-7` |
| 红线误报率 | 0.0（fp 0 / tn 17） | `clinic_baseline.json:7` |
| 挂号意图准确率 | 1.0（6/6） | `clinic_baseline.json:16-17` |
| **分诊准确率** | **0.0（hit 0 / total 5，`evaluated=true`）** | `clinic_baseline.json:21-32` |

分诊 5 条用例的真实返回：4 条**一个候选都没给出**（`实际 []`），
1 条给出了 3 个错误科室（`t04 期望 皮肤科，实际 ['急诊科','感染科','肛肠科']`）
——注意后者不是"没答"，而是**答错且看起来很有把握**，这比空结果更危险。

根因是**可核的数据级问题**，不是调参问题：

- 知识卡片 `data/clinic/cards.json` 共 **31,592** 张
  （symptom 12,577 / disease 14,726 / drug 4,289）；
- 但 `data/clinic/` 被 `.gitignore:53-54` 排除，**仓库不携带**，全新克隆的机器上
  这个文件根本不存在；
- 症状词典 `src/kubedocs/domains/clinic/data/symptom_dict.py` 去重后只有 **172** 个词项
  （28 个标准症状名 + 144 个口语变体去重成 146，加 24 个部位词，再加
  `KB_SYMPTOM_ALIAS` 的 4 个词），与卡片 `key` 的重叠是 **102**。

（这组数字我实测过两次并复现：172 词项 / 102 重叠。**曾经有一个"5673 键 / 仅 8 重叠"
的说法，那是错的、复现不出来，不要引用它。**）

评测器对这件事的处理是**诚实**的：缺语料时 `evaluate_triage()` 返回
`evaluated=False`（`evaluation/clinic.py`），而不是把"答不出来"计成 0 分。

**第二件事（已修复，行为反转了，别按旧文档推理）**：闸门现在**看**分诊。
原先 `gate_failures()` 只检查红线漏报与红线误报率——分诊 0 分**不会让闸门变红**，
所以"闸门通过"**不等于"分诊可用"**。现在：

- 分诊**未评测**（`evaluated=False`，或评测了却没有一条可评测用例）⇒ **闸门失败**
  （否则"删掉语料"会变成让闸门变绿的手段）；
- 分诊准确率 **< 60%**（`DEFAULT_TRIAGE_MIN_ACCURACY`，取值理由见该常量的说明：
  候选科室实测 40 个、随机 top-3 上界 7.5%，0.60 是它的 8 倍；在 5 条用例上等价于
  "至少答对 3 条"）⇒ **闸门失败**。

**⇒ 所以 `python scripts/eval_clinic.py --gate` 现在**（修复后实测）
**退出码 2、打印「闸门：不通过」**，而修复前是退出码 0 + 「闸门：通过」。
**这个红是预期的、也是对的**：`data/eval/clinic_baseline.json` 里本来就记着
`triage.accuracy = 0.0`——红色只是把一个一直存在的事实（分诊不可用）
暴露到退出码上，不是本次改动引入的回归。

**基线不许为了让闸门变绿而改**（那是本项目明令禁止的"把红灯涂成绿灯"）。
要变绿只有一条路：真的把分诊做对——补"口语主诉 → 临床体征"的映射层，或换一份
与症状词典词汇重合的语料。这件事仍然在**未做**清单里（见第 6 节）。

**红线的 1.0 也不要高估**：那是自建规则表在自建用例上的成绩，规则与用例同一作者、
同一时期（`data/eval/clinic_gold.jsonl` 26 条），**不是独立临床验证**，
也未在真实急诊分诊数据上评估过。

### S-5 提示词注入与工具滥用（中等，部分缓解）

- 检索到的片段与工具返回结果都被**直接拼进提示词**（`generate.py:534-536`、
  `tool_loop.py:322-327`）。语料来自 kubernetes.io（受信度较高），
  但 clinic 卡片是第三方派生数据。
- `/chat` 的控制流由模型决定（`tool_loop.py` 全文）。副作用工具的执行闸门是
  `_confirm_side_effect()`（`tool_loop.py:409-435`）：**默认拒绝**，只有 `confirm`
  回调点头才放行。而 `/chat` 传入的回调是
  `(lambda _call: True) if user_asked_to_book else None`（`clinic_react.py:211`）——
  也就是说，**只要用户这一轮的原文里出现挂号意图，闸门对该轮所有副作用工具全开**。
  这道闸门防的是"模型自作主张"，**不防**"调用方未经授权"（见 S-1）。
- 已缓解的部分：工具参数逐个走 pydantic 校验（`tool_loop.py:373-378`），
  工具名必须命中注册表，未知工具名会把可用列表回灌给模型（`tool_loop.py:397-405`）；
  工具执行体（号源三件套 / 检索）不接受任意 URL 或文件路径，因此**没有 SSRF 或
  任意文件读取面**；模型输出只当文本用，**没有** `eval` / `exec` / shell 调用
  （全仓无 `subprocess` / `shell=True` / `pickle` / `yaml.load` 使用）。

### S-6 依赖与供应链（未验证）

- `requirements.lock.txt` 是**本机环境**的闭包快照（由 `scripts/gen_lock.py` 生成），
  它**不含哈希**，也不做版本求解。用 `pip install -r requirements.lock.txt`
  复现的是"版本集合"，不是"逐字节相同的产物"。
- 锁文件生成时 **`eval` optional 组（`ragas` / `deepeval`）未安装，因此未纳入**
  （header 里有明写）。这两个包在当前代码里**没有任何 import**（全仓 grep 只命中
  `pyproject.toml` 的 `eval` optional 组与 `docs/adr/0001-技术选型.md:19`），
  属于"声明了但没用上"。
- 无法在离线环境验证"每个依赖都能从 PyPI 获取"：实测能确认的是——本机 141 个包
  **全部由 pip 安装、没有任何 `direct_url.json`**（即没有 VCS / 本地路径 / 目录安装），
  且整个环境里 `pip list` 无 `file:` 或 `@ url` 形式的条目。PyPI 的**可达性**未测试。
- CI 里**没有**依赖漏洞扫描（无 `pip-audit` / Dependabot 告警未被启用）。
  唯一沾边的是 `ruff` 的 `S` 规则集（flake8-bandit 子集）在阻塞门禁里
  （`pyproject.toml` 的 `[tool.ruff.lint] select`），它抓的是代码模式，不是依赖 CVE。

---

## 3. 已经封闭的攻击面（不是漏洞；列出来是为了说明哪些面被专门处理过）

| 面 | 现状 | 代码出处 |
|---|---|---|
| FTS5 `MATCH` 表达式注入 | 只抽取 `\w+` 词元并逐个双引号短语化，用户输入 `pod -limit` / `"` 不再改变语法、不再 500 | `store/textquery.py:92-112`（含对应单测） |
| 引用越界（模型编 `[99]`） | 越界标记被**剥除**并记入诊断；回答里一个引用都没有也会被标记 | `generate.py:193-214` |
| 检索为空却调 LLM | 直接拒答，**不调用模型**（可断言：单测断言打桩 LLM 一次都没被调用） | `generate.py:509-514` |
| 请求体越界 / 未知字段 | pydantic 校验，越界 422、`extra="forbid"` | `api/schemas.py:125-168` |
| 工具同名冲突 | 注册表按名索引，重名直接报错而不是静默覆盖 | `tool_loop.py:250-256` |
| 副作用工具默认行为 | **默认拒绝执行**，不是默认执行后再回滚 | `tool_loop.py:409-423` |
| 空索引 / 索引损坏 | 不返回 500、不泄露 traceback：记 `index_error` 并返回 200 + 说明 | `api/app.py:129-145, 241-252` |
| 容器内权限 | api 以非 root（`USER app`，uid 10001）运行 | `Dockerfile:50, 60` |

---

## 4. 部署者最低要求（照做，否则上面的缺口会直接变成事故）

1. **不要把它直接暴露到公网。** 先补认证（S-1），再考虑任何形式的对外服务。
2. **密钥必须由环境显式提供**，且只放在 `.env` 或容器环境变量里：
   - 密钥读取路径是 `os.environ`（`generate.py:404-417` 的 `required_api_key_env()` /
     `has_api_key()`）；`.env` 由 `config.load_env_file()` 导入 `os.environ`，
     **真实环境变量优先、`.env` 不覆盖已有值**（`config.py:35-63`，`override=False` 的语义）。
   - `.env` 里**留空的键等于未设置**（`config.py:59-61`）：这防止 `.env.example` 里成片的
     `KEY=` 把"没有密钥"伪装成"有密钥"。
3. **`.env` 永不入库。** 它已被 `.gitignore:1-4`（`.env` / `.env.*` / `!.env.example`）覆盖；
   提交前请自己再确认一次（`git status --short` 里不应出现 `.env`）。
   `docker-compose.yml:48-49` 用 `env_file: .env` 把密钥注入 api 容器——它不会被打印，
   但会出现在 `docker inspect` 的输出里，共享宿主机的场合要注意。
4. **钱包护栏**：`KUBEDOCS_LLM_DAILY_BUDGET_USD` 现在覆盖 `/ask`、CLI `ask` **和
   `/chat` 的工具循环**（S-2 已修复）。但仍有两条要你自己加：
   - **速率限制不在本项目里**（S-3），日上限只限制总额、不限制烧得多快；
   - **多副本下上限会变成 `副本数 × 预算`**（S-2 的残留限制），
     多副本部署要么在网关层做配额，要么把账本换成共享存储。
   `KUBEDOCS_LLM_UNPRICED_CALL_USD` 控制"模型算不出价"时每次的保守计入金额
   （默认 0.25 美元；置 0 = 不计金额但**仍然计数**）。
5. **离线 / 在线的差别**（这决定了"没有密钥能不能用"）：
   - **完全离线可用**：单测、检索、评测、闸门、索引体检。嵌入用
     `KUBEDOCS_EMBEDDING_PROVIDER=hashing`（确定性哈希，仅供管道自检，
     **不代表检索质量**）或已缓存的 `bge` + `KUBEDOCS_HF_OFFLINE=true`。
   - **受限网络**：`huggingface.co` 不可达；用 `KUBEDOCS_HF_ENDPOINT=https://hf-mirror.com`
     拉模型，之后置 `KUBEDOCS_HF_OFFLINE=true`（模型已缓存时联网校验会**挂死** >10 分钟，
     README「环境约束」一节有实测）。
   - **只有"生成回答"需要外网**：没有密钥时 `/ask` 返回检索结果 + 降级说明、
     `/chat` 回退到确定性图（红线与挂号仍可用），都不会崩（`api/app.py:644-661, 432-461`）。
6. **数据合规**：`data/clinic/` 的知识卡片不入库、且上游标注"仅用于学术研究"。
   要用在真实问诊场景，**先自己解决该数据集的授权**，本仓库不代为授权。
7. **公网部署前至少替换**：`docker-compose.yml:16` 的 Postgres 口令
   （`kubedocs/kubedocs`）、`:89-90` 的 `NEXTAUTH_SECRET` / `SALT` 占位值。
   compose 把 `5432` / `6379` / `8080` 直接映射到宿主机（`:21-22, 35-36, 54-55`），
   其中 Redis **没有设密码**。

---

## 5. 怎么上报

请使用 **GitHub Private vulnerability reporting**（仓库页 → `Security` →
`Advisories` → `Report a vulnerability`）。这是本仓库唯一的接收渠道，
同时也是**私密**渠道：请**不要**在公开 issue / PR 里贴出可利用细节。

请在报告里给出：受影响的文件与版本（commit 短哈希）、复现步骤、
你观察到的实际影响、以及你希望的处理方式。如果 7 天内没有回音，可以再开一个
**不含技术细节**的公开 issue 提醒一下——但请只说明"我有一个安全报告"。

本仓库**没有**安全邮箱列表（也不打算编一个出来）：Private reporting 是平台能力，
不依赖任何个人联系方式。

---

## 6. 未验证事项（诚实清单）

- `Dockerfile` 与 `docker-compose.yml` **从未在本机跑通过**（Docker Hub 不可达，
  见 `Dockerfile:2`、`docker-compose.yml:3`）。上面关于容器内权限、端口暴露的描述
  来自**文件内容**，不是运行观察。
- CI（`.github/workflows/ci.yml`）**从未在真实 runner 上执行过**（仓库无 remote）。
- 红线规则只有 26 条自建用例，**没有**外部/临床验证。
- 分诊链路的"接上真实卡片语料后能到多少分"**未测**：
  `data/clinic/cards.json` 存在（31,592 张卡），但 `triage.accuracy=0.0` 这份基线
  对应的运行环境与语料版本没有留档，**不能断言"有了语料就一定好"**。
- **分诊闸门现在是红的（预期）**：`scripts/eval_clinic.py --gate` 退出码 2，
  因为冻结基线里就记着 `triage.accuracy = 0.0`。这条红**没有被"修绿"**，
  也不该被修绿——只把"分诊不可用"暴露到了退出码上（见 S-4）。
- **多副本下的 LLM 花费上限未验证**：跨进程账本仍未原子（S-2），
  当前部署形态是单进程所以不生效，但**没有实测过多副本**。
- 依赖的 PyPI 可达性与 CVE 状态**未测**。
- 没有做过任何渗透测试、模糊测试或负载测试。
