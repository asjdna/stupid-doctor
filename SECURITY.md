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
  见 `.gitignore:41-42`，**不入库**）；
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
`GET /tools`、`POST /search`、`POST /ask`、`POST /chat`、`GET /`（`api/app.py:562-638`）。

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
（`MockAppointmentProvider`，`api/app.py:299-302`、`domains/clinic/mock.py`），
所以现在被占用的只是内存里的假号源，不是真实 HIS。换成真实号源库后这条立刻变成
"任何人都能替别人取号"。

**部署者必须做的**：把服务放在反向代理 / API 网关之后，由网关做认证与授权；
或者在 `create_app()` 里自己加一层 `Depends` 校验。本仓库**不提供**开箱可用的认证。

### S-2 `/chat` 路径没有 LLM 成本上限（严重，未修复）

仓库里**有**成本闸门，但它只覆盖一条路径：

- `generate.DailyBudget`（`generate.py:199-240`）在 `Answerer.answer()` 里
  `check()`（`generate.py:367`），默认上限 `llm_daily_budget_usd = 5.0`
  （`config.py` 的 `Settings.llm_daily_budget_usd`）。这条路径 = CLI `ask` 与 `POST /ask`。
- `POST /chat` 走 `agent/clinic_react.py` → `agent/llm_tool_caller.py`：
  `LiteLLMToolCaller` 只**累加** `self.cost`（`llm_tool_caller.py:35-36, 62-67`），
  **从不与 `llm_daily_budget_usd` 比较**；而 `ClinicReactRuntime.turn()` 每一轮都
  新建一个 caller（`clinic_react.py:108`），累加值随对象一起丢弃。

⇒ `/chat` 的花费**没有日上限**，唯一的上界是"每轮最多 5 次 LLM 轮次"
（`clinic_react.py:115-116` 的 `max_steps=5`，语义见 `tool_loop.py:118-125`）
乘以"请求数无上限"（见 S-3）。匿名 + 无速率限制 + 无成本闸门，三者叠加就是
"陌生人可以用你的密钥烧钱"。

**另外两个即使在 `/ask` 上也成立的性质**（如实说明，避免读者高估这个闸门）：

- 它是**日累计**上限，不是并发上限：`DailyBudget` 是"读 JSON → 加 → 写 JSON"
  （`generate.py:206-240`），**没有加锁**。多进程 / 多副本时各写各的，
  互相看不见 ⇒ 实际总花费上限是 `副本数 × 5 美元`。
- 成本依赖 `litellm.completion_cost()`；未知模型算不出价时按 0 记
  （`generate.py:316-319`），此时**闸门不会累加、也就永远拦不住**。

### S-3 无速率限制；`/chat` 会话状态无界增长（内存耗尽，未修复）

全仓没有任何限流/并发上限代码（`slowapi`、令牌桶、信号量、队列都不存在）。

`/chat` 的会话历史挂在进程内存里，键是**客户端提供的 `thread_id`**：

- `api/app.py:383` `app.state.clinic_history = {}`；
- `api/app.py:390-395` 每轮把该 `thread_id` 的最近 8 条消息写回去。

`thread_id` 只受长度限制（`schemas.py:159-160`，`max_length=64`），**数量不限**、
无 TTL、无淘汰。攻击者用不同 `thread_id` 反复调用即可让这个 dict 持续增长。

同一条性质在图路径上也成立：无 LLM 密钥时 `/chat` 回退到确定性图
（`api/app.py:399-429`），而 `_get_clinic_agent()` 构造 agent 时**没有传 checkpointer**
（`api/app.py:303`），于是 `kernel.py:229` 的默认分支生效——
`g.compile(checkpointer=checkpointer or InMemorySaver())`，状态同样只在内存里，
且同样按 `thread_id` 无界累积。

`thread_id` 由调用方**自带**这一点还有第二个后果：谁都能"续聊"到别人的会话上
（猜到/复用同一个 id 即可），**没有任何属主校验**。

**输入体积本身是有界的**（这部分做得对，如实记录）：`q ≤ 300`、`top_k ∈ [1,20]`、
`message ≤ 2000`、`thread_id ≤ 64`、`extra="forbid"`（`api/schemas.py:126-169`）。
所以暴涨的是**键的数量**，不是单个请求的体积。

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
- 但 `data/clinic/` 被 `.gitignore:41-42` 排除，**仓库不携带**，全新克隆的机器上
  这个文件根本不存在；
- 症状词典 `src/kubedocs/domains/clinic/data/symptom_dict.py` 去重后只有 **172** 个词项
  （28 个标准症状名 + 144 个口语变体去重成 146，加 24 个部位词，再加
  `KB_SYMPTOM_ALIAS` 的 4 个词），与卡片 `key` 的重叠是 **102**。

（这组数字我实测过两次并复现：172 词项 / 102 重叠。**曾经有一个"5673 键 / 仅 8 重叠"
的说法，那是错的、复现不出来，不要引用它。**）

评测器对这件事的处理是**诚实**的：缺语料时 `evaluate_triage()` 返回
`evaluated=False`（`evaluation/clinic.py:158-175`），而不是把"答不出来"计成 0 分。

**但必须知道的第二件事**：闸门**不看**分诊。`gate_failures()`（`evaluation/clinic.py:178-195`）
只检查红线漏报与红线误报率——分诊 0 分**不会让闸门变红**。
所以"`make check` 绿 / 闸门通过"**不等于"分诊可用"**。

**红线的 1.0 也不要高估**：那是自建规则表在自建用例上的成绩，规则与用例同一作者、
同一时期（`data/eval/clinic_gold.jsonl` 26 条），**不是独立临床验证**，
也未在真实急诊分诊数据上评估过。

### S-5 提示词注入与工具滥用（中等，部分缓解）

- 检索到的片段与工具返回结果都被**直接拼进提示词**（`generate.py:363-365`、
  `tool_loop.py:182-183`）。语料来自 kubernetes.io（受信度较高），
  但 clinic 卡片是第三方派生数据。
- `/chat` 的控制流由模型决定（`tool_loop.py` 全文）。副作用工具的执行闸门是
  `_confirm_side_effect()`（`tool_loop.py:257-280`）：**默认拒绝**，只有 `confirm`
  回调点头才放行。而 `/chat` 传入的回调是
  `(lambda _call: True) if user_asked_to_book else None`（`clinic_react.py:117`）——
  也就是说，**只要用户这一轮的原文里出现挂号意图，闸门对该轮所有副作用工具全开**。
  这道闸门防的是"模型自作主张"，**不防**"调用方未经授权"（见 S-1）。
- 已缓解的部分：工具参数逐个走 pydantic 校验（`tool_loop.py:223-227`），
  工具名必须命中注册表，未知工具名会把可用列表回灌给模型（`tool_loop.py:245-254`）；
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
| FTS5 `MATCH` 表达式注入 | 只抽取 `\w+` 词元并逐个双引号短语化，用户输入 `pod -limit` / `"` 不再改变语法、不再 500 | `store/sqlite_store.py:123-143`（含对应单测） |
| 引用越界（模型编 `[99]`） | 越界标记被**剥除**并记入诊断；回答里一个引用都没有也会被标记 | `generate.py:170-191` |
| 检索为空却调 LLM | 直接拒答，**不调用模型**（可断言：单测断言打桩 LLM 一次都没被调用） | `generate.py:347-352` |
| 请求体越界 / 未知字段 | pydantic 校验，越界 422、`extra="forbid"` | `api/schemas.py:126-169` |
| 工具同名冲突 | 注册表按名索引，重名直接报错而不是静默覆盖 | `tool_loop.py:129-135` |
| 副作用工具默认行为 | **默认拒绝执行**，不是默认执行后再回滚 | `tool_loop.py:257-267` |
| 空索引 / 索引损坏 | 不返回 500、不泄露 traceback：记 `index_error` 并返回 200 + 说明 | `api/app.py:118-133, 225-234` |
| 容器内权限 | api 以非 root（`USER app`，uid 10001）运行 | `Dockerfile:35, 45` |

---

## 4. 部署者最低要求（照做，否则上面的缺口会直接变成事故）

1. **不要把它直接暴露到公网。** 先补认证（S-1），再考虑任何形式的对外服务。
2. **密钥必须由环境显式提供**，且只放在 `.env` 或容器环境变量里：
   - 密钥读取路径是 `os.environ`（`generate.py:260-269` 的 `required_api_key_env()` /
     `has_api_key()`）；`.env` 由 `config.load_env_file()` 导入 `os.environ`，
     **真实环境变量优先、`.env` 不覆盖已有值**（`config.py:34-62`，`override=False` 的语义）。
   - `.env` 里**留空的键等于未设置**（`config.py:58-60`）：这防止 `.env.example` 里成片的
     `KEY=` 把"没有密钥"伪装成"有密钥"。
3. **`.env` 永不入库。** 它已被 `.gitignore:1-4`（`.env` / `.env.*` / `!.env.example`）覆盖；
   提交前请自己再确认一次（`git status --short` 里不应出现 `.env`）。
   `docker-compose.yml:48-49` 用 `env_file: .env` 把密钥注入 api 容器——它不会被打印，
   但会出现在 `docker inspect` 的输出里，共享宿主机的场合要注意。
4. **钱包护栏自己加**：`KUBEDOCS_LLM_DAILY_BUDGET_USD` 只覆盖 `/ask` 与 CLI `ask`（S-2）。
   要么给 `/chat` 也接上同一条闸门，要么在网关层做配额。
5. **离线 / 在线的差别**（这决定了"没有密钥能不能用"）：
   - **完全离线可用**：单测、检索、评测、闸门、索引体检。嵌入用
     `KUBEDOCS_EMBEDDING_PROVIDER=hashing`（确定性哈希，仅供管道自检，
     **不代表检索质量**）或已缓存的 `bge` + `KUBEDOCS_HF_OFFLINE=true`。
   - **受限网络**：`huggingface.co` 不可达；用 `KUBEDOCS_HF_ENDPOINT=https://hf-mirror.com`
     拉模型，之后置 `KUBEDOCS_HF_OFFLINE=true`（模型已缓存时联网校验会**挂死** >10 分钟，
     README「环境约束」一节有实测）。
   - **只有"生成回答"需要外网**：没有密钥时 `/ask` 返回检索结果 + 降级说明、
     `/chat` 回退到确定性图（红线与挂号仍可用），都不会崩（`api/app.py:601-614, 399-429`）。
6. **数据合规**：`data/clinic/` 的知识卡片不入库、且上游标注"仅用于学术研究"。
   要用在真实问诊场景，**先自己解决该数据集的授权**，本仓库不代为授权。
7. **公网部署前至少替换**：`docker-compose.yml:16` 的 Postgres 口令
   （`kubedocs/kubedocs`）、`:84-85` 的 `NEXTAUTH_SECRET` / `SALT` 占位值。
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
- 依赖的 PyPI 可达性与 CVE 状态**未测**。
- 没有做过任何渗透测试、模糊测试或负载测试。
