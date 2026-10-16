# 版本说明

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

本仓库尚未发布带 tag 的版本，`pyproject.toml` 中的 `0.1.0` 是当前开发版本号。

## [未发布]

### 问诊

- 交互式问诊页 `/consult`：多轮采问、对话式建议，急重症时顶部给出醒目提示。
- 科室建议与用药建议作为回答之后的延伸项，由用户在对话中按需追问；
  急重症情形下这两项随急救指引一并给出。
- 会话保留最近若干轮对话，追问时结论保持一致。

### 挂号

- 号源落 Postgres 的 `booking` schema：排班幂等生成（绝不覆盖已有余量）、停诊 / 开诊、改号量；
  占号是一个事务里的一句条件更新（并发不超卖），退号立即释放余量，
  队列号按「科室 + 日期 + 时段」从 1 起，挂号费以科室表为唯一真相源。
- 号源服务首次构造时幂等补一次排班；新增 `kubedocs schedule <generate|show|close|open|capacity>`；
  `kubedocs mcp` 新增 `--patient-id`（这个 MCP 会话代表哪位就诊人）。
- 挂号拆成"模型提议 → 患者点确认 → 服务端执行"：模型侧只有**只读**的 `propose_booking`
  （产出待确认单，号源不动）；写操作只走 `POST /consult/booking/confirm` 与
  `POST /consult/booking/cancel` 两个端点，确认请求只带 `proposal_id`，
  科室 / 日期 / 时段 / 费用都取自服务端那张单子；业务失败返回 409 + 一句人话。
- 待确认单一次性消费、15 分钟过期、进程内存储；一次性消费让双击与重放只生效一张。
- 就诊记录新增挂号关联（**单向**：就诊 → 号，权威状态在 `booking.appointments`），
  `GET /consult/encounters` 每条记录带 `bookings`。
- `/chat` 与 `/consult` 统一到同一套流程：不再按"这句话里有没有挂号两个字"决定要不要给确认回调。

### 诊疗判断

- 急诊判断依据"病症 + 严重程度"，并输出可审计的判断依据（病症、严重程度、理由）。
- 契约层校验自相矛盾：判出急重症却给常规建议会被判违规。
- 检查与用药建议必须引用真实存在的证据编号，没有依据的条目被丢弃。

### 判断与约束

- **判断权全部交给大模型**：急诊红线词表 + 三维门控、口语症状词表、语料症状键、
  症状→科室票数、检索式分诊、科室补全表全部降级为**证据提供者**——它们被装配成一个
  `FactSet`（「参考标准」面板），**一条结论都不下**。`RiskGuard` 不再是任何路径上的
  强制边：命中不覆盖 `urgency`、不短路流程，只作为线索进面板。
- **三层防护，各管一段**（安全靠纵深，不靠某一次分数）：
  **L1 语义理解**——大模型从六个角度（症状性质 / 组合共现 / 时间过程 / 严重度 / 高危人群 /
  情境诱因）识别危险信号，`red_flags` 每条带 `angle`（判断角度）、`likelihood`
  （`ruled_out|unlikely|possible|likely|very_likely`）、`discriminator`（能区分
  「要急救 / 不用急救」的那一个关键问题）、`ruled_out_by`（判为已排除时的依据）；
  **L2 主动问诊**——未排除的危险信号（`possible` 及以上，或 `unlikely` 却没写排除依据）
  而 `urgency` 不是 `emergency` 时 `followup_questions` 必须非空，模型没给追问就由
  `safety.fallback_question()` 补一句确定性的（优先用模型的 `discriminator`）并把 `action`
  置为 `ask`；**L3 确定性安全门**——纯代码执行三条不变量（见下）。
- **L3 的三条不变量**（`clinical/safety.py` 的 `INVARIANTS` 是唯一来源）：
  `no_medical_advice_without_evidence`（检索为空就不给医学内容）、
  `no_fabricated_booking_action`（不得声称服务端没执行过的挂号 / 退号 / 凭据号）、
  `emergency_judgment_requires_directive`（判出急症必须有可执行的急救动作）。第三条是唯一
  一条**补内容**的：`urgency=emergency`，或任何一条 `critical` 严重度 / `likely`、
  `very_likely` 的危险信号，而正文里没有 120 / 急诊 / 救护车这类可执行去向（判据
  `has_actionable_directive`）时，服务端追加固定指引 `EMERGENCY_DIRECTIVE` 并把 `action`
  升级为 `emergency`。三条都只做能被事实锚定的事，需要医学判断的一律留在 L1 / L2。
- **不确定性量化**：`ClinicalAssessment` 新增 `confidence`（0~1，**缺省 `None`**）与
  `information_gaps`。没给 `confidence` 记一条契约问题；低于 `CONFIDENCE_FLOOR`（0.6）时
  `information_gaps` 不能为空。`confidence` **不进任何阈值**——它只触发这条行为约束，
  并进校准统计（`evaluation/calibration.py` 的 Brier 分数与可靠性分箱，只报告不上门限）。
- **`safety` 段与运行期审计**：`POST /consult` 响应新增 `safety`（`{fired, ids, followup}`）、
  `confidence`、`information_gaps`、`followup_questions`，`red_flags[]` 补齐上面四个字段；
  `/chat` 的 SSE `message` 事件带同一份 `safety`。每次不变量触发向
  `data/index/safety_audit.jsonl` 追加一行 JSONL（只记 `invariant` / `action` / `detail` /
  `trace_id` / `ts`，**不记患者原话**），写失败不影响这一轮应答但会记 warning。
  `/consult/status` 的 `redline_source` 换成 `safety_model`（三层的一句话自述）、
  `safety_invariants`、`safety_audit_path` 与 `safety_audit_recent`。
- **红线闸门改为治理门禁**：`evaluation/clinic.py` 的 `gate_failures()` /
  `redline_gate_failures()` 与 `clinic_llm.py` 的 `llm_gate_failures()` 删除，换成
  `governance_failures(report, *, failed_calls=..., label=...)` 与
  `llm_governance_failures(report)`：`--gate` 只在**这次运行不能作为证据**时失败
  （一条可评测用例都没有，或有题目没拿到模型判断）。**安全不变量的触发也降为记录**：
  每条都已被安全门处置（换成只含事实的回答 / 补上急救指引），说明的是"模型这一轮需要被
  纠正"；硬信号由长期监听按 `alert` 报给运维。指标与参考目标（100% / ≤20% / ≥60%，
  常量改名 `TARGET_REDLINE_RECALL` / `TARGET_FALSE_POSITIVE_LIMIT` /
  `TARGET_TRIAGE_MIN_ACCURACY`，由 `reporting_targets()` 导出）继续进报告与档案，
  但**不判通过/不通过**。
- **临床治理落到文件与脚本**：新增 `data/governance/safety_assets.json`——五条安全资产
  （prompt / 危险信号词表 / 症状词表 / 科室表 / 金标集）各带 `sha256`、`version`、
  `effective_date`、`source`、`owner`、`rationale`，加上三条不变量的治理条目与
  `governance_gaps` 已知缺口；新增 `scripts/check_governance.py`：核对资产哈希、不变量与
  代码注册表一一对应、`implemented_in` 路径存在、每条不变量在 `tests/` 里有引用、
  `verified_by` 出处存在（`--update-hashes` **只刷哈希**，版本与理由由人写）。它已挂进
  `scripts/ci_local.py` 的 quality job。
- 面板同时是**递给模型的线索**：`/chat` 把 `render()` 拼进系统提示；`/consult` 递的是线索版
  （`render(defer_retrieval=True)`，知识依据留空、检索仍由模型自己决定），循环跑完后再用同一份
  完整事实**校验模型的输出**（`verify.py`）并随响应以 `reference` 留档。
- **参考标准面板**六块内容：急诊线索（含断言 / 经验者 / 时间性三维取值，以及"不触发"的出现处）、
  症状归一（原话词 → 标准词 → 语料键）、科室候选（票数 + 来源 + 可挂号科室清单）、
  知识依据、业务事实（没有已确认的挂号时明写"本轮服务端**没有**任何已确认的挂号/退号动作"）、
  线索装配问题。
- **输出约束层**（`clinical/verify.py`）：拿**同一份事实集**校验模型的每一处输出，分三级——
  `block` = 动作幻觉（`fabricated_booking` / `fabricated_cancellation`）、编造凭据号
  （`unknown_ticket`）、检索为空却给医学内容（`medical_without_evidence`）；
  `strip` = 编造依据编号（`fabricated_evidence`）、科室未接地（`department_not_grounded`）；
  `warn` = 未解释的红线线索（`red_line_unexplained`）、服务端没给过的金额（`unknown_amount`）。
- **三级处置**：① 带违规原因重生成一次；② 仍违规则**模型那段散文不上屏**，改由
  `facts_only_answer()` 拼一段只含事实的回答；③ 历次违规全部进响应里的 `contract_violations`
  （重生成后回正的也算，回正不等于没发生过）。`/consult` 与 `/chat` 都带同一形状的
  `reference` 面板留档与 `contract_violations`；`/chat` 的输出是一段自由文本，走
  `verify_prose()`，只做能被事实锚定的检查（动作幻觉 / 凭据号 / 金额 / 红线未解释），
  不从文本里猜科室。
- **先校验、再执行契约**：`contracts.enforce` 会删掉无依据的检查/用药建议，先跑它会让
  "检索为空却给医学内容"看不出异常，而散文里那句话照样上屏。
- **安全动作豁免依据要求**：建议就医 / 拨打 120 不是事实断言，检索为空时照样给。
- 科室归一化收掉口语后缀（`全科门诊` → `全科`、`呼吸内科门诊` → `呼吸内科`）。挂号与接地校验
  共用这套规则：`resolve_department()` 靠它认出"呼吸内科门诊"，`verify.py` 按**归一化后**的名字
  比较——拿原始字符串硬比会把合法科室判成"未接地"并从建议里剔除。规则属于域知识，
  `FactSet` 只收一个注入的可调用对象。
- 依据编号同时接受我们自己分配的 `K#` 与索引里的真实 `chunk_id`：只认前者会把**真实命中**
  判成"编造依据"。
- 确定性图（`agent/kernel.py` 的 `ClinicAgent`）退出请求路径，保留为"规则判断 vs 模型判断"的
  对照臂与回归测试。

### 无大模型时的行为

- **没有判断者就没有结论**：没配密钥回「未连接大模型，请连接大模型后重试。」；
  配了但这次调用失败（超时 / 限流 / 输出不可解析 / 预算用尽）回「大模型调用失败，请稍后重试。」。
  文案的唯一来源是 `kubedocs/llm_status.py`，两句刻意不合并。
- 取消全部兜底内容：主判失败不再"按现有证据给出有限结论"（`assessment` 留空）；
  `Answerer.answer()` 不把检索结果当成回答交付；`/ask` 没密钥时**在检索之前**就返回
  （`citations=[]`、`n_hits=0`）；CLI `ask` 打这句话并以退出码 **3** 结束；
  `/chat` 不回退到确定性图。
- 不受影响的入口（它们不需要大模型）：`/search`、`ingest`、`stats`、`/healthz`、`/metrics`、
  `domains`、`tools`，以及 MCP 工具——MCP 用客户端自己的模型。

### 摄取

- 新增多格式解析层（`parse.py`）：PDF / docx / xlsx / pptx / html / csv / txt / md
  都变成**带位置的块**；位置按格式定义（`p.7` / `Sheet1!A1:C20` / `slide 3` / 标题路径），
  并作为面包屑首段进入引用。解析库全部延迟 import，缺一个库只坏一种格式。
- 三条不变量：编码按 `utf-8-sig` → `utf-8` → `gb18030` 严格探测，**绝不静默丢字**
  （不用 `errors="ignore"`）；扫描件 PDF 点名跳过、绝不建空块；单文件失败不影响整批。

### 编排

- 工具由模型自行调用：代码提供工具清单与执行器，模型决定调哪些、调几次；
  轮次、工具白名单与证据编号由程序控制。
- 有副作用的工具默认不暴露，产生副作用前必须显式确认。

### 检索与存储

- 混合检索（BM25 + 向量 + RRF 融合）与可选重排。
- 切块改为**父子两层**：子块（默认 300 字符、重叠 60）进检索索引，父块（标题小节，
  上限 3000 字符）只在命中后作为完整语境交给模型；短碎片只在**同一个父块内**合并。
  三个后端都支持，老索引可原地升级（补列补表，缺父块时退回子块正文当语境）。
- 默认向量后端改为 `pgvector`。数据库不可达时降级为 `sqlite-vec`，并把原因写到
  `/healthz` 的 `store_fallback` 与 `cli stats` 的「存储后端 X（配置 Y）」；
  非法取值直接报错、绝不静默回退。
- 分诊采用两阶段检索（粗排候选池 → IDF 精排 → 科室投票），并叠加科室补全表；
  补回的标签带来源与置信度。
- 支持 sqlite / sqlite-vec / pgvector 三种向量后端，按配置切换。

### 评测

- 检索闸门与冻结基线：语料指纹或嵌入模型不符时拒绝比较，而不是给出假通过。
- 分诊评测支持按严重度与语言风格分层；提供急诊识别与问诊质量离线评测脚本。
- `scripts/eval_clinic.py --arm` 按引擎分臂（`{both,rule,llm,tool,reasoner}`）：**规则臂**
  （`rule`）测 `RiskGuard` + 规则分诊，确定性、离线、免费，是**线索质量**的读数（红线词表漏
  一条仍然是回归）；**工具循环臂**（`llm` / `tool`）跑 `pipeline.advance(原话)`，模型自己决定
  调不调检索，**这是产品路径**，需要索引 + 密钥、会花钱；**单发对照臂**（`reasoner`）用
  `ClinicalReasoner` 只做单发判断、不检索，便宜，用来对照"有工具 / 没工具"的差别；
  `both`（默认）等于规则臂 + 工具循环臂。报告与 JSON 里判断臂的键是 `llm`，新增
  `engine` 字段（`tool_loop` / `reasoner`），所以"跑的是哪条路"在档案里能直接读出来。
- **装配收敛到一处**：新增 `src/kubedocs/clinical/runtime.py`——`open_domain_index(settings)`
  按域打开索引并构造检索器（索引为空直接报错，而不是给一个"检索永远为空"的检索器），
  `build_clinical_assembly(...)` 把检索器接成一条可用的临床管线。`/consult`
  （`api/consult.py`）与评测的工具循环臂都调它：同一次工具注册、同一个面板建造器
  （`domains/clinic/reference.py` 新增 `make_reference_builder` 与 `booking_facts`）、
  同一本日预算账、同一套安全门。评测侧只注入 `MockAppointmentProvider` 与临时落点：
  不碰真实号源库、不落就诊记录、不写审计文件。
- **指标是记录，不判通过/不通过**：危险信号召回、误报率、分诊命中率来自自建小样本用例集，
  能说明趋势与回归，不能承担「这条回答安全」的责任。参考目标（召回 100% / 误报率 ≤20% /
  分诊 ≥60%）与实测值并排打印并进档案，由人看趋势。
- **闸门只卡一件事**（`governance_failures()` / `llm_governance_failures()`）：这次运行
  **不能作为证据**——一条可评测用例都没有，或有题目没拿到模型判断。删掉密钥、语料或索引
  不会让闸门变绿：没测不等于通过。**安全不变量的触发不判闸门**（它说明"模型需要被纠正"，
  每条都已被安全门处置），硬信号由长期监听按 `alert` 报给运维。
- **`--archive` 写治理档案**：把整份报告（金标指纹、两臂指标、参考目标、基线注记）写成带
  UTC 时间戳的 JSON，默认目录 `data/eval/clinic_archive/`，永不覆盖。写档案时**同时追加一行
  趋势记录**到 `data/eval/clinic_trend.jsonl`（`--trend-path` 可改落点）：记录由
  `src/kubedocs/evaluation/monitor.py` 定义，每行一条扁平 JSON，含时间戳、引擎、金标指纹、
  提示词版本、两臂关键指标、失败调用数、不变量触发数与调用量/花费，**不记患者原话**（与安全
  审计同一条纪律）。只在人要看的格式里留档、却不在机器要比的格式里留痕，趋势就只能靠人翻
  JSON。
- **长期监听**：新增 `scripts/monitor_clinic.py`——`--status`（只看历史，不花钱、不联网）、
  `--run`（跑一轮评测 → 留档 → 趋势 → 提醒）、`--follow --interval 3600`（常驻，容器边车 /
  cron 都行）。`--run` 与 `--follow` 复用 `eval_clinic.py` 的 `main()`，不另写一份评测。
  监听**不新增门限**：`alert`（不变量被触发 / 这一轮没跑成 / 有题目没拿到判断）让退出码变 1，
  `note`（召回下降、误报上升、分诊下降、单题花费上升、提示词或金标集变化）只打印、
  默认不改退出码，`--strict` 才硬拦。`note` 的判据是"低于最近 `--window`（默认 5）轮的
  最低值 / 高于最高值"，不与人拍的阈值比；工具循环臂与单发臂**分开比**（数量级不同）。
- **运行期可观测**：新增 Prometheus 计数
  `kubedocs_safety_invariant_fired_total{invariant,domain}`（每次安全门触发都计数，按不变量与
  入口分开）。它与审计日志出自同一个收口点——`clinical/safety.py` 的 `append_audit`：
  审计文件回答"发生了什么"，计数回答"最近是不是变多了"，两处各打一次迟早会静默漂移。
- **输出解析失败重试一次**：工具循环跑完之后模型偶尔吐散文而不是 JSON（实测约 2/31 轮），
  会带"请只输出 JSON"**重试一次**，仍失败才回 `LLM_FAILED_MESSAGE`——直接失败对患者是白等
  一场，而修不好时**绝不产出兜底结论**。
- **置信度校准**（`evaluation/calibration.py`）：把模型自报的 `confidence` 与实际判对率配对，
  给出 Brier 分数与可靠性分箱，只随报告与档案打印——小样本上把它写成门限等于用噪声判生死。
- 实测（工具循环臂 31 次调用，约 $0.035）：规则臂召回 100%（TP=9 FN=0）、误报 0%
  （FP=0 TN=17）、分诊 100%（5/5）；工具循环臂召回 100%（TP=9 FN=0）、误报率 23.53%
  （FP=4 TN=13）、分诊 100%（5/5）；确定性安全不变量 0 条；置信度校准 n=31、Brier 0.20
  （只报告）。工具循环臂"报了危险信号"的判据是**判成急症，或还有未排除的危险信号**——把
  看到过、判断不该上报的信号写成 `ruled_out` 是留痕而不是误报。剩下几条误报（`n02`
  「我父亲昨天说心口疼」、`n03`「我家孩子喘不上气，要不要去医院」）是**金标口径**（患者
  本人急症）与模型行为（对话里有人有危险就报）的分歧。这些差距改的是提示词与用例口径，
  不是门限。
- **不同轮次会波动**：另一次运行触发 2 条确定性安全不变量
  （`no_medical_advice_without_evidence` 与 `emergency_judgment_requires_directive`），
  **每一条都已被服务端处置过**（把那段散文换成只含事实的回答 / 补上固定急救指引）。
  报告里的"安全不变量触发"因此说明的是**"模型这一轮需要被纠正"**，
  **不是**"患者看到了不该看的内容"。
- clinic 的评测不在 `scripts/ci_local.py` 的三个 job 里：其中 `gate` 跑的是**检索**闸门
  （`eval --domain k8s_docs --gate`）。clinic 的治理门禁在 quality job
  （`scripts/check_governance.py`），评测本身单独跑 `scripts/eval_clinic.py --arm both`。
- `scripts/ci_local.py` 一条命令覆盖静态检查、依赖声明一致性、文档锚点、临床治理一致性、
  全量单元测试、全链路自检与检索闸门。

### 打包与部署

- 容器镜像支持 CPU 版 torch 配方（镜像体积显著小于默认 CUDA 路线）。
- 命令行工具覆盖摄取、检索、问答、评测、服务、排班与索引体检。

## [0.1.0] - 未发布

初始开发版本：双域（`k8s_docs` / `clinic`）检索问答与 Agent 平台，
包含混合检索、带引用回答、评测闸门、可观测与医疗问诊能力。
