# Changelog

本文件记录 kubedocs 的重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

**关于版本与日期，先把话说清楚**：仓库**没有 tag、没有 remote、也没有发布过任何制品**，
所以下面两节的版本号标注为 **未发布（no tag yet）**——`0.1.0` 来自 `pyproject.toml` 的
`version = "0.1.0"`，不是一次发布。**不写版本日期**，因为写任何一个日期都会是编的。
条目里的短哈希是真实提交，`git show <哈希>` 可以逐条核对。

> **本次修订说明（2026-10-12）**：此前本文件只归纳了**前 22 个提交**
> （提交日期区间 2026-09-24 ~ 2026-09-25），而仓库当时已经有 **33 个提交**——
> 2026-10-08 ~ 2026-10-11 那 **11 个提交没有进变更日志**。现在把它们补进 `[0.1.0]`
> 的对应小节，并新增一节记录本轮（10 项）修复。
> 同时更正一处旧说法：原先写"**没有删除项**——历史里没有任何功能或文件被移除"，
> 那个判断只在 22 个提交的范围内成立；`KUBEDOCS_ENV` / `is_prod` 这两个**假开关**
> 后来被删掉了（`4eb5e10`），本轮又删掉了另外 4 个。

---

## [未发布] - 本轮修复（2026-10-12）

一轮以"关闭真实缺陷 + 抹平文档与实现的自相矛盾"为目标的改动。每一条都标注了
**怎么验证**，而不是只写"已修复"。

### Fixed — 安全关键：追问闸门曾抢在模型判急重症之前（实测 4/4 次误拦）

**这是"红线交给模型"之后暴露出的洞，比模型漏判更严重。**

- **现象**：`「突发言语不清，右侧肢体无力两小时」`（教科书级卒中）连续 **4/4 次**
  返回 `action=ask` —— 患者拿到的是一句**追问**，而**模型根本没机会判断这是急症**。
- **成因**：`ClinicalPipeline.advance` 无条件先判 `intake.can_conclude()`，不满足就返回
  槽位追问。关键词护栏退场后，这道闸门成了"模型之前的最后一道关卡"，
  而它**只会看槽位填满没有，不会看急症**。
- **修法**：**有工具循环时不拦**——"够不够下结论"交回模型（提示词 R5 要求信息不足时
  把追问放进 `followup_questions`、把 `patient_summary` 写成一句自然的追问）。
  同时新增 `followup_questions` 非空 ⇒ `action=ask`，但**急重症优先**
  （判出急症时不给"先追问"的动作，必须先给急救指引）。
- **复验**（真机、真调模型、各 3 次）：教科书级卒中 **0/4 → 3/3**；
  心梗书面语 3/3；口语胸痛 3/3；普通主诉 0/3（正确地不报急诊）。
- **回归测试**：`test_intake_does_not_preempt_the_model_when_tool_loop_is_present`、
  `test_intake_gate_still_asks_without_tool_loop`、
  `test_followup_questions_become_an_ask_but_emergency_wins`。

### 附：本轮完整验证结果

| 验证线 | 结果 |
|---|---|
| 项目自带 CI（`scripts/ci_local.py`，3 个 job） | **3/3 通过，退出码 0** |
| 单元测试 | 1024 passed |
| ruff / format / mypy / `check_doc_refs` | 全绿 |
| 本机服务功能冒烟（8082） | `/consult` 多轮·红线·噪音·重置、`/search`、`/ask`、`/tools`、`/metrics`、`/domains`、`/patient`、`/console` 全部 HTTP 200 且行为正确 |
| 容器路线（8080） | `/healthz` 正常（18,822 chunks）；**`/consult` 404 —— 镜像早于本功能，未重建**（外部下载受阻，已知项） |

### Changed — 架构转向：底层约束是**大模型**，红线改由它判断，工具由它调度

**这是本仓库最根本的一次改动**，此前那些"安全性质"的落点大多要重新看一遍。

**1. 红线不再是关键词表，而是模型判出的「病症 + 严重程度」**

- 旧实现：`RiskGuard` 匹配 15 个口语词（胸痛/喘不上气/意识不清…）⇒ 渲染成
  「【红线信号｜MUST 处理】命中词『胸痛』…**必须**包含紧急处理指引」塞进提示词，
  并强制 `urgency=emergency`。问题不是不灵，而是**把医学判断压成了字符串匹配**：
  `胸口压着疼` 命中，`胸骨后压榨样疼痛放射至左肩` 反而漏掉；
  `突然看不清` 命中，而它可能只是干眼。
- 现在：提示词 R1 改为**由模型按病症（可能危及生命的急症方向）+ 严重程度
  （器官功能/生命体征受威胁的表现）判断**，输出到新契约 `RedFlag`
  （`condition` + `severity: critical|severe` + `why` **三者必填**，`evidence_ids` 可选）。
  **`evidence_ids` 允许为空是刻意的**：急重症常可由患者原话直接识别，
  强求卡片依据会逼模型去凑 id，反而害了 R3"不许编依据"。
- **关键词表退出判断链**：不进提示词（`evidence.render` 里一个字都不提）、
  不影响 `urgency`、不触发横幅。它只剩一个用途——**主判不可用时**给一句保守对话提示
  （`degraded` 非空 ⇒ 前端不弹横幅，所以不冒充模型结论）。
  `test_keyword_hit_is_not_rendered_into_the_prompt` 与
  `test_prompt_does_not_tell_the_model_to_obey_a_keyword_list` 两条回归钉住这件事。
- **安全性改由"契约自相矛盾检查"守住**：`red_flags` 非空而 `urgency` 不是 `emergency`
  ⇒ `validate_against` 记违规。判断权在模型，但"判了急症却给常规建议"这种输出会被拦下。
- **不一致仍要留痕**：关键词命中而模型未判急重症时，HTTP 层回传 `keyword_hit_conflict=true`
  并在对话正文**追加**一句风险提示（只进对话、不弹横幅）。判断权归模型，
  但"系统担心过这件事"不能被抹掉。
- **实测拿到的正是这次转向的意义**：`「胸骨后压榨样疼痛，向左肩放射，伴大汗」`
  在旧表下**一个词都不命中**（`keyword_hit` 为空），现在模型判出
  `red_flags=[急性冠脉综合征/critical/why=…]` 且 `emergency=true`。
  而口语的 `「我胸痛，喘不上气」` 判出 3 条 `critical`（含气胸、肺栓塞），也仍然正确。

**2. 工具由模型自己调度（`clinical/orchestration.py`）**

- 旧流程是代码写死的：`采集槽位 → 强制检索 → 强制分诊 → 主判`。
  这等于代码替模型决定了"要不要检索、检索什么"，模型只能对**代码给它的那批证据**作答。
- 现在代码只提供**工具清单 + 执行器**（`agent.tools.ToolRegistry` 仍是唯一真相源），
  模型自己决定调哪个、调几次（实测：普通主诉调了 8 次 `search_docs`；
  书面语胸痛调了 4 次检索 + 2 次 `check_citations`，第一次 `ok=False` 是它抓到了自己引用错，
  重试后通过——这正是工具循环的价值）。
- 代码只保留三件**非医学**的事：轮次上限（`MAX_TOOL_ROUNDS=4`，到顶明确要求"用现有依据作答"，
  **不静默截断**）、工具白名单（**副作用工具默认不给模型**，与 MCP server 同一约定）、
  以及**证据编号由代码分配**（`K1`、`K2`…，单一来源）。
  所以"模型可以不检索"不会架空 R3：不检索就没有编号可引用，也就给不出检查/用药建议。
- **工具循环里每一次模型调用都过同一本日预算账**（`BudgetedClient.complete_with_tools`）。
  漏了这一步等于给了一条"模型自己决定花多少钱"的路径。

**3. 过程中修掉两个真实缺陷（都是测试逼出来的）**

- **`ToolRegistry.get()` 抛的是 `ToolError` 而不是 `KeyError`**：只捕获 `KeyError` 时，
  "模型点了一个不存在的工具"会**把整轮问诊炸掉**——而模型偶尔写错工具名是常态。
  `test_unknown_tool_becomes_a_message_not_a_crash`。
- **`search_docs` 根本没注册进 clinic 域**：`app._register_tools` 故意不注册它
  （启动时注册会强制加载嵌入器，理由正当），但工具循环忘了补注册，
  于是 `tool_schemas` 在过滤掉副作用工具后**是空的** ⇒ 模型一个工具都没得调，
  退回用通用医学知识作答——**表面上"有回答"，实际把 R3 架空了**。
  实测复验时发现（`模型自己调用的工具 = []`），现在在 `consult._build` 里
  拿到 retriever 后补注册（此时嵌入器已加载，不增加成本）。

**4. 与产品形态的衔接**

- `emergency` = **模型这一轮的判断**且该结论真的出自模型（`degraded` 非空不算）。
- 结构化结论（含 `red_flags`）仍随 JSON 回传供后端留档，前端只渲染 `message`。
- 新增 `tool_calls` 字段（单独字段而**不是塞进 `problems`**：前者是正常行为，
  后者是契约问题，混在一起会让"这一轮有没有违规"读不出来）。

**验证**：1021 passed（新增 `tests/unit/test_tool_orchestration.py` 14 条 +
改写红线相关 8 条）；ruff/format/mypy/check_doc_refs 全绿。
真机复验（8082，真调模型）：4 个用例的行为、工具调用数与 `red_flags` 见上。
花费：当日累计 **$0.0886 / 上限 $2.00**。

### Changed — 问诊形态改为「只留对话」+ 横幅改由模型判断触发

**产品决定（用户明确要求），与上一版实现相反，故记清改动理由：**

- **主页只剩对话区**：科室 / 病症排序 / 检查 / 用药 / 小结 / 依据 六个面板**全部撤掉**，
  相关信息移入后台作为大模型的参考与判断依据，由它在对话正文里讲给用户听。
  `data/web/consult.html` 重写为纯对话界面（输入框 + 消息流 + 重新开始 + 急诊横幅）。
  结构化字段仍随 JSON 回传（后端留档 / 契约校验 / 排障），但**前端不渲染**——
  由 `TestPageIsDialogueOnly` 锁死，包含"面板 id / 区块标题 / 结构化字段引用都不许出现"三类断言。
- **提示词新增「三之二」节**：规定 `patient_summary` 就是患者看到的**全部内容**，
  要按"现在该怎么办 → 挂哪个科 → 可能方向排序（每条都讲清为什么，用于建立信任）→
  该做什么检查 → 能考虑哪类药（**不给剂量**）→ 什么情况必须立刻去医院"的顺序
  **连贯地讲**（不许写成清单）；长度上限由 4 句放宽到 10 句 / 约 450 字；
  并明确"只填结构化字段、没写进 `patient_summary`，等于患者什么都没得到"。
  实测：普通主诉的对话正文 313 字，把五项都讲到了且带理由；`prompt_version` 随之变化。
- **急诊横幅触发条件反转**：由"红线阳性 ⇒ 无条件弹（服务端强制）"改为
  **"模型这一轮判出 `urgency=emergency` 才弹"**，且该结论必须**真的出自模型**
  （`degraded` 非空 ⇒ 结论来自兜底，不算模型判断，不弹）。
  理由：本系统分工是"大模型主要判断，红线与分诊作为辅助信号"，
  横幅是给患者看的结论，就该跟着结论的产出者走。
- **同时堵上由此产生的静默失败路径**：红线阳性而模型判非急诊时（模型没照 R1 执行），
  服务端往对话正文**追加**一句固定风险提示（`RED_LINE_CONFLICT_NOTE`，含 120）并回传
  `red_line_conflict=true` —— **只进对话，不弹横幅**。横幅的触发权归模型，但危险信号不会消失。
  回归测试：`test_red_line_conflict_is_spoken_in_the_dialogue`、
  `test_red_line_alone_does_not_raise_the_banner`、`test_degraded_emergency_does_not_raise_the_banner`。
- **R1 未放宽**：提示词仍要求"看到【红线信号】就必须给紧急处理指引"，
  所以正常情况下红线仍会让模型判出 `emergency`、横幅仍会出现——
  变的只是**触发链**（模型判断）而不是**红线的作用**。若要把红线也降为纯建议，需要另外拍板。
- 前端不再自己判断急诊：页面里**没有** `red_line_positive` 与 `"urgent"` 字样，
  只渲染服务端给的 `emergency`（同一类断言也在测试里）。



- **`src/kubedocs/api/consult.py` + `data/web/consult.html`**：把 `ClinicalPipeline`
  真正装配起来的那个点（此前 `clinical/__init__.py` 只导出、不构造，所以"补全映射表 +14 点"
  这类收益线上拿不到）。四个端点：`GET /consult`（页面）、`POST /consult`（推进一轮）、
  `POST /consult/reset`、`GET /consult/status`。
- **页面内容**：对话 + 多轮采问、分诊科室（带置信度条）、可能病症（带可能性标签）、
  建议检查、用药提示（**只列药名类别不带剂量**）、小结与证据、原始 JSON 折叠区、
  顶部急诊横幅与降级提示、免责声明。
- **复用 clinic 域同一份 store / embedder**（不重复加载模型）；管线惰性装配一次并缓存。
- **`RetrievalTriage(department_map=load_department_map(min_confidence="high"))`**：
  把 §5.1 实测 +14 点的补全表接进线上路径。表缺失时**记 warning 并退化为解析正文**，
  不静默当成"没有这项收益"。
- **急诊红线由服务端强制**：`red_line_positive ⇒ emergency=true` 且带固定 `emergency_note`
  （120 / 急诊），**不看模型把 urgency 判成什么**——`to_payload()` 的这条逻辑有参数化回归测试
  （`routine` / `urgent` 两种都必须报急诊）。前端只渲染，不写第二份判断。
- **会话在进程内**：`session_id` 复用 `IntakeState`，条数上限 200、最久未用淘汰；
  **重启即丢、多副本不共享**——这一点由 `GET /consult/status` 直接吐出来，不假装有持久化。
- **索引不可用 = 503 + 可执行指引**，不是 500 堆栈（`klubedocs ingest` 建索引）。
- **实测（本机 8082，真调模型）**：普通口语主诉 → `action=answer`、`degraded=''`、
  科室 `中医科(0.6)/呼吸内科(0.5)/传染科(0.3)`、10 条证据、契约校验 0 问题；
  `「我胸痛，喘不上气」` → `action=emergency`、`emergency=true`、`red_line_word='胸痛'`、
  文案含 120；`「嗯嗯好的谢谢你」` → 不乱给科室。
  回归测试：`tests/unit/test_consult_api.py`（30 条，用假管线，不联网不花钱）。

> **注**：上面这条实测是**改形态之前**跑的（当时页面还有面板、横幅由红线强制）。
> 改形态之后复验的结果见 `Changed` 一节；两处数字都对得上，但**当前契约以后者为准**。

### Added — 分诊金标 C2：从 5 条扩到 71 条（按原型标准派生）

- **新增 `scripts/build_triage_gold.py`**：依据全部来自原型，不是我编的——
  `data/eval/prototype/department_scope.md`（**科室导诊范围标准**，显式给出「症状 → 科室」判据）、
  `phenomenon_to_symptom.json`（口语现象 → 规范症状）、`manual_annotation.csv`（853 条人工实体等价）。
  产出 `data/eval/triage_gold.jsonl`（`meta` 头 + 71 条，**照搬原型 `rag_gold.jsonl` 的金标规范**：
  `built_from` 语料指纹、抽样策略、`human_verified=false`、`disclaimer`；每条带 `source`/`canonical`）。
- **构造中三道质量闸，每道都被实际错例逼出来**：部位冲突（`头痛 -> 乳头疼`）、
  多科室归属/互为子串（`胸痛`、`腹痛`↔`下腹痛` 一律丢，答案必须唯一）、
  变体须对标准关键词也成立（`月经量少 -> 怕量少`）。
  `_match_canonical` 由"子串包含"改为**只用前缀匹配**——包含匹配曾造出
  `「心绞痛」-> 泌尿科` 这种概念错位的假金标。
- **`eval_triage.py` 新增 C2 口径**（`C2 标准派生`），与 A/B/C 分开报，不许混成一个数。
- **实测（实测：C2 = 71 条口语金标）**：`off 35/71 top-1、48/71 top-3(67.6%)`；
  **`high` 44/71、58/71(81.7%)**；`medium` 40/71、57/71(80.3%)。

### Fixed — 更正上一轮一条**读反了**的结论

- 上一轮我据 C（**只有 5 条**）的 `off 2 / high 3 / medium 4` 写下"语料补充没有提升、
  瓶颈不在语料覆盖率"。**扩到 71 条后结论反转**：口语口径上 top-1 **+12.7 点**、
  top-3 **+14.1 点**，语料补充**确实有效**；`medium 4 > high 3` 是噪声。
  已更正 `triage.py` 的 `DEFAULT_DEPARTMENT_MAP` 文档、方案 §5.1。
  **教训写进文档**：5 条样本里 1 条值 20%，连"变好还是变坏"都判不了——
  这就是"金标 ≥50 条是前置条件"的实证，不是流程洁癖。
- **默认只信 `high` 档**（图谱直接关系 + `SameAs` 别名 + 实体→疾病→科室）；
  含模糊归并的 `medium` 档实测更差，要显式打开并人工抽检。
- A 集上仍降 2 例（164→162）：补全让候选池的旁观卡片也投票，在"查询词就是卡片名"的
  A 集里稀释了正确卡片的票。真实但很小的代价，且 A 是上限集。
- C2 的 top-3 81.7% 已越过 80%，但它是 `human_verified=false` 的派生金标，
  **不据此宣布达标**——它证明的是"语料补充有效"，不是"分诊达到上线准确率"。

### Known gap — 补全的收益目前**只在评测里生效**

- `RetrievalTriage` 是**依赖注入**进 `ClinicalPipeline` 的，而产品路径里**还没有装配它的地方**
  （`clinical/__init__.py` 只导出、不构造）。所以"接上映射表能 +14 点"这个结论，
  **线上路径现在拿不到**。推荐写法已写进 `RetrievalTriage` 的类文档：
  `RetrievalTriage(department_map=load_department_map(min_confidence="high"))`。
  这与"临床管线尚未接到 HTTP"是同一个装配缺口，不另开条目。

### Changed — 分诊语料补充（根因、做法与实施）

- **查清了 12,863 张「未标注」卡片的根因**：不是解析问题，是 `cards.json` 生成阶段
  **丢掉了关系**——这些卡片的 `diseases` 字段本身就是「暂无」，所以在卡片内部做传递
  补全的成功率实测为 **0%**（同 key / 经 diseases / 经 drugs 三条路全 0）。
  它们是 `耳朵里疼 / 耳朵疼 / 耳朵很疼` 这类**口语变体**，源图谱里没有任何关系。
  验证：`scripts/clinic_derive_departments.py --report-only`。
- **回到原始三元组补全**：新增 `scripts/clinic_derive_departments.py`，从
  `medical-agent/data/raw/OpenCMKG/triples.txt` 推导出
  `data/clinic/department_map.json`（**带 `source` + `confidence`，可抽检、可回退**）：
  `SameAs` 别名 3,009 条（权威）+ 实体→疾病→科室 466 条（权威）+ 最近邻模糊 ≥0.5
  1,252 条（**有错例，需抽检**）= 未标注 key 的 **27.6%**，其余 **72.4% 图谱无记录**。
- **`RetrievalTriage` 支持查表**：新增 `department_map` 参数与 `load_department_map()`
  （`min_confidence` 分 `high`/`medium` 两档，文件缺失时退化为正文解析而**不抛异常**）。
  回归测试：`test_clinical.py::test_department_map_fills_placeholder_cards`、
  `::test_confidence_tier_controls_which_labels_are_trusted`。
- **实测结论见上面「更正」一节**：`eval_triage.py --department-map {off,high,medium}`
  在 A/B（各 200 条）上 **B 完全不变（153/196）**，A 的 top-1 降 2 例（164→162）；
  在 **C2（71 条口语金标）上 top-1 35→44、top-3 48→58**。
  映射表命中次数非零（A/B/C2 = 748/552/664 与 1085/750/894），
  **证明代码路径确实被触发**，不是空转造成的假差异。
- **关于余下 12,425 条是否花钱标注**：上一轮据"无提升"决定不追加 LLM 开销，
  该结论的前提（语料补充无效）**已被推翻**，所以这条决策**同时作废**——
  余下部分现在有理由再做，但**应先扩金标之外的第二批人工复核**，避免又一次读错。
- **补一道防假证据的闸**：`data/clinic/` 是 `.gitignore` 的，换台机器就没有映射表。
  `eval_triage.py --department-map {high,medium}` 现在**表为空时响亮失败（exit 2）**，
  而不是静默退化成 `off` 的数字再被记成 `high` 的结果——那正是"看起来测了、其实没测"。
  验证：把映射表换成空表跑 `--department-map high`，退出码 2 并打印修复指引。

### Fixed — 安全与正确性

- **急诊红线现在**每一轮**都判**：此前 `node_redline` 只看首轮的 `question`，而 resume
  轮被直接送回 `observe`——用户在回答追问时才说出"胸痛，喘不上气"**永远不会**转急诊。
  现在续答轮先过红线。回归测试：`test_kernel.py::test_red_line_is_checked_again_on_resume_turns`。
- **挂号意图能识别否定**：`"挂号" in "我不想挂号"` 曾为真，用户在**明确否认**时也会被占号
  （图路径实测出过票）。`pack.wants_booking` 现在走否定判定（复用红线的 `polarity`，
  并补上它缺的祈使否定「别」，同时排除"特别/别的"这类复合词）。
- **图路径必须先确认才占号**：新增 `confirm` 节点（真 `interrupt` 挂起）与
  `_after_confirm`；未确认绝不进 `book`。副作用工具的默认行为是**拒绝**，不是"先做了再说"。
- **号源归属来自调用方**：`patient_id` 经 `run_tool_loop(context=...)` 注入执行体，
  **不进工具 schema**（模型无法伪造）。此前它压根传不到执行体，**所有号都记在 `"anon"` 名下**。
- **LLM 账本真正原子**：读也在锁内、写入改成 `tmp` + 原子替换（`_atomic_replace`）。
  此前 `write_text` 先截断，并发读者会读到空文件 ⇒ 当作"今天没花过" ⇒ **闸门 fail-open**。
  实测 3 写 3 读并发 900 次记账：读到空文件 0 次、异常 0 次。
- **新增 `reserve()`，堵住 `check()`→`record()` 之间的并发窗口**：
  实测上限 $1.0、每次预留 $0.3 时，10 个并发请求只放行 **3** 个（此前会全部放行再各自记账）。
- **账本改成"一门账"**：路径移到 `Settings.llm_spend_path`（全局一份，不在域目录下）。
  此前按域分账 ⇒ 切换 `domain` 就能把上限翻成 N 份，且 clinic 会话的花费被记进 k8s_docs 目录。
  磁盘上已有的两本账（`data/index/llm_spend.json` 与 `data/index/k8s_docs/llm_spend.json`）
  已**合并求和**。
- **纯删除的一轮 `ingest` 补上 `finalize()`**：`todo` 为空时循环体不执行，
  `finalize()` 也就没被调用过，而 `delete_docs()` 只标记"脏"——结果是
  `vector_row ↔ 矩阵行`错位，重启后**静默返回错误内容**或越界 `IndexError`（已复现）。
  同时 `_reload_index` 现在校验 `vector_row` 连续性与矩阵行数，不一致就**大声失败**。
- **评测闸门缺字段 = 拒绝比较**：`check_gate` 先校验基线有
  `schema`/`gold_sha256`/`embedding_model`/`top_k`/`n_chunks`/`metrics`，缺任一即拒绝。
  此前 `int(base.get("top_k", 0))` 配 `if b_top_k and ...`，让"基线没有 top_k"整段**静默跳过**——
  实测用旧格式基线可以拿 `hit@20 = 0.9000` 通过绑定 `hit@10 = 0.8400` 的闸门（退出码 0）。
- **clinic 安全闸门不再被空集满足**：`redline_recall` 在分母为 0 时返回 `None`（此前是
  **1.0**，于是删掉全部红线用例会打印「红线召回 100.00% (TP=0 FN=0)」）；`gate_failures()`
  把"红线未评测"判为失败；clinic 基线**绑定金标指纹**（`gold_sha256` + `n_cases`，
  **指标值未改**）。
- **存储层的失败也降级**：此前 `try` 只包住 `embedder.encode`，`StoreError` 会直接抛出，
  既没有纯全文回退、也没有 `diagnostics.degraded` 里的原因。
- **引用标记正则放宽到 `\d+`**：`\d{1,3}` 会漏掉 `[1234]`，让它**静默留下**。
- **重叠尾段不再切进代码围栏**：`_safe_overlap_tail` 现在判断断点是否落在围栏内部，
  落在里面就放弃重叠。实测全量语料：围栏标记数为奇数的块 **581 → 25**（−95.7%），
  块数 18861 → **18811**。
- **`_get_clinic_react` 补上锁 + 双检**（与 `_get_clinic_agent` 同款）：并发首请求曾会构造
  两个运行时、两份互不相通的号源库存。

### Fixed — 双域（clinic）真正接通

- **`DomainPack` 自报嵌入模型与维度**（`embedding_model` / `embedding_dim`，与既有
  `normalizer_name` 同构），`cli` 在用户未显式覆盖时套用：`--domain clinic` 现在会自动用
  **jieba 归一化 + `BAAI/bge-small-zh-v1.5`（512 维）**。
- **`cmd_ingest` 改走 `pack.build_index()`**，`stats` 改用 `pack.corpus_root()`，
  `search`/`ask` 改走 `pack.retriever()`。此前 `--domain` **只换了索引目录**，
  语料/归一化/嵌入仍是 k8s 默认值——于是 `--domain clinic ingest` 会把**英文文档按
  identity 归一化、用英文模型编码**灌进 clinic 索引（本机那个 0 块的 clinic 库，
  元信息就是 `bge-small-en / 384 / identity`，正是这个缺陷的产物）。
- **clinic 索引已按中文模型全量建成并实测**：31,592 块 / 31,592 文档，
  `verify_fts()` 与 `verify_vec()` 都是 `ok`，中文查询真实命中。
- 修掉卡片引用 URL 多一层 `clinic/` 的拼接错误。

### Added — 观测（此前是"假开关"）

- 新增 `src/kubedocs/observability.py`：`configure_logging()`（`log_level` 真正生效）、
  `setup_tracing()`（真实 `TracerProvider`，console / otlp）、Prometheus 计数器与直方图、
  `span()` 上下文管理器（未启用时零开销空操作）。
- `/metrics` 端点 + 请求计数/耗时中间件（端点名取**路由模板**，避免标签基数爆炸）；
  检索单路延迟、LLM 调用次数（按能否定价分解）与累计花费都上了指标。
- `otel_enable`/`otel_exporter`/`log_level` 从"声明了没人读"变成**真实分支**；
  `otel_enable` **默认关**（console 导出器会往 stdout 打 span，测试与库场景是噪声）。

### Added — 文档锚点校验覆盖到"文件引用"

- `check_doc_refs.py` 新增一类校验：**纯 `.md` 文件引用必须存在**。
  这是此前的**整类盲区**（`文件:行号` 那条路要求"路径 + 冒号 + 数字"）——
  `docs/adr/0005-milvus-后端.md` 曾经并不存在、12 处引用它，而门禁一路绿灯。
  规则刻意收紧以避免噪声（只查首段是仓内顶层目录的路径、跳过示意与外部路径），
  代价写在脚本注释里。
- 新增 `docs/adr/0005-milvus-后端.md`（补上那 12 处悬空引用），含本轮**重测**的
  HNSW 召回/延迟数据：3000 块量级下 HNSW 与精确检索**逐位相同**（ef=16/64/128/256 都是 50/50），
  而单查询 ~0.7 s——远慢于 `sqlite-vec`。

### Changed — 陈旧文档与配置

- 删除 4 个**假开关**：`Settings.redis_url`、`langfuse_host` / `langfuse_public_key` /
  `langfuse_secret_key`；连带删除 `redis` 依赖（`types-redis`）、compose 里没人连的
  `cache`(redis) 与 `langfuse` 服务、以及 `.env.example` 里的对应项。
- `Makefile` 的 `eval` / `ingest` / `search` 补上 `--hf-offline`：README 自己实测"已缓存模型的
  联网校验会挂死 >10 分钟"，而 Makefile 此前只带 `--hf-endpoint`——照着它跑会在本机卡住。
- 重新锚定 `SECURITY.md` 的全部 `文件:行` 引用（代码改动会让行号整体位移），
  并把本轮修复的事实写进对应的 S-2 / S-3 / S-4 / S-5 与"已封闭攻击面"表。
- 更正 README/CONTRIBUTING/ci.yml 里若干**与代码不一致**的说法：
  API 端点 6→8、默认存储写错、pgvector 闸门"仍是红的"（早已修好）、
  `make check` 是否含排版门、concept 反常结论的机制（"3 条本来排第一"不成立，
  实为 **4 失 1 得**且没有一条是第 1 位）、`±11pp` 的样本口径（那是 50 条整体的 CI）。
- 清理游离文件：`.kd_probe1.sql`（未被 gitignore 的探针脚本）与未被跟踪的 `docker/`
  （其镜像选型取舍已记在 ADR-0004 第十节，没有丢失结论）。
- `.env` 里的真实密钥改为占位值。

---

### Docker 全栈实测（2026-10-13）

容器栈第一次真跑，并修掉 **3 个只有真起容器才会暴露的缺陷**：

- **容器内 `repo_root()` 解析错**（最严重）：包是 `pip install .` 装进 venv 的，
  `__file__` 上溯三层落到 `/opt/venv/lib/python3.10` ⇒ `data/` 指向一个不存在、
  也写不进去的目录。而 `/healthz` **照样返回 200**（"索引不可用"按设计不算存活失败），
  表现是"服务是绿的、`n_chunks` 是 0、`/search` 报
  `PermissionError: '/opt/venv/lib/python3.10/data'`"。
  修：`repo_root()` 支持 `KUBEDOCS_REPO_ROOT` 覆盖，compose 里设 `/app`
  （本地与 CI 不设它，行为与以前**逐字相同**）。
- **`HF_HOME` 落在非 root 用户读不到的地方**：容器以 `app`（uid 10001）运行，
  而 `/root` 是 0700 ⇒ 挂进卷的模型对 `app` **不可读**，容器内检索会静默降级成
  "没有向量路"。修：改到 `/home/app/.cache/huggingface` + `chown`，compose 挂载点同步。
- **依赖层与源码层耦合 + 无谓的编译器**：原来 `COPY src` 之后才 `pip install .`，
  改一行源码就要重下几 GB 依赖；builder 里还装了 `build-essential`（64 个包 / 87 MB），
  而依赖其实全是 manylinux wheel——受限网络下这一层实测 **>18 分钟仍没下完**。
  修：先装依赖（空占位包）、再 `COPY src` + `pip install --no-deps .`；
  编译器改成 `BUILD_TOOLS=1` 按需开启（默认不装）。

新增构建参数（**默认全部留空 = 官方源，行为与以前一致**）：`APT_MIRROR`、`PIP_INDEX_URL`、
`TORCH_CPU_FIND_LINKS` + `TORCH_CPU_VERSION`（CPU-only torch 配方；必须连版本一起给，
否则 PEP 440 下 pip 会挑数值更大的 CUDA 版）。

实测结论（细节见 README「Docker 全栈实测」与 SECURITY.md 第 6 节）：

- ✅ `docker compose up -d`：db + api 起来且 healthy；`/healthz` 报 **18811 块**；
  `/metrics` 200（`kubedocs_*` 指标按**路由模板**打标签、计数随请求增长）；
  `/search` **3 条真实命中、无降级**；`/chat` 的 SSE 里 `check_red_line` 命中「胸痛」
  并转 `status: emergency`。
- ✅ **pgvector 集成测试从 skip 变为真跑**：`test_pgvector_store.py` **15 条** +
  `test_store_contract.py` 的 pgvector 参数 **18 条** = **33 passed / 0 skipped**
  （此处原先写的是"90 passed / 0 skipped"，与任何可复现命令都对不上，已更正）；
  当前索引重灌进 pgvector 后全文/向量自检 ok、抽样 200 块向量**逐位一致**、
  字符总数两边相同，`cli --store pgvector eval --top-k 10 --gate` **退出码 0**
  （hybrid hit@10 0.8600 ≥ 冻结基线 0.8400）。
- ⚠️ **镜像"从零重建"没有在本机跑完**：重建需要的外网大文件下载**反复停滞**
  （同一构建里 tuna 能把 553 MB 的 torch 拉完，另一条 184 MB 的 wheel 却卡死 6 分钟零字节）。
  所以那次容器验证是在**现有镜像里挂载当前源码**跑的——修复本身被完整验证了，
  但"干净环境一次构建成功"这条没有跑通。
- ⚠️ 镜像 **9.91 GB**：PyPI 上的 `torch` 默认是 CUDA 版，会多拉约 2.5 GB 的
  `nvidia-*` / `triton` / `cuda-toolkit`，而这个服务是**纯 CPU** 的。CPU-only 配方
  实测能把 torch 从 553 MB+ 降到 **184 MB**，但那一次构建被打断，
  因此**不对"镜像能瘦到多少"下断言**。
- ⚠️ 基础镜像仍是**浮动 tag**（`python:3.10-slim` 实测解析到 **Debian 13**），
  与 compose 自己"版本都钉死"的纪律不一致；Docker 构建**不消费 `requirements.lock.txt`**
  （镜像里的 `langgraph` 1.2.15 / `litellm` 1.105.0 / `torch` 2.14.1 都比锁文件新）。
- ⚠️ 构建需要约 **15 GB 空闲磁盘**（中间层 + 9.91 GB 镜像）：本机 C 盘曾只剩 0.5 GB，
  构建直接以 `OSError: [Errno 5] Input/output error` 失败。

### 一键启动 + 控制台页（2026-10-14）

新增两个面向"人真的要用它"的东西，过程中又抓到 **3 个新缺陷**。

**新增：**

- **`start.cmd`（Windows 一键启动窗口）**：挑端口（8080 起往后找第一个空闲的）→ 起服务
  → 轮询 `/healthz` 直到服务**真的健康且索引非空** → 打开控制台页。服务跑在**前台**，
  所以 Ctrl+C 和关窗口都能真的停掉它，不会留下孤儿进程占着端口。
  8080 上已有可用实例时默认**附着**过去（`--force` 强制另起）；附着判据要求
  `n_chunks` 非零——实测遇到过"容器活着但后端是空表"的实例，附着上去比另起一个更糟。
- **`/console` 控制台页**（`data/web/console.html`）：与演示页 `/` 并存（职责不同，
  共用同一套设计令牌）、**零外部依赖**（离线可用，样式脚本全内联）。内容：索引体检、
  `/metrics` 摘要（含**非 2xx 单独计数**——只看端点会把 404 混进正常流量）、
  每条的**各路检索名次**（bm25/dense）、`/ask` 的降级原因、问诊 SSE 的工具轨迹与急诊高亮。
  新增路由 `GET /console`，与 `/` 共用 `_serve_page`（缺文件同样返回 **200** 说明页）。
  必须由同一个 app 提供：API 没开 CORS，页面要直接对同源发 `/search`、`/chat`。

**修复：**

- **`KUBEDOCS_DATABASE_URL` 是个假开关**。compose 给 api 设了它指向 `db` 服务，
  但 `Settings.store` 默认值是 `sqlite-vec`，而它**不读** `database_url` ⇒ 容器实际去读
  bind-mount 的 `./data/index/<domain>/kubedocs.db`，把旁边那个灌满 18811 块的 Postgres
  **完全晾着**（`db` 服务白起）。表现同样是"服务是绿的、`n_chunks` 是 0"：SQLite 跑在
  Windows bind-mount 上会报 `OperationalError: disk I/O error`，而 `/healthz` 照样 200。
  修：compose 里 `KUBEDOCS_STORE: pgvector` 与 `DATABASE_URL` **成对**给出。修后容器内
  `stats --verify` → 1720 文档 / 18811 块、FTS 与向量一致性均 `ok`，`/search`
  `overlap=13`（关键词路与向量路都参与融合）、`degraded` 为空。
- **`has_api_key()` 会被 `.env` 里的说明文本骗过**。它原先只判"非空"。本机 `.env`
  里留的是一句中文说明（38 字符 / 78 字节，**20 个非 ASCII 字符**），于是判为"已配置"
  ⇒ `/ask` 拿着它发起真实请求，LiteLLM 把它放进 HTTP 头时抛
  `'ascii' codec can't encode characters in position 9-24`，`degraded` 里回显的就是这条。
  **这比直接降级更坏**：用户以为模型坏了，实际是配置里写了一句中文。修：两条判据——
  `_PLACEHOLDER_MARKERS` 词表（`your-` / `placeholder` / `changeme` / `example` /
  `xxxx` / `<` / `>`）与**必须全是 ASCII**（密钥要进 `Authorization` 头，非 ASCII
  在那里根本无法编码；真密钥都是可打印 ASCII，不会误伤）。修后同一条请求变成干净的
  "未配置 LLM 密钥，仅返回检索结果"，且**不再发起网络调用**；日志明说
  "含 20 个非 ASCII 字符，不可能是密钥"。
- **预算 gauge 在"一次都没调用过"时谎报"不限"**。`kubedocs_llm_budget_usd` 原先只在
  `observe_llm_call()` 里被设值，于是在"配了 $2.00 上限、但还没调用过 LLM"的进程里
  它暴露的是 gauge 默认值 **0**，而这个 gauge 的约定恰恰是"**0 表示不限**"——
  控制台如实读指标，就会把一个真实存在的花费上限显示成"不限"。这不是少显示一个字段，
  是显示**错的**金额控制。修：`publish_config_gauges()` 在 `create_app` 装配点发布一次
  （与有没有密钥无关）。容器内实测启动即为 `kubedocs_llm_budget_usd 2.0`。

**新增 `tests/unit/test_observability.py`（12 项）**：此前**整个观测面没有任何一行测试**，
而 README 与 SECURITY 都把它当作已实现的能力对外声明（"开关不再是假开关"）——
声明的唯一证据是"手工 curl 过一次"，而手工 curl 证明不了回归：把中间件的标签从
**路由模板**改回原始 URL，页面照样出数，只有 Prometheus 侧的聚合会悄悄碎掉。
现在把口径写死成断言：日志级别真的落到 root logger、`/metrics` 的六个指标名真的出现在
**暴露文本**里、HTTP 计数按路由模板打标签（带查询串的两次请求必须累加到同一条序列）、
状态码进标签（否则 404 会混进正常流量）。

**实测（本机，16 GB / Windows 11）**：主机直跑与容器两条路都完整验过。检索冷启动
墙钟 **69 s**（加载嵌入模型，暖机 0.1 s）——`create_app()` 有意不加载模型以保持启动廉价，
所以 `/console` 的检索面板会把等待秒数实时抖出来，而不是让页面看上去"点了没反应"。
另记一条环境事实：本机同时跑 Docker Desktop 时提交内存只剩约 1.6 GB，此时**主机**进程
加载 torch 会以 `os error 1455`（paging file too small）失败、检索降级成纯 BM25
（HTTP 仍 200、`diagnostics.degraded` 写明原因）；内存宽裕后同一条路 `overlap=13`、
无降级。⇒ 属瞬时资源问题，不是代码问题。

### 自洽性检查：CI 本地流程 + 闸门精确判据 + 语料漂移（2026-10-14）

一轮以"查项目自身的逻辑自洽、完整流程能否走通、md 声明与代码是否冲突"为目标的改动。

**新增：**

- **`scripts/ci_local.py` + `make ci-local`**：本地按 CI 口径把**三个 job** 都跑一遍。
  此前本地只有 `make check`（= `quality` job 的静态+测试部分），`pipeline` 与 `gate`
  本地没有入口 —— 而这两个 job 在 CI 里都是**阻塞**的，所以"本地 `make check` 全绿"
  **并不等于** "CI 会绿"（何况仓库没有 remote，`ci.yml` 至今没在任何 runner 上跑过）。
  三条设计：**Postgres 不可达就直接失败**（CI 里那族用例有 service、必须真跑，而
  "skip"与"pass"在日志里长得几乎一样）；**失败即停**（与 `needs: [quality]` 一致）；
  pipeline 的 hashing 索引建到**临时目录**（不碰那份重建要 35 分钟的真实索引）。
  实测 **3/3 job 通过**，失败路径退出码 1。

**修复：**

- **`corpus_sha` 从未被比较**（`evaluation/harness.py`）。它的 docstring 一直写着
  "用于判断两次评测是否同一批数据"，但它只被**计算与落盘**（报告里有、基线里有），
  闸门里一次都没用过 —— 实际把关的只是 **2% 的块数容差**。
  实测触发：重拉上游语料后文档 1720→1722、块数 18811→18822（**+0.058%**），
  落在容差内 ⇒ **静默通过、一个字都不打印**，而 CONTRIBUTING 与 `ci.yml`
  都声称闸门会拒绝跨语料比较。修：`corpus_sha` 进必需字段并**精确**比较，
  不等即拒绝（退出码 2），逃生口 `--allow-corpus-drift` 放行但**留警告**；
  块数容差降级为"变了多少"的细节提供者（原先它是唯一判据，且近似）。
- **`eval_sets` 的 docstring 说反了**（`domains/base.py`）：原文是"内核据此跑评测与闸门"，
  但全仓只有 `cli domains` 的打印与 `/domains` 响应读它，评测走的是显式路径参数。
- **clinic 的 `eval_sets` 注释已过时**（`domains/clinic/domain.py`）：原注释说金标集
  "要在 P4 才建"，而 `data/eval/clinic_gold.jsonl`（**37 条**）与
  `data/eval/clinic_baseline.json` **都已存在**、`scripts/eval_clinic.py --gate` 也在跑。
  仍返回空，但理由更正为"clinic 金标是另一种形状、不走这个协议字段"。
- **过期数字**（写死的数会腐烂，而腐烂的注释会被当成现状）：
  SECURITY/CHANGELOG 里 pgvector 用例"**90 passed / 0 skipped**"与任何可复现命令都对不上
  → 改为 **33**（`test_pgvector_store.py` 15 + `test_store_contract.py` 的 pgvector 参数 18，
  含可复制的命令）；`ci.yml` 里写死的 18861 / "12 条" / "30 条" / "706 passed、736 passed"
  一律改成**判据或规则**（只保留"milvus 那一族可 skip"这类不随用例增删变化的表述）；
  Makefile 的"本机 Docker Hub 不可达、未实测" → 已实测（并注明 `up --build` 与
  "镜像里那份旧源码"的关系）；CONTRIBUTING 的"12 条"→15 条。

**尺子变更（按 CONTRIBUTING 第 5 节留档）：**

上游语料取自 `main`、没有钉版本，因此索引规模变化，基线重冻：

| | 重冻前 | 重冻后 |
|---|---|---|
| 文档 | 1720 | 1722 |
| 块数 | 18811 | 18822 |
| `corpus_sha` | `f6be8d659a348c60` | `f9ffa225825bd593` |
| hybrid hit@10 | 0.8400 | **0.8400（不变）** |
| mrr@10 | 0.5205 | 0.5239 |
| ndcg@10 | 0.5469 | 0.5495 |

**这是一次口径变更，不是把指标"修绿"**：hit@10 前后完全相同，mrr/ndcg 略升。
pgvector 侧按同一口径重灌（抽样 200 块向量逐位一致、字符总数 14762302 两边相同）后
hit@10 = **0.8600**、`--gate` 退出码 0。

**审计结论（这些项查过、没有改动）：** `Settings` 的 **49 个字段全部**有真实读者
（无假开关；注意自动扫描会对"字段名 vs 派生属性名"误报，`llm_fallbacks` 就是一个假阳性）；
退出码 `0/1/2/3` 与 README 声称**完全一致**；`agent/` **13** 个模块、
**4** 个 Store 后端实现、**9** 个端点，均与文档声明相符。

### 性能基准与评估（2026-10-14）

第一次测项目的**整体性能**，并把结果按基准输出的结构写进 README「性能基准」一节。

**新增 `scripts/bench_api.py`**：HTTP 层的并发 × 延迟分位 × 吞吐 × 错误率。
这是仓库里唯一从没测过的一面（SECURITY 第 6 节一直写着"没有做过负载测试"）。
两个口径**分开报**：客户端（含序列化与环回）与**服务端**（读 `/metrics` 直方图，
**按窗口差值**算分位——直接对累计值取分位会被"冷启动那一枪加载模型"污染，
实测第一次跑就是这么被骗的：报出"服务端 p95 = 750ms"而客户端只有 100ms）。
不引入 k6/locust：本机没有，且项目本来就依赖 httpx。

**修掉一个实测抓到的性能缺陷**：`/healthz` **每个请求**都调 `_DomainRuntime.stats()`，
而它里面那条 `SELECT COUNT(*) FROM chunks WHERE body LIKE '%```%'` 是 18822 行大文本列的
**全表扫描**（store 自己的连接上实测 17.31 ms，占 `stats()` 的 24.40 ms 的绝大部分——
同批 `COUNT(documents)` 与 `SELECT meta` 都只要 0.01 ms）。`/search` 也会走它
（`_index_note` 只为拼一句提示），且整段都在 store 锁里 ⇒ 每个请求过一次 ~20 ms
串行临界区。修法：`stats()` 加 **2 秒 TTL 缓存**（探针间隔 10 s，所以仍反映"最多 2 秒前"；
`cli stats --verify` 走 store 直连，语义一个字没变）。效果：

| 并发 | `/healthz` p50 前 → 后 | 吞吐/s 前 → 后 |
|---|---|---|
| 1 | 54.6 ms → **1.6 ms** | 19.0 → **581** |
| 8 | 415.5 ms → **8.2 ms** | 19.2 → **894** |
| 16 | 802.2 ms → **27.7 ms** | 19.6 → **495** |

⚠️ 同一修复对 `/search` **几乎没有帮助**（c=1 的 p50 82.2 → 80.4 ms）——它的瓶颈是
**查询编码**而不是这次扫描。这正是"先量再改"的价值。

**评估结论（详见 README 该节的表）**：

- **`/search` 在并发 1 就已 CPU 饱和**：吞吐 12~21 req/s，并发 1→8 吞吐不涨而 p50 线性上涨
  （本地 80 → 650 ms）。瓶颈是编码（dense-only p50 65 ms vs bm25-only 14 ms ⇒ 占端到端 ~80%）。
  要提吞吐就得做**量化/ONNX/更小模型或多副本**，而不是调批大小（已有消融证明批越大越慢）。
  多副本会踩到 S-2 的单进程预算账本 ⇒ 两者必须一起做。
- **换存储后端是权衡、不是升级**：`pgvector` 的 dense 比 `sqlite-vec` 快 4.3×
  （4.69 vs 20.03 ms，HNSW vs 穷举），但 keyword 慢 5.2×（58.60 vs 11.31 ms）。
- **一个如实记录、未解释的差距**：跨后端 top-10 一致率，向量路 90%（135/150，HNSW 本就不保证逐位相同），
  但**全文路集合一致率只有 34%（51/150）**。两个引擎都是 BM25（`verify_fts()` 返回 `ok`，
  不是降级路径），但 FTS5 与 `pg_search` 的分词/打分细节不同。记在这里而不是藏起来。
- **`sqlite`（旧 numpy 侧车）后端在当前索引上打不开**：`vector_row` 全是 -1，因为这份索引是
  `sqlite-vec` 建的，而 `chunk_vectors.npy` / `vector_row` 只有 `sqlite` 后端维护。
  它的报错**原先会误导**（建议 `ingest --rebuild`，而默认后端仍是 `sqlite-vec`，重建后照样打不开），
  已改成明确指出"重建必须指定同一个后端"。
- **交叉验证一致**：嵌入吞吐实测 **8.55 块/s**（117 ms/块）⇒ 外推全量 **36.7 分钟 / 18822 块**，
  与 README 早已记录的"约 35 分钟"吻合；`/healthz` 就绪 **1.6 s**、首个 `/search` **25.5~27.5 s**。

**没测什么（同样写进文档）**：长稳压（内存泄漏/GC/句柄/连接池）、多副本、LLM 路径
（`/ask`、`/chat` 要真花钱）、pgvector 在百万块量级、渗透与模糊测试。
本机同时跑着 Docker Desktop，数字**不可迁移**。

---

## [0.1.0] - 未发布（no tag yet）

首个版本的全部内容，由 **33 个提交**归纳（提交日期区间 2026-09-24 ~ 2026-10-11，
取自 `git log` 的 committer date）。

> 说明：其中 2026-10-08 ~ 2026-10-11 的 11 个提交原先**没有记进本文件**，
> 本次补入下面的 Added / Changed / Fixed。

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
- **开源门面 ＋ 可插拔向量存储后端**：`Store` 协议下的 `sqlite-vec`（默认）与 `pgvector`
  实现、`store/__init__.py` 的 `BACKENDS` 单一出口、按域分目录的索引布局（`c6a8ff0`）
- **文档锚点内容级校验器** `scripts/check_doc_refs.py`：`文件:行号` 引用必须指着它自己
  声称的东西（不只是"文件存在、行号不越界"），并重锚了 24 处失效引用（`4eb5e10`）
- CI 真实性：独立的 `gate` job（真实模型 + 冻结基线卡检索指标）+ `quality` job 加
  Postgres service 让 DB 集成测试**真跑**（`b415727`）

### Changed

- 回答上下文去冗余：**同一文档最多收录 2 段**（`config.max_chunks_per_doc` 默认 2）（`76471b8`）
- `/chat` 从确定性图切到 **LLM 驱动的工具循环**（模型自己选工具、自己决定步数），
  并留下真实运行记录 `scripts/demo_clinic_agent.py`（`be399a4`）
- `ruff format` 全量重排并接入 CI 阻塞门禁（`ruff check` 与 `ruff format --check` 是两道门）
  （`fcd7a55`）
- MIT 版权署名改为集合式约定 `The kubedocs Authors`（`e808e73`）
- **pgvector 全文路换 ParadeDB `pg_search` 真 BM25**：镜像从 `pgvector/pgvector:pg16`
  换成 `paradedb/paradedb:0.26.1-pg16`（**仍是 PG16**，数据目录原地上机）。
  hybrid hit@10 **0.7400 → 0.8600**（≥ 冻结基线 0.8400，`--gate` 退出码 0），
  关键字路 `EXPLAIN` 从 **42/50 条 `Seq Scan`** 变成 **0/50**，hybrid p50 120.1 → 72.6 ms
  （`5ceaa6f`）

### Fixed

- 健壮性与可用性收尾：LLM 调用失败改为**降级**（检索结果照常返回而不是整个请求崩）、
  嵌入批大小默认收紧、修正 CLI 全局参数位置（`f60a8f7`）
- **`.env` 里的密钥到不了模型调用**（pydantic-settings 不写 `os.environ`，而
  LiteLLM 与 `has_api_key()` 读的正是它）——README 描述的那套流程此前是**静默失效**的；
  同时修掉"测试可能真实花钱"的两个缺陷（`65537e0`）
- 修 A/B 脚本的 ruff 告警（`S110` / `C416` / `SIM105`）（`b11f2c0`）
- 修正 P5 对 self-ask 的错误判断，并在复合问题上做了一次**真正隔离"分解"**的 A/B
  （结论仍是负面）（`3fcd4c3`）
- **工具协议接线**：clinic 号源工具 + `bootstrap.register_domain_tools` 幂等注册
  ——在此之前**全仓库从未调用过 `register`**，整条工具链路是死的；同一次提交补上
  `agent/tool_loop.py` 与对话前端 `data/web/index.html`（`1090c71`）
- 修六项：工具循环丢 history / ADR-0004 的假设被证否 / `make help` / `ef_search` 默认 200 /
  `.dockerignore` / 文档残留（`aaffab1`）
- 修 `eval --gate --json` **静默跳过闸门**的缺陷（`1ad984d`）
- 修三个 bug：**闸门不看分诊** / `/chat` **无成本上限** / `thread_id` **无属主校验**
  ——三者都是"文档说有、代码没有"的缺口（`e1307cd`）
- 堵住评测闸门的一个**假绿洞**：`check_gate` 从不比对基线的 `top_k`
  （`f7db3d7`；本轮又把"基线缺字段"这条退路一并关掉）
- 修 `--json` 输出污染（stdout 上多一行人类旁白会让 `json.loads` 整体失败）等三处缺陷
  （`fcd7a55`）
- **删除 `KUBEDOCS_ENV` / `is_prod` 两个假开关**：全仓没有任何一行代码读它们，
  只做声明与校验，却让 CI 里的 `KUBEDOCS_ENV: ci` 看起来在起作用（`4eb5e10`）

### Removed

- 无**文件**被删除（`git log --diff-filter=D` 为空）。
- 但**有配置项被删除**（旧说法"没有删除项"不准确）：`KUBEDOCS_ENV` 与派生的 `is_prod`
  在 `4eb5e10` 被移除，`.env.example` 与 `ci.yml` 里的对应设置一并清掉。

---

## 怎么读这个 changelog

- 每条后面的短哈希指向**一个真实提交**，不是一组合并叙述：`git show 1090c71`。
- 负面结论（self-ask 无用、多跳更差）**照实写进来**，不粉饰成"优化了检索"。
- 想看每个数字是怎么来的，看 `README.md` 的「实测数字」「评测：怎么保证分数不是自欺」
  两节，以及 `docs/检索与评测口径.md`。
- 尚未完成的部分逐条记在 README 的 Roadmap 与 `SECURITY.md` 第 6 节，
  不写进本文件（还没做的事不是"变更"）。
