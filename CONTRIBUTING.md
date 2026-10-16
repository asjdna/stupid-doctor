# 参与开发

## 环境

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

开发不需要 GPU。嵌入模型首次使用时会下载（受限网络下可设 `KUBEDOCS_HF_ENDPOINT`）；
不下载模型也能跑大部分测试：`KUBEDOCS_EMBEDDING_PROVIDER=hashing`。

## 提交前要跑的检查

```bash
python -m pytest tests/unit -q     # 单元测试
python -m ruff check .             # 静态检查
python -m ruff format --check .    # 格式
python -m mypy src scripts         # 类型
python scripts/check_doc_refs.py   # 文档里的 文件:行号 引用必须指向正确内容
python scripts/check_deps.py       # src/ 里 import 的第三方包必须已在 pyproject 声明
```

一条命令跑完整套验证（静态检查、单元测试、全量摄取自检、检索闸门）：

```bash
python scripts/ci_local.py
```

`ci_local.py` 需要本机 Postgres 可达（`docker compose up -d db`，或设 `KUBEDOCS_DATABASE_URL`）。
它把"需要数据库却被跳过"视为失败——**跳过的测试和通过的测试在日志里长得一样**，
这是必须避免的。

## 代码风格

- 类型标注：公开函数与类都要有；`mypy src scripts` 是阻塞门。
- 文档字符串：说明**这个模块/函数做什么、怎么用、有什么约束**，面向第一次读代码的人。
  重点写清对外契约、不变量、参数语义与安全约束。
- 注释解释"为什么"，代码本身说明"是什么"。
- 行宽 110，格式由 `ruff format` 决定，不要手工对齐。

## 加一个域

1. 新建 `<domain>/`，声明语料路径、嵌入模型、检索器配置与工具；
2. 在 `domains.load_all()` 注册一行；
3. 若该域需要自己的工具，在域包里注册到 `ToolRegistry`。

域之间必须隔离：各自的索引目录、存储 schema、向量维度与归一化器。

## 加一个工具

在 `agent/tools.py` 注册一个 `ToolSpec`：名称、说明、参数模型、
**"什么时候不该用"**（必填，它比正向描述更能约束模型）、是否有副作用、执行体。

同一份注册表同时供给 HTTP `/tools`、Agent 工具循环与 MCP server（`kubedocs mcp`），
因此**不要另写一份工具定义**。

执行体有两种等价形状（`handler(params)` 收 pydantic 模型，或 `handler(**kwargs)`），
调用方按签名自动分派（`agent/tool_loop._invoke`，MCP 层复用同一份），选哪种都不会漏。

有副作用的工具默认不暴露；如需暴露，必须同时保证"调用前显式确认"。

## 加一步服务端编排

需要固定顺序时（采问 → 取证 → 判定 → 落库），改 `clinical/orchestration.py` 的工具循环，
而不是把流程写进提示词。能交给模型的判断就交给模型，代码只保留清单、执行器、轮次上限与白名单。

## 改提示词

提示词在 `clinical/prompts/system.md`。它是产品契约的一部分：

- 内容变更会改变提示词版本指纹，评测结果里会记录该指纹；
- 硬性条款由测试逐条断言存在，删条款会导致测试失败——这是有意的，
  如果你确实要删，请连同测试一起改，并在 PR 里说明理由。

## 怎么跑评测与监听

clinic 域的评测按引擎分臂（`--arm`），跑哪个臂决定这台机器要不要索引、要不要密钥、会不会花钱：

```bash
python scripts/eval_clinic.py --arm rule      # 规则臂：离线、免费、确定性（线索质量读数）
python scripts/eval_clinic.py --arm both      # 默认：规则臂 + 工具循环臂（产品路径，会花钱）
python scripts/eval_clinic.py --arm reasoner  # 单发对照臂：不检索，便宜（对照"有工具 / 没工具"）
```

工具循环臂（`--arm llm` / `--arm tool`）跑的是与 `/consult` 同一条路，需要索引 + 密钥。
报告与 JSON 里判断臂的键是 `llm`，`engine` 字段说明实际跑的是 `tool_loop` 还是 `reasoner`。

趋势与回归靠长期监听，而不是靠翻档案：

```bash
python scripts/monitor_clinic.py --status                  # 只看历史趋势（不花钱、不联网）
python scripts/monitor_clinic.py --run                     # 跑一轮：评测 → 留档 → 趋势 → 提醒
python scripts/monitor_clinic.py --follow --interval 3600  # 常驻（容器边车 / cron 都行）
```

`alert`（不变量被触发 / 这一轮没跑成 / 有题目没拿到判断）会让退出码变 1；`note`（召回下降、
误报上升、分诊下降、单题花费上升、提示词或金标集变化）只打印，默认不改退出码，`--strict`
才硬拦。`note` 与最近 `--window` 轮的实际区间比，工具循环臂与单发臂分开比。

**改装配只改 `src/kubedocs/clinical/runtime.py`**：`/consult` 与评测的工具循环臂共用
`build_clinical_assembly`（同一次工具注册、同一个面板建造器、同一本日预算账、同一套安全门）。
在两边各写一份装配，评测测的就是一个"产品不会走的形状"，而分数看起来仍然合理。评测侧的
差异只有三处，都是刻意的：号源用 `MockAppointmentProvider`、不落就诊记录、不写审计文件。

趋势记录（`data/eval/clinic_trend.jsonl`）与治理档案（`data/eval/clinic_archive/`）是同一轮
评测的两个落点：档案是给人看的完整报告（带 UTC 时间戳、永不覆盖），趋势记录是给监听比对的
扁平一行（`--archive` 时同时追加，`--trend-path` 可改落点）。两者都不记患者原话。

## 改评测口径

评测口径的分层与拒绝比较逻辑定义在 `docs/EVALUATION.md`。
调整口径时请同时更新基线（`eval --write-baseline`），并在 PR 里说明为什么旧基线不再适用。

clinic 域没有可调的门限：指标是记录，参考目标（召回 100% / 误报率 ≤20% / 分诊 ≥60%）只是
「差多远」的参照，不判通过/不通过。**要让某个指标变好，只能真的改实现**（提示词、检索、
用例集），改指标不是改门限。

改**安全资产**要留痕。安全资产指：提示词（`src/kubedocs/clinical/prompts/system.md`）、
危险信号词表（`src/kubedocs/domains/clinic/redline.py`）、症状词表
（`src/kubedocs/domains/clinic/data/symptom_dict.py`）、科室表
（`src/kubedocs/domains/clinic/data/departments.py`）、金标集
（`data/eval/clinic_gold.jsonl`）。改了其中任何一个，都必须更新
`data/governance/safety_assets.json` 里对应条目的 `sha256` / `version` / `rationale`：

```bash
python scripts/check_governance.py --update-hashes   # 只刷哈希
```

`--update-hashes` **只刷哈希**：版本号与「为什么改」必须由人写——机器能代填的只有哈希。

新增或删除**确定性安全不变量**要同时改三处，否则 CI 的 `scripts/check_governance.py` 会红：

1. `src/kubedocs/clinical/safety.py` 的 `INVARIANTS`（代码里的唯一来源）；
2. `data/governance/safety_assets.json` 的 `invariants` 条目（负责人 / 生效日期 / 理由 /
   `verified_by`）；
3. `tests/` 里至少一处引用它的 id——没有测试覆盖的不变量等于没人守。

分层与治理机制的完整说明见 [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md)。

## 提交信息

说明**改了什么、为什么**，以及**怎么验证**。涉及行为变更时给出可复现的验证命令。

## 报告问题

请附上：复现步骤、期望与实际结果、相关日志或指标，以及环境（域、后端、模型）。
涉及安全问题时见 [SECURITY.md](SECURITY.md)。
