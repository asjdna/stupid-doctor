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

## 改评测口径

评测口径的分层与拒绝比较逻辑定义在 `docs/EVALUATION.md`。
调整口径时请同时更新基线（`eval --write-baseline`），并在 PR 里说明为什么旧基线不再适用。

## 提交信息

说明**改了什么、为什么**，以及**怎么验证**。涉及行为变更时给出可复现的验证命令。

## 报告问题

请附上：复现步骤、期望与实际结果、相关日志或指标，以及环境（域、后端、模型）。
涉及安全问题时见 [SECURITY.md](SECURITY.md)。
