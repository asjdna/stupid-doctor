# 贡献指南

感谢你愿意花时间。这个仓库的纪律只有一条，其它都是它的推论：

> **没跑过的就不写"已完成"；数字必须是实测的。**

具体到提交：**门禁必须绿、评测与标注集的改动必须留档、`.env` 永不入库。**

---

## 1. 开发环境

需要 **Python ≥ 3.10**（`pyproject.toml` 的 `requires-python`）。仓库**不要求** Docker：
默认存储是嵌入式 SQLite + numpy，不需要任何外部进程。

```bash
python -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

> 想顺便复用系统里已装的重型依赖（torch 等）可以加 `--system-site-packages`，
> 但**不要**把这个当成默认要求——干净环境能装起来才是 CI 要验证的事。

如果你要用行业口径的评测包（`ragas` / `deepeval`，属于 `eval` optional 组），
再装一次：`pip install -e ".[eval]"`。**注意**：这一组在当前 `requirements.lock.txt`
里**没有被覆盖**（生成锁文件的机器上没装它，header 里有明写），装完请顺手重跑
`make lock`（见第 6 节）。

**不要**改 `.venv/` 里的任何文件，也不要把它提交（`.gitignore` 已覆盖）。

---

## 2. 跑门禁：`make check`

```bash
make check          # = lint + fmt + type + test + refs，与 CI 的 quality job 等价
```

五件事串起来，**全部阻塞**（非零退出码即失败）：

| 目标 | 命令 | 抓什么 |
|---|---|---|
| `make lint` | `ruff check .` | `E/F/W/B/I/UP/S/C4/RET/SIM`（含 flake8-bandit 的 `S` 子集） |
| `make fmt` | `ruff format --check .` | 排版漂移。`ruff check` 管规则、它管排版，**两道不同的门**都得在 |
| `make type` | `mypy src scripts` | `disallow_untyped_defs` 等，**新函数必须带类型注解** |
| `make test` | `pytest tests/unit -p no:cacheprovider -rs` | 单元测试；不需要网络与嵌入模型，秒级 |
| `make refs` | `python scripts/check_doc_refs.py` | 文档里的 `文件:行号` 引用是否指着它声称的东西 |

单独跑也一样：

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src scripts
python -m pytest tests/unit -p no:cacheprovider -rs
python scripts/check_doc_refs.py
```

> ⚠️ **直接调 pytest 时不要再加 `-q`**：`pyproject.toml` 的
> `addopts = "-q --strict-markers"` 里已经有一个，再叠一层会变成 `-qq`，把
> **最后那行汇总**（`N passed`）一起吞掉，你会看不到实际通过数。
>
> `-p no:cacheprovider` 只是为了不留下 `.pytest_cache`（`make test`、CI 与
> `make clean` 的清单都按这个来），与测试结果无关。
> `-rs` 把每条 skip 的**原因**打进日志：CI 里**允许** skip 的只有 milvus 那一族
> （`pymilvus` / `milvus-lite` 是可选依赖，`pyproject.toml` 没声明），出现别的 skip 就是回归。
> 这三处的命令是**同一份**：
> `Makefile` 的 `test` 目标、`.github/workflows/ci.yml`、以及上面这个代码块。

**没有 `make` 的机器**（原生的 Windows 环境常见）可以直接跑上面那几条命令——
它们才是真相，`make` 只是把它们串起来。

**单元测试绝不能联网、不能花钱**：`tests/` 用哈希向量与打桩 LLM。
唯一需要外部进程的是 `tests/unit/test_pgvector_store.py`（12 条）：它要一条真实
Postgres——CI 用 `pgvector/pgvector:pg16` service 起它（见 `ci.yml` 的 quality job，
同一条 service 也让 `test_store_contract.py` 的 `pgvector` 参数真正出现），
本地没起库时整个文件 skip 并把探测到的错误写进 `reason`（`-rs` 能看到）。
如果你要写一个"真的调模型/真的加载模型"的测试，标 `@pytest.mark.slow`
（`pyproject.toml` 的 markers 已声明 `integration` 与 `slow`，且开了 `--strict-markers`，
**没声明的 mark 会直接失败**）。

---

## 3. 提交信息风格（照着现有历史写）

这个仓库的历史**不用** `feat:` / `fix:` 这类 Conventional Commits 前缀。
真实风格是：**单行中文主题**，可用全角冒号分成"范围 + 内容"，括号里写证据或结论。
看 `git log --oneline` 就能确认：

```
里程碑 1：项目骨架 + 语料获取 + 切块器（含真实语料体检）
P0：双域平台地基（域协议 / 多域索引 / 中文归一化）+ 原生系统对接检测
回答层 + CLI 加固：带引用回答、引用核对、拒答、预算上限
修复 .env 密钥到不了模型调用 + 测试可能真实花钱的两个缺陷
修 A/B 脚本的 ruff（S110/C416/SIM105）：成本统计失败改为显式保留原值而非空 pass
P8：用例工厂 + 样本扩到 46 条 —— 多跳被**有统计意义地**证否，且依赖机制从未触发
```

要点：

1. **一行说清"改了什么 + 为什么"**；需要长解释就写正文，不要写成三行主题。
2. 用 `里程碑 N：` / `PN：` 标阶段；一次性小修直接用动词开头（`修` / `修复` / `接线`）。
3. **负面结论也要写进主题**。这个仓库的历史里有好几条"做了实验、结论是没用"的提交，
   它们是资产而不是失败——不要把它们压缩成"优化检索"这种含糊说法。
4. 不要写 `update`、`fix bug`、`misc`。
5. 引用规则号（`ruff` 的 `S110/C416/SIM105`）时写清楚，省得后人再查一遍。

---

## 4. 提交前检查清单

- [ ] `make check` 全绿（或三条命令各自绿）
- [ ] 新增/修改的函数有类型注解（`mypy` 会拦）
- [ ] 新增行为有单测；修复缺陷时**补一条能复现该缺陷的测试**
- [ ] `git status --short` 里**没有** `.env`、`data/raw/`、`data/index/`、`*.db`
- [ ] 改了依赖 → 重跑了 `make lock`（见第 6 节）
- [ ] 改了评测集 / 基线 / 提示词口径 → 留档了"改之前 vs 改之后"两个数字（见第 5 节）
- [ ] 文本文件是 **LF**（`.gitattributes` 已强制 `* text=auto eol=lf`）

---

## 5. 评测纪律（这个项目最容易自欺的地方）

1. **改了尺子就必须说。** 如果动了 `data/eval/**` 的标注集或基线，请照
   README 的「[标注集修正的留档](README.md#标注集修正的留档改了尺子就必须说)」一节的做法：
   **把修正前后的数字都留在文档里**，并写清每条标注为什么改。
   只留修完之后的漂亮数字，等于把尺子调松了再宣称进步。
2. **跨标注集/跨模型/跨语料的分数不可比**，闸门会拒绝比较（这是刻意的）。
   改了问题、嵌入模型或索引规模，就必须重建基线：`make eval-baseline`。
3. **不许把 `hashing` 向量的分数当质量结论**（它只擅长字面重合）。
   质量结论必须用真实模型跑，并写清跑它的机器与耗时。
4. **闸门绿 ≠ 什么都好**（但反过来说也成立：**闸门红 ≠ 一定是你改坏了**）。
   clinic 域的闸门现在**看分诊**：未评测或低于 60% 都判失败
   （`src/kubedocs/evaluation/clinic.py` 的 `gate_failures()`）。
   所以 `python scripts/eval_clinic.py --gate` **当前是红的（退出码 2）**，
   因为冻结基线里就记着 `triage.accuracy = 0.0`。
   **这条红不要"修"**：既不许改基线，也不许放宽门限——那是把红灯涂成绿灯。
   汇报时必须把"闸门红的原因是有据可查的既有缺口"说清楚（见 `SECURITY.md` S-4）。
   同理，**pgvector 后端**上的 `cli eval --store pgvector --gate` 也是红的
   （冻结基线的口径是 sqlite/sqlite-vec，而 pgvector 的 hybrid hit@10 是 0.74 而非 0.84），
   而**默认后端**的 `cli eval --gate` 是**绿的**（本机实测 hit@10 = 0.8400、退出码 0）。
   理由与四条候选路径见 ADR-0004 第五、九节。
5. **没有实测就不要写数字。** 宁可写"未验证"，这一条比任何格式要求都重要。

---

## 6. 改了依赖 → 必须重跑锁文件

`requirements.lock.txt` 由 `scripts/gen_lock.py` 从 `pyproject.toml` 的
`[project].dependencies` 与各 optional 组出发做**依赖闭包遍历**生成：

```bash
make lock                                     # 重新生成
python scripts/gen_lock.py --check            # 只查漂移（有漂移 → 退出码 1，不写文件）
```

为什么必须手动做这一步：**dependabot 只看得懂 `pyproject.toml`，不维护锁文件**。
不重跑的话，声明写 `>=`、锁文件还钉着旧版本，**没有任何东西会报警**。

两个已知边界，别对它期待过高：

- 它**不做版本求解、不写哈希**：版本取自**当前解释器里已装**的包，
  所以先 `pip install -e ".[dev]"` 再生成。换环境/换 Python 大版本请重跑。
- optional 组如果**没装**，会被**跳过并写进锁文件 header**（不凭空编造版本）。
  装了之后记得再跑一次 `make lock` 把它纳入。

---

## 7. 不要提交 `.env`

`.env` 里有真实密钥，且已被 `.gitignore`（`.env` / `.env.*` / `!.env.example`）覆盖。

- **不要** `git add -f .env`，不要把它复制成 `config.local.py` 之类再提交。
- **不要**在 issue / PR / 截图里贴出密钥、`llm_spend.json` 或完整的环境变量。
- 加了新的配置项：改 `.env.example`（它**必须**是空值占位），**不要**动 `.env`。
- 报 bug 需要贴日志时，先把密钥相关行删掉。

---

## 8. 提交 PR

- 从 `main` 切分支，一个 PR 只做一件事。
- PR 描述请照 `.github/PULL_REQUEST_TEMPLATE.md` 的清单填，重点是：
  **跑了什么命令、看到什么输出、哪些没跑**。
- 不要顺手改无关文件；`src/**`、`tests/**`、`data/**` 之外的脚手架文件
  （`Dockerfile` / `docker-compose.yml` / `.github/workflows/` / `Makefile`）改动
  请在描述里单独说明理由。
- **不要**给 README 加 CI badge：仓库没有 remote、CI 从未在真实 runner 上跑过，
  badge 会显示 failing，那是个假的"已就绪"信号。

有问题就开 issue，模板见 `.github/ISSUE_TEMPLATE/`。
