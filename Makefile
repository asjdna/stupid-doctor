# 常用入口。原则：`make check` 一条命令跑完所有静态闸门，与 CI 等价。
#
# 本机直跑（默认，不需要 Docker）与 compose 是两条**并列**的路径：
#   make ingest / make search / make eval   -> 本地嵌入式索引（默认 sqlite-vec，可切 sqlite / pgvector）
#   make up / make down / make logs         -> compose 全栈（需 Docker；**已实测**，见 README「Docker 全栈实测」）
# 详见 docs/adr/0002。

# venv 解释器按平台取路径，不能写死一种：
#   · Windows 的 venv 在 `Scripts\` 下，且 **cmd.exe 不把 `/` 当路径分隔符**——
#     实测 `.venv/Scripts/python -m ruff check .` 在 cmd 下报
#     “'.venv' 不是内部或外部命令”（mingw32-make 把整行交给 cmd 时才暴露）；
#   · POSIX 的 venv 在 `bin/` 下。
ifeq ($(OS),Windows_NT)
PY := .venv\Scripts\python
else
PY := .venv/bin/python
endif

# 为什么不再用 `RUN = $(PY) 2>/dev/null || $(PY_ALT)`：
# 那条写法假设 PATH 上一定有 POSIX shell（`2>/dev/null`、`||` 都是 sh 语法，且要求
# 路径能被 sh 解析）。本机 PATH 上没有任何 sh.exe，mingw32-make 于是落到 cmd.exe，
# 整行被当成一条 cmd 命令 → 实测 `mingw32-make check` 退出码 2。
# 平台上面已经显式判过，解释器路径就是确定的，不需要“先试错的再回退”：
# 回退链在 Windows 上不仅没用，还会把“路径写法不对”掩盖成一句看不懂的 cmd 报错。
# POSIX 用法不受影响：Linux/macOS 上 $(OS) != Windows_NT，PY 取 `.venv/bin/python`，
# 与原先回退链最终命中的是同一个文件（旧写法只是靠“第一条命令必然失败”绕到它）。
RUN = $(PY)
# 受限网络下模型下载源与离线加载。
# `--hf-offline` 是**必需**的，不是可选优化：README「环境约束」实测"已缓存模型的联网校验
# 会**挂死 >10 分钟**"，而离线载入 0.8s。此前 eval / ingest / search 只带了 `--hf-endpoint`，
# 于是照着 Makefile 跑会在本机卡住——Makefile 与 README 的命令因此是不一致的。
HF := --hf-endpoint https://hf-mirror.com --hf-offline

.PHONY: help install check ci-local lint fmt type refs test lock corpus cstats ingest ingest-hash search stats \
        gold eval eval-baseline up down logs clean

# help 也用 $(PY) -c 实现，与 clean 同一手法、同一理由：
#   · 原实现是 `grep -E ... | awk ...`，而本机 PATH 上**没有** grep / awk：
#     实测 `mingw32-make help` 打印 `'grep' is not recognized ...`、退出码 2；
#   · venv 解释器是本仓库的运行前提，一定在；正则与格式化用 Python 表达也更清楚。
# 帮助约定不变：`target:  ## 说明` 的“说明”就是被打印的那一列（`$(MAKEFILE_LIST)`
# 里仍然只有遵守该约定的行会出现在帮助里）。
# 关于那串转义：`$$` 是 make 的字面 `$`（正则的行尾锚点）；`\033` 由 Python 解释成
# 终端转义符，保持与原 awk 版本相同的青色高亮。`#` 在 recipe 行里会原样交给 shell
# （make 只在非 recipe 行把 `#` 当注释），所以正则里的 `##` 不需要特殊处理。
# `.split()` 不是多余的：本机实测 mingw32-make 3.82.90 的 `$(MAKEFILE_LIST)` 展开后
# **带一个前导空格**（直接 `open()` 会 FileNotFoundError: ' Makefile'），
# 而且 include 了别的 makefile 时它是空格分隔的多个文件名。
help:
	@$(PY) -c "import re;pat=re.compile(r'^([A-Za-z_-]+):.*?## (.*)$$');[print('  \033[36m'+m.group(1).ljust(16)+'\033[0m '+m.group(2)) for f in '$(MAKEFILE_LIST)'.split() for m in map(pat.match,open(f).read().splitlines()) if m]"

install:  ## 建 venv 并安装（含 dev 依赖）
	python -m venv --system-site-packages .venv
	$(PY) -m pip install -e ".[dev]"

check: lint fmt type test refs  ## 所有静态闸门（= ci.yml 的 quality job，不含 pipeline/gate）
	@echo "所有闸门通过"

# `make check` 只覆盖 quality job 的**静态+测试**部分：ci.yml 里另外两个 job
# （pipeline 全链路自检、gate 质量闸门）本地原先没有对应目标，于是"本地全绿"并不
# 等于"CI 会绿"。这个目标把三个 job 按 CI 的口径都跑一遍（见 scripts/ci_local.py）。
# 它还会**强制要求 Postgres 可达**：那族用例在 CI 里有 service 容器、必须真跑，
# 而 skip 与 pass 在日志里长得几乎一样。
ci-local:  ## 本地按 CI 口径跑三个 job（quality + pipeline + gate）
	$(RUN) scripts/ci_local.py

lint:  ## ruff（阻塞）
	$(RUN) -m ruff check .

# `ruff check`（规则）与 `ruff format --check`（排版）是**两道**门，与 ci.yml 的
# quality job 里那两步一一对应。这一步原先只有 CI 有：本地 `make check` 一路绿灯、
# 推到 CI 却挂在排版上——"本地与 CI 是两道不同的门"正是这个仓库自己明令禁止的事
# （见下面 refs 目标的注释）。补上它，`make check` 才真的等于 CI 的 quality job。
fmt:  ## ruff 排版检查（阻塞）
	$(RUN) -m ruff format --check .

type:  ## mypy（阻塞）
	$(RUN) -m mypy src scripts

# 文档里的 `文件:行号` 引用必须指着它自己声称的东西（内容级校验，不只是"行号不越界"）。
# 与 ci.yml 的 quality job 里那一步是同一条命令：本地与 CI 不能是两道不同的门。
# 目标名**不能叫 `docs`**：仓库里有一个同名目录，make 会认为"已是最新"而什么都不跑——
# 实测 `mingw32-make docs` 输出 `'docs' is up to date.`、退出码 0（假绿）。
refs:  ## 文档锚点校验（阻塞）
	$(RUN) scripts/check_doc_refs.py

# 只跑 tests/unit（本仓库 tests/ 下只有 unit，等价于全量），不加载模型。
# **不要在这里再叠加 `-q`**：pyproject.toml 的 addopts 已经是 `-q --strict-markers`，
# 两个叠成 `-qq` 之后 pytest 连 "N passed" 汇总行都不打，只剩一屏进度点——
# 汇总行恰恰是“这次到底跑了多少、跳了多少”的唯一出处。
# 加 `-p no:cacheprovider`：这里不需要缓存，make 也就不会顺手留下 .pytest_cache。
# 加 `-rs`：把每条 skip 的**原因**打出来。CI 里 skippable 的只有 milvus 那一族
# （可选依赖，pyproject 没声明），出现别的 skip 就算回归——不能藏在 "N skipped" 里。
# 本地没起 Postgres 时，tests/unit/test_pgvector_store.py 整文件 skip（CI 用 service 起库）。
test:  ## 单元测试（不需要网络与模型，秒级）
	$(RUN) -m pytest tests/unit -p no:cacheprovider -rs

lock:  ## 重跑依赖锁文件（改了 pyproject 依赖后必须做；只查漂移见 scripts/gen_lock.py --check）
	$(RUN) scripts/gen_lock.py

# ---------------------------------------------------------------- 数据
corpus:  ## 拉取 Kubernetes 官方文档（1720 篇 / 16.2MB）
	$(RUN) scripts/fetch_corpus.py

cstats:  ## 语料切块体检（产出可写进 README 的数字）
	$(RUN) scripts/chunk_stats.py

ingest:  ## 建索引（增量；首次全量约 35 分钟，嵌入是瓶颈）
	$(RUN) -m kubedocs.cli $(HF) ingest

ingest-hash:  ## 用确定性哈希向量建索引（离线、约 30 秒，仅供管道自检，不代表质量）
	$(RUN) -m kubedocs.cli --provider hashing ingest

search:  ## 混合检索，用法：make search Q="how do I set a pod cpu limit"
	$(RUN) -m kubedocs.cli $(HF) search "$(Q)" --explain

stats:  ## 索引体检（含 FTS 一致性自检）
	$(RUN) -m kubedocs.cli stats --verify

# ---------------------------------------------------------------- 评测
gold:  ## 标注集预检（对原始语料，秒级）
	$(RUN) scripts/check_gold.py

eval:  ## 评测 + 单路消融 + 卡闸门（质量闸门用真实模型跑）
	$(RUN) -m kubedocs.cli $(HF) eval --gate

eval-baseline:  ## 冻结新基线（改了标注集/模型/语料后必须重建）
	$(RUN) -m kubedocs.cli $(HF) eval --write-baseline

# ---------------------------------------------------------------- 容器（已实测）
# ⚠️ `up` 用 `--build`：不重建的话跑的是**镜像里那份旧源码**。本机镜像至今没能从零重建成功
# （外网大文件下载反复停滞，见 README），所以本地验证阶段是用覆盖文件把 `./src` 挂进容器的
# ——那套做法与"干净环境一次构建成功"不是一回事，别把它当成前者。
up:  ## 起全栈（需 Docker）
	docker compose up -d --build

down:  ## 停全栈
	docker compose down

logs:  ## 跟日志
	docker compose logs -f api

# clean 的 recipe 用 Python 实现，不用 `rm -rf`：
#   · cmd.exe 上没有 rm（写成 `rm -rf` 会得到 “'rm' 不是内部或外部命令”，make 直接报错）；
#   · cmd 内置的 del/rmdir 参数风格又是另一套（`/s /q`），写进去就把 POSIX 用法弄坏了。
#   · venv 里的解释器一定在（它是本仓库的运行前提），两条路都能用同一份实现。
# 只删构建缓存，**不动语料与索引**（data/raw 与 data/index 重建代价极高）。
clean:  ## 清缓存（不动语料与索引）
	$(PY) -c "import pathlib,shutil;[shutil.rmtree(p,ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in map(pathlib.Path,['.pytest_cache','.ruff_cache','.mypy_cache','htmlcov','.coverage'])]"
