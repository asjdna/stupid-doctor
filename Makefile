# 常用入口。`make check` 跑完所有静态闸门；整套验证（含全量摄取与检索闸门）用 `make ci-local`。
#
# 两条并列的使用路径：
#   make ingest / search / eval / stats   -> 本机嵌入式索引（默认 sqlite-vec）
#   make up / down / logs                 -> compose 全栈（需要 Docker）

# venv 解释器按平台取路径：Windows 在 Scripts\ 下，POSIX 在 bin/ 下。
# 不写"先试错再回退"的写法——那种写法依赖 PATH 上有 POSIX shell，在 Windows 上会
# 把"路径写法不对"掩盖成一句看不懂的 cmd 报错。
ifeq ($(OS),Windows_NT)
PY := .venv\Scripts\python
else
PY := .venv/bin/python
endif

RUN = $(PY)
# 受限网络下用镜像端点下载模型，并在模型已缓存后离线加载（联网校验会长时间挂起）。
HF := --hf-endpoint https://hf-mirror.com --hf-offline

.PHONY: help install check ci-local lint fmt type refs test lock corpus cstats ingest ingest-hash search stats \
        gold eval eval-baseline up down logs clean

# help 用 $(PY) -c 实现：本机 PATH 上不一定有 grep / awk，而 venv 解释器是本仓库的运行前提。
# 帮助约定不变：`target:  ## 说明` 里的说明就是被打印的列。
help:
	@$(PY) -c "import re;pat=re.compile(r'^([A-Za-z_-]+):.*?## (.*)$$');[print('  \033[36m'+m.group(1).ljust(16)+'\033[0m '+m.group(2)) for f in '$(MAKEFILE_LIST)'.split() for m in map(pat.match,open(f).read().splitlines()) if m]"

install:  ## 建 venv 并安装（含 dev 依赖）
	python -m venv --system-site-packages .venv
	$(PY) -m pip install -e ".[dev]"

check: lint fmt type test refs  ## 所有静态闸门（不含 pipeline / gate，那两个用 make ci-local）
	@echo "所有闸门通过"

# 跑完整套验证；它要求 Postgres 可达，因为 pgvector 那族用例必须真跑，skip 不算跑过。
ci-local:  ## 跑完整套验证（quality + pipeline + gate）
	$(RUN) scripts/ci_local.py

lint:  ## ruff（阻塞）
	$(RUN) -m ruff check .

# `ruff check`（规则）与 `ruff format --check`（排版）是两道独立的门。
fmt:  ## ruff 排版检查（阻塞）
	$(RUN) -m ruff format --check .

type:  ## mypy（阻塞）
	$(RUN) -m mypy src scripts

# 文档里的 `文件:行号` 引用要指对它声称的内容（内容级校验，不只是行号不越界）。
# 目标名不能叫 `docs`：仓库里有同名目录，make 会认为"已是最新"而什么都不跑。
refs:  ## 文档锚点校验（阻塞）
	$(RUN) scripts/check_doc_refs.py

# 只跑 tests/unit（tests/ 下目前只有 unit），不加载模型、不需要网络。
# 不要叠加第二个 `-q`（addopts 里已有一个），否则汇总行会被吞掉。
# `-rs` 把每条 skip 的原因打出来：可接受的 skip 只有 milvus 那一族（可选依赖）。
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

# ---------------------------------------------------------------- 容器
# `up` 带 `--build`：不重建的话跑的是镜像里那份旧源码。
up:  ## 起全栈（需 Docker）
	docker compose up -d --build

down:  ## 停全栈
	docker compose down

logs:  ## 跟日志
	docker compose logs -f api

# clean 用 Python 实现而不是 rm -rf：cmd.exe 上没有 rm，del/rmdir 的参数风格又是另一套。
# 只删构建缓存，不动语料与索引（它们重建代价很高）。
clean:  ## 清缓存（不动语料与索引）
	$(PY) -c "import pathlib,shutil;[shutil.rmtree(p,ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in map(pathlib.Path,['.pytest_cache','.ruff_cache','.mypy_cache','htmlcov','.coverage'])]"
