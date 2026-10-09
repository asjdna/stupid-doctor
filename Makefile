# 常用入口。原则：`make check` 一条命令跑完所有静态闸门，与 CI 等价。
#
# 本机直跑（默认，不需要 Docker）与 compose 是两条**并列**的路径：
#   make ingest / make search / make eval   -> 本地嵌入式索引（默认 sqlite-vec，可切 sqlite / pgvector）
#   make up / make down / make logs         -> compose 全栈（需 Docker；本机 Docker Hub 不可达，未实测）
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
# 受限网络下模型下载源与离线加载（模型已缓存后必须离线，否则联网校验会挂死）
HF := --hf-endpoint https://hf-mirror.com

.PHONY: help install check lint type test lock corpus cstats ingest ingest-hash search stats \
        gold eval eval-baseline up down logs clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## 建 venv 并安装（含 dev 依赖）
	python -m venv --system-site-packages .venv
	$(PY) -m pip install -e ".[dev]"

check: lint type test  ## 跑完所有静态闸门（与 CI 等价）
	@echo "所有闸门通过"

lint:  ## ruff（阻塞）
	$(RUN) -m ruff check .

type:  ## mypy（阻塞）
	$(RUN) -m mypy src scripts

# 只跑 tests/unit（本仓库 tests/ 下只有 unit，等价于全量），不加载模型。
# **不要在这里再叠加 `-q`**：pyproject.toml 的 addopts 已经是 `-q --strict-markers`，
# 两个叠成 `-qq` 之后 pytest 连 "N passed" 汇总行都不打，只剩一屏进度点——
# 汇总行恰恰是“这次到底跑了多少、跳了多少”的唯一出处。
# 加 `-p no:cacheprovider`：这里不需要缓存，make 也就不会顺手留下 .pytest_cache。
test:  ## 单元测试（零依赖、秒级，不加载模型）
	$(RUN) -m pytest tests/unit -p no:cacheprovider

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

# ---------------------------------------------------------------- 容器（未实测）
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
