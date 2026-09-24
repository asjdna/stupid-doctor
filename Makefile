# 常用入口。原则：`make check` 一条命令跑完所有静态闸门，与 CI 等价。
#
# 本机直跑（默认，不需要 Docker）与 compose 是两条**并列**的路径：
#   make ingest / make search / make eval   -> 本地嵌入式索引（SQLite+FTS5+numpy）
#   make up / make ingest-docker            -> compose 全栈（需 Docker；本机 Docker Hub 不可达，未实测）
# 详见 docs/adr/0002。

PY := .venv/Scripts/python
PY_ALT := .venv/bin/python
RUN = $(PY) 2>/dev/null || $(PY_ALT)
# 受限网络下模型下载源与离线加载（模型已缓存后必须离线，否则联网校验会挂死）
HF := --hf-endpoint https://hf-mirror.com

.PHONY: help install check lint type test corpus cstats ingest ingest-hash search stats \
        gold eval eval-baseline up down logs clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## 建 venv 并安装（含 dev 依赖）
	python -m venv --system-site-packages .venv
	$(PY) -m pip install -e ".[dev]" 2>/dev/null || $(PY_ALT) -m pip install -e ".[dev]"

check: lint type test  ## 跑完所有静态闸门（与 CI 等价）
	@echo "所有闸门通过"

lint:  ## ruff（阻塞）
	$(RUN) -m ruff check .

type:  ## mypy（阻塞）
	$(RUN) -m mypy src scripts

test:  ## 单元测试（零依赖、秒级，不加载模型）
	$(RUN) -m pytest tests/unit -q

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

clean:  ## 清缓存（不动语料与索引）
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
