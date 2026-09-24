# 常用入口。原则：`make check` 一条命令跑完所有闸门，与 CI 等价。
.PHONY: help install check lint type test itest up down logs ingest eval clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## 建 venv 并安装（含 dev 依赖）
	python -m venv --system-site-packages .venv
	.venv/Scripts/python -m pip install -e ".[dev]" 2>/dev/null || .venv/bin/python -m pip install -e ".[dev]"

check: lint type test  ## 跑完所有闸门（与 CI 等价）
	@echo "所有闸门通过"

lint:  ## ruff
	.venv/Scripts/python -m ruff check . 2>/dev/null || .venv/bin/python -m ruff check .

type:  ## mypy
	.venv/Scripts/python -m mypy src scripts 2>/dev/null || .venv/bin/python -m mypy src scripts

test:  ## 单元测试（零依赖、秒级）
	.venv/Scripts/python -m pytest tests/unit -q 2>/dev/null || .venv/bin/python -m pytest tests/unit -q

itest:  ## 集成测试（需 docker compose up 起的 db/redis）
	.venv/Scripts/python -m pytest -m integration -q 2>/dev/null || .venv/bin/python -m pytest -m integration -q

corpus:  ## 拉取 Kubernetes 官方文档
	.venv/Scripts/python scripts/fetch_corpus.py 2>/dev/null || .venv/bin/python scripts/fetch_corpus.py

cstats:  ## 语料切块体检（产出可写进 README 的数字）
	.venv/Scripts/python scripts/chunk_stats.py 2>/dev/null || .venv/bin/python scripts/chunk_stats.py

up:  ## 起全栈
	docker compose up -d --build

down:  ## 停全栈
	docker compose down

logs:  ## 跟日志
	docker compose logs -f api

ingest:  ## 灌库（切块 + 嵌入 + 入库）
	docker compose exec api kubedocs ingest

eval:  ## 评测闸门
	docker compose exec api kubedocs eval --gate

clean:  ## 清缓存与派生数据（不动语料）
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
