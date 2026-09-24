# 多阶段构建：builder 装依赖并编译轮子，runtime 只带运行所需，镜像更小、攻击面更少。
# ⚠️ 本文件尚未在本机实测（写它时本机还没装 Docker）。标为「未验证」。

# ---------- builder ----------
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 编译期依赖（psycopg / numpy 等有 C 扩展）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# 用 venv 承载依赖，运行时整目录拷过去即可（避免把 build-essential 带进最终镜像）
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# ---------- runtime ----------
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # 模型与语料落到挂载卷，容器重建不重下
    HF_HOME=/root/.cache/huggingface

# 运行期只需要 libpq（psycopg 的动态库）+ curl（healthcheck 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts

# 非 root 运行（最小权限原则）
USER app

EXPOSE 8080

# 生产用 gunicorn+uvicorn worker 更稳；这里保持单进程以简化本地复现
CMD ["uvicorn", "kubedocs.api:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
