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
# LICENSE 必须一起 COPY：pyproject.toml 用的是 PEP 639 的 `license-files = ["LICENSE"]`。
# 本机实测（setuptools 84.0.0 / py3.10，`pip wheel --no-deps`）：
#   · 不 COPY LICENSE：构建**不会失败**，只有
#       SetuptoolsDeprecationWarning: Cannot find any files for the given pattern.
#       Pattern 'LICENSE' did not match any files.
#     但产物是**错的**——wheel 里没有 `.dist-info/licenses/LICENSE`，
#     METADATA 里也没有 `License-File: LICENSE`（只剩 `License-Expression: MIT`）。
#   · COPY 上之后：`adding license file 'LICENSE'`，wheel 里多出
#     `kubedocs-0.1.0.dist-info/licenses/LICENSE`，METADATA 多出 `License-File: LICENSE`。
# PEP 639 对 build tool 的要求是 “MUST raise an error if any individual user-specified
# pattern does not match at least one file”——眼下只是 DeprecationWarning，意味着
# 将来会变成硬失败。所以这一行既是让**现在的产物**带上许可证，也是让**将来的构建**不炸。
# 放 builder 阶段：本文件里只有 builder 会执行构建（下面的 `pip install .`），
# runtime 只从 builder 拷已装好的 venv，不再读 pyproject，所以那里不需要 LICENSE。
COPY LICENSE ./
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
# 目标必须是 `kubedocs.api.app:app`（模块里 `app = create_app()`，api/app.py:643）：
# `kubedocs.api` 这个包**没有** `app` 属性——它的 `__init__.py` 用 PEP 562 的
# 模块级 `__getattr__` 只惰性暴露 `create_app`，别的名字一律抛 AttributeError
# （这样 `import kubedocs.api` 不会顺带打开一份索引库）。
# 实测：`uvicorn kubedocs.api:app` → ImportFromStringError: Attribute "app" not found
# in module "kubedocs.api"；换成下面这个目标即可正常导入。
CMD ["uvicorn", "kubedocs.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
