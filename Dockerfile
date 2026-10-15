# 多阶段构建：builder 装依赖并编译轮子，runtime 只带运行所需，镜像更小、攻击面更少。
# 已实测可用：`docker compose up -d` 起 api + Postgres，健康检查通过，检索与 `/consult` 正常。
# 下面各段注释记录的是构建参数的由来与取舍（受限网络、镜像体积、层缓存、非 root 运行）。

# ---------- 受限网络下的源（可选）----------
# 这两个 ARG **默认留空 = 用 Debian / PyPI 官方源，行为与不传时完全一致**；
# 受限网络下（apt 或 pip 取包被中断）传镜像站即可构建：
#   docker compose build --build-arg APT_MIRROR=mirrors.aliyun.com \
#                        --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple api
ARG APT_MIRROR=""
ARG PIP_INDEX_URL=""
# CPU-only torch 的可选配方（为什么需要它见 builder 里的长注释）。默认**不启用**。
ARG TORCH_CPU_FIND_LINKS=""
ARG TORCH_CPU_VERSION=""

# ---------- builder ----------
FROM python:3.10-slim AS builder

ARG APT_MIRROR
ARG PIP_INDEX_URL
ARG TORCH_CPU_FIND_LINKS
ARG TORCH_CPU_VERSION
# 默认**不**装编译器：本项目依赖都以 manylinux wheel 提供，正常构建不需要编译器，
# 而 `build-essential` 会多拖进 64 个包 / 87MB。需要从源码编译依赖时
# （换架构、或引入没有 wheel 的包）置 1：
#   docker compose build --build-arg BUILD_TOOLS=1 api
ARG BUILD_TOOLS=""

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 编译期依赖（**可选**，见上面的 BUILD_TOOLS 说明）
RUN set -eux; \
    if [ "$BUILD_TOOLS" = "1" ]; then \
        if [ -n "$APT_MIRROR" ]; then \
            for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
                if [ -f "$f" ]; then \
                    sed -i "s|deb.debian.org|$APT_MIRROR|g; s|security.debian.org|$APT_MIRROR|g" "$f"; \
                fi; \
            done; \
        fi; \
        apt-get update; \
        apt-get install -y --no-install-recommends build-essential libpq-dev; \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /build
COPY pyproject.toml README.md ./
# LICENSE 必须一起 COPY：pyproject.toml 用的是 PEP 639 的 `license-files = ["LICENSE"]`。
# 不 COPY 它时构建**不会失败**，只有一个 `SetuptoolsDeprecationWarning`
# （Pattern 'LICENSE' did not match any files），但产物是**错的**——wheel 里没有
# `.dist-info/licenses/LICENSE`，METADATA 里也没有 `License-File: LICENSE`
# （只剩 `License-Expression: MIT`）。PEP 639 要求构建工具在 pattern 匹配不到文件时
# **必须报错**，眼下只是警告，意味着将来会变成硬失败：这一行既让现在的产物带上许可证，
# 也让将来的构建不炸。
# 放 builder 阶段：本文件里只有 builder 会执行构建（下面的 `pip install .`），
# runtime 只从 builder 拷已装好的 venv，不再读 pyproject，所以那里不需要 LICENSE。
COPY LICENSE ./
# 用 venv 承载依赖，运行时整目录拷过去即可（避免把 build-essential 带进最终镜像）。
#
# ## 为什么分两步装（依赖层 / 源码层分开）
#
# PyPI 上的 `torch` 默认是 **CUDA 版**，会连带拉 `nvidia-cublas / cudnn / nccl /
# cusparse / cusparselt / triton / cuda-toolkit / cuda-bindings …` 约 2.5 GB，
# 而这个服务是**纯 CPU** 的（README 的吞吐数字就是纯 CPU 测出来的）。装一个约 1 GB
# 的服务却下 2.5 GB 的 GPU 运行时，是纯粹的浪费——所以给出可选配方。
#
# ### CPU-only torch 的可选配方（`TORCH_CPU_FIND_LINKS` + `TORCH_CPU_VERSION`）
#
# **必须显式指定版本**，不能只给一个 `--find-links`：PEP 440 里 `2.14.1` 比
# `2.9.1+cpu` 新，pip 会照样挑 PyPI 上的 CUDA 版。所以版本要一起给：
#
#     export TORCH_CPU_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
#     export TORCH_CPU_VERSION=2.9.1+cpu     # 该目录里 cp310 的最新 CPU 版
#
# 两个都留空（默认）时**完全不执行这一步**，行为与不传时完全一致。
# 阿里云的 `pytorch-wheels/cpu/` 目录可用（cp310 有 54 个 `+cpu` wheel）；
# 官方 `download.pytorch.org` 在受限网络下常常传输挂死，故默认不用。
#
# ### 依赖层 / 源码层分开
#
# 先只装依赖（用一个空的占位包），**再** COPY 真正的 src 并用 `--no-deps` 装包。
# 这样以后改 `src/` 不会让那几 GB 重下一遍——层缓存才有意义。
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
    && mkdir -p src/kubedocs && : > src/kubedocs/__init__.py \
    && if [ -n "$TORCH_CPU_FIND_LINKS" ]; then \
           /opt/venv/bin/pip install --find-links "$TORCH_CPU_FIND_LINKS" \
               ${PIP_INDEX_URL:+--extra-index-url "$PIP_INDEX_URL"} \
               "torch${TORCH_CPU_VERSION:+==$TORCH_CPU_VERSION}"; \
       fi \
    && /opt/venv/bin/pip install ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} .

# 真正的源码：这一层只覆盖 `src/`，上面那个几 GB 的依赖层不受影响。
# `--no-deps`：依赖已经装好了，这里只把包本身换成真代码。
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# ---------- runtime ----------
FROM python:3.10-slim AS runtime

ARG APT_MIRROR

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # 模型缓存必须放在 **app 用户读得到** 的地方：容器以非 root 运行（下面 USER app），
    # 而 `/root` 是 0700——缓存若落在那里，app 用户读不到，容器内的检索会静默降级成
    # "没有向量路"。
    HF_HOME=/home/app/.cache/huggingface

# 运行期只需要 libpq（psycopg 的动态库）+ curl（healthcheck 用）
RUN set -eux; \
    if [ -n "$APT_MIRROR" ]; then \
        for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
            if [ -f "$f" ]; then \
                sed -i "s|deb.debian.org|$APT_MIRROR|g; s|security.debian.org|$APT_MIRROR|g" "$f"; \
            fi; \
        done; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends libpq5 curl; \
    rm -rf /var/lib/apt/lists/*; \
    useradd -m -u 10001 app; \
    mkdir -p /home/app/.cache/huggingface; \
    chown -R app:app /home/app/.cache

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts

# 非 root 运行（最小权限原则）
USER app

EXPOSE 8080

# 生产用 gunicorn+uvicorn worker 更稳；这里保持单进程以简化本地复现。
# 目标必须是 `kubedocs.api.app:app`（模块末尾的 `app = create_app()`）。
# `kubedocs.api` 这个包**没有** `app` 属性——它的 `__init__.py` 用 PEP 562 的
# 模块级 `__getattr__` 只惰性暴露 `create_app`，别的名字一律抛 AttributeError
# （这样 `import kubedocs.api` 不会顺带打开一份索引库）；所以写成
# `uvicorn kubedocs.api:app` 会报 ImportFromStringError: Attribute "app" not found。
CMD ["uvicorn", "kubedocs.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
