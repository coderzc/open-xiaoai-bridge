FROM python:3.12-slim AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 更新源
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libopus-dev libportaudio2 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

# 设置环境变量
ENV BASH_ENV=/root/.bash_env
RUN touch "$BASH_ENV"
RUN echo '. "$BASH_ENV"' >> "$HOME/.bashrc"
RUN echo '[ -s "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"' >> "$BASH_ENV"

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY examples/bridge .
COPY packages/client-rust ./client-rust

# 构建
RUN sed -i 's/\.\.\/\.\.\/packages\///g' Cargo.toml \
    && cargo build --release

# 安装锁定依赖（保持 pyproject.toml 与 uv.lock 一致）
RUN uv sync --locked --no-install-project --no-editable

# 构建 wheel 并安装
RUN uv run maturin build --release && uv remove maturin


FROM python:3.12-slim

WORKDIR /app

# 更新源
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CLI=true

COPY --from=builder /app/.venv /app/.venv
COPY examples/bridge/main.py .
COPY examples/bridge/config.py .
COPY examples/bridge/core ./core


# Ensure sherpa_onnx can locate onnxruntime shared library at runtime.
RUN set -eux; \
    ort_dir="/app/.venv/lib/python3.12/site-packages/onnxruntime/capi"; \
    if [ ! -f "$ort_dir/libonnxruntime.so" ]; then \
      ln -s "$(basename "$(ls "$ort_dir"/libonnxruntime.so.* | head -n 1)")" "$ort_dir/libonnxruntime.so"; \
    fi

ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/onnxruntime/capi

# 暴露 API 服务器端口
EXPOSE 9092

# 先初始化关键词模型，然后启动主程序
CMD ["/bin/bash", "-c", "source /app/.venv/bin/activate && python core/services/audio/kws/keywords.py && python main.py"]
