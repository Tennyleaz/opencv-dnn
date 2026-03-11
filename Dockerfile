# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-trixie

# System dependencies commonly required by opencv-python on Debian
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Headless by default (container is intended for --out)
ENV MPLBACKEND=Agg

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy source (includes model/ when committed)
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []
USER nonroot

# Default command; append args in `docker run ...`
ENTRYPOINT ["uv", "run", "main.py"]
CMD ["--image", "test.jpg", "--out", "out.jpg"]