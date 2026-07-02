# Generic worker image. ONE recipe for every worker; the only differences are two
# build args, so adding a worker needs no new Dockerfile:
#   EXTRA  -> [project.optional-dependencies] group to install   (e.g. bittensor)
#   WORKER -> [project.scripts] console script to run as the cmd (e.g. chain-watcher)
# BUILD FROM THE REPO ROOT — compose sets `context: .` — so pyproject.toml /
# uv.lock / src/ are in scope and the `tau` package can be installed.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

ARG EXTRA
ARG WORKER
# Only the task-solver needs git in the image (it clones task repos on the host
# before handing them to the sandbox). Opt-in so the other worker images stay slim.
ARG INSTALL_GIT=false
RUN if [ "$INSTALL_GIT" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends git ca-certificates \
        && rm -rf /var/lib/apt/lists/*; \
    fi

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    # Persist the chosen worker so the runtime CMD can read it (ARG is build-only).
    WORKER=${WORKER}

WORKDIR /app

# Dependency layer (cached): the worker's extra + core deps, project not yet built.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra "${EXTRA}"

# Project source, then install `tau` itself.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra "${EXTRA}"

# Run the selected console script as PID 1 (exec, so signals reach it).
CMD ["sh", "-c", "exec ${WORKER}"]
