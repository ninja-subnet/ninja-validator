# Migration runner. Installs the `tau` package (for tau.db.models, the schema
# source of truth) plus Alembic, waits for the DB elsewhere, and applies
# migrations. BUILD FROM THE REPO ROOT — compose sets `context: .` — so that
# pyproject.toml / uv.lock / src/ are in scope.
FROM python:3.12-slim

# Bring in the uv binary (pinned to the build-backend range in pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Put the project venv on PATH so `alembic` resolves without `uv run`.
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Dependency layer — cached across source edits. Installs the `migrate` extra's
#    deps (alembic + the core sqlalchemy/psycopg) but NOT the project itself.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra migrate

# 2) Project source, then install the `tau` package on top of the cached deps.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra migrate

# 3) Alembic config + migrations. These are ops artifacts, not part of the
#    package, so they are copied in directly. script_location=alembic in
#    alembic.ini resolves to /app/alembic.
COPY deploy/migrate/alembic.ini ./alembic.ini
COPY deploy/migrate/alembic ./alembic

# Bring the schema up to the latest revision. Idempotent; safe to re-run.
CMD ["alembic", "upgrade", "head"]
