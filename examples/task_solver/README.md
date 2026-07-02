# Task-solver dry run

Exercise the `task-solver` worker end to end: it qualifies CANDIDATE tasks by running
a **king** agent, then produces **challenger** solutions for active challenges — a
couple of duels' worth of work for the judge.

Agents are read from a **local submissions directory** — one folder per submission,
named by its **submission id**, each a bundle whose entry point is `agent.py` (the
ninja shape: `agent.py` + an `agent/` package). Point the worker at it with
`TAU_SUBMISSIONS_DIR`.

The sample task repo is cloned via `file://`, so no GitHub token is needed.

## Prerequisites

- Docker running (the solver spawns sandbox containers via the local daemon).
- Postgres up: `docker compose up -d db`
- Deps: `uv sync --extra task-solver --extra migrate`
- Run the worker **on the host** (not the compose `task-solver` service) for this dry
  run, so it can reach the `file://` sample repo and the local Docker socket. (In
  compose/docker-out-of-docker the work tree is streamed into the sandbox over the
  daemon API, so no shared work dir is needed — only the submissions dir is mounted.)

Use a dedicated dry-run DB (the local `arena` may hold a pre-refactor schema):

```bash
CREDS=$(uv run python -c "import os;from dotenv import load_dotenv;load_dotenv('.env');print(f\"{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@localhost:{os.environ.get('POSTGRES_PORT','5432')}\")")
uv run python -c "from sqlalchemy import create_engine,text;c=create_engine('postgresql+psycopg://${CREDS}/postgres').connect().execution_options(isolation_level='AUTOCOMMIT');c.execute(text('DROP DATABASE IF EXISTS tau_dryrun'));c.execute(text('CREATE DATABASE tau_dryrun'))"
export DATABASE_URL="postgresql+psycopg://${CREDS}/tau_dryrun"
( cd deploy/migrate && uv run --project ../.. alembic upgrade head )
```

## A) Real submissions

```bash
export TAU_SUBMISSIONS_DIR=/home/robert/repos/s66_repositories/submissions   # folders = submission ids

# Seed: first folder becomes the king, the rest become challengers; 2 CANDIDATE tasks.
uv run python examples/task_solver/seed_duel.py --submissions-dir "$TAU_SUBMISSIONS_DIR"

# Run the solver. The ninja agents call the model through the proxy, so set a real
# upstream (OpenRouter shown; or LLM_PROVIDER=ninja for our backend).
OPENROUTER_API_KEY=sk-or-... SOLVER_MODEL=deepseek/deepseek-v4-flash \
  MAX_CONTAINERS=4 TAU_SOLVER_POLL_SECONDS=5 LOG_LEVEL=INFO uv run task-solver
# Ctrl-C once the qualify + challenger-solve lines are logged.

uv run python examples/task_solver/show_state.py
```

The agent inside the sandbox only ever sees the proxy URL + a per-solve token; the
real upstream key is injected by the proxy and never enters the container. Each solve
runs with the configured per-solve budget (`SOLVER_MAX_*`) — set one when spending
real tokens on untrusted code.

## B) Token-free dry run (no real submissions, no LLM)

Scaffolds two sample bundles (a deterministic noop `agent.py`) into a submissions dir,
so you can exercise the whole pipeline at zero token cost:

```bash
export TAU_SUBMISSIONS_DIR=/tmp/tau-sample-submissions
uv run python examples/task_solver/seed_duel.py --scaffold --submissions-dir "$TAU_SUBMISSIONS_DIR"

# OPENROUTER_API_KEY can be a dummy: the noop agent never calls the proxy.
OPENROUTER_API_KEY=sk-dummy MAX_CONTAINERS=8 TAU_SOLVER_POLL_SECONDS=5 uv run task-solver
uv run python examples/task_solver/show_state.py
```

Expected: both tasks `QUALIFIED`, a king `task_solution` per task, and a challenger
`task_solution` per (task, challenger). Run `judge-worker` next to fill `judgements`.

## Notes

- `seed_duel.py` is idempotent (clears its discovered ids and rebuilds the sample
  repo). Re-run to reset a duel; solutions are write-once (`ON CONFLICT DO NOTHING`).
- A submission folder missing `agent.py` is skipped with a warning (task left for a
  retry), never silently marked solved.
- `agents/noop_agent.py` (token-free) and `agents/swe_agent.py` (calls the model) are
  single-file sample bundles you can copy into a submissions folder as `agent.py`.
