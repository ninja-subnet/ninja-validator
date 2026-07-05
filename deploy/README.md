# deploy

Docker assets for the validator stack. Everything here is built/run from the
**repo root** via `compose.yaml` — that is the one directory that holds both the
`tau` package (`pyproject.toml`, `uv.lock`, `src/`) and these deployment files,
so it is the build context for every image that installs the package.

```
deploy/
├── db/                 # PostgreSQL 17: tuned config + first-boot init script
│   ├── postgresql.conf #   mounted into the official postgres image
│   └── 00_init.sh      #   extensions, settings, read-only monitor role
├── migrate/            # one-shot migration runner image
│   ├── Dockerfile      #   installs tau[migrate], runs `alembic upgrade head`
│   └── alembic/        #   alembic.ini, env.py, versions/
└── worker/             # generic worker image (one recipe, EXTRA/WORKER build args)
    └── Dockerfile
```

## Quick start

Run from the repo root (where `compose.yaml` lives):

```bash
cp .env.example .env        # then edit the passwords
docker compose up -d db     # start postgres (runs the init script on first boot)
docker compose up migrate   # build the migrate image, apply migrations, exit
```

`docker compose up` (no service) does both: starts the DB, waits for it to be
healthy, runs migrations, and leaves the DB running.

Connect from your host:

```bash
psql "postgresql://appuser:...@localhost:5432/arena"
```

## Schema

Generated from the ERD plus migration refinements. Core tables include
`submissions`, `kings`, `challenges`, `tasks`, `duel_task_solutions`,
`task_solutions`, `judgements`, and `registrations`. Notable modeling choices:

- Composite primary keys on `challenges`, `duel_task_solutions`, `task_solutions`,
  and `judgements`.
- `duel_task_solutions` is scoped by `(task_id, challenger_submission_id,
  submission_id)`, so the king is solved fresh for each challenge rather than reused
  from a task-wide cache.
- `tasks` hangs off a king via `king_id`, carrying a `pool_type` discriminator.
- `registrations` had no key in the ERD, so it gets a surrogate `id` plus a
  `(uid, block)` uniqueness guard.
- Every foreign key is indexed (Postgres does not do this automatically), and
  `tasks.problem_statement` has a trigram GIN index for fuzzy search.

## Working with migrations

Migrations target the SQLAlchemy models in `src/tau/db/models.py` — the migrate
image installs the `tau` package and `deploy/migrate/alembic/env.py` imports
`Base` from there, so the models are a single source of truth (no duplicate copy).

Run Alembic from `deploy/migrate/` (where `alembic.ini` lives) with the `tau`
package installed in your environment and `DATABASE_URL` pointing at the DB
(use `localhost` and the published `POSTGRES_PORT` from your host):

```bash
cd deploy/migrate
alembic revision --autogenerate -m "add column X"   # new revision from model changes
alembic upgrade head                                 # apply
alembic downgrade -1                                 # roll back
```

The migrate Docker image is rebuilt automatically; after editing models and
adding a revision, re-run `docker compose up migrate` from the repo root.

## Adding a worker

Workers are containers that install the same `tau` package and run a worker
entrypoint. There is no per-worker Dockerfile — `deploy/worker/Dockerfile` is
parameterized:

1. Add `tau/workers/<name>.py` exposing a `main()`.
2. Add a `[project.scripts]` line in `pyproject.toml`, and fill in the worker's
   `[project.optional-dependencies]` extra, then `uv lock`.
3. Copy the commented worker service block in `compose.yaml`, setting the
   `EXTRA` (which extra to install) and `WORKER` (which console script to run).

## Tuning notes

`postgresql.conf` is sized for ~8 GB RAM / 4 vCPUs (matching the
`deploy.resources` limits in `compose.yaml`). The formulas in the comments
show how to scale each value. `shm_size: 1gb` in compose prevents the classic
"could not resize shared memory segment" failures under parallel queries.

`pg_stat_statements` is preloaded and exposed via the read-only `monitor` role —
use it to find your actual slow queries before tuning further:

```sql
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;
```

For production, put a connection pooler (PgBouncer) in front rather than raising
`max_connections`.
