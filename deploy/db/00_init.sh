#!/usr/bin/env bash
# =============================================================================
# 00_init.sh
# Runs exactly once, on first cluster initialization, as the superuser. Use it
# for things that must exist BEFORE migrations and that need superuser rights
# (extensions, roles, db-level GUCs). Table/schema DDL lives in Alembic so it
# stays versioned — do not create application tables here.
#
# The official postgres entrypoint exports POSTGRES_USER / POSTGRES_DB and a
# working libpq environment, plus any vars from .env (e.g. MONITOR_PASSWORD).
# We forward the ones we need to psql with -v so :'NAME' substitution works.
# =============================================================================
set -euo pipefail

: "${MONITOR_PASSWORD:=monitor}"   # fallback if not provided via .env

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v db_name="$POSTGRES_DB" \
     -v monitor_password="$MONITOR_PASSWORD" <<-'EOSQL'

    -- gen_random_uuid(), digest(), crypt(), etc.
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- Backing view for query-bottleneck analysis. The library itself is
    -- preloaded via shared_preload_libraries in postgresql.conf.
    CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

    -- Trigram indexes for fast fuzzy / ILIKE text search.
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    -- Per-database defaults applied to every connection to this database.
    ALTER DATABASE :"db_name" SET timezone TO 'UTC';
    ALTER DATABASE :"db_name" SET statement_timeout TO '60s';
    ALTER DATABASE :"db_name" SET idle_in_transaction_session_timeout TO '60s';
    ALTER DATABASE :"db_name" SET lock_timeout TO '10s';

    -- Read-only monitoring role: lets dashboards / on-call inspect
    -- pg_stat_statements and pg_stat_* without any write access.
    -- (psql does not substitute :'vars' inside DO $$..$$ blocks, so build the
    --  statement in plain SQL and run it with \gexec — idempotent.)
    SELECT format('CREATE ROLE monitor LOGIN PASSWORD %L', :'monitor_password')
    WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'monitor')
    \gexec

    GRANT pg_monitor TO monitor;
    GRANT CONNECT ON DATABASE :"db_name" TO monitor;
    GRANT USAGE ON SCHEMA public TO monitor;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO monitor;

EOSQL

echo "00_init.sh: extensions, db settings and monitor role configured."
