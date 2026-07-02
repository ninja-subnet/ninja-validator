"""Connection plumbing: resolve the DB URL and hand out sessions.

The URL is read from the environment the same way the migration template does
(`tau/deploy/db/alembic/env.py`) so a worker and Alembic always agree on which
database they are talking to: `DATABASE_URL` wins, otherwise it is assembled from
the standard `POSTGRES_*` vars.

`session_scope()` is the unit of work used throughout `database.py`: it commits on
success, rolls back on any exception, and always closes the session — workers never
touch a `Session` directly.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker


def database_url() -> str:
    """Return the SQLAlchemy URL, mirroring the migration template's resolution.

    Prefers `DATABASE_URL`; falls back to `POSTGRES_USER/PASSWORD/HOST/PORT/DB`.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Build an Engine with a pre-ping pool (drops connections killed server-side)."""
    return create_engine(url or database_url(), pool_pre_ping=True, echo=echo, future=True)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """A `sessionmaker` bound to `engine`. `expire_on_commit=False` keeps returned
    rows usable after the unit of work closes."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# -- async variants ----------------------------------------------------------------
# The async-native workers (e.g. the task-generator's GeneratorDb) use these; the
# psycopg3 driver and the `postgresql+psycopg://` URL above serve both worlds, so
# `database_url()` is shared. The sync helpers above stay for `database.py`.


def create_async_db_engine(url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    """Build an async Engine with a pre-ping pool (drops connections killed server-side)."""
    return create_async_engine(url or database_url(), pool_pre_ping=True, echo=echo)


def async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """An `async_sessionmaker` bound to `engine`. `expire_on_commit=False` keeps
    returned rows usable after the unit of work closes."""
    return async_sessionmaker(bind=engine, expire_on_commit=False)


@asynccontextmanager
async def async_session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Async transactional scope: commit on success, roll back on error, always close."""
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
