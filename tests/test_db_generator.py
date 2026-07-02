"""Integration tests for the task-generator DB seam.

Needs a Postgres *server* — set ``TAU_TEST_DATABASE_URL`` to a throwaway database
on it. The suite CREATEs that database on first run if it is missing, then drops +
recreates the schema around each test. Skipped unless the server is reachable, so
it never errors on a missing DB. Keep the database name DISTINCT from your dev
``POSTGRES_DB``: the per-test drop_all would wipe whatever it points at. Run with,
e.g.:

    TAU_TEST_DATABASE_URL=postgresql+psycopg://user:pw@localhost:5432/tau_pytest \\
        uv run pytest tests/test_db_generator.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import make_url, select, text
from sqlalchemy.exc import IntegrityError

from tau.db import GenerationMetrics, GeneratorDb, PoolDeficit, PoolType, TaskStatus
from tau.db.engine import (
    async_session_factory,
    async_session_scope,
    create_async_db_engine,
    create_db_engine,
)
from tau.db.models import Base, King, Submission, Task, TaskGenerationFailure
from tau.pools import PoolTargets

# Load the project-root .env so TAU_TEST_DATABASE_URL can live there rather than
# being exported by hand. load_dotenv keeps any already-exported var (override=False).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_TEST_URL = os.environ.get("TAU_TEST_DATABASE_URL")


def _maintenance_url(url: str) -> str:
    """Same server/creds as *url*, but the always-present ``postgres`` database.

    CREATE DATABASE for the test DB has to run from some *other* database; the
    default ``postgres`` one is guaranteed to exist on a stock cluster.
    """
    return make_url(url).set(database="postgres").render_as_string(hide_password=False)


def _server_reachable(url: str | None) -> bool:
    """True if the Postgres *server* in *url* accepts a connection.

    Probes the server's ``postgres`` database, not the test database (which the
    suite creates on demand) — so a missing test DB does not skip the suite, only a
    missing/unreachable server does. One-off ``SELECT 1`` at collection.
    """
    if not url:
        return False
    try:
        engine = create_db_engine(_maintenance_url(url))
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _server_reachable(_TEST_URL),
    reason="TAU_TEST_DATABASE_URL unset or its Postgres server unreachable — skipping DB tests",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database() -> None:
    """Create the dedicated test database once per session if it does not exist.

    CREATE DATABASE cannot run inside a transaction, so this connects to the
    server's ``postgres`` database in AUTOCOMMIT. The database named in
    ``TAU_TEST_DATABASE_URL`` is the suite's own; the per-test fixture below then
    drops/recreates only its *tables*.
    """
    assert _TEST_URL is not None  # guaranteed by the skip marker above
    db_name = make_url(_TEST_URL).database
    engine = create_db_engine(_maintenance_url(_TEST_URL))
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            already = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not already:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        engine.dispose()


@pytest_asyncio.fixture
async def db() -> AsyncIterator[GeneratorDb]:
    engine = create_async_db_engine(_TEST_URL)
    async with engine.begin() as conn:
        # The trigram index on tasks.problem_statement needs this extension; the
        # migration/initdb create it, but create_all() here does not.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    # Seed one submission + king for the generator to attach tasks to.
    sessions = async_session_factory(engine)
    async with async_session_scope(sessions) as session:
        session.add(Submission(submission_id="sub-king", block=100, hotkey="hk"))
        session.add(King(king_id="sub-king"))  # king_from filled by the DB default
    generator = GeneratorDb(_TEST_URL)
    try:
        yield generator
    finally:
        await generator.aclose()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


_METRICS = GenerationMetrics(
    model="test/model",
    fetch_seconds=1.0,
    llm_seconds=2.0,
    llm_attempt=1,
    rejected_duplicate=0,
    rejected_structural=0,
    rejected_quality=0,
    rejected_fetch_error=0,
)


async def _insert(
    db: GeneratorDb,
    *,
    fingerprint: str,
    pool: PoolType = PoolType.POOL_ONE,
    metrics: GenerationMetrics = _METRICS,
) -> bool:
    return await db.insert_task_candidate(
        task_id=f"task-{fingerprint}",
        king_id="sub-king",
        pool_type=int(pool),
        problem_statement="Do the thing.",
        reference_patch="diff --git a/x b/x",
        repo_clone_url="https://github.com/octo/repo.git",
        parent_sha="b" * 40,
        commit_sha="a" * 40,
        content_fingerprint=fingerprint,
        metrics=metrics,
    )


async def test_pending_pool_deficits_empty_without_king(db: GeneratorDb) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        await session.delete(await session.get(King, "sub-king"))
    # No reigning king -> nothing to generate.
    assert await db.pending_pool_deficits(PoolTargets()) == []


async def test_insert_is_idempotent_on_fingerprint(db: GeneratorDb) -> None:
    assert await _insert(db, fingerprint="fp-1") is True
    assert await _insert(db, fingerprint="fp-1") is False  # ON CONFLICT DO NOTHING
    assert await db.fingerprint_exists("fp-1") is True
    assert await db.fingerprint_exists("fp-unknown") is False


async def test_inserted_task_is_a_candidate(db: GeneratorDb) -> None:
    await _insert(db, fingerprint="fp-2")
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task-fp-2")
        assert task is not None
        assert task.status_id == int(TaskStatus.CANDIDATE)
        assert task.commit_sha == "a" * 40


async def test_pending_pool_deficits_reports_deficit_in_order(db: GeneratorDb) -> None:
    targets = PoolTargets(pool_one=2, pool_two=1)
    # Empty -> full deficit for both pools, pool one first, tagged with the king.
    assert await db.pending_pool_deficits(targets) == [
        PoolDeficit("sub-king", PoolType.POOL_ONE, 2),
        PoolDeficit("sub-king", PoolType.POOL_TWO, 1),
    ]
    # One pool-one task lands -> its deficit drops by one.
    await _insert(db, fingerprint="fp-p1a", pool=PoolType.POOL_ONE)
    assert await db.pending_pool_deficits(targets) == [
        PoolDeficit("sub-king", PoolType.POOL_ONE, 1),
        PoolDeficit("sub-king", PoolType.POOL_TWO, 1),
    ]
    # Fill pool one -> only pool two remains.
    await _insert(db, fingerprint="fp-p1b", pool=PoolType.POOL_ONE)
    assert await db.pending_pool_deficits(targets) == [
        PoolDeficit("sub-king", PoolType.POOL_TWO, 1)
    ]
    # Fill pool two -> nothing left.
    await _insert(db, fingerprint="fp-p2", pool=PoolType.POOL_TWO)
    assert await db.pending_pool_deficits(targets) == []


async def test_pending_pool_deficits_ignores_disqualified(db: GeneratorDb) -> None:
    targets = PoolTargets(pool_one=1, pool_two=1)
    # A disqualified task must not count toward a pool's target.
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add(
            Task(
                task_id="dq-task",
                king_id="sub-king",
                pool_type=int(PoolType.POOL_ONE),
                problem_statement="x",
                status_id=int(TaskStatus.DISQUALIFIED),
                repo_clone_url="u",
                parent_sha="p",
                commit_sha="c",
                reference_patch="r",
                content_fingerprint="fp-dq",
            )
        )
    # Pool one still needs its one task despite the disqualified row.
    assert await db.pending_pool_deficits(targets) == [
        PoolDeficit("sub-king", PoolType.POOL_ONE, 1),
        PoolDeficit("sub-king", PoolType.POOL_TWO, 1),
    ]


async def test_insert_records_metrics_and_utc_created_at(db: GeneratorDb) -> None:
    metrics = GenerationMetrics(
        model="deepseek/x",
        fetch_seconds=1.5,
        llm_seconds=3.2,
        llm_attempt=2,
        rejected_duplicate=1,
        rejected_structural=0,
        rejected_quality=4,
        rejected_fetch_error=0,
    )
    await _insert(db, fingerprint="fp-m", metrics=metrics)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task-fp-m")
        assert task is not None
        assert task.model == "deepseek/x"
        assert task.llm_attempt == 2
        assert task.rejected_quality == 4
        assert task.rejected_duplicate == 1
        # created_at is a UTC-aware timestamp filled by the DB default.
        assert task.created_at is not None and task.created_at.tzinfo is not None


async def test_record_generation_failure_appends_row(db: GeneratorDb) -> None:
    await db.record_generation_failure(
        king_id="sub-king",
        pool_type=int(PoolType.POOL_ONE),
        repo_full_name="octo/repo",
        commit_sha="a" * 40,
        model="deepseek/x",
        attempts=2,
        reason="invalid json from model",
    )
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        rows = (await session.scalars(select(TaskGenerationFailure))).all()
    assert len(rows) == 1
    failure = rows[0]
    assert failure.king_id == "sub-king"
    assert failure.commit_sha == "a" * 40
    assert failure.attempts == 2
    assert failure.reason == "invalid json from model"
    # created_at is a UTC-aware timestamp filled by the DB default.
    assert failure.created_at is not None and failure.created_at.tzinfo is not None


async def test_check_constraint_rejects_out_of_range_status(db: GeneratorDb) -> None:
    with pytest.raises(IntegrityError):
        async with async_session_scope(db._sessions) as session:  # noqa: SLF001
            session.add(
                Task(
                    task_id="bad-task",
                    king_id="sub-king",
                    pool_type=int(PoolType.POOL_ONE),
                    problem_statement="x",
                    status_id=99,  # outside {0,1,2} -> ck_tasks_status_id violation
                    repo_clone_url="u",
                    parent_sha="p",
                    commit_sha="c",
                    reference_patch="r",
                    content_fingerprint="fp-bad",
                )
            )
