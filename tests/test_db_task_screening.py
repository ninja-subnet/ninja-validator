"""PostgreSQL coverage for task-screen state transitions."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import make_url, text

from tau.db.engine import async_session_scope, create_async_db_engine, create_db_engine
from tau.db.generator import GeneratorDb, PoolDeficit
from tau.db.models import Base, King, Submission, Task, TaskScreening
from tau.db.status import PoolType, TaskStatus
from tau.db.task_screening import TaskScreeningDb
from tau.pools import PoolTargets

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_TEST_URL = os.environ.get("TAU_TEST_DATABASE_URL")


def _maintenance_url(url: str) -> str:
    return make_url(url).set(database="postgres").render_as_string(hide_password=False)


def _server_reachable(url: str | None) -> bool:
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
    reason="TAU_TEST_DATABASE_URL unset or unreachable",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database() -> None:
    assert _TEST_URL is not None
    name = make_url(_TEST_URL).database
    engine = create_db_engine(_maintenance_url(_TEST_URL))
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            if not conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            ).scalar():
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        engine.dispose()


@pytest_asyncio.fixture
async def db() -> AsyncIterator[TaskScreeningDb]:
    engine = create_async_db_engine(_TEST_URL)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    seam = TaskScreeningDb(_TEST_URL)
    try:
        yield seam
    finally:
        await seam.aclose()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


async def _add_king(db: TaskScreeningDb, king_id: str, at: dt.datetime) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add(
            Submission(
                submission_id=king_id,
                block=1,
                hotkey=f"hk-{king_id}",
                status_id=1,
            )
        )
        await session.flush()
        session.add(King(king_id=king_id, king_from=at))


async def _add_screening(
    db: TaskScreeningDb,
    task_id: str,
    *,
    king_id: str,
    status: TaskStatus = TaskStatus.PENDING_SCREEN,
    next_retry_at: dt.datetime | None = None,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add(
            Task(
                task_id=task_id,
                pool_type=1,
                problem_statement=f"problem {task_id}",
                status_id=int(status),
                king_id=king_id,
                repo_clone_url="https://example.invalid/repo.git",
                parent_sha="a" * 40,
                commit_sha="b" * 40,
                reference_patch="reference patch",
                content_fingerprint=f"fingerprint-{task_id}",
            )
        )
        await session.flush()
        session.add(
            TaskScreening(
                task_id=task_id,
                king_submission_id=king_id,
                qualification_solution=f"solution {task_id}",
                next_retry_at=next_retry_at,
            )
        )


async def _make_due(db: TaskScreeningDb, task_id: str) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        row = await session.get(TaskScreening, task_id)
        assert row is not None
        row.next_retry_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)


async def _fail(db: TaskScreeningDb):
    return await db.save_error(
        task_id="task",
        king_submission_id="king",
        max_failed_runs=3,
        retry_base_seconds=60,
        retry_max_seconds=90,
    )


async def test_pending_requests_filter_king_status_and_backoff(
    db: TaskScreeningDb,
) -> None:
    now = dt.datetime.now(dt.UTC)
    await _add_king(db, "old", now - dt.timedelta(hours=1))
    await _add_king(db, "current", now)
    await _add_screening(db, "old-task", king_id="old")
    await _add_screening(db, "wanted", king_id="current")
    await _add_screening(
        db,
        "deferred",
        king_id="current",
        next_retry_at=now + dt.timedelta(minutes=5),
    )
    await _add_screening(db, "final", king_id="current", status=TaskStatus.QUALIFIED)

    assert [row.task_id for row in await db.pending_requests()] == ["wanted"]
    assert {
        row.task_id for row in await db.pending_requests(include_deferred=True)
    } == {
        "wanted",
        "deferred",
    }


@pytest.mark.parametrize(
    ("score", "outcome", "status"),
    [
        (0.70, "qualified", TaskStatus.QUALIFIED),
        (0.701, "disqualified", TaskStatus.DISQUALIFIED),
    ],
)
async def test_save_decision_transitions_strict_boundary(
    db: TaskScreeningDb, score: float, outcome: str, status: TaskStatus
) -> None:
    await _add_king(db, "king", dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")
    reason = "score_at_or_below_max" if outcome == "qualified" else "score_above_max"

    assert await db.save_decision(
        task_id="task",
        king_submission_id="king",
        outcome=outcome,  # type: ignore[arg-type]
        king_score=score,
        max_score=0.70,
        reason=reason,
        model="test/model",
    )
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(status)
    assert row is not None and (row.king_score, row.reason) == (score, reason)


async def test_failures_back_off_then_disqualify_and_refill(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")

    first = await _fail(db)
    assert first.state == "retry" and first.failed_runs == 1
    assert first.next_retry_at is not None and await db.pending_requests() == []
    await _make_due(db, "task")
    second = await _fail(db)
    assert second.state == "retry" and second.failed_runs == 2
    await _make_due(db, "task")
    third = await _fail(db)
    assert third.state == "exhausted" and third.failed_runs == 3

    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(TaskStatus.DISQUALIFIED)
    assert row is not None and row.reason == "screening_exhausted"

    generator = GeneratorDb(_TEST_URL)
    try:
        deficits = await generator.pending_pool_deficits(PoolTargets(1, 1))
    finally:
        await generator.aclose()
    assert PoolDeficit("king", PoolType.POOL_ONE, 1) in deficits


async def test_concurrent_failures_count_once_per_backoff_window(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")

    results = await asyncio.gather(*(_fail(db) for _ in range(3)))
    assert [result.state for result in results].count("retry") == 1
    assert [result.state for result in results].count("stale") == 2
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        row = await session.get(TaskScreening, "task")
    assert row is not None and row.failed_runs == 1


async def test_final_write_wins_and_late_results_are_ignored(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")
    common = {
        "task_id": "task",
        "king_submission_id": "king",
        "max_score": 0.70,
        "model": "test/model",
    }
    assert await db.save_decision(
        **common,
        outcome="qualified",
        king_score=0.4,
        reason="score_at_or_below_max",
    )
    assert not await db.save_decision(
        **common,
        outcome="disqualified",
        king_score=0.9,
        reason="score_above_max",
    )
    assert (await _fail(db)).state == "stale"


async def test_dethroned_king_result_is_ignored(db: TaskScreeningDb) -> None:
    now = dt.datetime.now(dt.UTC)
    await _add_king(db, "old", now)
    await _add_screening(db, "task", king_id="old")
    await _add_king(db, "new", now + dt.timedelta(seconds=1))

    assert not await db.save_decision(
        task_id="task",
        king_submission_id="old",
        outcome="qualified",
        king_score=0.1,
        max_score=0.70,
        reason="score_at_or_below_max",
        model="test/model",
    )
