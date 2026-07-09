"""Integration coverage for the task-screening DB seam and status guards."""

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
    reason="TAU_TEST_DATABASE_URL unset or its Postgres server unreachable",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database() -> None:
    assert _TEST_URL is not None
    db_name = make_url(_TEST_URL).database
    engine = create_db_engine(_maintenance_url(_TEST_URL))
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
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


async def _add_king(db: TaskScreeningDb, king_id: str, *, at: dt.datetime) -> None:
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
    outcome: str = "pending",
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
                qualification_duration_seconds=1.25,
                qualification_exit_reason="completed",
                outcome=outcome,
                next_retry_at=next_retry_at,
            )
        )


async def test_pending_requests_only_returns_current_king_pending_rows(
    db: TaskScreeningDb,
) -> None:
    now = dt.datetime.now(dt.UTC)
    await _add_king(db, "old", at=now - dt.timedelta(hours=1))
    await _add_king(db, "current", at=now)
    await _add_screening(db, "old-task", king_id="old")
    await _add_screening(db, "wanted", king_id="current")
    await _add_screening(
        db,
        "already-final",
        king_id="current",
        status=TaskStatus.QUALIFIED,
        outcome="qualified",
    )

    requests = await db.pending_requests()

    assert [request.task_id for request in requests] == ["wanted"]
    assert requests[0].king_submission_id == "current"
    assert requests[0].qualification_solution == "solution wanted"


async def test_pending_requests_filters_backoff_unless_explicitly_included(
    db: TaskScreeningDb,
) -> None:
    now = dt.datetime.now(dt.UTC)
    await _add_king(db, "king", at=now)
    await _add_screening(
        db,
        "deferred",
        king_id="king",
        next_retry_at=now + dt.timedelta(minutes=5),
    )

    assert await db.pending_requests() == []
    assert [
        request.task_id for request in await db.pending_requests(include_deferred=True)
    ] == ["deferred"]


@pytest.mark.parametrize(
    ("score", "outcome", "status"),
    [
        (0.70, "qualified", TaskStatus.QUALIFIED),
        (0.701, "disqualified", TaskStatus.DISQUALIFIED),
    ],
)
async def test_save_decision_obeys_strict_above_threshold_boundary(
    db: TaskScreeningDb,
    score: float,
    outcome: str,
    status: TaskStatus,
) -> None:
    await _add_king(db, "king", at=dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")

    saved = await db.save_decision(
        task_id="task",
        king_submission_id="king",
        outcome=outcome,  # type: ignore[arg-type]
        king_score=score,
        max_score=0.70,
        reason="score_at_or_below_max" if outcome == "qualified" else "score_above_max",
        model="test/model",
        rationale="scored",
        attempts=2,
        duration_seconds=0.5,
    )

    assert saved is True
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(status)
    assert row is not None
    assert row.outcome == outcome
    assert row.king_score == score
    assert row.max_score == 0.70
    assert row.error is None


async def test_blocked_decision_disqualifies_without_fabricating_score(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", at=dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")

    saved = await db.save_decision(
        task_id="task",
        king_submission_id="king",
        outcome="disqualified",
        king_score=None,
        max_score=0.70,
        reason="prompt_injection",
        model="static/prompt-injection",
        rationale="suspicious instruction",
        attempts=0,
        duration_seconds=0.01,
    )

    assert saved is True
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(TaskStatus.DISQUALIFIED)
    assert row is not None and row.king_score is None
    assert row.reason == "prompt_injection"


async def test_save_error_records_telemetry_and_leaves_task_pending(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", at=dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")

    result = await db.save_error(
        task_id="task",
        king_submission_id="king",
        max_score=0.70,
        model="fallback/model",
        error="provider unavailable",
        attempts=4,
        duration_seconds=2.0,
        max_failed_runs=3,
        retry_base_seconds=60,
        retry_max_seconds=900,
    )

    assert result.saved is True
    assert result.exhausted is False
    assert result.failed_runs == 1
    assert result.cumulative_attempts == 4
    assert result.next_retry_at is not None
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(TaskStatus.PENDING_SCREEN)
    assert row is not None and row.outcome == "pending"
    assert row.error == "provider unavailable"
    assert row.attempts == 4
    assert row.failed_runs == 1
    assert row.next_retry_at == result.next_retry_at
    assert row.king_score is None


async def test_repeated_failures_back_off_then_atomically_disqualify(
    db: TaskScreeningDb,
) -> None:
    now = dt.datetime.now(dt.UTC)
    await _add_king(db, "king", at=now)
    await _add_screening(db, "task", king_id="king")

    first = await db.save_error(
        task_id="task",
        king_submission_id="king",
        max_score=0.70,
        model="primary/model",
        error="first failure",
        attempts=2,
        duration_seconds=3.0,
        max_failed_runs=3,
        retry_base_seconds=60,
        retry_max_seconds=90,
    )
    assert first.saved and not first.exhausted
    assert first.next_retry_at is not None
    first_delay = (first.next_retry_at - dt.datetime.now(dt.UTC)).total_seconds()
    assert 55 <= first_delay <= 60
    assert await db.pending_requests() == []

    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        row = await session.get(TaskScreening, "task", with_for_update=True)
        assert row is not None
        row.next_retry_at = now - dt.timedelta(seconds=1)

    second = await db.save_error(
        task_id="task",
        king_submission_id="king",
        max_score=0.70,
        model="fallback/model",
        error="second failure",
        attempts=3,
        duration_seconds=4.0,
        max_failed_runs=3,
        retry_base_seconds=60,
        retry_max_seconds=90,
    )
    assert second.saved and not second.exhausted
    assert second.next_retry_at is not None
    second_delay = (second.next_retry_at - dt.datetime.now(dt.UTC)).total_seconds()
    assert 85 <= second_delay <= 90  # 120s exponential delay capped at 90s

    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        row = await session.get(TaskScreening, "task", with_for_update=True)
        assert row is not None
        row.next_retry_at = now - dt.timedelta(seconds=1)

    third = await db.save_error(
        task_id="task",
        king_submission_id="king",
        max_score=0.70,
        model="fallback/model",
        error="third failure",
        attempts=1,
        duration_seconds=5.0,
        max_failed_runs=3,
        retry_base_seconds=60,
        retry_max_seconds=90,
    )
    assert third.saved and third.exhausted
    assert third.failed_runs == 3
    assert third.cumulative_attempts == 6
    assert third.next_retry_at is None

    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(TaskStatus.DISQUALIFIED)
    assert row is not None
    assert row.outcome == "disqualified"
    assert row.reason == "screening_exhausted"
    assert row.error == "third failure"
    assert row.attempts == 6
    assert row.failed_runs == 3
    assert row.score_duration_seconds == pytest.approx(12.0)
    assert row.next_retry_at is None

    generator = GeneratorDb(_TEST_URL)
    try:
        deficits = await generator.pending_pool_deficits(
            PoolTargets(pool_one=1, pool_two=1)
        )
    finally:
        await generator.aclose()
    assert PoolDeficit("king", PoolType.POOL_ONE, 1) in deficits


async def test_simultaneous_worker_failures_count_once_per_backoff_window(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", at=dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")

    results = await asyncio.gather(
        *(
            db.save_error(
                task_id="task",
                king_submission_id="king",
                max_score=0.70,
                model=f"worker-{index}",
                error="shared provider outage",
                attempts=1,
                duration_seconds=1.0,
                max_failed_runs=3,
                retry_base_seconds=60,
                retry_max_seconds=900,
            )
            for index in range(3)
        )
    )

    assert sum(result.saved for result in results) == 1
    assert not any(result.exhausted for result in results)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(TaskStatus.PENDING_SCREEN)
    assert row is not None
    assert row.failed_runs == 1
    assert row.attempts == 1
    assert row.next_retry_at is not None


async def test_stale_result_after_king_change_is_ignored(
    db: TaskScreeningDb,
) -> None:
    now = dt.datetime.now(dt.UTC)
    await _add_king(db, "old", at=now)
    await _add_screening(db, "task", king_id="old")
    await _add_king(db, "new", at=now + dt.timedelta(seconds=1))

    saved = await db.save_decision(
        task_id="task",
        king_submission_id="old",
        outcome="qualified",
        king_score=0.1,
        max_score=0.70,
        reason="score_at_or_below_max",
        model="test/model",
        rationale="late",
        attempts=1,
        duration_seconds=0.1,
    )

    assert saved is False
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        task = await session.get(Task, "task")
        row = await session.get(TaskScreening, "task")
    assert task is not None and task.status_id == int(TaskStatus.PENDING_SCREEN)
    assert row is not None and row.outcome == "pending"
    assert row.king_score is None


async def test_final_decision_is_idempotent_first_writer_wins(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", at=dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")
    common = {
        "task_id": "task",
        "king_submission_id": "king",
        "max_score": 0.70,
        "model": "test/model",
        "attempts": 1,
        "duration_seconds": 0.1,
    }

    assert await db.save_decision(
        **common,
        outcome="qualified",
        king_score=0.4,
        reason="score_at_or_below_max",
        rationale="first",
    )
    assert not await db.save_decision(
        **common,
        outcome="disqualified",
        king_score=0.9,
        reason="score_above_max",
        rationale="second",
    )

    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        row = await session.get(TaskScreening, "task")
    assert row is not None
    assert row.outcome == "qualified"
    assert row.king_score == 0.4
    assert row.rationale == "first"


async def test_late_retry_error_cannot_overwrite_final_decision(
    db: TaskScreeningDb,
) -> None:
    await _add_king(db, "king", at=dt.datetime.now(dt.UTC))
    await _add_screening(db, "task", king_id="king")
    assert await db.save_decision(
        task_id="task",
        king_submission_id="king",
        outcome="qualified",
        king_score=0.4,
        max_score=0.70,
        reason="score_at_or_below_max",
        model="test/model",
        rationale="final",
        attempts=1,
        duration_seconds=0.1,
    )

    result = await db.save_error(
        task_id="task",
        king_submission_id="king",
        max_score=0.70,
        model="late/model",
        error="late timeout",
        attempts=2,
        duration_seconds=1.0,
        max_failed_runs=3,
        retry_base_seconds=60,
        retry_max_seconds=900,
    )

    assert result.saved is False
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        row = await session.get(TaskScreening, "task")
    assert row is not None
    assert row.outcome == "qualified"
    assert row.model == "test/model"
    assert row.error is None
