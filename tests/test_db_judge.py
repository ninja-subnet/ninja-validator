"""Integration coverage for judge persistence and task-screen correlation."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import AsyncIterator
from hashlib import sha256
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import make_url, select, text

from tau.db.engine import async_session_scope, create_async_db_engine, create_db_engine
from tau.db.judge import JudgeDb
from tau.db.models import (
    Base,
    Challenge,
    Judgement,
    King,
    Submission,
    Task as TaskRow,
    TaskScreening,
)
from tau.db.status import ChallengeStatus, PoolType, SubmissionStatus, TaskStatus
from tau.judging import Judgment, Solution, Task

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
async def db() -> AsyncIterator[JudgeDb]:
    engine = create_async_db_engine(_TEST_URL)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    judge_db = JudgeDb(_TEST_URL)
    try:
        yield judge_db
    finally:
        await judge_db.aclose()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


async def _seed_screened_round(db: JudgeDb) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add_all(
            [
                Submission(
                    submission_id="king",
                    block=1,
                    hotkey="hk-king",
                    status_id=int(SubmissionStatus.ELIGIBLE),
                ),
                Submission(
                    submission_id="challenger",
                    block=2,
                    hotkey="hk-challenger",
                    status_id=int(SubmissionStatus.ELIGIBLE),
                ),
            ]
        )
        await session.flush()
        session.add(King(king_id="king", king_from=dt.datetime.now(dt.UTC)))
        await session.flush()
        session.add_all(
            [
                Challenge(
                    challenger_submission_id="challenger",
                    king_id="king",
                    status=int(ChallengeStatus.POOL_ONE),
                ),
                TaskRow(
                    task_id="task",
                    pool_type=int(PoolType.POOL_ONE),
                    problem_statement="problem",
                    status_id=int(TaskStatus.QUALIFIED),
                    king_id="king",
                    repo_clone_url="https://example.invalid/repo.git",
                    parent_sha="a" * 40,
                    commit_sha="b" * 40,
                    reference_patch="reference patch",
                    content_fingerprint="fingerprint-task",
                ),
            ]
        )
        await session.flush()
        session.add(
            TaskScreening(
                task_id="task",
                king_submission_id="king",
                qualification_solution="qualification patch",
                qualification_duration_seconds=10.0,
                qualification_exit_reason="completed",
                king_score=0.20,
                max_score=0.70,
                outcome="qualified",
                reason="score_at_or_below_max",
                model="test/screener",
            )
        )


async def test_save_judgment_returns_privacy_safe_comparison_only_for_fresh_write(
    db: JudgeDb,
) -> None:
    await _seed_screened_round(db)
    task = Task("task", "problem", "reference patch")
    king = Solution("king", "fresh duel patch")
    challenger = Solution("challenger", "challenger patch")
    judgment = Judgment(
        winner="king",
        king_score=0.85,
        challenger_score=0.15,
        model="test/judge",
    )

    comparison = await db.save_judgment(
        task,
        king,
        challenger,
        judgment,
        attempts=1,
        duration_seconds=2.5,
    )

    assert comparison is not None
    assert comparison.screening_king_score == 0.20
    assert comparison.duel_king_score == 0.85
    assert comparison.duel_minus_screen_king_score_delta == pytest.approx(0.65)
    assert comparison.screening_model == "test/screener"
    assert comparison.duel_model == "test/judge"
    assert (
        comparison.qualification_patch_sha256
        == sha256(b"qualification patch").hexdigest()
    )
    assert comparison.duel_patch_sha256 == sha256(b"fresh duel patch").hexdigest()
    assert comparison.qualification_patch_matches_duel_patch is False

    # The write-once retry neither overwrites the verdict nor emits another context.
    duplicate = await db.save_judgment(
        task,
        Solution("king", "different retry patch"),
        challenger,
        Judgment("challenger", 0.0, 1.0, model="test/retry"),
        attempts=2,
        duration_seconds=3.0,
    )
    assert duplicate is None

    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        saved = (
            await session.execute(
                select(Judgement).where(
                    Judgement.task_id == "task",
                    Judgement.king_submission_id == "king",
                    Judgement.challenger_submission_id == "challenger",
                )
            )
        ).scalar_one()
    assert saved.llm_winner == "king"
    assert saved.king_score == 0.85
    assert saved.model == "test/judge"


async def test_degraded_neutral_judgment_is_persisted_but_not_compared(
    db: JudgeDb,
) -> None:
    await _seed_screened_round(db)

    comparison = await db.save_judgment(
        Task("task", "problem", "reference patch"),
        Solution("king", "fresh duel patch"),
        Solution("challenger", "challenger patch"),
        Judgment(
            winner="tie",
            king_score=0.5,
            challenger_score=0.5,
            model="fallback/neutral",
            error="all judge routes unavailable",
        ),
        attempts=4,
        duration_seconds=300.0,
    )

    assert comparison is None
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        saved = (
            await session.execute(select(Judgement).where(Judgement.task_id == "task"))
        ).scalar_one()
    assert saved.error == "all judge routes unavailable"
    assert saved.king_score == 0.5


async def test_screening_must_match_both_task_and_king(db: JudgeDb) -> None:
    await _seed_screened_round(db)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        session.add(
            Submission(
                submission_id="other-king",
                block=3,
                hotkey="hk-other-king",
                status_id=int(SubmissionStatus.ELIGIBLE),
            )
        )
        await session.flush()
        screening = await session.get(TaskScreening, "task")
        assert screening is not None
        screening.king_submission_id = "other-king"

    comparison = await db.save_judgment(
        Task("task", "problem", "reference patch"),
        Solution("king", "fresh duel patch"),
        Solution("challenger", "challenger patch"),
        Judgment("king", 0.85, 0.15, model="test/judge"),
        attempts=1,
        duration_seconds=2.5,
    )

    assert comparison is None


async def test_disabled_screen_score_still_returns_patch_comparison(
    db: JudgeDb,
) -> None:
    await _seed_screened_round(db)
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        screening = await session.get(TaskScreening, "task")
        assert screening is not None
        screening.king_score = None

    comparison = await db.save_judgment(
        Task("task", "problem", "reference patch"),
        Solution("king", "fresh duel patch"),
        Solution("challenger", "challenger patch"),
        Judgment("king", 0.85, 0.15, model="test/judge"),
        attempts=1,
        duration_seconds=2.5,
    )

    assert comparison is not None
    assert comparison.screening_king_score is None
    assert comparison.duel_minus_screen_king_score_delta is None
    assert comparison.duel_patch_sha256 == sha256(b"fresh duel patch").hexdigest()
