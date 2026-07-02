"""Integration tests for the submission qualification DB seam."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import make_url, select, text

from tau.db import QualificationDb, SubmissionStatus
from tau.db.engine import (
    async_session_scope,
    create_async_db_engine,
    create_db_engine,
)
from tau.db.models import (
    Base,
    Challenge,
    King,
    Registration,
    Submission,
    SubmissionQualification,
)
from tau.qualification import QualificationOutcome, SecurityQualificationResult

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_TEST_URL = os.environ.get("TAU_TEST_DATABASE_URL")
_BLOCK_DATE = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


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
    reason="TAU_TEST_DATABASE_URL unset or its Postgres server unreachable - skipping DB tests",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database() -> None:
    assert _TEST_URL is not None
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
async def db() -> AsyncIterator[QualificationDb]:
    engine = create_async_db_engine(_TEST_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    qualification_db = QualificationDb(_TEST_URL)
    try:
        yield qualification_db
    finally:
        await qualification_db.aclose()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


def _submission(
    session,
    sid: str,
    *,
    hotkey: str,
    block: int,
    status: SubmissionStatus,
    agent_files: str | None = None,
) -> None:
    session.add(
        Submission(
            submission_id=sid,
            block=block,
            hotkey=hotkey,
            status_id=int(status),
            agent_files=agent_files,
        )
    )


def _registration(session, *, uid: int, hotkey: str, block: int) -> None:
    session.add(
        Registration(
            uid=uid,
            ss58_hot=hotkey,
            ss58_cold=f"cold-{uid}",
            block=block,
            block_date=_BLOCK_DATE,
        )
    )


async def test_next_candidate_returns_first_unverified_inside_queue_head(
    db: QualificationDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(
            session,
            "s-eligible",
            hotkey="hk-eligible",
            block=10,
            status=SubmissionStatus.ELIGIBLE,
        )
        _submission(
            session,
            "s-unverified",
            hotkey="hk-unverified",
            block=20,
            status=SubmissionStatus.UNVERIFIED,
            agent_files="agent.py\nagent/model.py",
        )
        _submission(
            session,
            "s-later",
            hotkey="hk-later",
            block=30,
            status=SubmissionStatus.UNVERIFIED,
        )
        _registration(session, uid=1, hotkey="hk-eligible", block=10)
        _registration(session, uid=2, hotkey="hk-unverified", block=20)
        _registration(session, uid=3, hotkey="hk-later", block=30)

    assert await db.next_candidate(window_size=1) is None
    candidate = await db.next_candidate(window_size=2)
    assert candidate is not None
    assert candidate.submission_id == "s-unverified"
    assert candidate.agent_files == "agent.py\nagent/model.py"


async def test_next_candidate_skips_terminal_stale_king_and_challenged_rows(
    db: QualificationDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(
            session,
            "s-disqualified",
            hotkey="hk-disqualified",
            block=1,
            status=SubmissionStatus.DISQUALIFIED,
        )
        _submission(
            session,
            "s-king",
            hotkey="hk-king",
            block=2,
            status=SubmissionStatus.UNVERIFIED,
        )
        _submission(
            session,
            "s-challenged",
            hotkey="hk-challenged",
            block=3,
            status=SubmissionStatus.UNVERIFIED,
        )
        _submission(
            session,
            "s-stale",
            hotkey="hk-stale",
            block=4,
            status=SubmissionStatus.UNVERIFIED,
        )
        _submission(
            session,
            "s-next",
            hotkey="hk-next",
            block=5,
            status=SubmissionStatus.UNVERIFIED,
        )
        session.add(King(king_id="s-king"))
        session.add(
            Challenge(
                challenger_submission_id="s-challenged",
                king_id="s-king",
                status=0,
            )
        )
        _registration(session, uid=1, hotkey="hk-disqualified", block=1)
        _registration(session, uid=2, hotkey="hk-king", block=2)
        _registration(session, uid=3, hotkey="hk-challenged", block=3)
        _registration(session, uid=4, hotkey="hk-stale", block=10)
        _registration(session, uid=5, hotkey="hk-next", block=5)

    candidate = await db.next_candidate(window_size=1)
    assert candidate is not None
    assert candidate.submission_id == "s-next"


async def test_save_qualification_records_details_and_sets_status(
    db: QualificationDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(
            session,
            "s",
            hotkey="hk",
            block=1,
            status=SubmissionStatus.UNVERIFIED,
        )
    result = SecurityQualificationResult(
        verdict="pass",
        overall_score=91,
        security_score=98,
        summary="Clean.",
        reasons=("No suspicious IO.",),
        risks=(),
        required_changes=(),
        model="model/security",
    )

    saved = await db.save_qualification(
        submission_id="s",
        result=result,
        outcome=QualificationOutcome.QUALIFIED,
        base_files_available=True,
        duration_seconds=1.25,
    )

    assert saved is True
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        submission = await session.get(Submission, "s")
        row = await session.get(SubmissionQualification, "s")
    assert submission is not None
    assert submission.status_id == int(SubmissionStatus.ELIGIBLE)
    assert row is not None
    assert row.outcome == "qualified"
    assert row.verdict == "pass"
    assert row.model == "model/security"
    assert row.reasons == "No suspicious IO."
    assert row.base_files_available is True
    assert row.duration_seconds == 1.25


async def test_save_qualification_is_guarded_on_unverified_status(
    db: QualificationDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(
            session,
            "s",
            hotkey="hk",
            block=1,
            status=SubmissionStatus.ELIGIBLE,
        )
    result = SecurityQualificationResult(verdict="fail", summary="Nope.")

    saved = await db.save_qualification(
        submission_id="s",
        result=result,
        outcome=QualificationOutcome.DISQUALIFIED,
        base_files_available=False,
    )

    assert saved is False
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        submission = await session.get(Submission, "s")
        rows = (await session.scalars(select(SubmissionQualification))).all()
    assert submission is not None
    assert submission.status_id == int(SubmissionStatus.ELIGIBLE)
    assert rows == []


async def test_save_error_records_retryable_error_without_status_transition(
    db: QualificationDb,
) -> None:
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        _submission(
            session,
            "s",
            hotkey="hk",
            block=1,
            status=SubmissionStatus.UNVERIFIED,
        )

    saved = await db.save_error(
        submission_id="s",
        error="upstream unavailable",
        base_files_available=False,
        model="model/security",
        duration_seconds=0.1,
    )

    assert saved is True
    async with async_session_scope(db._sessions) as session:  # noqa: SLF001
        submission = await session.get(Submission, "s")
        row = await session.get(SubmissionQualification, "s")
    assert submission is not None
    assert submission.status_id == int(SubmissionStatus.UNVERIFIED)
    assert row is not None
    assert row.outcome == "error"
    assert row.error == "upstream unavailable"
