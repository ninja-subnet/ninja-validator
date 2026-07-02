"""Database operations for the submission qualification worker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Row, func, select, update
from sqlalchemy.dialects.postgresql import insert

from tau.qualification import (
    QualificationOutcome,
    SecurityQualificationResult,
    security_failures,
    security_risk_categories,
)

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import SubmissionStatus

_QUEUE_HEAD_STATUSES = (
    int(SubmissionStatus.UNVERIFIED),
    int(SubmissionStatus.ELIGIBLE),
)


@dataclass(frozen=True, slots=True)
class QualificationCandidate:
    submission_id: str
    hotkey: str
    block: int
    agent_files: str | None = None


class QualificationDb:
    """Qualification worker's view of the database (one per worker process)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def next_candidate(self, *, window_size: int) -> QualificationCandidate | None:
        """First UNVERIFIED submission inside the live challenger queue head.

        The live queue mirrors the duel-resolver's challenger filter, except it
        includes only UNVERIFIED and ELIGIBLE rows. Terminal rows are skipped
        because they can never become challengers. If the first ``window_size`` live
        rows are already eligible, there is nothing for this worker to do.
        """
        if window_size <= 0:
            return None
        meta = _current_metagraph()
        queue_head = (
            select(
                models.Submission.submission_id,
                models.Submission.hotkey,
                models.Submission.block,
                models.Submission.agent_files,
                models.Submission.status_id,
            )
            .join(meta, meta.c.ss58_hot == models.Submission.hotkey)
            .where(
                models.Submission.status_id.in_(_QUEUE_HEAD_STATUSES),
                # No older than the current registration (stale after re-registration).
                meta.c.block <= models.Submission.block,
                models.Submission.submission_id.not_in(select(models.King.king_id)),
                models.Submission.submission_id.not_in(
                    select(models.Challenge.challenger_submission_id)
                ),
            )
            .order_by(models.Submission.block, models.Submission.submission_id)
            .limit(window_size)
            .subquery()
        )
        stmt = (
            select(queue_head)
            .where(queue_head.c.status_id == int(SubmissionStatus.UNVERIFIED))
            .order_by(queue_head.c.block, queue_head.c.submission_id)
            .limit(1)
        )
        async with async_session_scope(self._sessions) as session:
            row = (await session.execute(stmt)).first()
        return None if row is None else _row_to_candidate(row)

    async def save_qualification(
        self,
        *,
        submission_id: str,
        result: SecurityQualificationResult,
        outcome: QualificationOutcome,
        base_files_available: bool,
        failures: Sequence[str] | None = None,
        duration_seconds: float | None = None,
    ) -> bool:
        """Persist a final qualification and transition the submission status.

        Returns False if the submission is no longer UNVERIFIED. The guarded update
        keeps the operation safe across restarts or manual status edits even though
        the first worker version is expected to be single-instance.
        """
        new_status = _submission_status_for_outcome(outcome)
        failure_lines = list(failures) if failures is not None else security_failures(result)
        values = _qualification_values(
            submission_id=submission_id,
            outcome=outcome.value,
            result=result,
            base_files_available=base_files_available,
            failures=failure_lines,
            duration_seconds=duration_seconds,
        )
        async with async_session_scope(self._sessions) as session:
            updated = await session.execute(
                update(models.Submission)
                .where(
                    models.Submission.submission_id == submission_id,
                    models.Submission.status_id == int(SubmissionStatus.UNVERIFIED),
                )
                .values(status_id=int(new_status))
                .returning(models.Submission.submission_id)
            )
            if updated.first() is None:
                return False
            await session.execute(_upsert_qualification(values))
            return True

    async def save_error(
        self,
        *,
        submission_id: str,
        error: str,
        base_files_available: bool,
        model: str | None = None,
        duration_seconds: float | None = None,
    ) -> bool:
        """Record a non-final qualification error while leaving status UNVERIFIED."""
        values = {
            "submission_id": submission_id,
            "outcome": "error",
            "verdict": None,
            "overall_score": None,
            "security_score": None,
            "model": model,
            "summary": None,
            "reasons": None,
            "risks": None,
            "risk_categories": None,
            "failures": None,
            "required_changes": None,
            "base_files_available": base_files_available,
            "error": error,
            "duration_seconds": duration_seconds,
        }
        async with async_session_scope(self._sessions) as session:
            exists = (
                await session.scalars(
                    select(models.Submission.submission_id)
                    .where(
                        models.Submission.submission_id == submission_id,
                        models.Submission.status_id == int(SubmissionStatus.UNVERIFIED),
                    )
                    .limit(1)
                )
            ).first()
            if exists is None:
                return False
            await session.execute(_upsert_qualification(values))
            return True


def _submission_status_for_outcome(outcome: QualificationOutcome) -> SubmissionStatus:
    match outcome:
        case QualificationOutcome.QUALIFIED:
            return SubmissionStatus.ELIGIBLE
        case QualificationOutcome.DISQUALIFIED:
            return SubmissionStatus.DISQUALIFIED
        case QualificationOutcome.NEEDS_REVIEW:
            return SubmissionStatus.NEEDS_REVIEW


def _qualification_values(
    *,
    submission_id: str,
    outcome: str,
    result: SecurityQualificationResult,
    base_files_available: bool,
    failures: Sequence[str],
    duration_seconds: float | None,
) -> dict[str, Any]:
    return {
        "submission_id": submission_id,
        "outcome": outcome,
        "verdict": result.verdict,
        "overall_score": result.overall_score,
        "security_score": result.security_score,
        "model": result.model,
        "summary": result.summary,
        "reasons": _lines(result.reasons),
        "risks": _lines(result.risks),
        "risk_categories": _lines(sorted(security_risk_categories(result))),
        "failures": _lines(failures),
        "required_changes": _lines(result.required_changes),
        "base_files_available": base_files_available,
        "error": None,
        "duration_seconds": duration_seconds,
    }


def _upsert_qualification(values: dict[str, Any]):
    insert_stmt = insert(models.SubmissionQualification).values(**values)
    update_values = {
        key: value for key, value in values.items() if key != "submission_id"
    }
    update_values["updated_at"] = func.now()
    return insert_stmt.on_conflict_do_update(
        index_elements=["submission_id"],
        set_=update_values,
    )


def _lines(values: Sequence[str]) -> str | None:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return "\n".join(cleaned) if cleaned else None


def _row_to_candidate(row: Row[Any]) -> QualificationCandidate:
    return QualificationCandidate(
        submission_id=row.submission_id,
        hotkey=row.hotkey,
        block=row.block,
        agent_files=row.agent_files,
    )


def _current_metagraph():
    """Current metagraph as (ss58_hot, block): latest registration per uid."""
    return (
        select(models.Registration.ss58_hot, models.Registration.block)
        .distinct(models.Registration.uid)
        .order_by(models.Registration.uid, models.Registration.block.desc())
        .subquery()
    )
