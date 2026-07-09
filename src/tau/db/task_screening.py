"""Database operations for the task-screening worker.

The task solver writes a king qualification solve to ``task_screenings`` and moves
the task to ``PENDING_SCREEN``.  This seam exposes only those rows that still belong
to the reigning king, then atomically records either a final screening decision or
bounded retry/error telemetry.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import exists, func, select, update

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import TaskStatus

FinalScreeningOutcome = Literal["qualified", "disqualified"]


@dataclass(frozen=True, slots=True)
class TaskScreenRequest:
    """One qualification-time king patch awaiting an independent score."""

    task_id: str
    king_submission_id: str
    problem_statement: str
    reference_patch: str
    qualification_solution: str


@dataclass(frozen=True, slots=True)
class ScreeningFailureSave:
    """Outcome of atomically recording one failed screening run."""

    saved: bool
    exhausted: bool = False
    failed_runs: int | None = None
    cumulative_attempts: int | None = None
    next_retry_at: dt.datetime | None = None


class _StaleScreening(RuntimeError):
    """Internal rollback signal when a guarded two-row decision loses a race."""


class TaskScreeningDb:
    """Task-screener's focused async view of the validator database."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def pending_requests(
        self, *, include_deferred: bool = False
    ) -> list[TaskScreenRequest]:
        """Return all still-pending screening rows for the reigning king.

        Rows whose exponential backoff has not elapsed are omitted unless
        ``include_deferred`` is requested (used by disabled mode to immediately
        drain every pending row). Concurrency is bounded by the worker.
        """
        reigning_king_id = _reigning_king_id()
        stmt = (
            select(
                models.Task.task_id,
                models.Task.king_id.label("king_submission_id"),
                models.Task.problem_statement,
                models.Task.reference_patch,
                models.TaskScreening.qualification_solution,
            )
            .join(
                models.TaskScreening,
                models.TaskScreening.task_id == models.Task.task_id,
            )
            .where(
                models.Task.king_id == reigning_king_id,
                models.Task.status_id == int(TaskStatus.PENDING_SCREEN),
                models.TaskScreening.king_submission_id == models.Task.king_id,
                models.TaskScreening.outcome == "pending",
            )
            .order_by(models.Task.created_at, models.Task.task_id)
        )
        if not include_deferred:
            stmt = stmt.where(
                (models.TaskScreening.next_retry_at.is_(None))
                | (models.TaskScreening.next_retry_at <= func.now())
            )
        async with async_session_scope(self._sessions) as session:
            rows = (await session.execute(stmt)).all()
        return [
            TaskScreenRequest(
                task_id=row.task_id,
                king_submission_id=row.king_submission_id,
                problem_statement=row.problem_statement,
                reference_patch=row.reference_patch,
                qualification_solution=row.qualification_solution,
            )
            for row in rows
        ]

    async def save_decision(
        self,
        *,
        task_id: str,
        king_submission_id: str,
        outcome: FinalScreeningOutcome,
        king_score: float | None,
        max_score: float,
        reason: str,
        model: str | None,
        rationale: str | None,
        attempts: int,
        duration_seconds: float,
    ) -> bool:
        """Atomically persist a final decision and transition the task.

        The screening row is claimed with a guarded update before the task status is
        changed.  A second worker, a stale result after a king change, or any manual
        task transition returns ``False`` and changes neither row.
        """
        if outcome not in ("qualified", "disqualified"):
            raise ValueError("outcome must be 'qualified' or 'disqualified'")
        if not 0 <= max_score <= 1:
            raise ValueError("max_score must be between 0 and 1")
        if king_score is not None and not 0 <= king_score <= 1:
            raise ValueError("king_score must be between 0 and 1")
        if attempts < 0:
            raise ValueError("attempts must be >= 0")
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")
        new_status = (
            TaskStatus.QUALIFIED if outcome == "qualified" else TaskStatus.DISQUALIFIED
        )
        eligible_task = _eligible_task_exists(
            task_id=task_id, king_submission_id=king_submission_id
        )
        try:
            async with async_session_scope(self._sessions) as session:
                claimed = await session.execute(
                    update(models.TaskScreening)
                    .where(
                        models.TaskScreening.task_id == task_id,
                        models.TaskScreening.king_submission_id == king_submission_id,
                        models.TaskScreening.outcome == "pending",
                        eligible_task,
                    )
                    .values(
                        king_score=king_score,
                        max_score=max_score,
                        outcome=outcome,
                        reason=reason,
                        model=model,
                        rationale=rationale,
                        error=None,
                        attempts=models.TaskScreening.attempts + attempts,
                        score_duration_seconds=(
                            func.coalesce(
                                models.TaskScreening.score_duration_seconds, 0.0
                            )
                            + duration_seconds
                        ),
                        next_retry_at=None,
                    )
                    .returning(models.TaskScreening.task_id)
                )
                if claimed.first() is None:
                    return False

                transitioned = await session.execute(
                    update(models.Task)
                    .where(
                        models.Task.task_id == task_id,
                        models.Task.king_id == king_submission_id,
                        models.Task.king_id == _reigning_king_id(),
                        models.Task.status_id == int(TaskStatus.PENDING_SCREEN),
                    )
                    .values(status_id=int(new_status))
                    .returning(models.Task.task_id)
                )
                if transitioned.first() is None:
                    # Raising rolls back the screening-row claim as well.
                    raise _StaleScreening
        except _StaleScreening:
            return False
        return True

    async def save_error(
        self,
        *,
        task_id: str,
        king_submission_id: str,
        max_score: float,
        model: str | None,
        error: str,
        attempts: int,
        duration_seconds: float,
        max_failed_runs: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
    ) -> ScreeningFailureSave:
        """Record one failed run, scheduling retry or terminally dropping the task.

        The screening row is locked before either outcome. On the terminal run the
        task transition and audit fields commit together, so a pool slot can never
        remain permanently occupied by an unscreenable task.
        """
        if not 0 <= max_score <= 1:
            raise ValueError("max_score must be between 0 and 1")
        if attempts < 0:
            raise ValueError("attempts must be >= 0")
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")
        if max_failed_runs < 1:
            raise ValueError("max_failed_runs must be >= 1")
        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be positive")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError("retry_max_seconds must be >= retry_base_seconds")
        eligible_task = _eligible_task_exists(
            task_id=task_id, king_submission_id=king_submission_id
        )
        now = dt.datetime.now(dt.UTC)
        try:
            async with async_session_scope(self._sessions) as session:
                row = (
                    await session.execute(
                        select(models.TaskScreening)
                        .where(
                            models.TaskScreening.task_id == task_id,
                            models.TaskScreening.king_submission_id
                            == king_submission_id,
                            models.TaskScreening.outcome == "pending",
                            (models.TaskScreening.next_retry_at.is_(None))
                            | (models.TaskScreening.next_retry_at <= func.now()),
                            eligible_task,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if row is None:
                    return ScreeningFailureSave(saved=False)

                failed_runs = row.failed_runs + 1
                cumulative_attempts = row.attempts + attempts
                row.king_score = None
                row.max_score = max_score
                row.model = model
                row.rationale = None
                row.error = error
                row.attempts = cumulative_attempts
                row.failed_runs = failed_runs
                row.score_duration_seconds = (
                    row.score_duration_seconds or 0.0
                ) + duration_seconds

                if failed_runs >= max_failed_runs:
                    transitioned = await session.execute(
                        update(models.Task)
                        .where(
                            models.Task.task_id == task_id,
                            models.Task.king_id == king_submission_id,
                            models.Task.king_id == _reigning_king_id(),
                            models.Task.status_id == int(TaskStatus.PENDING_SCREEN),
                        )
                        .values(status_id=int(TaskStatus.DISQUALIFIED))
                        .returning(models.Task.task_id)
                    )
                    if transitioned.first() is None:
                        raise _StaleScreening
                    row.outcome = "disqualified"
                    row.reason = "screening_exhausted"
                    row.next_retry_at = None
                    return ScreeningFailureSave(
                        saved=True,
                        exhausted=True,
                        failed_runs=failed_runs,
                        cumulative_attempts=cumulative_attempts,
                    )

                delay_seconds = min(
                    retry_max_seconds,
                    retry_base_seconds * (2 ** (failed_runs - 1)),
                )
                next_retry_at = now + dt.timedelta(seconds=delay_seconds)
                row.reason = None
                row.next_retry_at = next_retry_at
                return ScreeningFailureSave(
                    saved=True,
                    failed_runs=failed_runs,
                    cumulative_attempts=cumulative_attempts,
                    next_retry_at=next_retry_at,
                )
        except _StaleScreening:
            return ScreeningFailureSave(saved=False)


def _reigning_king_id():
    return (
        select(models.King.king_id)
        .order_by(models.King.king_from.desc())
        .limit(1)
        .scalar_subquery()
    )


def _eligible_task_exists(*, task_id: str, king_submission_id: str):
    return exists(
        select(models.Task.task_id).where(
            models.Task.task_id == task_id,
            models.Task.king_id == king_submission_id,
            models.Task.king_id == _reigning_king_id(),
            models.Task.status_id == int(TaskStatus.PENDING_SCREEN),
        )
    )
