"""Database operations for the task-screening worker.

The task solver writes a king qualification solve to ``task_screenings`` and moves
the task to ``PENDING_SCREEN``.  This seam exposes only those rows that still belong
to the reigning king, then atomically records either a final screening decision or
retryable error telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import exists, select, update

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


class _StaleScreening(RuntimeError):
    """Internal rollback signal when a guarded two-row decision loses a race."""


class TaskScreeningDb:
    """Task-screener's focused async view of the validator database."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def pending_requests(self) -> list[TaskScreenRequest]:
        """Return all still-pending screening rows for the reigning king.

        Returning the full wanted set lets the worker cancel obsolete in-flight
        calls when a task is manually changed, deleted, or belongs to a dethroned
        king. Concurrency is bounded by the worker rather than by this query.
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
                        attempts=attempts,
                        score_duration_seconds=duration_seconds,
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
    ) -> bool:
        """Record retryable scorer failure telemetry without admitting the task."""
        if not 0 <= max_score <= 1:
            raise ValueError("max_score must be between 0 and 1")
        eligible_task = _eligible_task_exists(
            task_id=task_id, king_submission_id=king_submission_id
        )
        async with async_session_scope(self._sessions) as session:
            saved = await session.execute(
                update(models.TaskScreening)
                .where(
                    models.TaskScreening.task_id == task_id,
                    models.TaskScreening.king_submission_id == king_submission_id,
                    models.TaskScreening.outcome == "pending",
                    eligible_task,
                )
                .values(
                    king_score=None,
                    max_score=max_score,
                    reason=None,
                    model=model,
                    rationale=None,
                    error=error,
                    attempts=attempts,
                    score_duration_seconds=duration_seconds,
                )
                .returning(models.TaskScreening.task_id)
            )
            return saved.first() is not None


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
