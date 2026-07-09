"""Database seam for qualification-time task screening."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import TaskStatus

FinalScreeningOutcome = Literal["qualified", "disqualified"]
FailureState = Literal["stale", "retry", "exhausted"]


@dataclass(frozen=True, slots=True)
class TaskScreenRequest:
    task_id: str
    king_submission_id: str
    problem_statement: str
    reference_patch: str
    qualification_solution: str


@dataclass(frozen=True, slots=True)
class ScreeningFailureSave:
    state: FailureState
    failed_runs: int = 0
    next_retry_at: dt.datetime | None = None


class TaskScreeningDb:
    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def pending_requests(
        self, *, include_deferred: bool = False
    ) -> list[TaskScreenRequest]:
        """Return the reigning king's PENDING_SCREEN rows that are due."""
        stmt = (
            select(
                models.Task.task_id,
                models.Task.king_id.label("king_submission_id"),
                models.Task.problem_statement,
                models.Task.reference_patch,
                models.TaskScreening.qualification_solution,
            )
            .join(models.TaskScreening)
            .where(
                models.Task.king_id == _reigning_king_id(),
                models.Task.status_id == int(TaskStatus.PENDING_SCREEN),
                models.TaskScreening.king_submission_id == models.Task.king_id,
            )
            .order_by(models.Task.created_at, models.Task.task_id)
        )
        if not include_deferred:
            stmt = stmt.where(_retry_is_due())
        async with async_session_scope(self._sessions) as session:
            rows = (await session.execute(stmt)).all()
        return [TaskScreenRequest(*row) for row in rows]

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
    ) -> bool:
        """Atomically save a final screen and transition its pending task."""
        if outcome not in ("qualified", "disqualified"):
            raise ValueError("invalid screening outcome")
        if king_score is not None and not 0 <= king_score <= 1:
            raise ValueError("king_score must be between 0 and 1")
        async with async_session_scope(self._sessions) as session:
            task = await _lock_pending_task(session, task_id, king_submission_id)
            if task is None:
                return False
            screening = await session.get(models.TaskScreening, task_id)
            if screening is None:
                raise RuntimeError(f"pending task {task_id} has no screening row")
            screening.king_score = king_score
            screening.max_score = max_score
            screening.reason = reason
            screening.model = model
            screening.next_retry_at = None
            task.status_id = int(
                TaskStatus.QUALIFIED
                if outcome == "qualified"
                else TaskStatus.DISQUALIFIED
            )
        return True

    async def save_error(
        self,
        *,
        task_id: str,
        king_submission_id: str,
        max_failed_runs: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
    ) -> ScreeningFailureSave:
        """Count one due failed run, then back off or terminally drop the task."""
        now = dt.datetime.now(dt.UTC)
        async with async_session_scope(self._sessions) as session:
            task = await _lock_pending_task(
                session, task_id, king_submission_id, require_due=True
            )
            if task is None:
                return ScreeningFailureSave("stale")
            screening = await session.get(models.TaskScreening, task_id)
            if screening is None:
                raise RuntimeError(f"pending task {task_id} has no screening row")

            screening.failed_runs += 1
            if screening.failed_runs >= max_failed_runs:
                task.status_id = int(TaskStatus.DISQUALIFIED)
                screening.reason = "screening_exhausted"
                screening.next_retry_at = None
                return ScreeningFailureSave("exhausted", screening.failed_runs)

            delay = min(
                retry_max_seconds,
                retry_base_seconds * 2 ** (screening.failed_runs - 1),
            )
            screening.reason = None
            screening.next_retry_at = now + dt.timedelta(seconds=delay)
            return ScreeningFailureSave(
                "retry", screening.failed_runs, screening.next_retry_at
            )


async def _lock_pending_task(
    session,
    task_id: str,
    king_submission_id: str,
    *,
    require_due: bool = False,
):
    stmt = select(models.Task).where(
        models.Task.task_id == task_id,
        models.Task.king_id == king_submission_id,
        models.Task.king_id == _reigning_king_id(),
        models.Task.status_id == int(TaskStatus.PENDING_SCREEN),
    )
    if require_due:
        stmt = stmt.join(models.TaskScreening).where(_retry_is_due())
    return (await session.execute(stmt.with_for_update())).scalar_one_or_none()


def _retry_is_due():
    return (models.TaskScreening.next_retry_at.is_(None)) | (
        models.TaskScreening.next_retry_at <= func.now()
    )


def _reigning_king_id():
    return (
        select(models.King.king_id)
        .order_by(models.King.king_from.desc())
        .limit(1)
        .scalar_subquery()
    )
