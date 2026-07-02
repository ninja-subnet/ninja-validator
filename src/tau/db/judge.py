"""Database operations for the judge worker."""

from __future__ import annotations

from typing import Any

from sqlalchemy import ColumnElement, Row, and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import aliased

from tau.judging.types import Judgment, Solution, Task

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import TaskStatus

# One unit of judge work: (task, king_solution, challenger_solution).
JudgeRequest = tuple[Task, Solution, Solution]


class JudgeDb:
    """Judge worker's view of the database (one per worker process)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def pending_judge_requests(self) -> list[JudgeRequest]:
        """King/challenger solution pairs awaiting a judgment, each with the task's
        reference patch. Empty when there is nothing to judge."""
        king_sol = aliased(models.TaskSolution)
        chal_sol = aliased(models.TaskSolution)
        # The reigning king is the one with the latest king_from.
        reigning_king_id = (
            select(models.King.king_id)
            .order_by(models.King.king_from.desc())
            .limit(1)
            .scalar_subquery()
        )
        stmt = (
            select(
                models.Task.task_id,
                models.Task.problem_statement,
                models.Task.reference_patch,
                models.King.king_id.label("king_submission_id"),
                king_sol.solution.label("king_patch"),
                models.Challenge.challenger_submission_id,
                chal_sol.solution.label("chal_patch"),
            )
            .select_from(models.King)
            .join(models.Challenge, models.Challenge.king_id == models.King.king_id)
            .join(models.Task, models.Task.king_id == models.King.king_id)
            .join(
                king_sol,
                # kings.king_id holds the king's submission_id -> its own solution row.
                and_(
                    king_sol.task_id == models.Task.task_id,
                    king_sol.submission_id == models.King.king_id,
                ),
            )
            .join(
                chal_sol,
                and_(
                    chal_sol.task_id == models.Task.task_id,
                    chal_sol.submission_id == models.Challenge.challenger_submission_id,
                ),
            )
            .outerjoin(
                models.Judgement,
                and_(
                    models.Judgement.task_id == models.Task.task_id,
                    models.Judgement.king_submission_id == models.King.king_id,
                    models.Judgement.challenger_submission_id
                    == models.Challenge.challenger_submission_id,
                ),
            )
            .where(
                models.King.king_id == reigning_king_id,
                models.Task.status_id == int(TaskStatus.QUALIFIED),
                _in_active_pool(models.Task, models.Challenge),
                models.Judgement.task_id.is_(None),
            )
            # oldest qualified task first, task_id breaking ties for determinism
            .order_by(models.Task.created_at, models.Task.task_id)
        )
        async with async_session_scope(self._sessions) as session:
            rows = (await session.execute(stmt)).all()
        return [_row_to_request(row) for row in rows]

    async def save_judgment(
        self,
        task: Task,
        king_solution: Solution,
        challenger_solution: Solution,
        judgment: Judgment,
        *,
        attempts: int,
        duration_seconds: float,
    ) -> None:
        """Persist the judgment for one (task, king, challenger) triple.

        `attempts`/`duration_seconds` are worker telemetry (LLM tries spent and
        wall-clock time), not part of the judging verdict.
        """
        stmt = (
            insert(models.Judgement)
            .values(
                task_id=task.task_id,
                king_submission_id=king_solution.submission_id,
                challenger_submission_id=challenger_solution.submission_id,
                llm_winner=judgment.winner,
                king_score=judgment.king_score,
                challenger_score=judgment.challenger_score,
                model=judgment.model,
                rationale=judgment.rationale,
                error=judgment.error,
                attempts=attempts,
                duration_seconds=duration_seconds,
            )
            # write-once: keep the first verdict, never clobber on a retry/race
            .on_conflict_do_nothing(
                index_elements=[
                    "task_id",
                    "king_submission_id",
                    "challenger_submission_id",
                ],
            )
        )
        async with async_session_scope(self._sessions) as session:
            await session.execute(stmt)


def _in_active_pool(
    task: type[models.Task], challenge: type[models.Challenge]
) -> ColumnElement[bool]:
    # challenges.status names the pool currently dueling (1/2); judge only that
    # pool's tasks. pending(0)/finished never equal a pool_type, so they drop out.
    return task.pool_type == challenge.status


def _row_to_request(row: Row[Any]) -> JudgeRequest:
    task = Task(
        task_id=row.task_id,
        problem_statement=row.problem_statement,
        reference_patch=row.reference_patch,
    )
    king = Solution(submission_id=row.king_submission_id, patch=row.king_patch or "")
    challenger = Solution(
        submission_id=row.challenger_submission_id, patch=row.chal_patch or ""
    )
    return task, king, challenger
