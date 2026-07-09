"""Database operations for the judge worker."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
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


@dataclass(frozen=True, slots=True)
class TaskScreenDuelComparison:
    """Privacy-safe telemetry joining a task screen to a newly saved duel verdict.

    The qualification and duel patches are deliberately reduced to hashes plus an
    equality bit here, at the persistence boundary. Patch bodies must never reach
    the observability event emitted by the judge worker.
    """

    task_id: str
    king_submission_id: str
    challenger_submission_id: str
    screening_king_score: float | None
    duel_king_score: float
    duel_minus_screen_king_score_delta: float | None
    screening_model: str | None
    duel_model: str
    qualification_patch_sha256: str
    duel_patch_sha256: str
    qualification_patch_matches_duel_patch: bool


class JudgeDb:
    """Judge worker's view of the database (one per worker process)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def pending_judge_requests(self) -> list[JudgeRequest]:
        """Fresh challenge-scoped solution pairs awaiting a judgment.

        Empty when there is nothing to judge. The king side comes from
        ``duel_task_solutions`` for this challenge, not the qualification-time
        task-wide cache.
        """
        king_sol = aliased(models.DuelTaskSolution)
        chal_sol = aliased(models.DuelTaskSolution)
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
                and_(
                    king_sol.task_id == models.Task.task_id,
                    king_sol.challenger_submission_id
                    == models.Challenge.challenger_submission_id,
                    king_sol.submission_id == models.King.king_id,
                ),
            )
            .join(
                chal_sol,
                and_(
                    chal_sol.task_id == models.Task.task_id,
                    chal_sol.challenger_submission_id
                    == models.Challenge.challenger_submission_id,
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
    ) -> TaskScreenDuelComparison | None:
        """Persist the judgment for one (task, king, challenger) triple.

        `attempts`/`duration_seconds` are worker telemetry (LLM tries spent and
        wall-clock time), not part of the judging verdict. When this call wins the
        write-once insert and a matching task-screen row exists, return a
        privacy-safe comparison for observability. A retry/race that loses the
        insert returns ``None`` so it cannot emit duplicate comparison telemetry.
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
            .returning(models.Judgement.task_id)
        )
        async with async_session_scope(self._sessions) as session:
            inserted_task_id = (await session.execute(stmt)).scalar_one_or_none()
            if inserted_task_id is None:
                return None
            # The judge's degraded fallback persists a neutral 0.5/0.5 row so the
            # duel can progress, but it is not evidence about solution quality.
            if judgment.error is not None:
                return None

            screen_stmt = select(
                models.TaskScreening.king_score,
                models.TaskScreening.model,
                models.TaskScreening.qualification_solution,
            ).where(
                models.TaskScreening.task_id == task.task_id,
                models.TaskScreening.king_submission_id == king_solution.submission_id,
            )
            screening = (await session.execute(screen_stmt)).one_or_none()
            if screening is None:
                return None

            return _task_screen_duel_comparison(
                task_id=task.task_id,
                king_submission_id=king_solution.submission_id,
                challenger_submission_id=challenger_solution.submission_id,
                screening_king_score=screening.king_score,
                duel_king_score=judgment.king_score,
                screening_model=screening.model,
                duel_model=judgment.model,
                qualification_patch=screening.qualification_solution,
                duel_patch=king_solution.patch,
            )


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


def _task_screen_duel_comparison(
    *,
    task_id: str,
    king_submission_id: str,
    challenger_submission_id: str,
    screening_king_score: float | None,
    duel_king_score: float,
    screening_model: str | None,
    duel_model: str,
    qualification_patch: str,
    duel_patch: str,
) -> TaskScreenDuelComparison:
    """Build score-drift telemetry without retaining either patch body."""
    return TaskScreenDuelComparison(
        task_id=task_id,
        king_submission_id=king_submission_id,
        challenger_submission_id=challenger_submission_id,
        screening_king_score=screening_king_score,
        duel_king_score=duel_king_score,
        duel_minus_screen_king_score_delta=(
            None
            if screening_king_score is None
            else duel_king_score - screening_king_score
        ),
        screening_model=screening_model,
        duel_model=duel_model,
        qualification_patch_sha256=_patch_sha256(qualification_patch),
        duel_patch_sha256=_patch_sha256(duel_patch),
        qualification_patch_matches_duel_patch=qualification_patch == duel_patch,
    )


def _patch_sha256(patch: str) -> str:
    return sha256(patch.encode("utf-8")).hexdigest()
