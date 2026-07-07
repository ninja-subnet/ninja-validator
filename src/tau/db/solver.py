"""Database operations for the task-solver worker.

A focused, self-contained slice of the DB seam (the same pattern as
``GeneratorDb``), kept **sync** because the orchestrator is sequential blocking
docker/git work — async would buy nothing. Two reads drive the worker's two phases:

  * ``next_qualification_jobs`` — CANDIDATE tasks of the reigning king, to be run
    against the king's own agent (viability check);
  * ``next_duel_jobs`` — QUALIFIED tasks in active challenges where either side lacks
    a fresh challenge-scoped solution, to be run against the king or challenger agent.

Each job carries the *submission_id* whose agent should run; the worker resolves the
actual agent files from the local submissions directory (folder == submission id).
Writes (``finish_qualification`` / ``save_duel_task_solution``) are idempotent via
``ON CONFLICT``. It deliberately avoids the ``tau.bittensor`` imports that gate the
full ``database.py``, so it can ship independently.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import aliased

from tau.pools import PoolTargets

from . import models
from .engine import create_db_engine, session_factory, session_scope
from .status import TaskStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SolveJob:
    """One sandboxed solve: run *submission_id*'s agent on *task_id*.

    Used for qualification; duel solves use ``DuelSolveJob`` so they can carry the
    challenge identity. The agent files live on disk under
    ``<submissions_dir>/<submission_id>/``.
    """

    task_id: str
    submission_id: str
    problem_statement: str
    repo_clone_url: str
    base_commit: str  # tasks.parent_sha — the state to check out (before the fix)


@dataclass(frozen=True, slots=True)
class DuelSolveJob:
    """One fresh duel solve, scoped to a specific challenge.

    ``submission_id`` is the agent to run. ``challenger_submission_id`` is the
    challenge identity, used to keep king solutions fresh per challenger instead of
    cached globally by task.
    """

    task_id: str
    submission_id: str
    challenger_submission_id: str
    problem_statement: str
    repo_clone_url: str
    base_commit: str


class SolverDb:
    """Task-solver's view of the database (one per worker process, sync)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_db_engine(url, echo=echo)
        self._sessions = session_factory(self._engine)

    def close(self) -> None:
        self._engine.dispose()

    # -- phase A: qualification ----------------------------------------------
    def next_qualification_jobs(self, limit: int) -> list[SolveJob]:
        """CANDIDATE tasks of the reigning king, to be solved by the king's agent.

        The reigning king is the ``kings`` row with the latest ``king_from`` (matches
        ``GeneratorDb``). Returns ``[]`` if no king reigns.
        """
        if limit <= 0:
            return []
        with session_scope(self._sessions) as session:
            # A king's king_id IS its submission id (1:1 FK), so it doubles as the
            # submission whose agent runs for qualification.
            king_id = session.scalars(
                select(models.King.king_id)
                .order_by(models.King.king_from.desc())
                .limit(1)
            ).first()
            if king_id is None:
                return []
            rows = session.execute(
                select(
                    models.Task.task_id,
                    models.Task.problem_statement,
                    models.Task.repo_clone_url,
                    models.Task.parent_sha,
                )
                .where(
                    models.Task.king_id == king_id,
                    models.Task.status_id == int(TaskStatus.CANDIDATE),
                )
                .order_by(models.Task.pool_type, models.Task.created_at)
                .limit(limit)
            ).all()
        return [
            SolveJob(
                task_id=row.task_id,
                submission_id=king_id,
                problem_statement=row.problem_statement,
                repo_clone_url=row.repo_clone_url,
                base_commit=row.parent_sha,
            )
            for row in rows
        ]

    def finish_qualification(
        self,
        *,
        task_id: str,
        king_submission_id: str,
        qualified: bool,
        solution: str,
        duration: float,
        exit_reason: str,
    ) -> None:
        """Flip a CANDIDATE task to QUALIFIED/DISQUALIFIED.

        Qualification is only a task-quality gate. Duel comparison solutions are
        written later to ``duel_task_solutions`` so the king is re-run under the same
        current inference conditions as the challenger for each challenge.
        """
        _ = (king_submission_id, solution, duration, exit_reason)
        status = TaskStatus.QUALIFIED if qualified else TaskStatus.DISQUALIFIED
        with session_scope(self._sessions) as session:
            session.execute(
                update(models.Task)
                .where(models.Task.task_id == task_id)
                .values(status_id=int(status))
            )

    # -- phase B: fresh duel solves ------------------------------------------
    def next_duel_jobs(
        self,
        limit: int,
        *,
        require_full_pool: bool = False,
        pool_targets: PoolTargets | None = None,
    ) -> list[DuelSolveJob]:
        """Fresh king/challenger solves still missing for active challenge rounds.

        A row is scoped by ``(task_id, challenger_submission_id, submission_id)``.
        That means the king runs again for each challenger rather than reusing the
        qualification solve or a prior challenger's duel solve.
        When ``require_full_pool`` is true, active challenge pools are ignored until
        the matching pool has reached its configured count of QUALIFIED tasks.
        """
        if limit <= 0:
            return []
        if pool_targets is None:
            pool_targets = PoolTargets()
        king_sol = aliased(models.DuelTaskSolution)
        chal_sol = aliased(models.DuelTaskSolution)
        stmt = (
            select(
                models.Task.task_id,
                models.Task.problem_statement,
                models.Task.repo_clone_url,
                models.Task.parent_sha,
                models.Challenge.king_id.label("king_submission_id"),
                models.Challenge.challenger_submission_id,
                king_sol.task_id.label("king_solution_task_id"),
                chal_sol.task_id.label("challenger_solution_task_id"),
            )
            .select_from(models.Challenge)
            .join(
                models.Task,
                and_(
                    models.Task.king_id == models.Challenge.king_id,
                    models.Task.pool_type == models.Challenge.status,
                    models.Task.status_id == int(TaskStatus.QUALIFIED),
                ),
            )
            .outerjoin(
                king_sol,
                and_(
                    king_sol.task_id == models.Task.task_id,
                    king_sol.challenger_submission_id
                    == models.Challenge.challenger_submission_id,
                    king_sol.submission_id == models.Challenge.king_id,
                ),
            )
            .outerjoin(
                chal_sol,
                and_(
                    chal_sol.task_id == models.Task.task_id,
                    chal_sol.challenger_submission_id
                    == models.Challenge.challenger_submission_id,
                    chal_sol.submission_id == models.Challenge.challenger_submission_id,
                ),
            )
            .where(
                models.Challenge.status.in_((1, 2)),
                or_(king_sol.task_id.is_(None), chal_sol.task_id.is_(None)),
            )
            .order_by(
                models.Task.created_at,
                models.Task.task_id,
                models.Challenge.challenger_submission_id,
            )
            .limit(limit * 2)
        )
        if require_full_pool:
            qualified_count = (
                select(func.count(models.Task.task_id))
                .where(
                    models.Task.king_id == models.Challenge.king_id,
                    models.Task.pool_type == models.Challenge.status,
                    models.Task.status_id == int(TaskStatus.QUALIFIED),
                )
                .correlate(models.Challenge)
                .scalar_subquery()
            )
            pool_target = case(
                (models.Challenge.status == 1, pool_targets.pool_one),
                (models.Challenge.status == 2, pool_targets.pool_two),
                else_=pool_targets.pool_two,
            )
            stmt = stmt.where(qualified_count >= pool_target)
        with session_scope(self._sessions) as session:
            rows = session.execute(stmt).all()
        jobs: list[DuelSolveJob] = []
        for row in rows:
            missing_king = row.king_solution_task_id is None
            missing_challenger = row.challenger_solution_task_id is None
            if missing_king and missing_challenger:
                if len(jobs) + 2 > limit:
                    continue
                for side in _duel_side_order(
                    task_id=row.task_id,
                    king_submission_id=row.king_submission_id,
                    challenger_submission_id=row.challenger_submission_id,
                ):
                    jobs.append(_duel_job_for_side(row, side=side))
            elif missing_king and len(jobs) < limit:
                jobs.append(_duel_job_for_side(row, side="king"))
            elif missing_challenger and len(jobs) < limit:
                jobs.append(_duel_job_for_side(row, side="challenger"))
            if len(jobs) >= limit:
                break
        return jobs

    def save_duel_task_solution(
        self,
        *,
        task_id: str,
        challenger_submission_id: str,
        submission_id: str,
        solution: str,
        duration: float,
        exit_reason: str,
    ) -> None:
        """Insert a fresh challenge-scoped solution row.

        Idempotent on ``(task_id, challenger_submission_id, submission_id)`` so a
        retry/race keeps the first terminal solve for that side of that round.
        """
        with session_scope(self._sessions) as session:
            session.execute(
                insert(models.DuelTaskSolution)
                .values(
                    task_id=task_id,
                    challenger_submission_id=challenger_submission_id,
                    submission_id=submission_id,
                    solution=solution,
                    duration=duration,
                    exit_reason=exit_reason,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        "task_id",
                        "challenger_submission_id",
                        "submission_id",
                    ]
                )
            )

    def save_task_solution(
        self,
        *,
        task_id: str,
        submission_id: str,
        solution: str,
        duration: float,
        exit_reason: str,
    ) -> None:
        """Insert a legacy task-grain solution row.

        The live duel path writes ``duel_task_solutions`` instead; this remains for
        compatibility with older local tooling.
        """
        with session_scope(self._sessions) as session:
            session.execute(
                insert(models.TaskSolution)
                .values(
                    task_id=task_id,
                    submission_id=submission_id,
                    solution=solution,
                    duration=duration,
                    exit_reason=exit_reason,
                )
                .on_conflict_do_nothing(index_elements=["task_id", "submission_id"])
            )


def _duel_side_order(
    *,
    task_id: str,
    king_submission_id: str,
    challenger_submission_id: str,
) -> tuple[str, str]:
    """Return a stable, roughly balanced side order for a duel task."""
    key = "\0".join((challenger_submission_id, king_submission_id, task_id))
    first_byte = hashlib.blake2b(key.encode("utf-8"), digest_size=1).digest()[0]
    if first_byte % 2 == 0:
        return ("king", "challenger")
    return ("challenger", "king")


def _duel_job_for_side(row: Any, *, side: str) -> DuelSolveJob:
    if side == "king":
        submission_id = row.king_submission_id
    elif side == "challenger":
        submission_id = row.challenger_submission_id
    else:  # pragma: no cover - defensive only; callers use the two literals above.
        raise ValueError(f"unknown duel side: {side}")
    return DuelSolveJob(
        task_id=row.task_id,
        submission_id=submission_id,
        challenger_submission_id=row.challenger_submission_id,
        problem_statement=row.problem_statement,
        repo_clone_url=row.repo_clone_url,
        base_commit=row.parent_sha,
    )
