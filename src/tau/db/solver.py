"""Database operations for the task-solver worker.

A focused, self-contained slice of the DB seam (the same pattern as
``GeneratorDb``), kept **sync** because the orchestrator is sequential blocking
docker/git work — async would buy nothing. Two reads drive the worker's two phases:

  * ``next_qualification_jobs`` — CANDIDATE tasks of the reigning king, to be run
    against the king's own agent (viability check);
  * ``next_challenger_jobs`` — QUALIFIED tasks in active challenges whose challenger
    has no solution yet, to be run against the challenger's agent (judge fodder).

Each job carries the *submission_id* whose agent should run; the worker resolves the
actual agent files from the local submissions directory (folder == submission id).
Writes (``finish_qualification`` / ``save_task_solution``) are safe across
replicas: qualification finish is guarded on ``CANDIDATE`` (first transition wins),
and solution rows use ``ON CONFLICT DO NOTHING``. It deliberately avoids the
``tau.bittensor`` imports that gate the
full ``database.py``, so it can ship independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import and_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import aliased

from . import models
from .engine import create_db_engine, session_factory, session_scope
from .status import TaskStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SolveJob:
    """One sandboxed solve: run *submission_id*'s agent on *task_id*.

    Used uniformly by both phases — for qualification ``submission_id`` is the
    king's submission; for challenger-solve it is the challenger's. The agent files
    live on disk under ``<submissions_dir>/<submission_id>/``.
    """

    task_id: str
    submission_id: str
    problem_statement: str
    repo_clone_url: str
    base_commit: str  # tasks.parent_sha — the state to check out (before the fix)


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
    ) -> bool:
        """Flip a CANDIDATE task to QUALIFIED/DISQUALIFIED; store the king solution on pass.

        One transaction. Returns False if the task is no longer ``CANDIDATE`` (another
        replica or an earlier finish won the race). On QUALIFIED the king's
        ``task_solutions`` row is inserted (``ON CONFLICT DO NOTHING``) so a duel/judge
        has the king side; on DISQUALIFIED only the status changes.
        """
        status = TaskStatus.QUALIFIED if qualified else TaskStatus.DISQUALIFIED
        with session_scope(self._sessions) as session:
            updated = session.execute(
                update(models.Task)
                .where(
                    models.Task.task_id == task_id,
                    models.Task.status_id == int(TaskStatus.CANDIDATE),
                )
                .values(status_id=int(status))
                .returning(models.Task.task_id)
            )
            if updated.first() is None:
                return False
            if qualified:
                session.execute(
                    insert(models.TaskSolution)
                    .values(
                        task_id=task_id,
                        submission_id=king_submission_id,
                        solution=solution,
                        duration=duration,
                        exit_reason=exit_reason,
                    )
                    .on_conflict_do_nothing(index_elements=["task_id", "submission_id"])
                )
            return True

    # -- phase B: challenger solve -------------------------------------------
    def next_challenger_jobs(self, limit: int) -> list[SolveJob]:
        """QUALIFIED tasks in active challenges whose challenger lacks a solution.

        Anti-join mirror of ``v_active_unsolved_tasks`` but at ``(task, challenger)``
        grain (the view is task-grain and can't express "this challenger has no
        solution").
        """
        if limit <= 0:
            return []
        chal_sol = aliased(models.TaskSolution)
        stmt = (
            select(
                models.Task.task_id,
                models.Task.problem_statement,
                models.Task.repo_clone_url,
                models.Task.parent_sha,
                models.Challenge.challenger_submission_id,
            )
            .select_from(models.Challenge)
            .join(models.King, models.King.king_id == models.Challenge.king_id)
            .join(
                models.Task,
                and_(
                    models.Task.king_id == models.King.king_id,
                    models.Task.pool_type == models.Challenge.status,
                    models.Task.status_id == int(TaskStatus.QUALIFIED),
                ),
            )
            .outerjoin(
                chal_sol,
                and_(
                    chal_sol.task_id == models.Task.task_id,
                    chal_sol.submission_id == models.Challenge.challenger_submission_id,
                ),
            )
            .where(
                models.Challenge.status.in_((1, 2)),
                chal_sol.task_id.is_(None),
            )
            .order_by(models.Task.created_at, models.Challenge.challenger_submission_id)
            .limit(limit)
        )
        with session_scope(self._sessions) as session:
            rows = session.execute(stmt).all()
        return [
            SolveJob(
                task_id=row.task_id,
                submission_id=row.challenger_submission_id,
                problem_statement=row.problem_statement,
                repo_clone_url=row.repo_clone_url,
                base_commit=row.parent_sha,
            )
            for row in rows
        ]

    def save_task_solution(
        self,
        *,
        task_id: str,
        submission_id: str,
        solution: str,
        duration: float,
        exit_reason: str,
    ) -> None:
        """Insert a solution row (idempotent on the ``(task_id, submission_id)`` PK)."""
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
