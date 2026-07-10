"""Database operations for the task-generator worker.

A focused, self-contained slice of the DB seam: read the reigning king and latest
block, check pool fill + dedup, and insert ``CANDIDATE`` tasks. It deliberately
does NOT depend on the chain-watcher snapshot types (``tau.bittensor``) that block
the full ``database.py``, so it can ship ahead of them. All writes are idempotent
via Postgres ``ON CONFLICT``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import exists, func, select
from sqlalchemy.dialects.postgresql import insert

from tau.pools import PoolTargets

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import PoolType, TaskStatus


@dataclass(frozen=True, slots=True)
class PoolDeficit:
    """One unit of generation work: *deficit* tasks to add to *pool* for *king_id*."""

    king_id: str
    pool: PoolType
    deficit: int


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Write-once generation telemetry stored on the task row (observability only)."""

    model: str
    fetch_seconds: float
    llm_seconds: float
    llm_attempt: int
    # Commits discarded (by reason) before the winning one was found.
    rejected_duplicate: int
    rejected_structural: int
    rejected_quality: int
    rejected_fetch_error: int


class GeneratorDb:
    """Task-generator's view of the database (one per worker process)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        # create_async_engine is lazy (no connection), so construction stays sync.
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def pending_pool_deficits(
        self,
        targets: PoolTargets,
        *,
        qualification_inflight_target: int,
    ) -> list[PoolDeficit]:
        """Generation work that keeps qualification load steady until pools fill.

        Finds the reigning king (latest ``king_from``) and its per-pool
        non-DISQUALIFIED task counts in a single session, so the king and its
        counts are a consistent snapshot. A pool is complete only when its
        ``QUALIFIED`` count reaches its target. Until then, the configured number
        of real ``CANDIDATE``/``PENDING_SCREEN`` tasks is divided evenly among all
        incomplete pools, keeping the solver loaded even near the final slots.

        Returns one :class:`PoolDeficit` per pool still under target, in ``PoolType``
        order (POOL_ONE first, so earlier pools fill first). An **empty list**
        means there is nothing to generate — either no king reigns or every pool
        is already full — and the worker should idle.
        """
        if qualification_inflight_target < 1:
            raise ValueError("qualification_inflight_target must be >= 1")
        king_stmt = (
            select(models.King.king_id)
            .order_by(models.King.king_from.desc())
            .limit(1)
        )
        async with async_session_scope(self._sessions) as session:
            king_id = (await session.scalars(king_stmt)).first()
            if king_id is None:
                return []  # no reigning king -> generate nothing
            qualified: dict[PoolType, int] = dict.fromkeys(PoolType, 0)
            active: dict[PoolType, int] = dict.fromkeys(PoolType, 0)
            count_stmt = (
                select(models.Task.pool_type, models.Task.status_id, func.count())
                .where(
                    models.Task.king_id == king_id,
                    models.Task.status_id != int(TaskStatus.DISQUALIFIED),
                )
                .group_by(models.Task.pool_type, models.Task.status_id)
            )
            for pool_type, status_id, count in await session.execute(count_stmt):
                try:
                    pool = PoolType(pool_type)
                except ValueError:
                    continue  # unknown pool_type — ignore
                if status_id == int(TaskStatus.QUALIFIED):
                    qualified[pool] += count
                elif status_id in (
                    int(TaskStatus.CANDIDATE),
                    int(TaskStatus.PENDING_SCREEN),
                ):
                    active[pool] += count

        incomplete = [
            pool for pool in PoolType if qualified[pool] < targets.target(pool)
        ]
        if not incomplete:
            return []
        share, remainder = divmod(qualification_inflight_target, len(incomplete))
        return [
            PoolDeficit(king_id=king_id, pool=pool, deficit=deficit)
            for index, pool in enumerate(incomplete)
            if (
                deficit := share + int(index < remainder) - active[pool]
            ) > 0
        ]

    async def fingerprint_exists(self, content_fingerprint: str) -> bool:
        stmt = select(
            exists().where(models.Task.content_fingerprint == content_fingerprint)
        )
        async with async_session_scope(self._sessions) as session:
            return bool(await session.scalar(stmt))

    async def insert_task_candidate(
        self,
        *,
        task_id: str,
        king_id: str,
        pool_type: int,
        problem_statement: str,
        reference_patch: str,
        repo_clone_url: str,
        parent_sha: str,
        commit_sha: str,
        content_fingerprint: str,
        metrics: GenerationMetrics,
    ) -> bool:
        """Insert a CANDIDATE task (with its generation telemetry); True iff a row was added.

        ``ON CONFLICT (content_fingerprint) DO NOTHING`` makes this idempotent and
        race-safe across workers: a duplicate mined commit is silently skipped
        (returns False). Insertion is detected via ``RETURNING`` — on conflict no
        row comes back — rather than ``rowcount`` (which the async psycopg driver
        reports as -1 for this statement). ``created_at`` is filled by the DB default.
        """
        stmt = (
            insert(models.Task)
            .values(
                task_id=task_id,
                king_id=king_id,
                pool_type=int(pool_type),
                problem_statement=problem_statement,
                reference_patch=reference_patch,
                repo_clone_url=repo_clone_url,
                parent_sha=parent_sha,
                commit_sha=commit_sha,
                content_fingerprint=content_fingerprint,
                status_id=int(TaskStatus.CANDIDATE),
                model=metrics.model,
                fetch_seconds=metrics.fetch_seconds,
                llm_seconds=metrics.llm_seconds,
                llm_attempt=metrics.llm_attempt,
                rejected_duplicate=metrics.rejected_duplicate,
                rejected_structural=metrics.rejected_structural,
                rejected_quality=metrics.rejected_quality,
                rejected_fetch_error=metrics.rejected_fetch_error,
            )
            .on_conflict_do_nothing(index_elements=["content_fingerprint"])
            .returning(models.Task.task_id)
        )
        async with async_session_scope(self._sessions) as session:
            result = await session.execute(stmt)
            return result.first() is not None

    async def record_generation_failure(
        self,
        *,
        king_id: str,
        pool_type: int,
        repo_full_name: str,
        commit_sha: str,
        model: str,
        attempts: int,
        reason: str | None,
    ) -> None:
        """Append one row to ``task_generation_failures``.

        Records a commit abandoned because the LLM failed to describe it on every
        attempt -- it produces no ``tasks`` row, so the give-up would otherwise be
        invisible. Append-only event log: written here, never read by the pipeline.
        """
        stmt = insert(models.TaskGenerationFailure).values(
            king_id=king_id,
            pool_type=int(pool_type),
            repo_full_name=repo_full_name,
            commit_sha=commit_sha,
            model=model,
            attempts=attempts,
            reason=reason,
        )
        async with async_session_scope(self._sessions) as session:
            await session.execute(stmt)
