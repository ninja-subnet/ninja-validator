"""Database operations for the duel-resolver worker."""

from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import Subquery, and_, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from tau.duel import (
    ActiveChallenge,
    ChallengeSnapshot,
    DuelScoringMethod,
    Tally,
    TokenEfficiencyConfig,
    TokenEfficiencyRound,
    calculate_token_efficiency,
)
from tau.pools import PoolTargets

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import ChallengeStatus, DuelOutcome, PoolType, SubmissionStatus, TaskStatus

_ACTIVE_STATUSES = (int(ChallengeStatus.POOL_ONE), int(ChallengeStatus.POOL_TWO))


class DuelResolverDb:
    """Duel-resolver's view of the database (one per worker process)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_async_db_engine(url, echo=echo)
        self._sessions = async_session_factory(self._engine)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def snapshot(
        self,
        targets: PoolTargets,
        *,
        token_efficiency: TokenEfficiencyConfig | None = None,
    ) -> ChallengeSnapshot:
        """Read the arena state `decide` needs, in one transaction.

        With no reigning king the snapshot is empty (the worker waits). The next
        challenger is resolved only when no challenge is active, since `decide`
        ignores it while a duel is running.
        """
        async with async_session_scope(self._sessions) as session:
            king_id = await self._reigning_king(session)
            if king_id is None:
                return ChallengeSnapshot(None, None, None, task_pools_ready=False)
            active = await self._active_challenge(
                session,
                king_id,
                targets,
                token_efficiency or TokenEfficiencyConfig(),
            )
            if active is not None:
                return ChallengeSnapshot(king_id, active, None)
            task_pools_ready = await self._task_pools_ready(session, king_id, targets)
            next_challenger = (
                await self._next_challenger(session) if task_pools_ready else None
            )
            return ChallengeSnapshot(
                king_id,
                None,
                next_challenger,
                task_pools_ready=task_pools_ready,
            )

    # -- writes (guarded conditional transitions) ----------------------------------

    async def open_challenge(self, king_id: str, challenger_submission_id: str) -> bool:
        """Open a POOL_ONE challenge of *king_id* by *challenger_submission_id*.

        Idempotent: a challenger that already has a challenge row conflicts and the
        insert is a no-op (returns False).
        """
        stmt = (
            insert(models.Challenge)
            .values(
                challenger_submission_id=challenger_submission_id,
                king_id=king_id,
                status=int(ChallengeStatus.POOL_ONE),
            )
            .on_conflict_do_nothing(index_elements=["challenger_submission_id"])
            .returning(models.Challenge.challenger_submission_id)
        )
        async with async_session_scope(self._sessions) as session:
            return (await session.execute(stmt)).first() is not None

    async def advance_pool(
        self,
        challenge: ActiveChallenge,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
        token_efficiency: TokenEfficiencyConfig | None = None,
    ) -> bool:
        """POOL_ONE -> POOL_TWO (challenger won pool 1), recording the verdict, only if
        still in the pool the snapshot observed."""
        async with async_session_scope(self._sessions) as session:
            if not await self._set_status(session, challenge, ChallengeStatus.POOL_TWO):
                return False
            await self._record_resolution(
                session,
                challenge,
                DuelOutcome.CHALLENGER_WON,
                scoring_method,
                round_win_margin,
                mean_score_margin,
                token_efficiency or TokenEfficiencyConfig(),
            )
            return True

    async def close_challenge(
        self,
        challenge: ActiveChallenge,
        outcome: DuelOutcome,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
        token_efficiency: TokenEfficiencyConfig | None = None,
    ) -> bool:
        """Close the challenge (king holds or challenger left), recording *outcome*,
        only if still in the observed pool."""
        async with async_session_scope(self._sessions) as session:
            if not await self._set_status(session, challenge, ChallengeStatus.CLOSED):
                return False
            await self._record_resolution(
                session,
                challenge,
                outcome,
                scoring_method,
                round_win_margin,
                mean_score_margin,
                token_efficiency or TokenEfficiencyConfig(),
            )
            return True

    async def promote(
        self,
        challenge: ActiveChallenge,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
        token_efficiency: TokenEfficiencyConfig | None = None,
    ) -> bool:
        """Crown the challenger and close the challenge (challenger won pool 2), in one
        transaction.

        The verdict and the king row are written only after the guarded close succeeds,
        so a stale promote (the challenge left POOL_TWO since the snapshot) crowns
        nobody and records nothing.
        """
        async with async_session_scope(self._sessions) as session:
            if not await self._set_status(session, challenge, ChallengeStatus.CLOSED):
                return False
            await self._record_resolution(
                session,
                challenge,
                DuelOutcome.CHALLENGER_WON,
                scoring_method,
                round_win_margin,
                mean_score_margin,
                token_efficiency or TokenEfficiencyConfig(),
            )
            await session.execute(
                insert(models.King)
                .values(king_id=challenge.challenger_submission_id)
                .on_conflict_do_nothing(index_elements=["king_id"])
            )
            return True

    async def _set_status(
        self,
        session: AsyncSession,
        challenge: ActiveChallenge,
        new_status: ChallengeStatus,
    ) -> bool:
        """Set the challenge's status to *new_status*, guarded on it still being in the
        pool the snapshot observed (TOCTOU-safe). Returns whether a row changed."""
        stmt = (
            update(models.Challenge)
            .where(
                models.Challenge.challenger_submission_id
                == challenge.challenger_submission_id,
                models.Challenge.status == int(challenge.pool),
            )
            .values(status=int(new_status))
            .returning(models.Challenge.challenger_submission_id)
        )
        return (await session.execute(stmt)).first() is not None

    async def _record_resolution(
        self,
        session: AsyncSession,
        challenge: ActiveChallenge,
        outcome: DuelOutcome,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
        token_efficiency: TokenEfficiencyConfig,
    ) -> None:
        """Append the pool's verdict to ``duel_resolutions`` with the tally and
        thresholds it was decided on. Idempotent: one row per (challenge, pool)."""
        token_bonus_applied = (
            token_efficiency.enabled and scoring_method is DuelScoringMethod.MEAN
        )
        king_boost = challenge.tally.king_token_boost if token_bonus_applied else 0.0
        challenger_boost = (
            challenge.tally.challenger_token_boost if token_bonus_applied else 0.0
        )
        await session.execute(
            insert(models.DuelResolution)
            .values(
                challenger_submission_id=challenge.challenger_submission_id,
                pool_type=int(challenge.pool),
                outcome=int(outcome),
                challenger_wins=challenge.tally.wins,
                challenger_losses=challenge.tally.losses,
                ties=challenge.tally.ties,
                best_of=challenge.pool_target,
                scoring_method=scoring_method.value,
                round_win_margin=round_win_margin,
                mean_score_margin=mean_score_margin,
                king_score_mean=challenge.tally.king_score_mean,
                challenger_score_mean=challenge.tally.challenger_score_mean,
                score_mean_delta=challenge.tally.score_mean_delta,
                score_mean_rounds=challenge.tally.score_mean_rounds,
                token_bonus_enabled=token_bonus_applied,
                token_score_tolerance=token_efficiency.score_tolerance,
                token_min_score=token_efficiency.min_score,
                token_bonus_multiplier=token_efficiency.bonus_multiplier,
                king_total_tokens=challenge.tally.king_total_tokens,
                challenger_total_tokens=challenge.tally.challenger_total_tokens,
                token_comparison_rounds=challenge.tally.token_comparison_rounds,
                king_token_savings_mean=(
                    challenge.tally.king_token_savings_mean
                    if token_bonus_applied
                    else 0.0
                ),
                challenger_token_savings_mean=(
                    challenge.tally.challenger_token_savings_mean
                    if token_bonus_applied
                    else 0.0
                ),
                king_token_boost=king_boost,
                challenger_token_boost=challenger_boost,
                king_combined_score=challenge.tally.king_score_mean + king_boost,
                challenger_combined_score=(
                    challenge.tally.challenger_score_mean + challenger_boost
                ),
                combined_score_delta=(
                    challenge.tally.score_mean_delta + challenger_boost - king_boost
                ),
            )
            .on_conflict_do_nothing(
                index_elements=["challenger_submission_id", "pool_type"]
            )
        )

    async def _reigning_king(self, session: AsyncSession) -> str | None:
        """The king with the latest king_from, or None if none reigns."""
        stmt = (
            select(models.King.king_id).order_by(models.King.king_from.desc()).limit(1)
        )
        return (await session.scalars(stmt)).first()

    async def _task_pools_ready(
        self, session: AsyncSession, king_id: str, targets: PoolTargets
    ) -> bool:
        """Whether both task pools have enough qualified rounds to start a duel."""
        rows = await session.execute(
            select(models.Task.pool_type, func.count(models.Task.task_id))
            .where(
                models.Task.king_id == king_id,
                models.Task.status_id == int(TaskStatus.QUALIFIED),
                models.Task.pool_type.in_(
                    (int(PoolType.POOL_ONE), int(PoolType.POOL_TWO))
                ),
            )
            .group_by(models.Task.pool_type)
        )
        counts = {int(pool): int(count) for pool, count in rows.all()}
        return (
            counts.get(int(PoolType.POOL_ONE), 0) >= targets.pool_one
            and counts.get(int(PoolType.POOL_TWO), 0) >= targets.pool_two
        )

    async def _active_challenge(
        self,
        session: AsyncSession,
        king_id: str,
        targets: PoolTargets,
        token_efficiency: TokenEfficiencyConfig,
    ) -> ActiveChallenge | None:
        """The challenge dueling *king_id* (status POOL_ONE/POOL_TWO), or None.

        At most one challenge is active per king (the resolver opens one at a time).
        """
        row = (
            await session.execute(
                select(
                    models.Challenge.challenger_submission_id,
                    models.Challenge.status,
                )
                .where(
                    models.Challenge.king_id == king_id,
                    models.Challenge.status.in_(_ACTIVE_STATUSES),
                )
                .limit(1)
            )
        ).first()
        if row is None:
            return None
        pool = PoolType(row.status)
        challenger_id = row.challenger_submission_id
        return ActiveChallenge(
            challenger_submission_id=challenger_id,
            king_submission_id=king_id,
            pool=pool,
            pool_target=targets.target(pool),
            tally=await self._tally(
                session,
                king_id,
                challenger_id,
                pool,
                targets.target(pool),
                token_efficiency,
            ),
            challenger_registered=await self._challenger_registered(
                session, challenger_id
            ),
        )

    async def _tally(
        self,
        session: AsyncSession,
        king_id: str,
        challenger_id: str,
        pool: PoolType,
        pool_target: int,
        token_efficiency: TokenEfficiencyConfig,
    ) -> Tally:
        """Challenger-perspective outcomes and score means over the active pool."""
        conditions = (
            models.Task.king_id == king_id,
            models.Task.pool_type == int(pool),
            models.Judgement.king_submission_id == king_id,
            models.Judgement.challenger_submission_id == challenger_id,
        )
        stmt = (
            select(models.Judgement.llm_winner, func.count())
            .join(models.Task, models.Task.task_id == models.Judgement.task_id)
            .where(*conditions)
            .group_by(models.Judgement.llm_winner)
        )
        counts = {
            winner: count for winner, count in (await session.execute(stmt)).all()
        }
        king_solution = aliased(models.DuelTaskSolution)
        challenger_solution = aliased(models.DuelTaskSolution)
        round_stmt = (
            select(
                models.Judgement.king_score,
                models.Judgement.challenger_score,
                models.Judgement.error,
                king_solution.usage_summary["total_tokens"].label("king_tokens"),
                challenger_solution.usage_summary["total_tokens"].label(
                    "challenger_tokens"
                ),
            )
            .join(models.Task, models.Task.task_id == models.Judgement.task_id)
            .outerjoin(
                king_solution,
                and_(
                    king_solution.task_id == models.Judgement.task_id,
                    king_solution.challenger_submission_id == challenger_id,
                    king_solution.submission_id == king_id,
                ),
            )
            .outerjoin(
                challenger_solution,
                and_(
                    challenger_solution.task_id == models.Judgement.task_id,
                    challenger_solution.challenger_submission_id == challenger_id,
                    challenger_solution.submission_id == challenger_id,
                ),
            )
            .where(*conditions)
        )
        round_rows = (await session.execute(round_stmt)).all()
        scored_rows = [
            (float(row.king_score), float(row.challenger_score))
            for row in round_rows
            if row.king_score is not None and row.challenger_score is not None
        ]
        scored = len(scored_rows)
        king_score_mean = (
            sum(king_score for king_score, _ in scored_rows) / scored if scored else 0.0
        )
        challenger_score_mean = (
            sum(challenger_score for _, challenger_score in scored_rows) / scored
            if scored
            else 0.0
        )
        token_stats = calculate_token_efficiency(
            (
                TokenEfficiencyRound(
                    king_score=(
                        float(row.king_score) if row.king_score is not None else None
                    ),
                    challenger_score=(
                        float(row.challenger_score)
                        if row.challenger_score is not None
                        else None
                    ),
                    king_tokens=_usage_total_tokens(row.king_tokens),
                    challenger_tokens=_usage_total_tokens(row.challenger_tokens),
                    judgement_valid=row.error is None,
                )
                for row in round_rows
            ),
            pool_target=pool_target,
            config=token_efficiency,
        )
        king_total_tokens, challenger_total_tokens = await self._pool_token_totals(
            session,
            king_id=king_id,
            challenger_id=challenger_id,
            pool=pool,
            minimum_rows=sum(counts.values()),
        )
        # Anything that is not a decisive king/challenger win remains a tally tie.
        # Judge-failure fallback scores still preserve the existing quality mean,
        # but cannot create a token bonus.
        return Tally(
            wins=counts.get("challenger", 0),
            losses=counts.get("king", 0),
            ties=sum(
                n
                for winner, n in counts.items()
                if winner not in ("king", "challenger")
            ),
            king_score_mean=king_score_mean,
            challenger_score_mean=challenger_score_mean,
            score_mean_delta=challenger_score_mean - king_score_mean,
            score_mean_rounds=scored,
            king_total_tokens=king_total_tokens,
            challenger_total_tokens=challenger_total_tokens,
            token_comparison_rounds=token_stats.token_comparison_rounds,
            king_token_savings_mean=token_stats.king_savings_mean,
            challenger_token_savings_mean=token_stats.challenger_savings_mean,
            king_token_boost=token_stats.king_boost,
            challenger_token_boost=token_stats.challenger_boost,
        )

    async def _pool_token_totals(
        self,
        session: AsyncSession,
        *,
        king_id: str,
        challenger_id: str,
        pool: PoolType,
        minimum_rows: int,
    ) -> tuple[int | None, int | None]:
        """Total all solves run in the pool, including solves not yet judged.

        A missing usage value makes that side unavailable instead of exposing a
        partial sum. ``minimum_rows`` also protects older/incomplete data where a
        judgement exists without its challenge-scoped solution row.
        """
        rows = (
            await session.execute(
                select(
                    models.DuelTaskSolution.submission_id,
                    models.DuelTaskSolution.usage_summary["total_tokens"].label(
                        "total_tokens"
                    ),
                )
                .join(
                    models.Task,
                    models.Task.task_id == models.DuelTaskSolution.task_id,
                )
                .where(
                    models.Task.king_id == king_id,
                    models.Task.pool_type == int(pool),
                    models.DuelTaskSolution.challenger_submission_id == challenger_id,
                    models.DuelTaskSolution.submission_id.in_((king_id, challenger_id)),
                )
            )
        ).all()
        totals = {king_id: 0, challenger_id: 0}
        counts = {king_id: 0, challenger_id: 0}
        complete = {king_id: True, challenger_id: True}
        for submission_id, value in rows:
            counts[submission_id] += 1
            tokens = _usage_total_tokens(value)
            if tokens is None:
                complete[submission_id] = False
            else:
                totals[submission_id] += tokens

        def total(submission_id: str) -> int | None:
            if counts[submission_id] < minimum_rows or not complete[submission_id]:
                return None
            return totals[submission_id]

        return total(king_id), total(challenger_id)

    async def _challenger_registered(
        self, session: AsyncSession, challenger_id: str
    ) -> bool:
        """True if the challenger is on a current registration of its hotkey.

        Its hotkey must be in the current metagraph and the submission no older than
        that registration -- a re-registered hotkey makes a prior submission stale.
        """
        meta = _current_metagraph()
        stmt = (
            select(models.Submission.submission_id)
            .join(meta, meta.c.ss58_hot == models.Submission.hotkey)
            .where(
                models.Submission.submission_id == challenger_id,
                meta.c.block <= models.Submission.block,
            )
            .limit(1)
        )
        return (await session.scalars(stmt)).first() is not None

    async def _next_challenger(self, session: AsyncSession) -> str | None:
        """Oldest eligible submission, on a current registration of its hotkey, that
        is neither a king nor a (past or present) challenger -- one go, ever."""
        meta = _current_metagraph()
        stmt = (
            select(models.Submission.submission_id)
            .join(meta, meta.c.ss58_hot == models.Submission.hotkey)
            .where(
                models.Submission.status_id == int(SubmissionStatus.ELIGIBLE),
                # No older than the current registration (stale after re-registration).
                meta.c.block <= models.Submission.block,
                models.Submission.submission_id.not_in(select(models.King.king_id)),
                models.Submission.submission_id.not_in(
                    select(models.Challenge.challenger_submission_id)
                ),
            )
            .order_by(models.Submission.block)
            .limit(1)
        )
        return (await session.scalars(stmt)).first()


def _current_metagraph() -> Subquery:
    """The current metagraph as (ss58_hot, block): the latest registration per uid
    (highest block). Mirrors the ``v_current_metagraph`` view."""
    return (
        select(models.Registration.ss58_hot, models.Registration.block)
        .distinct(models.Registration.uid)
        .order_by(models.Registration.uid, models.Registration.block.desc())
        .subquery()
    )


def _usage_total_tokens(value: object) -> int | None:
    """Accept a sanitized non-negative JSON scalar and reject legacy bad values."""
    if isinstance(value, Mapping):
        value = value.get("total_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
