"""Database operations for the duel-resolver worker."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Float,
    Subquery,
    and_,
    case,
    cast,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from tau.duel import (
    ActiveChallenge,
    ChallengeSnapshot,
    DEFAULT_TOKEN_EFFICIENCY_CLIP,
    DEFAULT_TOKEN_QUALITY_FLOOR,
    DuelScoringMethod,
    Tally,
    token_adjusted_score_delta,
)
from tau.pools import PoolTargets

from . import models
from .engine import async_session_factory, async_session_scope, create_async_db_engine
from .status import ChallengeStatus, DuelOutcome, PoolType, SubmissionStatus

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
        token_quality_floor: float = DEFAULT_TOKEN_QUALITY_FLOOR,
        token_efficiency_clip: float = DEFAULT_TOKEN_EFFICIENCY_CLIP,
    ) -> ChallengeSnapshot:
        """Read the arena state `decide` needs, in one transaction.

        With no reigning king the snapshot is empty (the worker waits). The next
        challenger is resolved only when no challenge is active, since `decide`
        ignores it while a duel is running.
        """
        async with async_session_scope(self._sessions) as session:
            king_id = await self._reigning_king(session)
            if king_id is None:
                return ChallengeSnapshot(None, None, None)
            active = await self._active_challenge(
                session,
                king_id,
                targets,
                token_quality_floor=token_quality_floor,
                token_efficiency_clip=token_efficiency_clip,
            )
            next_challenger = (
                None if active is not None else await self._next_challenger(session)
            )
            return ChallengeSnapshot(king_id, active, next_challenger)

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
        token_weight: float,
        token_quality_floor: float,
        token_efficiency_clip: float,
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
                token_weight,
                token_quality_floor,
                token_efficiency_clip,
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
        token_weight: float,
        token_quality_floor: float,
        token_efficiency_clip: float,
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
                token_weight,
                token_quality_floor,
                token_efficiency_clip,
            )
            return True

    async def promote(
        self,
        challenge: ActiveChallenge,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
        token_weight: float,
        token_quality_floor: float,
        token_efficiency_clip: float,
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
                token_weight,
                token_quality_floor,
                token_efficiency_clip,
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
        token_weight: float,
        token_quality_floor: float,
        token_efficiency_clip: float,
    ) -> None:
        """Append the pool's verdict to ``duel_resolutions`` with the tally and
        thresholds it was decided on. Idempotent: one row per (challenge, pool)."""
        adjusted_score_delta = challenge.tally.score_mean_delta
        if scoring_method is DuelScoringMethod.TOKEN_EFFICIENCY:
            adjusted_score_delta = token_adjusted_score_delta(
                score_mean_delta=challenge.tally.score_mean_delta,
                token_efficiency_mean=challenge.tally.token_efficiency_mean,
                quality_band=mean_score_margin,
                token_weight=token_weight,
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
                token_weight=token_weight,
                token_quality_floor=token_quality_floor,
                token_efficiency_clip=token_efficiency_clip,
                token_efficiency_mean=challenge.tally.token_efficiency_mean,
                token_usage_rounds=challenge.tally.token_usage_rounds,
                token_usage_penalty_rounds=challenge.tally.token_usage_penalty_rounds,
                adjusted_score_delta=adjusted_score_delta,
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

    async def _active_challenge(
        self,
        session: AsyncSession,
        king_id: str,
        targets: PoolTargets,
        *,
        token_quality_floor: float,
        token_efficiency_clip: float,
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
                token_quality_floor=token_quality_floor,
                token_efficiency_clip=token_efficiency_clip,
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
        *,
        token_quality_floor: float,
        token_efficiency_clip: float,
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
        king_tokens = cast(
            king_solution.usage_summary["total_tokens"].astext, BigInteger
        )
        challenger_tokens = cast(
            challenger_solution.usage_summary["total_tokens"].astext, BigInteger
        )
        king_usage_ok = case(
            (
                and_(
                    king_solution.exit_reason == "completed",
                    king_tokens > 0,
                    cast(
                        king_solution.usage_summary["request_count"].astext,
                        BigInteger,
                    )
                    == func.jsonb_array_length(king_solution.usage_summary["requests"]),
                ),
                1,
            ),
            else_=0,
        )
        challenger_usage_ok = case(
            (
                and_(
                    challenger_solution.exit_reason == "completed",
                    challenger_tokens > 0,
                    cast(
                        challenger_solution.usage_summary["request_count"].astext,
                        BigInteger,
                    )
                    == func.jsonb_array_length(
                        challenger_solution.usage_summary["requests"]
                    ),
                ),
                1,
            ),
            else_=0,
        )
        quality_ok = (
            func.least(
                models.Judgement.king_score,
                models.Judgement.challenger_score,
            )
            >= token_quality_floor
        )
        both_usage_ok = and_(king_usage_ok == 1, challenger_usage_ok == 1)
        one_usage_missing = (king_usage_ok + challenger_usage_ok) == 1
        relative_efficiency = (king_tokens - challenger_tokens) / cast(
            king_tokens + challenger_tokens, Float
        )
        clipped_efficiency = func.greatest(
            -token_efficiency_clip,
            func.least(token_efficiency_clip, relative_efficiency),
        )
        token_efficiency = case(
            (and_(quality_ok, both_usage_ok), clipped_efficiency),
            (
                and_(quality_ok, king_usage_ok == 1, challenger_usage_ok == 0),
                -token_efficiency_clip,
            ),
            (
                and_(quality_ok, king_usage_ok == 0, challenger_usage_ok == 1),
                token_efficiency_clip,
            ),
            else_=0.0,
        )
        score_stmt = (
            select(
                func.avg(models.Judgement.king_score),
                func.avg(models.Judgement.challenger_score),
                func.count(),
                func.avg(token_efficiency),
                func.sum(case((both_usage_ok, 1), else_=0)),
                func.sum(case((and_(quality_ok, one_usage_missing), 1), else_=0)),
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
            .where(
                *conditions,
                models.Judgement.king_score.is_not(None),
                models.Judgement.challenger_score.is_not(None),
            )
        )
        (
            king_mean,
            challenger_mean,
            score_count,
            token_mean,
            token_usage_count,
            token_penalty_count,
        ) = (await session.execute(score_stmt)).one()
        scored = int(score_count or 0)
        king_score_mean = float(king_mean) if scored else 0.0
        challenger_score_mean = float(challenger_mean) if scored else 0.0
        # Anything that is not a decisive king/challenger win is a tie (including a
        # judge-failure tie); the resolver never reads judgements.error.
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
            token_efficiency_mean=float(token_mean or 0.0),
            token_usage_rounds=int(token_usage_count or 0),
            token_usage_penalty_rounds=int(token_penalty_count or 0),
        )

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
