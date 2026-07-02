"""Database operations for the duel-resolver worker."""

from __future__ import annotations

from sqlalchemy import Subquery, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from tau.duel import (
    ActiveChallenge,
    ChallengeSnapshot,
    DuelScoringMethod,
    Tally,
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

    async def snapshot(self, targets: PoolTargets) -> ChallengeSnapshot:
        """Read the arena state `decide` needs, in one transaction.

        With no reigning king the snapshot is empty (the worker waits). The next
        challenger is resolved only when no challenge is active, since `decide`
        ignores it while a duel is running.
        """
        async with async_session_scope(self._sessions) as session:
            king_id = await self._reigning_king(session)
            if king_id is None:
                return ChallengeSnapshot(None, None, None)
            active = await self._active_challenge(session, king_id, targets)
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
            )
            return True

    async def promote(
        self,
        challenge: ActiveChallenge,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
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
    ) -> None:
        """Append the pool's verdict to ``duel_resolutions`` with the tally and
        thresholds it was decided on. Idempotent: one row per (challenge, pool)."""
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
        self, session: AsyncSession, king_id: str, targets: PoolTargets
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
            tally=await self._tally(session, king_id, challenger_id, pool),
            challenger_registered=await self._challenger_registered(
                session, challenger_id
            ),
        )

    async def _tally(
        self, session: AsyncSession, king_id: str, challenger_id: str, pool: PoolType
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
        score_stmt = (
            select(
                func.avg(models.Judgement.king_score),
                func.avg(models.Judgement.challenger_score),
                func.count(),
            )
            .join(models.Task, models.Task.task_id == models.Judgement.task_id)
            .where(
                *conditions,
                models.Judgement.king_score.is_not(None),
                models.Judgement.challenger_score.is_not(None),
            )
        )
        king_mean, challenger_mean, score_count = (await session.execute(score_stmt)).one()
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
