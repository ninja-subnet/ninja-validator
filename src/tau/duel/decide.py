"""The duel resolver's decision function."""

from __future__ import annotations

from typing import assert_never

from tau.db.status import PoolType

from .actions import (
    Action,
    AdvancePool,
    CloseChallenge,
    CloseReason,
    Nothing,
    OpenChallenge,
    Promote,
    WaitReason,
)
from .predicates import (
    challenger_cannot_catch,
    challenger_is_unbeatable,
    challenger_wins_by_mean_score,
)
from .scoring import DuelScoringMethod
from .snapshot import ActiveChallenge, ChallengeSnapshot


def decide(
    snapshot: ChallengeSnapshot,
    *,
    scoring_method: DuelScoringMethod = DuelScoringMethod.ROUND_WINS,
    round_win_margin: int = 0,
    mean_score_margin: float = 0.0,
) -> Action:
    """Map *snapshot* to the one action due this tick, cheapest states first."""
    if snapshot.reigning_king_submission_id is None:
        return Nothing(WaitReason.NO_KING)

    active = snapshot.active_challenge
    if active is None:
        # A king with no active challenge: start one if a challenger is waiting.
        if snapshot.next_challenger_submission_id is None:
            return Nothing(WaitReason.NO_CHALLENGER)
        return OpenChallenge(
            snapshot.reigning_king_submission_id,
            snapshot.next_challenger_submission_id,
        )

    # An active challenge: a deregistered challenger forfeits regardless of its tally.
    if not active.challenger_registered:
        return CloseChallenge(active, CloseReason.CHALLENGER_DEREGISTERED)

    match scoring_method:
        case DuelScoringMethod.ROUND_WINS:
            return decide_with_round_wins_scoring_method(active, round_win_margin)
        case DuelScoringMethod.MEAN:
            return decide_with_mean_scoring_method(active, mean_score_margin)
        case _:
            assert_never(scoring_method)


def decide_with_round_wins_scoring_method(
    active: ActiveChallenge, round_win_margin: int
) -> Action:
    tally = active.tally
    remaining = active.remaining

    if challenger_is_unbeatable(tally.wins, tally.losses, remaining, round_win_margin):
        # Pool cleared: pool #1 advances, pool #2 wins the crown.
        if active.pool is PoolType.POOL_ONE:
            return AdvancePool(active)
        return Promote(active)
    if challenger_cannot_catch(tally.wins, tally.losses, remaining, round_win_margin):
        return CloseChallenge(active, CloseReason.KING_DEFENDED)
    return Nothing(WaitReason.DUEL_IN_PROGRESS)


def decide_with_mean_scoring_method(
    active: ActiveChallenge, mean_score_margin: float
) -> Action:
    tally = active.tally
    remaining = active.remaining

    if remaining > 0:
        return Nothing(WaitReason.DUEL_IN_PROGRESS)

    if not challenger_wins_by_mean_score(
        score_mean_delta=tally.score_mean_delta,
        score_mean_rounds=tally.score_mean_rounds,
        margin=mean_score_margin,
    ):
        return CloseChallenge(active, CloseReason.KING_DEFENDED)

    if active.pool is PoolType.POOL_ONE:
        return AdvancePool(active)

    return Promote(active)
