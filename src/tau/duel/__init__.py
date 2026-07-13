"""Pure duel-resolver logic: the snapshot, the action variants, and the decision."""

from __future__ import annotations

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
from .decide import decide
from .predicates import (
    challenger_cannot_catch,
    challenger_is_unbeatable,
    challenger_wins_by_mean_score,
    challenger_wins,
)
from .scoring import DEFAULT_MEAN_SCORE_MARGIN, DuelScoringMethod
from .snapshot import ActiveChallenge, ChallengeSnapshot, Tally
from .token_efficiency import (
    TokenEfficiencyConfig,
    TokenEfficiencyRound,
    TokenEfficiencyStats,
    calculate_token_efficiency,
)

__all__ = [
    "Action",
    "ActiveChallenge",
    "AdvancePool",
    "ChallengeSnapshot",
    "CloseChallenge",
    "CloseReason",
    "Nothing",
    "OpenChallenge",
    "Promote",
    "Tally",
    "TokenEfficiencyConfig",
    "TokenEfficiencyRound",
    "TokenEfficiencyStats",
    "WaitReason",
    "DuelScoringMethod",
    "DEFAULT_MEAN_SCORE_MARGIN",
    "challenger_cannot_catch",
    "challenger_is_unbeatable",
    "challenger_wins_by_mean_score",
    "challenger_wins",
    "calculate_token_efficiency",
    "decide",
]
