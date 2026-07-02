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
from .scoring import DuelScoringMethod
from .snapshot import ActiveChallenge, ChallengeSnapshot, Tally

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
    "WaitReason",
    "DuelScoringMethod",
    "challenger_cannot_catch",
    "challenger_is_unbeatable",
    "challenger_wins_by_mean_score",
    "challenger_wins",
    "decide",
]
