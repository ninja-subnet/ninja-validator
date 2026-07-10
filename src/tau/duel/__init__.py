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
    challenger_wins_by_token_efficiency,
    challenger_wins_by_mean_score,
    challenger_wins,
)
from .scoring import (
    DEFAULT_MEAN_SCORE_MARGIN,
    DEFAULT_TOKEN_EFFICIENCY_CLIP,
    DEFAULT_TOKEN_QUALITY_FLOOR,
    DEFAULT_TOKEN_WEIGHT,
    DuelScoringMethod,
    token_adjusted_score_delta,
)
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
    "DEFAULT_MEAN_SCORE_MARGIN",
    "DEFAULT_TOKEN_EFFICIENCY_CLIP",
    "DEFAULT_TOKEN_QUALITY_FLOOR",
    "DEFAULT_TOKEN_WEIGHT",
    "challenger_cannot_catch",
    "challenger_is_unbeatable",
    "challenger_wins_by_mean_score",
    "challenger_wins_by_token_efficiency",
    "challenger_wins",
    "decide",
    "token_adjusted_score_delta",
]
