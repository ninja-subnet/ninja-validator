"""Duel scoring modes shared by config, decision logic, and DB writes."""

from __future__ import annotations

from enum import StrEnum

DEFAULT_MEAN_SCORE_MARGIN: float = 0.05
DEFAULT_TOKEN_WEIGHT: float = 0.30
DEFAULT_TOKEN_QUALITY_FLOOR: float = 0.70
DEFAULT_TOKEN_EFFICIENCY_CLIP: float = 0.50


class DuelScoringMethod(StrEnum):
    """How a completed pool decides whether the challenger beat the king."""

    ROUND_WINS = "round_wins"
    MEAN = "mean"
    TOKEN_EFFICIENCY = "token_efficiency"


def token_adjusted_score_delta(
    *,
    score_mean_delta: float,
    token_efficiency_mean: float,
    quality_band: float,
    token_weight: float,
) -> float:
    """Blend quality and token efficiency only inside the near-equal band.

    Both inputs are challenger-perspective deltas. Outside ``quality_band`` raw
    quality is authoritative. Inside it, ``token_weight`` controls the efficiency
    share and ``1 - token_weight`` the quality share.
    """
    if abs(score_mean_delta) > quality_band:
        return score_mean_delta
    return (
        1.0 - token_weight
    ) * score_mean_delta + token_weight * token_efficiency_mean
