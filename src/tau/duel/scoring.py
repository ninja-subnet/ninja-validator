"""Duel scoring modes shared by config, decision logic, and DB writes."""

from __future__ import annotations

from enum import StrEnum


class DuelScoringMethod(StrEnum):
    """How a completed pool decides whether the challenger beat the king."""

    ROUND_WINS = "round_wins"
    MEAN = "mean"
