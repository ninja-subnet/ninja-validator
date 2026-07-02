"""Weight distribution domain logic: the king emission split and the cadence gate."""

from __future__ import annotations

from .chain import WeightChain
from .compute import KING_EMISSION_SHARES, compute_weights, king_emission_shares
from .schedule import SetDecision, blocks_until_next_epoch, should_set
from .types import (
    MetagraphView,
    PollState,
    RecentKing,
    SubnetParams,
    WeightPlan,
)

__all__ = [
    "KING_EMISSION_SHARES",
    "MetagraphView",
    "PollState",
    "RecentKing",
    "SetDecision",
    "SubnetParams",
    "WeightChain",
    "WeightPlan",
    "blocks_until_next_epoch",
    "compute_weights",
    "king_emission_shares",
    "should_set",
]
