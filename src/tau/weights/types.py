"""Data contracts for weight setting."""

from __future__ import annotations

import dataclasses as dc


@dc.dataclass(frozen=True, slots=True)
class RecentKing:
    king_id: str
    hotkey: str


@dc.dataclass(frozen=True, slots=True)
class MetagraphView:
    uids: tuple[int, ...]
    uid_by_hotkey: dict[str, int]


@dc.dataclass(frozen=True, slots=True)
class SubnetParams:
    """Startup/per-epoch chain state: our uid and the subnet's weight timing."""

    uid: int
    tempo: int
    weights_rate_limit: int


@dc.dataclass(frozen=True, slots=True)
class PollState:
    """Per-tick chain state: the tip and blocks since our uid last set weights."""

    current_block: int
    blocks_since_last_update: int


@dc.dataclass(frozen=True, slots=True)
class WeightPlan:
    uids: tuple[int, ...]
    weights: tuple[float, ...]
    submittable: bool
    skip_reason: str | None
    summary: str
