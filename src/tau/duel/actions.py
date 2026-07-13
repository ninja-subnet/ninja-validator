"""Output of the duel decision: the actions `decide` can return."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from .snapshot import ActiveChallenge


class CloseReason(StrEnum):
    """Why a challenge was closed with the king holding."""

    KING_DEFENDED = "king_defended"
    CHALLENGER_DEREGISTERED = "challenger_deregistered"


class WaitReason(StrEnum):
    """Why nothing is due -- what the resolver is waiting on."""

    NO_KING = "no_king"  # no reigning king
    POOLS_NOT_READY = "pools_not_ready"  # both qualified task pools are not full
    NO_CHALLENGER = "no_challenger"  # no eligible challenger to open against
    DUEL_IN_PROGRESS = "duel_in_progress"  # active duel not yet decided


@dataclass(frozen=True, slots=True)
class Nothing:
    """Wait -- nothing is due this tick; `reason` says what for."""

    reason: WaitReason


@dataclass(frozen=True, slots=True)
class OpenChallenge:
    """Open a fresh challenge: the challenger contests the reigning king in pool #1."""

    king_submission_id: str
    challenger_submission_id: str


@dataclass(frozen=True, slots=True)
class AdvancePool:
    """Advance the challenge from pool #1 to pool #2."""

    challenge: ActiveChallenge


@dataclass(frozen=True, slots=True)
class Promote:
    """Crown the challenger and close the challenge."""

    challenge: ActiveChallenge


@dataclass(frozen=True, slots=True)
class CloseChallenge:
    """Close the challenge with the king holding."""

    challenge: ActiveChallenge
    reason: CloseReason


Action: TypeAlias = Nothing | OpenChallenge | AdvancePool | Promote | CloseChallenge
