"""Input data for the duel decision: a consistent read of the arena state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tau.db.status import PoolType


@dataclass(frozen=True, slots=True)
class Tally:
    """Active-pool outcome and score aggregates, from the challenger's perspective."""

    wins: int
    losses: int
    ties: int
    king_score_mean: float = 0.0
    challenger_score_mean: float = 0.0
    score_mean_delta: float = 0.0
    score_mean_rounds: int = 0
    # Mean challenger-perspective token efficiency across all scored rounds.
    # Per-round values are positive when the challenger used fewer cumulative
    # prompt+completion tokens and are clipped before averaging.
    token_efficiency_mean: float = 0.0
    # Both sides supplied finalized, positive cumulative usage for these rounds.
    token_usage_rounds: int = 0
    # Exactly one side supplied finalized usage, so the incomplete side received
    # the configured worst per-round efficiency value.
    token_usage_penalty_rounds: int = 0

    @property
    def judged(self) -> int:
        """Rounds judged so far."""
        return self.wins + self.losses + self.ties


@dataclass(frozen=True, slots=True)
class ActiveChallenge:
    """A challenge currently dueling the reigning king."""

    challenger_submission_id: str
    king_submission_id: str
    pool: PoolType  # the pool being dueled
    pool_target: int  # best-of-N round count for this pool
    tally: Tally
    challenger_registered: bool  # still in the current metagraph?

    @property
    def remaining(self) -> int:
        """Rounds not yet judged in this pool, floored at zero."""
        return max(0, self.pool_target - self.tally.judged)


@dataclass(frozen=True, slots=True)
class ChallengeSnapshot:
    """One consistent read of the state `decide` needs."""

    reigning_king_submission_id: str | None
    active_challenge: ActiveChallenge | None
    next_challenger_submission_id: str | None
