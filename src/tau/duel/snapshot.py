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
    # None means at least one judged task had no trustworthy usage count. This keeps
    # a partial sum from being presented as the pool total.
    king_total_tokens: int | None = None
    challenger_total_tokens: int | None = None
    token_comparison_rounds: int = 0
    king_token_savings_mean: float = 0.0
    challenger_token_savings_mean: float = 0.0
    king_token_boost: float = 0.0
    challenger_token_boost: float = 0.0

    @property
    def judged(self) -> int:
        """Rounds judged so far."""
        return self.wins + self.losses + self.ties

    @property
    def king_combined_score(self) -> float:
        """King quality mean after its token-efficiency boost."""
        return self.king_score_mean + self.king_token_boost

    @property
    def challenger_combined_score(self) -> float:
        """Challenger quality mean after its token-efficiency boost."""
        return self.challenger_score_mean + self.challenger_token_boost

    @property
    def combined_score_delta(self) -> float:
        """Challenger-minus-king delta after both token boosts."""
        return (
            self.score_mean_delta + self.challenger_token_boost - self.king_token_boost
        )


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
    task_pools_ready: bool = True
