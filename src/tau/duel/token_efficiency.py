"""Pure per-task token-efficiency scoring for mean-score duels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TokenEfficiencyConfig:
    """Knobs for the symmetric token bonus applied to each side's quality mean."""

    enabled: bool = False
    score_tolerance: float = 0.05
    min_score: float = 0.20
    bonus_multiplier: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not 0 <= self.score_tolerance <= 1:
            raise ValueError("score_tolerance must be between 0 and 1")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if not math.isfinite(self.bonus_multiplier) or self.bonus_multiplier < 0:
            raise ValueError("bonus_multiplier must be >= 0")


@dataclass(frozen=True, slots=True)
class TokenEfficiencyRound:
    """Quality and raw proxy-observed token counts for one judged task."""

    king_score: float | None
    challenger_score: float | None
    king_tokens: int | None
    challenger_tokens: int | None
    judgement_valid: bool = True


@dataclass(frozen=True, slots=True)
class TokenEfficiencyStats:
    """Pool-level token totals, average savings, and resulting score boosts."""

    king_total_tokens: int | None
    challenger_total_tokens: int | None
    token_comparison_rounds: int
    king_savings_mean: float
    challenger_savings_mean: float
    king_boost: float
    challenger_boost: float


def calculate_token_efficiency(
    rounds: Iterable[TokenEfficiencyRound],
    *,
    pool_target: int,
    config: TokenEfficiencyConfig,
) -> TokenEfficiencyStats:
    """Calculate symmetric bonuses, dividing savings by the full pool target."""
    if pool_target <= 0:
        raise ValueError("pool_target must be positive")

    king_total = 0
    challenger_total = 0
    king_total_complete = True
    challenger_total_complete = True
    comparison_rounds = 0
    king_savings = 0.0
    challenger_savings = 0.0

    for round_ in rounds:
        king_tokens = _valid_tokens(round_.king_tokens)
        challenger_tokens = _valid_tokens(round_.challenger_tokens)
        if king_tokens is not None:
            king_total += king_tokens
        else:
            king_total_complete = False
        if challenger_tokens is not None:
            challenger_total += challenger_tokens
        else:
            challenger_total_complete = False

        if king_tokens is None or challenger_tokens is None:
            continue
        comparison_rounds += 1
        if not config.enabled:
            continue
        if (
            not round_.judgement_valid
            or round_.king_score is None
            or round_.challenger_score is None
        ):
            continue

        king_savings += _task_saving(
            own_score=round_.king_score,
            opponent_score=round_.challenger_score,
            own_tokens=king_tokens,
            opponent_tokens=challenger_tokens,
            config=config,
        )
        challenger_savings += _task_saving(
            own_score=round_.challenger_score,
            opponent_score=round_.king_score,
            own_tokens=challenger_tokens,
            opponent_tokens=king_tokens,
            config=config,
        )

    denominator = float(pool_target)
    king_savings_mean = king_savings / denominator
    challenger_savings_mean = challenger_savings / denominator
    return TokenEfficiencyStats(
        king_total_tokens=king_total if king_total_complete else None,
        challenger_total_tokens=(
            challenger_total if challenger_total_complete else None
        ),
        token_comparison_rounds=comparison_rounds,
        king_savings_mean=king_savings_mean,
        challenger_savings_mean=challenger_savings_mean,
        king_boost=king_savings_mean * config.bonus_multiplier,
        challenger_boost=challenger_savings_mean * config.bonus_multiplier,
    )


def _task_saving(
    *,
    own_score: float,
    opponent_score: float,
    own_tokens: int,
    opponent_tokens: int,
    config: TokenEfficiencyConfig,
) -> float:
    """Return one side's saving for one task; zero means no bonus contribution."""
    if own_score < config.min_score:
        return 0.0
    if own_score < opponent_score - config.score_tolerance:
        return 0.0
    if opponent_tokens <= 0:
        return 0.0
    return max(0.0, 1.0 - own_tokens / opponent_tokens)


def _valid_tokens(value: int | None) -> int | None:
    """Treat missing, boolean, and negative values as unavailable usage."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
