"""Tunable configuration for the duel-resolver worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tau.duel import DuelScoringMethod
from tau.utils.env import env_float, env_int, env_str


@dataclass(frozen=True, slots=True)
class DuelResolverConfig:
    scoring_method: DuelScoringMethod = DuelScoringMethod.ROUND_WINS
    # Margin for round-win scoring (`wins > losses + margin`).
    round_win_margin: int = 0
    # Margin for mean-score scoring (`challenger_mean - king_mean >= margin`).
    mean_score_margin: float = 0.05
    # Idle sleep between poll ticks (seconds).
    poll_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.scoring_method, DuelScoringMethod):
            raise ValueError("scoring_method must be a DuelScoringMethod")
        if self.round_win_margin < 0:
            raise ValueError("round_win_margin must be >= 0")
        if self.mean_score_margin < 0:
            raise ValueError("mean_score_margin must be >= 0")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DuelResolverConfig:
        """Build a config from ``TAU_DUEL_*`` env vars, falling back to defaults."""
        env = os.environ if environ is None else environ
        d = cls()

        round_win_margin = env_int(env, "TAU_DUEL_ROUND_WIN_MARGIN", d.round_win_margin)
        scoring_method = DuelScoringMethod(
            env_str(env, "TAU_DUEL_SCORING_METHOD", d.scoring_method.value)
        )
        mean_score_margin = env_float(
            env, "TAU_DUEL_MEAN_SCORE_MARGIN", d.mean_score_margin
        )
        poll_seconds = env_float(env, "TAU_DUEL_POLL_SECONDS", d.poll_seconds)
        return cls(
            scoring_method=scoring_method,
            round_win_margin=round_win_margin,
            mean_score_margin=mean_score_margin,
            poll_seconds=poll_seconds,
        )
