"""Tunable configuration for the duel-resolver worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tau.duel import DEFAULT_MEAN_SCORE_MARGIN, DuelScoringMethod
from tau.utils.env import env_bool, env_float, env_int, env_str


@dataclass(frozen=True, slots=True)
class DuelResolverConfig:
    scoring_method: DuelScoringMethod = DuelScoringMethod.ROUND_WINS
    # Margin for round-win scoring (`wins > losses + margin`).
    round_win_margin: int = 0
    # Margin for mean-score scoring (`challenger_mean - king_mean >= margin`).
    mean_score_margin: float = DEFAULT_MEAN_SCORE_MARGIN
    # Idle sleep between poll ticks (seconds).
    poll_seconds: float = 5.0
    # Optional GitHub publication for promoted local submission bundles.
    promotion_publish_repo: str | None = None
    promotion_publish_branch: str = "main"
    promotion_github_token: str | None = None
    promotion_publish_required: bool = False
    promotion_submissions_dir: Path = Path("submissions")
    promotion_http_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.scoring_method, DuelScoringMethod):
            raise ValueError("scoring_method must be a DuelScoringMethod")
        if self.round_win_margin < 0:
            raise ValueError("round_win_margin must be >= 0")
        if self.mean_score_margin < 0:
            raise ValueError("mean_score_margin must be >= 0")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if not self.promotion_publish_branch.strip():
            raise ValueError("promotion_publish_branch must not be blank")
        if self.promotion_publish_required and not self.promotion_enabled:
            raise ValueError(
                "promotion publishing is required but repo/token are not configured"
            )
        if self.promotion_http_timeout <= 0:
            raise ValueError("promotion_http_timeout must be positive")

    @property
    def promotion_enabled(self) -> bool:
        return bool(
            self.promotion_publish_repo
            and self.promotion_publish_repo.strip()
            and self.promotion_github_token
            and self.promotion_github_token.strip()
        )

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
        publish_repo = env_str(
            env,
            "TAU_PROMOTION_PUBLISH_REPO",
            env_str(env, "VALIDATE_PUBLISH_REPO", ""),
        )
        publish_branch = env_str(
            env,
            "TAU_PROMOTION_PUBLISH_BRANCH",
            env_str(env, "VALIDATE_PUBLISH_BASE", d.promotion_publish_branch),
        )
        github_token = env_str(
            env,
            "TAU_PROMOTION_GITHUB_TOKEN",
            env_str(
                env,
                "GITHUB_MERGE_TOKEN",
                env_str(env, "GITHUB_TOKEN_UNARBOS", ""),
            ),
        )
        return cls(
            scoring_method=scoring_method,
            round_win_margin=round_win_margin,
            mean_score_margin=mean_score_margin,
            poll_seconds=poll_seconds,
            promotion_publish_repo=publish_repo or None,
            promotion_publish_branch=publish_branch,
            promotion_github_token=github_token or None,
            promotion_publish_required=env_bool(
                env, "TAU_PROMOTION_PUBLISH_REQUIRED", d.promotion_publish_required
            ),
            promotion_submissions_dir=Path(
                env_str(
                    env,
                    "TAU_PROMOTION_SUBMISSIONS_DIR",
                    env_str(
                        env,
                        "TAU_SUBMISSIONS_DIR",
                        str(d.promotion_submissions_dir),
                    ),
                )
            ),
            promotion_http_timeout=env_float(
                env, "TAU_PROMOTION_GITHUB_TIMEOUT", d.promotion_http_timeout
            ),
        )
