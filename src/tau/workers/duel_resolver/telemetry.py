"""Logging and Axiom helpers for the duel-resolver worker."""

from __future__ import annotations

import logging
from typing import assert_never

from tau.axiom import get_axiom
from tau.db.status import DuelOutcome
from tau.duel import (
    Action,
    ActiveChallenge,
    AdvancePool,
    CloseChallenge,
    CloseReason,
    Nothing,
    OpenChallenge,
    Promote,
)

from .config import DuelResolverConfig

log = logging.getLogger(__name__)


class TickLog:
    """Edge-triggered text logging for resolver work/idle state."""

    def __init__(self) -> None:
        self._idle_reason: str | None = None

    def action(self, applied: bool, message: str) -> None:
        self._idle_reason = None
        if applied:
            log.info(message)
        else:
            log.info("%s -- skipped (state changed since snapshot)", message)

    def idle(self, reason: str) -> None:
        if reason != self._idle_reason:
            log.info("idle: %s", reason)
            self._idle_reason = reason


def emit_axiom(action: Action, config: DuelResolverConfig) -> None:
    """Mirror an applied duel transition to Axiom; idle/no-op actions never reach here."""
    match action:
        case OpenChallenge(
            king_submission_id=king, challenger_submission_id=challenger
        ):
            get_axiom().info(
                source="duel-resolver",
                event_type="challenge_opened",
                king_submission_id=king,
                challenger_submission_id=challenger,
                **_config_fields(config),
            )
        case AdvancePool(challenge=challenge):
            get_axiom().info(
                source="duel-resolver",
                event_type="pool_advanced",
                **_config_fields(config),
                **_challenge_fields(challenge),
            )
        case Promote(challenge=challenge):
            get_axiom().info(
                source="duel-resolver",
                event_type="king_promoted",
                **_config_fields(config),
                **_challenge_fields(challenge),
            )
        case CloseChallenge(challenge=challenge, reason=reason):
            get_axiom().info(
                source="duel-resolver",
                event_type="challenge_closed",
                reason=str(reason),
                outcome=close_outcome(reason).name,
                **_config_fields(config),
                **_challenge_fields(challenge),
            )
        case Nothing():
            pass


def close_outcome(reason: CloseReason) -> DuelOutcome:
    """The duel_resolutions outcome recorded when a challenge closes for *reason*."""
    match reason:
        case CloseReason.KING_DEFENDED:
            return DuelOutcome.KING_WON
        case CloseReason.CHALLENGER_DEREGISTERED:
            return DuelOutcome.CHALLENGER_DEREGISTERED
        case _ as unreachable:
            assert_never(unreachable)


def _challenge_fields(challenge: ActiveChallenge) -> dict[str, object]:
    """Structured identity + current tally shared by challenge transition events."""
    return {
        "challenger_submission_id": challenge.challenger_submission_id,
        "king_submission_id": challenge.king_submission_id,
        "pool": challenge.pool.name,
        "pool_target": challenge.pool_target,
        "wins": challenge.tally.wins,
        "losses": challenge.tally.losses,
        "ties": challenge.tally.ties,
        "king_score_mean": challenge.tally.king_score_mean,
        "challenger_score_mean": challenge.tally.challenger_score_mean,
        "score_mean_delta": challenge.tally.score_mean_delta,
        "score_mean_rounds": challenge.tally.score_mean_rounds,
        "token_efficiency_mean": challenge.tally.token_efficiency_mean,
        "token_usage_rounds": challenge.tally.token_usage_rounds,
        "token_usage_penalty_rounds": challenge.tally.token_usage_penalty_rounds,
    }


def _config_fields(config: DuelResolverConfig) -> dict[str, object]:
    """Structured scoring config shared by applied transition events."""
    return {
        "scoring_method": config.scoring_method.value,
        "round_win_margin": config.round_win_margin,
        "mean_score_margin": config.mean_score_margin,
        "token_weight": config.token_weight,
        "token_quality_floor": config.token_quality_floor,
        "token_efficiency_clip": config.token_efficiency_clip,
    }
