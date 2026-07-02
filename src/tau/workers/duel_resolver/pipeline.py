"""The duel-resolver poll loop: snapshot -> decide -> apply one action, repeat.

The resolver is the sole writer of `challenges`/`kings`. Each tick reads a
`ChallengeSnapshot`, asks the pure `decide` for the one action due, and applies it
through the guarded `DuelResolverDb` writes. Level-triggered: nothing is carried
between ticks, so a stale read self-heals next tick.
"""

from __future__ import annotations

import asyncio
import logging
from typing import assert_never

from tau.axiom import get_axiom
from tau.db import DuelResolverDb
from tau.duel import (
    Action,
    AdvancePool,
    CloseChallenge,
    Nothing,
    OpenChallenge,
    Promote,
    decide,
)
from tau.pools import PoolTargets

from .config import DuelResolverConfig
from .telemetry import TickLog, close_outcome, emit_axiom

log = logging.getLogger(__name__)


async def run_duel_resolver(
    *,
    db: DuelResolverDb,
    targets: PoolTargets,
    config: DuelResolverConfig,
    stop: asyncio.Event,
) -> None:
    """Poll, decide, and apply one action per tick until *stop* is set."""
    log.info(
        "duel resolver running: poll %.0fs scoring=%s",
        config.poll_seconds,
        config.scoring_method.value,
    )
    ticklog = TickLog()
    while not stop.is_set():
        try:
            snapshot = await db.snapshot(targets)
            action = decide(
                snapshot,
                scoring_method=config.scoring_method,
                round_win_margin=config.round_win_margin,
                mean_score_margin=config.mean_score_margin,
            )
            await _apply(db, action, ticklog, config=config)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            log.exception("duel resolver tick failed")
            get_axiom().exception("duel-resolver", "unexpected_error", error=str(ex))
        await _sleep_until_stop(stop, config.poll_seconds)


async def _apply(
    db: DuelResolverDb,
    action: Action,
    ticklog: TickLog,
    *,
    config: DuelResolverConfig,
) -> None:
    """Apply one action via the guarded writes, reporting the outcome to *ticklog*.

    The `assert_never` makes a future `Action` variant a type error here, not a
    silent no-op.
    """
    match action:
        case Nothing(reason=reason):
            ticklog.idle(reason)
            return
        case OpenChallenge(
            king_submission_id=king, challenger_submission_id=challenger
        ):
            applied = await db.open_challenge(king, challenger)
            ticklog.action(applied, f"opened challenge: {challenger} vs king {king}")
        case AdvancePool(challenge=challenge):
            applied = await db.advance_pool(
                challenge,
                scoring_method=config.scoring_method,
                round_win_margin=config.round_win_margin,
                mean_score_margin=config.mean_score_margin,
            )
            ticklog.action(
                applied, f"advanced to pool two: {challenge.challenger_submission_id}"
            )
        case Promote(challenge=challenge):
            applied = await db.promote(
                challenge,
                scoring_method=config.scoring_method,
                round_win_margin=config.round_win_margin,
                mean_score_margin=config.mean_score_margin,
            )
            ticklog.action(
                applied, f"promoted new king: {challenge.challenger_submission_id}"
            )
        case CloseChallenge(challenge=challenge, reason=reason):
            applied = await db.close_challenge(
                challenge,
                close_outcome(reason),
                scoring_method=config.scoring_method,
                round_win_margin=config.round_win_margin,
                mean_score_margin=config.mean_score_margin,
            )
            ticklog.action(
                applied,
                f"closed challenge ({reason}): {challenge.challenger_submission_id}",
            )
        case _ as unreachable:
            assert_never(unreachable)
    if applied:
        emit_axiom(action, config)


async def _sleep_until_stop(stop: asyncio.Event, seconds: float) -> None:
    """Sleep up to *seconds*, waking early if *stop* is set."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass
