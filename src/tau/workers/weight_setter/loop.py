"""The weight-setter poll loop: gate on the chain's rate limit + epoch boundary, then
compute and submit. Level-triggered; `step` is the testable unit, `run` the loop."""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import logging
import time
from collections.abc import Callable
from typing import Literal

from tau.axiom import get_axiom
from tau.bittensor.types import BLOCK_SECONDS
from tau.db.weight_setter import WeightSetterDb
from tau.weights.chain import WeightChain
from tau.weights.compute import compute_weights
from tau.weights.schedule import should_set
from tau.weights.types import SubnetParams, WeightPlan

from .config import WeightSetterConfig

logger = logging.getLogger(__name__)

Action = Literal["wait", "skip", "set", "failed"]


@dc.dataclass(frozen=True, slots=True)
class StepResult:
    action: Action
    reason: str
    epoch_blocks: int | None = None
    rate_off_blocks: int | None = None
    next_try_block: int | None = None
    next_try_blocks: int | None = None
    block: int | None = None
    num_kings: int | None = None
    plan: WeightPlan | None = None


def step(
    chain: WeightChain,
    db: WeightSetterDb,
    config: WeightSetterConfig,
    params: SubnetParams,
) -> StepResult:
    poll = chain.poll(config.netuid, params.uid)
    d = should_set(
        current_block=poll.current_block,
        tempo=params.tempo,
        netuid=config.netuid,
        blocks_since_last_update=poll.blocks_since_last_update,
        weights_rate_limit=params.weights_rate_limit,
        set_margin=config.set_margin,
    )
    if not d.proceed:
        return StepResult(
            "wait",
            "",
            epoch_blocks=d.epoch_blocks,
            rate_off_blocks=d.rate_off_blocks,
            next_try_block=d.next_try_block,
            next_try_blocks=d.next_try_block - poll.current_block,
        )

    meta = chain.metagraph(config.netuid)
    kings = [] if config.burn_mode else db.recent_kings(config.window)
    plan = compute_weights(kings, meta, window=config.window, burn_uid=config.burn_uid)

    if not plan.submittable:
        return StepResult(
            "skip",
            plan.skip_reason or "unsubmittable vector",
            block=poll.current_block,
            plan=plan,
        )
    if not chain.set_weights(config.netuid, list(plan.uids), list(plan.weights)):
        return StepResult(
            "failed", "set_weights rejected", block=poll.current_block, plan=plan
        )
    return StepResult(
        "set",
        f"block {poll.current_block}: {plan.summary}",
        epoch_blocks=d.epoch_blocks,
        block=poll.current_block,
        num_kings=len(kings),
        plan=plan,
    )


def run(
    chain: WeightChain,
    db: WeightSetterDb,
    config: WeightSetterConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    params = chain.params(config.netuid)
    logger.info(
        "weight setter running: netuid=%d uid=%d tempo=%d rate_limit=%d "
        "window=%d set_margin=%d burn_mode=%s poll=%.1fs",
        config.netuid,
        params.uid,
        params.tempo,
        params.weights_rate_limit,
        config.window,
        config.set_margin,
        config.burn_mode,
        config.poll_seconds,
    )
    get_axiom().info(
        source="weight-setter",
        event_type="init_worker",
        netuid=config.netuid,
        uid=params.uid,
        tempo=params.tempo,
        weights_rate_limit=params.weights_rate_limit,
        window=config.window,
        set_margin=config.set_margin,
        burn_mode=config.burn_mode,
        poll_seconds=config.poll_seconds,
    )
    ticklog = _TickLog()
    try:
        while True:
            try:
                result = step(chain, db, config, params)
                ticklog.report(result)
                _emit_axiom(result, config, params)
                if result.action == "set":
                    params = chain.params(config.netuid)
            except Exception as ex:
                logger.exception(f"weight setter tick failed: {ex}")
                get_axiom().exception(
                    "weight-setter", "unexpected_error", exception=str(ex)
                )
            sleep(config.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        get_axiom().info(
            source="weight-setter",
            event_type="exit_worker",
            netuid=config.netuid,
            uid=params.uid,
        )


def _emit_axiom(
    result: StepResult, config: WeightSetterConfig, params: SubnetParams
) -> None:
    """Mirror a step's outcome to Axiom; the routine 'wait' heartbeat is not sent."""
    if result.action == "set" and result.plan is not None:
        get_axiom().info(
            source="weight-setter",
            event_type="weights_set",
            netuid=config.netuid,
            uid=params.uid,
            block=result.block,
            epoch_blocks=result.epoch_blocks,
            next_epoch_seconds=(result.epoch_blocks or 0) * BLOCK_SECONDS,
            window=config.window,
            burn_mode=config.burn_mode,
            num_kings=result.num_kings,
            summary=result.plan.summary,
            uids=list(result.plan.uids),
            weights=list(result.plan.weights),
        )
    elif result.action == "skip":
        get_axiom().warn(
            source="weight-setter",
            event_type="weights_skipped",
            netuid=config.netuid,
            block=result.block,
            reason=result.reason,
        )
    elif result.action == "failed":
        get_axiom().error(
            source="weight-setter",
            event_type="weights_rejected",
            netuid=config.netuid,
            block=result.block,
            reason=result.reason,
        )


def _human(seconds: int) -> str:
    if seconds < 90:
        return f"~{seconds}s"
    if seconds < 5400:
        return f"~{seconds / 60:.0f}m"
    return f"~{seconds / 3600:.1f}h"


class _TickLog:
    def __init__(self) -> None:
        self._wait_key: int | None = None

    def report(self, result: StepResult) -> None:
        if result.action == "wait":
            if result.next_try_block != self._wait_key:
                self._wait_key = result.next_try_block
                seconds = (result.next_try_blocks or 0) * BLOCK_SECONDS
                at = (dt.datetime.now() + dt.timedelta(seconds=seconds)).strftime(
                    "%H:%M"
                )
                logger.info(
                    "waiting: rate-off ~%db, epoch ~%db, next try block %d in %s (%s)",
                    result.rate_off_blocks,
                    result.epoch_blocks,
                    result.next_try_block,
                    _human(seconds),
                    at,
                )
            return
        self._wait_key = None
        if result.action == "set":
            logger.info(
                "set weights: %s; next epoch in %s",
                result.reason,
                _human((result.epoch_blocks or 0) * BLOCK_SECONDS),
            )
        elif result.action == "skip":
            logger.warning("skipped set_weights: %s", result.reason)
        else:
            logger.error("set_weights failed: %s", result.reason)
