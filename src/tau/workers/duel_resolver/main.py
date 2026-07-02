"""Entry point and dependency wiring for the duel-resolver worker.

Builds the DB seam, installs signal handlers for a graceful stop, and hands them to
the pipeline's run loop.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db import DuelResolverDb
from tau.pools import PoolTargets
from tau.utils.logging import configure_logging

from .config import DuelResolverConfig
from .pipeline import run_duel_resolver

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = DuelResolverConfig.from_env()
    targets = PoolTargets.from_env()
    asyncio.run(_serve(config, targets))


async def _serve(config: DuelResolverConfig, targets: PoolTargets) -> None:
    async with AsyncExitStack() as stack:
        db = DuelResolverDb()
        stack.push_async_callback(db.aclose)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. Windows / restricted loop
                pass
        log.info("duel resolver starting")
        get_axiom().info(
            source="duel-resolver",
            event_type="init_worker",
            poll_seconds=config.poll_seconds,
            scoring_method=config.scoring_method.value,
            round_win_margin=config.round_win_margin,
            mean_score_margin=config.mean_score_margin,
            pool_one_target=targets.pool_one,
            pool_two_target=targets.pool_two,
        )
        try:
            await run_duel_resolver(db=db, targets=targets, config=config, stop=stop)
        finally:
            get_axiom().info(source="duel-resolver", event_type="exit_worker")
            log.info("duel resolver stopped")


if __name__ == "__main__":
    main()
