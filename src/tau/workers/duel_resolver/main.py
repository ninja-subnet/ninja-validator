"""Entry point and dependency wiring for the duel-resolver worker.

Builds the DB seam and installs signal handlers: SIGINT/SIGTERM stop the worker,
while SIGUSR1/SIGUSR2 pause/resume opening new challenges.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db import DuelResolverDb
from tau.github import (
    GitHubClient,
    GitHubPromotionPublisher,
    PromotionPublishConfig,
)
from tau.pools import PoolTargets
from tau.utils.logging import configure_logging

from .config import DuelResolverConfig
from .pipeline import run_duel_resolver

log = logging.getLogger(__name__)


def _signal_handlers(
    stop: asyncio.Event, new_challenges_paused: asyncio.Event
) -> tuple[tuple[int, Callable[[], None]], ...]:
    return (
        (signal.SIGTERM, stop.set),
        (signal.SIGINT, stop.set),
        (signal.SIGUSR1, new_challenges_paused.set),
        (signal.SIGUSR2, new_challenges_paused.clear),
    )


def main() -> None:
    configure_logging()
    config = DuelResolverConfig.from_env()
    targets = PoolTargets.from_env()
    asyncio.run(_serve(config, targets))


async def _serve(config: DuelResolverConfig, targets: PoolTargets) -> None:
    async with AsyncExitStack() as stack:
        db = DuelResolverDb()
        stack.push_async_callback(db.aclose)
        promotion_publisher = None
        if config.promotion_enabled:
            github_client = GitHubClient.create(
                token=config.promotion_github_token,
                timeout=config.promotion_http_timeout,
            )
            stack.push_async_callback(github_client.aclose)
            promotion_publisher = GitHubPromotionPublisher(
                github_client,
                PromotionPublishConfig(
                    submissions_dir=config.promotion_submissions_dir,
                    repo=config.promotion_publish_repo or "",
                    branch=config.promotion_publish_branch,
                ),
            )
        stop = asyncio.Event()
        new_challenges_paused = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig, callback in _signal_handlers(stop, new_challenges_paused):
            try:
                loop.add_signal_handler(sig, callback)
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
            token_bonus_enabled=config.token_efficiency.enabled,
            token_score_tolerance=config.token_score_tolerance,
            token_min_score=config.token_min_score,
            token_bonus_multiplier=config.token_bonus_multiplier,
            pool_one_target=targets.pool_one,
            pool_two_target=targets.pool_two,
        )
        try:
            await run_duel_resolver(
                db=db,
                targets=targets,
                config=config,
                stop=stop,
                new_challenges_paused=new_challenges_paused,
                promotion_publisher=promotion_publisher,
            )
        finally:
            get_axiom().info(source="duel-resolver", event_type="exit_worker")
            log.info("duel resolver stopped")


if __name__ == "__main__":
    main()
