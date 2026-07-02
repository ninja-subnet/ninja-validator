"""Entry point and dependency wiring for the task-generator worker.

Builds the long-lived collaborators (DB, commit sampler, LLM client), installs
signal handlers for a graceful drain, and hands them to the pipeline's run().
"""

from __future__ import annotations

import asyncio
import logging
import random
import signal
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db import GeneratorDb
from tau.github import CommitSampler, GitHubClient, GitHubConfig, GitHubTokenRotator
from tau.openrouter import OpenRouterClient
from tau.pools import PoolTargets
from tau.utils.logging import configure_logging

from .config import GeneratorConfig
from .dummy import DummyTaskClient
from .pipeline import run

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = GeneratorConfig.from_env()
    targets = PoolTargets.from_env()
    asyncio.run(_serve(config, targets))


def _build_llm(config: GeneratorConfig) -> OpenRouterClient | DummyTaskClient:
    """Pick the LLM client: the token-free dummy when enabled, else OpenRouter."""
    if config.use_dummy_llm:
        # Record a "dummy/" marker (keeping the emulated model) so tasks.model makes
        # it obvious these descriptions were fabricated, not produced by a real model.
        dummy_model = f"dummy/{config.generator_model}"
        log.warning(
            "task-generator using DummyTaskClient (model=%s): fabricating descriptions "
            "locally, no real LLM calls (no tokens spent)",
            dummy_model,
        )
        return DummyTaskClient(
            model=dummy_model,
            timeout=config.llm_timeout,
            config=config.dummy,
        )
    return OpenRouterClient(
        config.openrouter_api_key,
        model=config.generator_model,
        timeout=int(config.llm_timeout),
    )


async def _serve(config: GeneratorConfig, targets: PoolTargets) -> None:
    github_config = GitHubConfig.from_env()
    async with AsyncExitStack() as stack:
        db = GeneratorDb()
        stack.push_async_callback(db.aclose)
        client = await stack.enter_async_context(
            GitHubClient.create(
                token_rotator=GitHubTokenRotator.from_env(),
                timeout=github_config.http_timeout,
            )
        )
        llm = await stack.enter_async_context(_build_llm(config))
        sampler = CommitSampler(rng=random.Random(), client=client, config=github_config)

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. Windows / restricted loop
                pass

        log.info("task-generator starting (model=%s)", config.generator_model)
        get_axiom().info(
            source="task-generator",
            event_type="init_worker",
            generator_model=config.generator_model,
            describe_concurrency=config.describe_concurrency,
            poll_seconds=config.poll_seconds,
            llm_attempts=config.llm_attempts,
            llm_timeout=config.llm_timeout,
            use_dummy_llm=config.use_dummy_llm,
            pool_one_target=targets.pool_one,
            pool_two_target=targets.pool_two,
        )
        try:
            await run(
                db=db, sampler=sampler, llm=llm, config=config, targets=targets, stop=stop
            )
        finally:
            get_axiom().info(source="task-generator", event_type="exit_worker")
            log.info("task-generator stopped")
