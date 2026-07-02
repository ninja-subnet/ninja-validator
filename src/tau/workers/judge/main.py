"""Entry point and dependency wiring for the judge worker.

Builds the long-lived collaborators (DB, per-model LLM clients), installs signal
handlers for a graceful stop, and hands them to the pipeline's run loop.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db import JudgeDb
from tau.openrouter import OpenRouterClient
from tau.utils.logging import configure_logging

from .config import JudgeWorkerConfig
from .dummy import DummyJudgeClient
from .pipeline import run_judge_worker

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = JudgeWorkerConfig.from_env()
    asyncio.run(_serve(config))


async def _serve(config: JudgeWorkerConfig) -> None:
    async with AsyncExitStack() as stack:
        db = JudgeDb()
        stack.push_async_callback(db.aclose)
        clients = [
            await stack.enter_async_context(client)
            for client in _build_judge_clients(config)
        ]
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. Windows / restricted loop
                pass
        log.info("judge worker starting (model=%s)", config.model)
        get_axiom().info(
            source="judge",
            event_type="init_worker",
            model=config.model,
            fallback_models=list(config.fallback_models),
            concurrency=config.concurrency,
            attempts=config.attempts,
            poll_seconds=config.poll_seconds,
            total_timeout_seconds=config.total_timeout_seconds,
            use_dummy_llm=config.use_dummy_llm,
        )
        try:
            await run_judge_worker(db=db, clients=clients, config=config, stop=stop)
        finally:
            get_axiom().info(source="judge", event_type="exit_worker")
            log.info("judge worker stopped")


def _build_judge_clients(
    config: JudgeWorkerConfig,
) -> list[OpenRouterClient | DummyJudgeClient]:
    """One configured client per model: primary first, then fallbacks.

    The primary (cache-capable) model carries reasoning; fallbacks do not — what
    used to be per-call branching is now folded into client construction. Each
    client owns its own connection pool (an OpenRouterClient implementation
    detail), so the worker never touches the transport.
    """
    if config.use_dummy_llm:
        # "dummy/" marker (keeping the emulated model) so judgements.model shows the
        # verdict was fabricated, not produced by a real model. No fallbacks needed.
        dummy_model = f"dummy/{config.model}"
        log.warning(
            "judge worker using DummyJudgeClient (model=%s): fabricating verdicts "
            "locally, no real LLM calls (no tokens spent)",
            dummy_model,
        )
        return [
            DummyJudgeClient(
                model=dummy_model,
                timeout=float(config.timeout_seconds),
                config=config.dummy,
            )
        ]
    clients: list[OpenRouterClient | DummyJudgeClient] = []
    for model in (config.model, *config.fallback_models):
        is_primary = model == config.model
        clients.append(
            OpenRouterClient(
                config.openrouter_api_key,
                model=model,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                reasoning=config.reasoning if is_primary else None,
                timeout=config.timeout_seconds,
            )
        )
    return clients


if __name__ == "__main__":
    main()
