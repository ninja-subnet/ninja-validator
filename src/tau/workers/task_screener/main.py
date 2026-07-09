"""Entry point and dependency wiring for the task-screener worker."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db.task_screening import TaskScreeningDb
from tau.openrouter import OpenRouterClient
from tau.utils.logging import configure_logging

from .config import TaskScreenerConfig, TaskScreenMode
from .pipeline import run_task_screener

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = TaskScreenerConfig.from_env()
    asyncio.run(_serve(config))


async def _serve(config: TaskScreenerConfig) -> None:
    async with AsyncExitStack() as stack:
        db = TaskScreeningDb()
        stack.push_async_callback(db.aclose)
        clients = [
            await stack.enter_async_context(client)
            for client in _build_screen_clients(config)
        ]
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. Windows / restricted loop
                pass

        log.info(
            "task screener starting (mode=%s, model=%s)", config.mode, config.model
        )
        get_axiom().info(
            source="task-screener",
            event_type="init_worker",
            mode=config.mode,
            model=config.model,
            fallback_models=list(config.fallback_models),
            provider=config.provider,
            fallback_provider=config.fallback_provider,
            max_king_score=config.max_king_score,
            concurrency=config.concurrency,
            attempts=config.attempts,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            poll_seconds=config.poll_seconds,
            total_timeout_seconds=config.total_timeout_seconds,
            max_failed_runs=config.max_failed_runs,
            retry_base_seconds=config.retry_base_seconds,
            retry_max_seconds=config.retry_max_seconds,
        )
        try:
            await run_task_screener(db=db, clients=clients, config=config, stop=stop)
        finally:
            get_axiom().info(source="task-screener", event_type="exit_worker")
            log.info("task screener stopped")


def _build_screen_clients(config: TaskScreenerConfig) -> list[OpenRouterClient]:
    """Build primary and alternate-provider fallback scorer clients."""
    if config.mode is TaskScreenMode.DISABLED:
        return []
    clients: list[OpenRouterClient] = []
    for index, model in enumerate((config.model, *config.fallback_models)):
        is_primary = index == 0
        clients.append(
            OpenRouterClient(
                config.openrouter_api_key,
                model=model,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                # Alternate models may not accept the primary model's reasoning
                # controls. Keep them for same-model provider failover; disable
                # them only when the configured fallback model is different.
                reasoning=(
                    config.reasoning if is_primary or model == config.model else None
                ),
                provider=config.provider if is_primary else config.fallback_provider,
                timeout=config.timeout_seconds,
            )
        )
    return clients


if __name__ == "__main__":
    main()
