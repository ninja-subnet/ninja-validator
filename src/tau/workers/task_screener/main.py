"""Entry point and dependency wiring for the task-screener worker."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db.task_screening import TaskScreeningDb
from tau.utils.logging import configure_logging
from tau.workers.judge.main import _build_judge_clients

from .config import TaskScreenerConfig
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
        configured = _build_judge_clients(config.llm) if config.llm else []
        clients = [await stack.enter_async_context(client) for client in configured]
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. Windows / restricted loop
                pass

        model = config.llm.model if config.llm else None
        log.info("task screener starting (mode=%s, model=%s)", config.mode, model)
        get_axiom().info(
            source="task-screener",
            event_type="init_worker",
            mode=config.mode,
            model=model,
            max_king_score=config.max_king_score,
            concurrency=config.concurrency,
            poll_seconds=config.poll_seconds,
            max_failed_runs=config.max_failed_runs,
            retry_base_seconds=config.retry_base_seconds,
            retry_max_seconds=config.retry_max_seconds,
            pool_one_target=config.pool_targets.pool_one,
            pool_two_target=config.pool_targets.pool_two,
        )
        try:
            await run_task_screener(db=db, clients=clients, config=config, stop=stop)
        finally:
            get_axiom().info(source="task-screener", event_type="exit_worker")
            log.info("task screener stopped")


if __name__ == "__main__":
    main()
