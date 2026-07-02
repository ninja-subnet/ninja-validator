"""Entry point and dependency wiring for the submission qualification worker."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import AsyncExitStack

from tau.axiom import get_axiom
from tau.db import QualificationDb
from tau.openrouter import OpenRouterClient
from tau.utils.logging import configure_logging

from .config import QualificationWorkerConfig
from .loop import run

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = QualificationWorkerConfig.from_env()
    asyncio.run(_serve(config))


async def _serve(config: QualificationWorkerConfig) -> None:
    async with AsyncExitStack() as stack:
        db = QualificationDb()
        stack.push_async_callback(db.aclose)
        client = await stack.enter_async_context(
            OpenRouterClient(
                config.openrouter_api_key,
                model=config.security.model,
                temperature=0,
                max_tokens=config.llm_max_tokens,
                timeout=config.llm_timeout_seconds,
            )
        )
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # e.g. Windows / restricted loop
                pass

        log.info("qualification worker starting (model=%s)", config.security.model)
        get_axiom().info(
            source="qualification",
            event_type="init_worker",
            model=config.security.model,
            submissions_dir=str(config.submissions_dir),
            base_path=str(config.base_path) if config.base_path else None,
            window_size=config.window_size,
            poll_seconds=config.poll_seconds,
            llm_timeout_seconds=config.llm_timeout_seconds,
            llm_max_tokens=config.llm_max_tokens,
        )
        try:
            await run(db=db, client=client, config=config, stop=stop)
        finally:
            get_axiom().info(source="qualification", event_type="exit_worker")
            log.info("qualification worker stopped")


if __name__ == "__main__":
    main()
