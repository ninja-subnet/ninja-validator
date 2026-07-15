"""Entry point and wiring for the benchmark worker.

Builds the DB seam + config, installs signal handlers for a graceful stop, and
hands them to the loop. Sync throughout: each king's benchmark is a blocking
subprocess, so asyncio would add nothing.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

from .config import BenchmarkConfig
from .db import BenchmarkDb
from .loop import run

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "docker"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if level == "DEBUG" else logging.WARNING)


def main() -> None:
    _configure_logging()
    config = BenchmarkConfig.from_env()
    config.results_dir.mkdir(parents=True, exist_ok=True)

    db = BenchmarkDb()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:  # not in the main thread (e.g. some test runners)
            pass

    log.info(
        "benchmark-worker starting (submissions_dir=%s results_dir=%s bench_repo=%s "
        "model=%s slice=%s workers=%d poll=%.0fs)",
        config.submissions_dir, config.results_dir, config.bench_repo_dir,
        config.model, config.slice_spec, config.agent_workers, config.poll_seconds,
    )
    if not config.openrouter_api_key:
        log.warning("OPENROUTER_API_KEY is not set; kings cannot be benchmarked until it is")
    try:
        run(db=db, config=config, stop=stop)
        log.info("benchmark-worker stopped")
    finally:
        db.close()


if __name__ == "__main__":
    main()
