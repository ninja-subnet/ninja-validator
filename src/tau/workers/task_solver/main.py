"""Entry point and dependency wiring for the task-solver worker.

Builds the long-lived collaborators (Docker client, DB, sandbox image), installs
signal handlers for a graceful stop, and hands them to the threaded scheduler.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

import docker

from tau.db import SolverDb
from tau.sandbox import ensure_sandbox_image
from tau.sandbox.network import ensure_shared_network, self_container

from .config import SolverConfig
from .loop import run

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    http_level = logging.DEBUG if level == "DEBUG" else logging.WARNING
    for noisy in ("httpx", "httpcore", "urllib3", "docker"):
        logging.getLogger(noisy).setLevel(http_level)


def main() -> None:
    _configure_logging()
    config = SolverConfig.from_env()

    client = docker.from_env()
    db = SolverDb()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:  # not in the main thread (e.g. some test runners)
            pass

    try:
        image_tag = ensure_sandbox_image(client, config.sandbox)
        # Join the shared sandbox network once up front (production/in-container only).
        # Doing it here — not per solve — keeps the orchestrator's own DNS/egress
        # stable and avoids a first-tick race where concurrent solves all connect it.
        self_ctr = self_container(client)
        if self_ctr is not None:
            ensure_shared_network(client, self_ctr)
            log.info("shared sandbox network ready (orchestrator attached)")
        log.info(
            "task-solver starting (provider=%s model=%s max_containers=%d poll=%.0fs "
            "submissions_dir=%s budget=%s)",
            config.upstream.name,
            config.sandbox.model,
            config.max_containers,
            config.poll_seconds,
            config.submissions_dir,
            "set" if config.budget else "none",
        )
        run(db=db, client=client, config=config, image_tag=image_tag, stop=stop)
        log.info("task-solver stopped")
    finally:
        db.close()
        client.close()


if __name__ == "__main__":
    main()
