"""Shared logging setup for the worker entry points."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Root logging at LOG_LEVEL (default INFO); quiet noisy third-party loggers.

    httpx/httpcore (one line per request) and sqlalchemy (SQL echo + ORM mapper
    config) stay at WARNING unless LOG_LEVEL=DEBUG. Importing the bittensor SDK
    raises the level of loggers that already exist to CRITICAL, so any `tau` logger
    created before the SDK is reset to inherit the root level again.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    quiet_level = logging.DEBUG if level == "DEBUG" else logging.WARNING
    for noisy in ("httpx", "httpcore", "sqlalchemy"):
        logging.getLogger(noisy).setLevel(quiet_level)
    for name in list(logging.root.manager.loggerDict):
        if name == "tau" or name.startswith("tau."):
            logging.getLogger(name).setLevel(logging.NOTSET)
