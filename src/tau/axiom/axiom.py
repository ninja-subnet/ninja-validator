"""Best-effort structured event ingestion to Axiom.co (optional observability).

A thin sink shared by all workers. It is deliberately forgiving:

* it stays disabled (a no-op client) unless both ``AXIOM_TOKEN`` and ``AXIOM_DATASET`` are set;
* ingestion NEVER raises (a telemetry backend must not take down a worker).

Use the process-wide :func:`get_axiom` singleton so the client is built once. Event
helpers take primitives (not domain objects), so this module stays decoupled from
``tau.db`` / ``tau.sandbox`` and import-cycle-free.

Config (env): ``AXIOM_TOKEN``, ``AXIOM_DATASET``, ``AXIOM_ENVIRON`` (free-form tag).
"""

from __future__ import annotations

import logging
import traceback
from functools import lru_cache
from typing import Any

from .axiom_client import AxiomClientInterface
from .env_factory import create_axiom_client_from_env
from .labels import EventType, Severity, Source

log = logging.getLogger(__name__)


class Axiom:
    """An Axiom event sink that no-ops unless fully configured; never raises on ingest."""

    def __init__(self, axiom_client: AxiomClientInterface) -> None:
        self._client = axiom_client

    def emit(
        self, severity: Severity, source: Source, event_type: EventType, **kwargs: Any
    ) -> None:
        """Log an event at an explicit *severity* (the base that info/warn/error wrap)."""
        self._client.ingest(
            severity=severity,
            source=source,
            event_type=event_type,
            details=kwargs,
        )

    def exception(
        self,
        source: Source,
        event_type: EventType,
        severity: Severity = Severity.WARNING,
        **kwargs: Any,
    ) -> None:
        self._client.ingest(
            severity=severity,
            source=source,
            event_type=event_type,
            details={
                "exception": traceback.format_exc(),
            }
            | kwargs,
        )

    def error(self, source: Source, event_type: EventType, **kwargs: Any) -> None:
        self._client.ingest(
            severity=Severity.ERROR,
            source=source,
            event_type=event_type,
            details=kwargs,
        )

    def warn(self, source: Source, event_type: EventType, **kwargs: Any) -> None:
        self._client.ingest(
            severity=Severity.WARNING,
            source=source,
            event_type=event_type,
            details=kwargs,
        )

    def info(self, source: Source, event_type: EventType, **kwargs: Any) -> None:
        self._client.ingest(
            severity=Severity.INFO,
            source=source,
            event_type=event_type,
            details=kwargs,
        )


@lru_cache(maxsize=1)
def get_axiom() -> Axiom:
    """The process-wide Axiom sink (constructed once, on first use)."""
    client = create_axiom_client_from_env()
    return Axiom(client)
