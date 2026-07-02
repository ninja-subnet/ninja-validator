import abc
import logging

import axiom_py

from .labels import Details, EventType, Severity, Source

log = logging.getLogger(__name__)


class AxiomClientInterface(abc.ABC):
    @abc.abstractmethod
    def ingest(
        self,
        source: Source,
        event_type: EventType,
        severity: Severity,
        details: Details,
    ) -> None: ...


class AxiomClientNoop(AxiomClientInterface):
    def ingest(
        self,
        source: Source,
        event_type: EventType,
        severity: Severity,
        details: Details,
    ) -> None:
        pass


class AxiomClient(AxiomClientInterface):
    def __init__(self, dataset: str, environ: str, token: str) -> None:
        self._dataset = dataset
        self._environ = environ
        self._client = axiom_py.Client(token)

    def ingest(
        self,
        source: Source,
        event_type: EventType,
        severity: Severity,
        details: Details,
    ) -> None:
        payload = {
            "source": source,
            "environ": self._environ,
            "event_type": event_type,
            "severity": severity,
            **details,
        }
        try:
            self._client.ingest_events(self._dataset, [payload])
        except Exception as exc:  # noqa: BLE001 — telemetry must not break a worker
            log.warning("Axiom ingest failed: %s", exc)
