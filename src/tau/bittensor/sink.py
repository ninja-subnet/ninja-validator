"""Where the worker publishes the state it reads — the outbound seam.

The validator's workers communicate only through the database, so the worker does
not write rows itself: it hands each (head, metagraph) snapshot to a `SnapshotSink`.
The production DB sink — `tau.db.adapters.DatabaseSnapshotSink`, which appends a
`registrations` row only for uids whose hot/cold keys changed (see
`tau.db.models.Registration`) — satisfies this Protocol structurally, the same way
the judge leaves its LLM transport injectable.
`LoggingSink` is a trivial implementation so the worker can be exercised end-to-end
without a database.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from .types import ChainHead, MetagraphSnapshot

logger = logging.getLogger(__name__)


@runtime_checkable
class SnapshotSink(Protocol):
    """A destination for chain snapshots (in production: the database)."""

    def publish(self, head: ChainHead, metagraph: MetagraphSnapshot) -> None:
        """Persist one block + its metagraph snapshot. Must be idempotent per block."""
        ...


class LoggingSink:
    """A no-storage sink that just logs each snapshot — for local runs and tests."""

    def publish(self, head: ChainHead, metagraph: MetagraphSnapshot) -> None:
        logger.info(
            "block=%d time=%s netuid=%d neurons=%d",
            head.block,
            head.timestamp.isoformat(),
            metagraph.netuid,
            len(metagraph.neurons),
        )
