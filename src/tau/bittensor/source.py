"""The interface for receiving chain state — the worker's only view of a data source.

`ChainSource` is the seam that makes the data source interchangeable: the worker
(worker.py) is written entirely against this protocol, never against the bittensor
SDK. The live implementation is `finney.BittensorChainSource`, but any object with
these two methods works just as well — a replay reading historical snapshots out of
the database, a recorded fixture, or an in-memory fake in tests.

Implementations return the plain data contracts from types.py; the SDK / substrate
types stay behind this boundary.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import ChainHead, MetagraphSnapshot


@runtime_checkable
class ChainSource(Protocol):
    """A read-only source of subnet chain state."""

    def head(self) -> ChainHead:
        """Return the current chain tip (latest block number + its timestamp)."""
        ...

    def metagraph(self, netuid: int, block: int | None = None) -> MetagraphSnapshot:
        """Return the metagraph snapshot for `netuid`, at `block` if given else the tip."""
        ...
