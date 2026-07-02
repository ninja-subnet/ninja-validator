"""Chain-watcher worker package: poll subnet 66 and publish registration snapshots.

The package mirrors the rest of the codebase's seam-first shape: a pure worker loop
(`worker.run`) wired between two injectable boundaries — a `ChainSource` (where chain
state comes from) and a `SnapshotSink` (where it goes). The live source is
`BittensorChainSource`; the production sink is `tau.db.adapters.DatabaseSnapshotSink`.
"""

from .sink import LoggingSink, SnapshotSink
from .source import ChainSource
from .types import (
    BLOCK_SECONDS,
    FINNEY,
    NETUID,
    POLL_INTERVAL_SECONDS,
    ChainHead,
    MetagraphSnapshot,
    NeuronInfo,
)
from .worker import run, step

__all__ = [
    "BittensorChainSource",
    "ChainSource",
    "SnapshotSink",
    "LoggingSink",
    "ChainHead",
    "MetagraphSnapshot",
    "NeuronInfo",
    "NETUID",
    "FINNEY",
    "BLOCK_SECONDS",
    "POLL_INTERVAL_SECONDS",
    "run",
    "step",
]


def __getattr__(name: str) -> object:
    if name == "BittensorChainSource":
        from .finney import BittensorChainSource
        return BittensorChainSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
