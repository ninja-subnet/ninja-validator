"""Adapters that present the database as the seams the workers already expect.

The chain watcher (`tau.bittensor.worker.run`) takes a `SnapshotSink` and writes
through it, knowing nothing about the database. `DatabaseSnapshotSink` is the
production sink the watcher's docstring refers to as "left unwired": it satisfies
the `SnapshotSink` Protocol structurally and forwards each snapshot to a `Database`.
Wiring is a one-liner at the entry point; the worker itself is untouched.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from tau.bittensor.types import ChainHead, MetagraphSnapshot

from .interface import Database


class DatabaseSnapshotSink:
    """A `bittensor.sink.SnapshotSink` that persists each snapshot via a `Database`.

    `block_time` resolves a registration block height to its on-chain time; it is
    wired from the chain source (`BittensorChainSource.registration_block_time`) so the
    DB can stamp `block_date` without depending on the chain itself. The DB invokes it
    only for changed uids, so the (archive) lookup happens only on a real registration.
    """

    def __init__(self, db: Database, block_time: Callable[[int], dt.datetime]) -> None:
        self._db = db
        self._block_time = block_time

    def publish(self, head: ChainHead, metagraph: MetagraphSnapshot) -> None:
        self._db.publish_snapshot(head, metagraph, self._block_time)
