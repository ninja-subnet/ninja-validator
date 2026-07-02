"""The chain-watcher worker: poll a `ChainSource`, publish each new block's state.

Like the judge, the orchestration logic here is pure and the IO is injected — the
worker holds a `ChainSource` (where state comes from) and a `SnapshotSink` (where it
goes), and depends on neither the bittensor SDK nor the database directly. `step()`
is the testable unit: advance-or-skip for one tick. `run()` is the loop around it.

Run it against the live chain with:

    from .finney import BittensorChainSource
    from .sink import LoggingSink
    run(BittensorChainSource(), LoggingSink())

and swap either argument for a different source (a DB replay, a fake) or a real
database sink (`tau.db.adapters.DatabaseSnapshotSink`) without touching this file.
The wired-up production entry point lives in `tau.workers.chain_watcher`.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .sink import SnapshotSink
from .source import ChainSource
from .types import NETUID, POLL_INTERVAL_SECONDS

logger = logging.getLogger(__name__)


def step(
    source: ChainSource,
    sink: SnapshotSink,
    netuid: int,
    last_block: int | None,
) -> int | None:
    """Process one poll: if the tip advanced past `last_block`, snapshot and publish.

    Returns the block now considered last-seen (the new tip if it advanced, else
    `last_block` unchanged). Pure with respect to `source`/`sink` — no sleeping, no
    looping — so it can be driven directly in tests with fakes.
    """
    head = source.head()
    if last_block is not None and head.block <= last_block:
        logger.debug("tip unchanged: last_block=%s head.block=%s", last_block, head.block)
        return last_block

    metagraph = source.metagraph(netuid, head.block)
    logger.debug("tip advanced: last_block=%s head.block=%s", last_block, head.block)
    sink.publish(head, metagraph)
    logger.info("snapshot published: block=%s netuid=%d", head.block, netuid)
    return head.block


def run(
    source: ChainSource,
    sink: SnapshotSink,
    *,
    netuid: int = NETUID,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll `source` forever, publishing each newly seen block to `sink`.

    A transient query error is logged and retried on the next tick rather than
    killing the worker — a single bad block must never stop the watcher. Stops
    cleanly on KeyboardInterrupt. `sleep` is injected so tests can run the loop
    deterministically or break out of it.
    """
    last_block: int | None = None
    try:
        while True:
            try:
                last_block = step(source, sink, netuid, last_block)
                logger.debug("poll complete: last_block=%s", last_block)
            except Exception as exc:  # noqa: BLE001 — one bad tick must not kill the worker
                logger.exception("poll failed: %s", exc)
            sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
