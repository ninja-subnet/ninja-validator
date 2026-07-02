"""Chain-watcher worker entry point: stream subnet 66 registrations into the DB.

This is the production wiring the `tau.bittensor` package leaves injectable. It
reads the live Finney chain through a `BittensorChainSource` and persists each new
block's metagraph as `registrations` rows via a `DatabaseSnapshotSink` backed by the
env-resolved `Database`. The polling loop, the advance-or-skip logic, and the
per-tick error handling all live in `tau.bittensor.worker.run`; this module only
constructs the source/sink pair and hands them to it.

Run as the `chain-watcher` console script (see `[project.scripts]`); the deploy
image launches it with `EXTRA=bittensor` so the SDK is present.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from tau.bittensor.finney import BittensorChainSource
from tau.bittensor.types import ARCHIVE, FINNEY, NETUID, POLL_INTERVAL_SECONDS
from tau.bittensor.worker import run
from tau.db import DatabaseSnapshotSink, connect

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChainWatcherConfig:
    network: str = FINNEY
    # Archive node for historical block times (registration blocks are usually older
    # than the lite node's pruned-state window). See BittensorChainSource.
    archive_network: str = ARCHIVE
    netuid: int = NETUID
    poll_interval: float = POLL_INTERVAL_SECONDS


def _config_from_env() -> ChainWatcherConfig:
    """Build the worker config, letting the environment override the defaults."""
    return ChainWatcherConfig(
        network=os.getenv("BITTENSOR_NETWORK", FINNEY),
        archive_network=os.getenv("BITTENSOR_ARCHIVE_NETWORK", ARCHIVE),
        netuid=int(os.getenv("NETUID", NETUID)),
        poll_interval=float(os.getenv("POLL_INTERVAL_SECONDS", POLL_INTERVAL_SECONDS)),
    )


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = _config_from_env()

    db = connect()
    try:
        source = BittensorChainSource(
            network=config.network, archive_network=config.archive_network
        )
        # The DB stamps each registration's block_date by resolving its block height's
        # on-chain time through the source's (archive-backed, cached) resolver — invoked
        # only for changed uids, so the archive is hit only on a real registration.
        sink = DatabaseSnapshotSink(db, source.registration_block_time)
        log.info(
            "chain-watcher starting network=%s archive=%s netuid=%d poll=%.1fs",
            config.network,
            config.archive_network,
            config.netuid,
            config.poll_interval,
        )
        run(source, sink, netuid=config.netuid, poll_interval=config.poll_interval)
    finally:
        db.close()


if __name__ == "__main__":
    main()
