"""Live `ChainSource` backed by the bittensor SDK / substrate — the only file here
that touches the network or the SDK.

Ported from the reference loader (test_s66_loading.py): connect with
`bt.Subtensor(network=...)`, read the tip with `get_current_block()` +
`get_block_hash()` + the `Timestamp.Now` storage query, and read the metagraph with
`subtensor.metagraph(netuid, block)`. SDK objects (tensors, the substrate handle)
never escape this module — callers get the plain types from types.py, which is what
lets the worker treat this as just one of several interchangeable sources.
"""
from __future__ import annotations

import datetime as dt
import logging

from .types import ARCHIVE, FINNEY, ChainHead, MetagraphSnapshot, NeuronInfo

import bittensor as bt

logger = logging.getLogger(__name__)


def _to_snapshot(
    netuid: int,
    meta,  # noqa: ANN001 — meta is a bittensor metagraph
    timestamp: dt.datetime,
) -> MetagraphSnapshot:
    """Map an SDK metagraph to the plain `MetagraphSnapshot`, stamped with `timestamp`.

    Each neuron carries its registration block *height* (`meta.block_at_registration[i]`,
    the uid's on-chain `BlockAtRegistration`) — read straight from the already-fetched
    payload, no chain lookup. Its wall-clock time is resolved later and only for uids
    that actually change (see `BittensorChainSource.registration_block_time`). A uid whose
    registration block is missing from the SDK payload falls back to the snapshot block.

    Pure: it takes the already-fetched SDK object, so the SDK->contract mapping is
    testable without a live node.
    """
    count = meta.n.item()
    snapshot_block = meta.block.item()
    # block_at_registration is a list[int] indexed by uid (see bittensor MetagraphInfo);
    # guard against it being absent so a degraded payload can't crash the watcher.
    reg_blocks = getattr(meta, "block_at_registration", None) or []
    neurons = tuple(
        NeuronInfo(
            uid=int(meta.uids[i]),
            hotkey=str(meta.hotkeys[i]),
            coldkey=str(meta.coldkeys[i]),
            stake=float(meta.stake[i]),
            block_at_registration=int(reg_blocks[i]) if i < len(reg_blocks) else snapshot_block,
        )
        for i in range(count)
    )
    return MetagraphSnapshot(
        netuid=netuid, block=snapshot_block, timestamp=timestamp, neurons=neurons
    )


class BittensorChainSource:
    """Reads subnet chain state from a live Bittensor node.

    `network` is anything `bt.Subtensor` accepts — a named network ("finney",
    "test", "local") or a node URL ("ws://localhost:9944"). Defaults to Finney.

    A neuron's registration block is usually older than the lite node's pruned-state
    window, so its on-chain time can't be read from `network`. Those lookups go to a
    separate `archive_network` node (full history), connected lazily on first use and
    memoised — a block's time never changes, so each distinct registration block costs
    one archive RPC for the worker's lifetime. Pass `subtensor` / `archive` to inject
    already-connected (or fake) clients for tests.
    """

    def __init__(
        self,
        network: str = FINNEY,
        *,
        archive_network: str = ARCHIVE,
        subtensor: bt.Subtensor | None = None,
        archive: bt.Subtensor | None = None,
    ) -> None:
        self._network = network
        self._archive_network = archive_network
        self._subtensor = subtensor or bt.Subtensor(network=network)
        logger.info("Connected to %s", network)
        # Archive connection is deferred until a historical block time is needed, so a
        # worker that never sees a registration never pays for it.
        self._archive = archive
        self._block_time_cache: dict[int, dt.datetime] = {}

    def _archive_subtensor(self) -> bt.Subtensor:
        """The archive client for historical block times, connected on first use."""
        if self._archive is None:
            self._archive = bt.Subtensor(network=self._archive_network)
            logger.info("Connected to archive %s for historical block times", self._archive_network)
        return self._archive

    @staticmethod
    def _query_block_time(subtensor: bt.Subtensor, block: int) -> dt.datetime:
        """Wall-clock time (UTC) of `block`, from its `Timestamp.Now` storage value."""
        block_hash = subtensor.get_block_hash(block)
        # On-chain time is milliseconds since the epoch (Timestamp.Now storage).
        raw_ms = subtensor.substrate.query("Timestamp", "Now", block_hash=block_hash).value
        return dt.datetime.fromtimestamp(int(raw_ms) / 1000, tz=dt.UTC)

    def registration_block_time(self, block: int) -> dt.datetime:
        """On-chain time of a registration `block`, via the archive node, memoised.

        The DB calls this only for uids whose registration changed (a handful in steady
        state, the whole subnet once on an empty-DB backfill), so the archive is touched
        only when there is a new registration to timestamp — never on an unchanged tick.
        A block's time never changes, so it is cached: the cache stays bounded by the
        number of distinct registrations, not the tick count.
        """
        cached = self._block_time_cache.get(block)
        if cached is None:
            cached = self._query_block_time(self._archive_subtensor(), block)
            self._block_time_cache[block] = cached
        return cached

    def head(self) -> ChainHead:
        block = self._subtensor.get_current_block()
        # The head is recent, so its time is on the lite node — no archive needed.
        return ChainHead(block=block, timestamp=self._query_block_time(self._subtensor, block))

    def metagraph(self, netuid: int, block: int | None = None) -> MetagraphSnapshot:
        meta = self._subtensor.metagraph(netuid=netuid, block=block)
        # Stamp the snapshot with the (recent) block it actually reflects, via the lite
        # node. Per-neuron registration *heights* come from the metagraph payload itself
        # (no lookup); their times are resolved later, only for changed uids.
        snapshot = _to_snapshot(
            netuid, meta, self._query_block_time(self._subtensor, meta.block.item())
        )
        logger.debug(
            "fetched metagraph at block %d: %d neurons", snapshot.block, len(snapshot.neurons)
        )
        return snapshot
