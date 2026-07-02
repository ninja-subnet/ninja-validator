"""Data contracts for chain state — pure, no IO, no SDK types.

These are the immutable snapshots a `source.ChainSource` produces and the worker
publishes. They are deliberately framed in plain Python (no bittensor / substrate
objects) so the worker and the DB sink never depend on how the data was fetched —
that is what makes one source interchangeable with another (see source.py).
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt

# Subnet 66 on the Bittensor mainnet, polled from the Finney network by default.
NETUID = 66
FINNEY = "finney"
# A neuron's registration block is usually older than a lite node's ~300-block state
# window, so its on-chain time is read from an archive node (full history) instead.
ARCHIVE = "archive"

# Blocks are ~12s apart; polling at half that comfortably catches every head.
BLOCK_SECONDS = 12
POLL_INTERVAL_SECONDS = BLOCK_SECONDS / 2


@dc.dataclass(frozen=True, slots=True)
class ChainHead:
    """The current chain tip: its block number and wall-clock time (UTC)."""

    block: int
    timestamp: dt.datetime


@dc.dataclass(frozen=True, slots=True)
class NeuronInfo:
    """One neuron registered on the subnet.

    `block_at_registration` is *when this uid was registered* (the on-chain
    `BlockAtRegistration` height for the uid) — not the block of the snapshot we read it
    from. Stamping a registration with its own block, rather than the block we happened
    to first observe it, keeps the recorded block correct across worker downtime and the
    initial backfill, when the snapshot block can be far ahead of when the neuron
    actually registered. The height comes straight from the metagraph (no chain lookup);
    its wall-clock time is resolved separately and only for uids that actually change
    (see `BittensorChainSource.registration_block_time` and `Database.publish_snapshot`).
    """

    uid: int
    hotkey: str
    coldkey: str
    stake: float
    block_at_registration: int


@dc.dataclass(frozen=True, slots=True)
class MetagraphSnapshot:
    """The subnet's uid -> hot/cold key assignments captured at a single block.

    `block` and `timestamp` describe the block the metagraph data actually reflects
    (queried at that block's hash), not the head the worker happened to poll — so a
    registration is stamped with when it was true on-chain even if the watcher saw it
    late, restarted, or started clean.
    """

    netuid: int
    block: int
    timestamp: dt.datetime
    neurons: tuple[NeuronInfo, ...]
