from __future__ import annotations

import datetime as dt


from tau.bittensor import ChainHead, MetagraphSnapshot, NeuronInfo, run, step
from tau.bittensor.finney import BittensorChainSource, _to_snapshot
from tau.db.adapters import DatabaseSnapshotSink
from tau.db.database import _changed_rows, _resolve_block_dates

NETUID = 66

# A metagraph's on-chain time is deliberately a different day from the head's, so a
# test can tell "stamped from the metagraph" apart from "stamped from the head".
SNAP_TIME = dt.datetime(2026, 6, 22, 12, 0, tzinfo=dt.UTC)


def _head(block: int) -> ChainHead:
    return ChainHead(block=block, timestamp=dt.datetime(2026, 6, 19, tzinfo=dt.UTC))


def _snapshot(block: int, n: int = 2) -> MetagraphSnapshot:
    neurons = tuple(
        NeuronInfo(
            uid=i,
            hotkey=f"hot{i}",
            coldkey=f"cold{i}",
            stake=float(i),
            block_at_registration=block - i,
        )
        for i in range(n)
    )
    return MetagraphSnapshot(netuid=NETUID, block=block, timestamp=SNAP_TIME, neurons=neurons)


class RecordingSink:
    """A SnapshotSink that just remembers what it was handed."""

    def __init__(self) -> None:
        self.published: list[tuple[ChainHead, MetagraphSnapshot]] = []

    def publish(self, head: ChainHead, metagraph: MetagraphSnapshot) -> None:
        self.published.append((head, metagraph))


class ScriptedSource:
    """A ChainSource that yields a queued sequence of heads.

    A queued item may be an Exception, which `head()` raises — used to prove the
    loop survives a bad tick. `metagraph()` always returns a snapshot for the block.
    """

    def __init__(self, heads: list[ChainHead | Exception]) -> None:
        self._heads = list(heads)
        self.metagraph_calls: list[tuple[int, int | None]] = []

    def head(self) -> ChainHead:
        item = self._heads.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def metagraph(self, netuid: int, block: int | None = None) -> MetagraphSnapshot:
        self.metagraph_calls.append((netuid, block))
        return _snapshot(block or 0)


class StopAfter:
    """A `sleep` stand-in that breaks `run`'s loop after `n` ticks."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self, _seconds: float) -> None:
        self.calls += 1
        if self.calls >= self.n:
            raise KeyboardInterrupt


# --- step() ------------------------------------------------------------------


def test_step_publishes_and_returns_new_tip_when_block_advances() -> None:
    source = ScriptedSource([_head(5)])
    sink = RecordingSink()

    new_last = step(source, sink, NETUID, last_block=None)

    assert new_last == 5
    assert source.metagraph_calls == [(NETUID, 5)]
    assert len(sink.published) == 1
    head, snapshot = sink.published[0]
    assert head.block == 5
    assert snapshot.block == 5


def test_step_skips_when_tip_has_not_advanced() -> None:
    source = ScriptedSource([_head(5)])
    sink = RecordingSink()

    new_last = step(source, sink, NETUID, last_block=5)

    assert new_last == 5
    assert source.metagraph_calls == []  # never queried the metagraph
    assert sink.published == []


# --- run() -------------------------------------------------------------------


def test_run_publishes_each_new_block_and_skips_repeats() -> None:
    # tick 1: block 5 (new) -> publish; tick 2: block 5 (repeat) -> skip;
    # tick 3: block 6 (new) -> publish; then sleep breaks the loop.
    source = ScriptedSource([_head(5), _head(5), _head(6)])
    sink = RecordingSink()

    run(source, sink, netuid=NETUID, poll_interval=0, sleep=StopAfter(3))

    assert [head.block for head, _ in sink.published] == [5, 6]


def test_run_survives_a_failing_tick_and_continues() -> None:
    source = ScriptedSource([_head(5), RuntimeError("boom"), _head(6)])
    sink = RecordingSink()

    run(source, sink, netuid=NETUID, poll_interval=0, sleep=StopAfter(3))

    # The error tick published nothing, but the watcher kept going.
    assert [head.block for head, _ in sink.published] == [5, 6]


# --- DatabaseSnapshotSink ----------------------------------------------------


class FakeDatabase:
    def __init__(self) -> None:
        self.snapshots: list[tuple[ChainHead, MetagraphSnapshot, object]] = []

    def publish_snapshot(self, head: ChainHead, metagraph: MetagraphSnapshot, block_time) -> None:
        self.snapshots.append((head, metagraph, block_time))


def test_database_snapshot_sink_forwards_snapshot_and_resolver() -> None:
    db = FakeDatabase()
    resolver = REG_TIMES.__getitem__  # any Callable[[int], datetime]
    sink = DatabaseSnapshotSink(db, resolver)
    head, snapshot = _head(7), _snapshot(7)

    sink.publish(head, snapshot)

    # The sink forwards both the snapshot and the block-time resolver to the DB.
    assert db.snapshots == [(head, snapshot, resolver)]


# --- _changed_rows (registration-history change detection) -------------------


def _neuron(uid: int, hot: str, cold: str, *, reg_block: int = 100) -> NeuronInfo:
    return NeuronInfo(
        uid=uid,
        hotkey=hot,
        coldkey=cold,
        stake=0.0,
        block_at_registration=reg_block,
    )


def test_changed_rows_inserts_everything_when_no_prior_history() -> None:
    # Distinct registration blocks prove each row carries its own neuron's reg block,
    # not a single snapshot block shared across the batch.
    neurons = [
        _neuron(0, "hkA", "ckA", reg_block=100),
        _neuron(1, "hkB", "ckB", reg_block=150),
    ]

    rows = _changed_rows({}, neurons)

    assert [(r["uid"], r["ss58_hot"], r["ss58_cold"]) for r in rows] == [
        (0, "hkA", "ckA"),
        (1, "hkB", "ckB"),
    ]
    assert [r["block"] for r in rows] == [100, 150]
    # block_date is resolved later (per-height archive lookup), not here; inserted_at
    # is the DB's job (server_default now()). Neither is a value _changed_rows supplies.
    assert all("block_date" not in r and "inserted_at" not in r for r in rows)


def test_changed_rows_skips_unchanged_neurons() -> None:
    latest = {0: ("hkA", "ckA"), 1: ("hkB", "ckB")}
    neurons = [_neuron(0, "hkA", "ckA"), _neuron(1, "hkB", "ckB")]

    rows = _changed_rows(latest, neurons)

    assert rows == []  # identical metagraph -> no writes


def test_changed_rows_detects_hot_cold_and_new_uid_changes() -> None:
    latest = {0: ("hkA", "ckA"), 1: ("hkB", "ckB"), 2: ("hkC", "ckC")}
    neurons = [
        _neuron(0, "hkA", "ckA"),  # unchanged -> skipped
        _neuron(1, "hkB2", "ckB"),  # hotkey changed -> row
        _neuron(2, "hkC", "ckC2"),  # coldkey changed -> row
        _neuron(3, "hkD", "ckD"),  # brand-new uid -> row
    ]

    rows = _changed_rows(latest, neurons)

    assert sorted(r["uid"] for r in rows) == [1, 2, 3]


def test_resolve_block_dates_fills_each_row_from_its_block_height() -> None:
    rows = [{"uid": 0, "block": 100}, {"uid": 1, "block": 150}]
    calls: list[int] = []

    def block_time(block: int) -> dt.datetime:
        calls.append(block)
        return dt.datetime.fromtimestamp(block, tz=dt.UTC)

    _resolve_block_dates(rows, block_time, scope="initial load")

    # One lookup per changed row, keyed by that row's own registration height.
    assert calls == [100, 150]
    assert rows[0]["block_date"] == dt.datetime.fromtimestamp(100, tz=dt.UTC)
    assert rows[1]["block_date"] == dt.datetime.fromtimestamp(150, tz=dt.UTC)


# --- finney SDK->contract mapping (pure, no live node) -----------------------


class _Scalar:
    """Stands in for a 0-dim tensor: `.item()` unwraps the Python scalar."""

    def __init__(self, value) -> None:
        self._value = value

    def item(self):
        return self._value


class _FakeMeta:
    n = _Scalar(2)
    block = _Scalar(123)
    uids = [0, 1]
    hotkeys = ["hkA", "hkB"]
    coldkeys = ["ckA", "ckB"]
    stake = [1.5, 2.5]
    block_at_registration = [50, 60]


# Distinct registration-block times, used as a block-time resolver in the tests below.
REG_TIMES = {50: dt.datetime(2026, 1, 2, tzinfo=dt.UTC), 60: dt.datetime(2026, 1, 3, tzinfo=dt.UTC)}


def test_to_snapshot_maps_sdk_meta_and_carries_registration_heights() -> None:
    snapshot = _to_snapshot(NETUID, _FakeMeta(), SNAP_TIME)

    assert snapshot.netuid == NETUID
    assert snapshot.block == 123  # from meta.block, the block the data reflects
    assert snapshot.timestamp == SNAP_TIME
    # Registration heights are read straight from the payload; no time lookup happens here.
    assert snapshot.neurons == (
        NeuronInfo(uid=0, hotkey="hkA", coldkey="ckA", stake=1.5, block_at_registration=50),
        NeuronInfo(uid=1, hotkey="hkB", coldkey="ckB", stake=2.5, block_at_registration=60),
    )


def test_to_snapshot_falls_back_to_snapshot_block_when_reg_blocks_absent() -> None:
    class _NoRegMeta(_FakeMeta):
        block_at_registration = []  # degraded payload: SDK gave us no reg blocks

    snapshot = _to_snapshot(NETUID, _NoRegMeta(), SNAP_TIME)

    # Each neuron falls back to the snapshot block (123) rather than crashing.
    assert [n.block_at_registration for n in snapshot.neurons] == [123, 123]


# --- BittensorChainSource (which node each block-time lookup hits) ------------


class _Value:
    """Stands in for a substrate storage query result: `.value` is the raw scalar."""

    def __init__(self, value) -> None:
        self.value = value


class _FakeSubtensor:
    """Minimal fake Subtensor that records the blocks it was asked the time for.

    A block's on-chain time is faked as `block` seconds past the epoch, so a resolved
    `registration_date` reveals which block (and thus which node) it came from.
    """

    def __init__(self, *, current_block: int = 0, meta=None) -> None:
        self._current_block = current_block
        self._meta = meta
        self.time_queries: list[int] = []
        self.substrate = self  # subtensor.substrate.query(...) lands back here

    def get_current_block(self) -> int:
        return self._current_block

    def get_block_hash(self, block: int) -> int:
        return block  # use the block number itself as its "hash"

    def query(self, _module: str, _name: str, block_hash: int) -> _Value:
        self.time_queries.append(block_hash)
        return _Value(block_hash * 1000)  # Timestamp.Now is milliseconds

    def metagraph(self, netuid: int, block: int | None = None):  # noqa: ARG002
        return self._meta


def test_source_metagraph_carries_registration_heights_without_archive() -> None:
    lite = _FakeSubtensor(current_block=123, meta=_FakeMeta())
    archive = _FakeSubtensor()
    source = BittensorChainSource(subtensor=lite, archive=archive)

    snapshot = source.metagraph(NETUID, 123)

    # Heights come straight from the metagraph payload — the archive is never touched,
    # and only the recent snapshot block's time is read, from the lite node.
    assert [n.block_at_registration for n in snapshot.neurons] == [50, 60]
    assert archive.time_queries == []
    assert lite.time_queries == [123]


def test_source_registration_block_time_resolves_from_archive_and_caches() -> None:
    lite = _FakeSubtensor(current_block=123, meta=_FakeMeta())
    archive = _FakeSubtensor()
    source = BittensorChainSource(subtensor=lite, archive=archive)

    first = source.registration_block_time(50)
    again = source.registration_block_time(50)

    # Block 50's time comes from the archive (50 -> 50_000 ms -> 50s past epoch)...
    assert first == dt.datetime.fromtimestamp(50, tz=dt.UTC)
    assert again == first
    # ...and is cached: two calls, one archive RPC, lite node never consulted.
    assert archive.time_queries == [50]
    assert lite.time_queries == []


def test_source_head_uses_lite_node_only() -> None:
    lite = _FakeSubtensor(current_block=200)
    archive = _FakeSubtensor()
    source = BittensorChainSource(subtensor=lite, archive=archive)

    head = source.head()

    assert head.block == 200
    assert head.timestamp == dt.datetime.fromtimestamp(200, tz=dt.UTC)
    assert archive.time_queries == []  # head is recent; archive untouched
