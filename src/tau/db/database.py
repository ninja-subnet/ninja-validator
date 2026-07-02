"""SQLAlchemy implementation of the `Database` middleware, plus the `connect()`
factory workers call to obtain one.

All writes use Postgres `INSERT ... ON CONFLICT` so the operations are idempotent
on their natural keys, matching the guarantees promised in `interface.py`. The
worker-facing methods accept and return the workers' own dataclasses; the mapping
to and from rows lives entirely here.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from tau.bittensor.types import ChainHead, MetagraphSnapshot

from . import models
from .engine import create_db_engine, session_factory, session_scope

logger = logging.getLogger(__name__)

# Log backfill progress every this many resolved timestamps. The empty-DB load resolves
# one archive lookup per changed uid (potentially the whole subnet, sequentially), so
# without progress lines the worker would look hung while it grinds through them.
_PROGRESS_EVERY = 25


def _changed_rows(
    latest: dict[int, tuple[str, str]],
    neurons,  # noqa: ANN001 — Iterable[bittensor.types.NeuronInfo]
) -> list[dict]:
    """Pure core of the registration history: rows to insert for changed neurons.

    `latest` maps uid -> last recorded `(ss58_hot, ss58_cold)`. A neuron yields a row
    only if it has no prior record or its hot/cold pair differs from `latest`; an
    unchanged neuron is skipped. Separated from the DB I/O so the change-detection
    can be tested without a database.

    Each row's `block` is the neuron's *registration* block height, not the snapshot
    block — so a row records when the uid actually registered on-chain even if the worker
    first saw it much later (after downtime or on the initial backfill). `block_date` is
    filled in by the caller, which resolves that height's wall-clock time (an archive
    lookup) only for these already-filtered changed rows.
    """
    return [
        {
            "uid": n.uid,
            "ss58_hot": n.hotkey,
            "ss58_cold": n.coldkey,
            "block": n.block_at_registration,
        }
        for n in neurons
        if latest.get(n.uid) != (n.hotkey, n.coldkey)
    ]


def _resolve_block_dates(
    rows: list[dict],
    block_time: Callable[[int], dt.datetime],
    *,
    scope: str,
) -> None:
    """Fill each changed row's `block_date` from its registration block height, in place.

    `block_time` turns a height into its on-chain time — a (cached) archive lookup. This
    runs only over the already-diffed changed rows, so a quiet tick does no lookups at
    all; an empty-DB backfill resolves one per uid sequentially, which is why it logs
    progress (every `_PROGRESS_EVERY`) — otherwise the worker looks hung mid-backfill.
    """
    total = len(rows)
    logger.info(
        "resolving %d registration timestamp(s) from archive [%s]", total, scope
    )
    for i, row in enumerate(rows, start=1):
        row["block_date"] = block_time(row["block"])
        if total >= _PROGRESS_EVERY and (i % _PROGRESS_EVERY == 0 or i == total):
            logger.info("  resolved %d/%d registration timestamps", i, total)


class SqlDatabase:
    """Concrete `Database` backed by a SQLAlchemy engine (satisfies the Protocol)."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_db_engine(url, echo=echo)
        self._sessions = session_factory(self._engine)

    # --- chain watcher seam --------------------------------------------------
    def publish_snapshot(
        self,
        head: ChainHead,
        metagraph: MetagraphSnapshot,
        block_time: Callable[[int], dt.datetime],
    ) -> None:
        """Append a registration row only for uids whose hot/cold key pair changed.

        The table is a *history* of key assignments, not a per-block dump: each tick
        we diff the snapshot against the latest recorded keys per uid and insert a
        row only where the `(ss58_hot, ss58_cold)` pair differs (or the uid is new).
        An unchanged metagraph — the common case, since blocks are ~12s apart and
        registrations are rare — produces zero writes. Stake is not part of the
        schema, so it is dropped.

        Each row's `block` is the neuron's own *registration* block height (see
        `NeuronInfo`), so a registration carries when the uid actually registered, not
        the block the watcher happened to see it at (which matters across missed blocks,
        restarts, and clean starts). `block_time` resolves that height to its on-chain
        wall-clock `block_date`; because it is called only for the already-diffed changed
        rows, the (archive) lookup happens solely when there is a new registration to
        record — never on an unchanged tick. `head` is the polled tip, kept for the sink
        contract. The DB stamps `inserted_at` itself at write time.
        """
        if not metagraph.neurons:
            logger.warning(
                "block %d: empty metagraph, nothing to publish", metagraph.block
            )
            return

        with session_scope(self._sessions) as session:
            latest = self._latest_keys(session)
            rows = _changed_rows(latest, metagraph.neurons)
            if not rows:
                logger.info(
                    "block %d: no registration changes (%d neurons)",
                    metagraph.block,
                    len(metagraph.neurons),
                )
                return
            scope = "initial load" if not latest else "update"
            _resolve_block_dates(rows, block_time, scope=scope)
            # do-nothing (not do-update) on the (uid, block) key: a conflict means
            # this block was already recorded for the uid, so the history stands.
            stmt = insert(models.Registration)
            session.execute(
                stmt.values(rows).on_conflict_do_nothing(
                    index_elements=["uid", "block"]
                )
            )
            logger.info(
                "block %d: recorded %d registration change(s) [%s] for uid(s) %s",
                metagraph.block,
                len(rows),
                scope,
                sorted(r["uid"] for r in rows),
            )

    @staticmethod
    def _latest_keys(session) -> dict[int, tuple[str, str]]:  # noqa: ANN001 — SQLAlchemy Session
        """Map each uid to its most recently recorded `(ss58_hot, ss58_cold)` pair.

        `DISTINCT ON (uid)` ordered by descending block keeps only the newest row per
        uid — the current known assignment to diff the incoming snapshot against.
        """
        stmt = (
            select(
                models.Registration.uid,
                models.Registration.ss58_hot,
                models.Registration.ss58_cold,
            )
            .distinct(models.Registration.uid)
            .order_by(models.Registration.uid, models.Registration.block.desc())
        )
        return {row.uid: (row.ss58_hot, row.ss58_cold) for row in session.execute(stmt)}

    def close(self) -> None:
        self._engine.dispose()


def connect(url: str | None = None, *, echo: bool = False) -> SqlDatabase:
    """Open a `Database` against `url` (defaults to the env-resolved URL).

    This is the entry point a worker's wiring calls, e.g.::

        from tau.db import connect
        from tau.db.adapters import DatabaseSnapshotSink
        from tau.bittensor.finney import BittensorChainSource
        from tau.bittensor.worker import run

        db = connect()
        run(BittensorChainSource(), DatabaseSnapshotSink(db))
    """
    return SqlDatabase(url, echo=echo)
