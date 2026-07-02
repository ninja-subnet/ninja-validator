"""Database Protocol for the chain watcher -- a small `runtime_checkable` seam so
the SQLAlchemy implementation in `database.py` and test fakes are interchangeable.

The task-generator and judge workers use their own focused seams
(`generator.GeneratorDb`, `judge.JudgeDb`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from tau.bittensor.types import ChainHead, MetagraphSnapshot


@runtime_checkable
class Database(Protocol):
    """The persistence operations the chain watcher needs.

    Implementations must be safe to call repeatedly: writes are idempotent on the
    natural keys described per method, so a worker that retries a tick or replays
    a block never creates duplicates.
    """

    # --- chain watcher seam --------------------------------------------------
    def publish_snapshot(
        self,
        head: ChainHead,
        metagraph: MetagraphSnapshot,
        block_time: Callable[[int], dt.datetime],
    ) -> None:
        """Append registration history for uids whose hot/cold key pair changed.

        The snapshot is diffed against the latest recorded keys per uid; a row is
        written only where the `(ss58_hot, ss58_cold)` pair is new or changed, so an
        unchanged metagraph (the common case) produces no writes and the table stays
        a compact history rather than a per-block dump. Idempotent per block: a
        re-seen block conflicts on its `(uid, block)` key and is left untouched. The
        `MetagraphSnapshot` carries per-neuron stake, but the schema stores only
        `uid -> hot/cold key`, so stake is intentionally dropped here.

        `block_time` maps a registration block height to its on-chain wall-clock time;
        it is invoked only for the changed rows, so the (potentially slow archive)
        lookup happens solely when there is a new registration to timestamp.
        """
        ...

    # --- judge worker seam ---------------------------------------------------
    def pending_judge_requests(self) -> list[JudgeRequest]:
        """Return the king/challenger solution pairs still awaiting a judgment.

        A pair is pending when a challenge ties a challenger to a king, the king
        owns a task, both sides have a `task_solutions` row for that task, and no
        `judgements` row records the outcome yet.
        """
        ...

    def save_judgment(
        self,
        task: Task,
        king_solution: Solution,
        challenger_solution: Solution,
        judgment: Judgment,
    ) -> None:
        """Persist a completed judgment for one (task, king, challenger) triple.

        Idempotent on the judgment's composite key — re-judging overwrites the row.
        The `(task, king_solution, challenger_solution)` triple is exactly the
        `JudgeRequest` the worker consumed, so the caller already has it in hand.
        """
        ...

    def close(self) -> None:
        """Release the underlying connection pool. Safe to call more than once."""
        ...
