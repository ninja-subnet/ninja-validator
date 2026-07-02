"""WorkQueue: the fetch/describe queues plus in-flight accounting for the worker.

One object the pipeline coroutines drive so they stay thin. A unit of work is a
FetchRequest; the method names below are its lifecycle.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from tau.db import PoolDeficit
from tau.db.status import PoolType
from tau.github import CommitCandidate, RejectReason


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """A request to produce one task for (king_id, pool)."""

    king_id: str
    pool: PoolType


@dataclass(frozen=True, slots=True)
class FetchedCommit:
    """A sampled commit waiting to be described: its request plus sampling telemetry."""

    request: FetchRequest
    candidate: CommitCandidate
    fetch_seconds: float = 0.0
    rejections: Counter[RejectReason] = field(default_factory=Counter)
    # Dedup key the fetcher reserved before admitting this commit; released when
    # the describe unit settles or is dropped.
    fingerprint: str = ""


class WorkQueue:
    """In-flight accounting, cancellation policy, and the fetch/describe queues."""

    def __init__(self, describe_capacity: int) -> None:
        self.current_king: str | None = None
        self._in_flight: dict[tuple[str, PoolType], int] = defaultdict(int)
        self._cancel: dict[tuple[str, PoolType], int] = defaultdict(int)
        # Fingerprints of commits currently being fetched/described, so the same
        # commit (e.g. mined from two forks) is not described by two units at once.
        self._inflight_fingerprints: set[str] = set()
        # Both queues are priority queues keyed by pool: POOL_ONE (1) drains before
        # POOL_TWO (2), so the primary pool always fills first -- including a
        # re-queued retry, which jumps ahead of any waiting retest-pool work. The
        # sequence counter is a tiebreaker that preserves FIFO order within a pool
        # and avoids comparing the (unorderable) payloads on a priority tie.
        self._seq = itertools.count()
        self._to_fetch: asyncio.PriorityQueue[tuple[int, int, FetchRequest]] = (
            asyncio.PriorityQueue()
        )
        # Bounded, so the fetcher waits (back-pressure) instead of out-running the
        # describers.
        self._to_describe: asyncio.PriorityQueue[tuple[int, int, FetchedCommit]] = (
            asyncio.PriorityQueue(maxsize=describe_capacity)
        )

    def _priority(self, req: FetchRequest) -> tuple[int, int]:
        """Queue ordering key: by pool (POOL_ONE first), then insertion order."""
        return (int(req.pool), next(self._seq))

    # -- controller ----------------------------------------------------------------

    def sync_to_deficits(self, deficits: list[PoolDeficit]) -> int:
        """Bring in-flight work in line with the latest per-pool deficits.

        Launches fetch requests for under-target pools and queues cancellations for
        over-target ones; returns the number launched.
        """
        # The reigning king is whoever this poll is about; empty deficits (no king,
        # or all pools full) clear it, marking every in-flight request unwanted.
        self.current_king = deficits[0].king_id if deficits else None
        king = self.current_king
        if king is None:
            return 0
        # Still-needed task count per pool from this poll (0 for pools not listed).
        needed = {d.pool: d.deficit for d in deficits if d.king_id == king}
        launched = 0
        for pool in PoolType:
            # How many more (+) or fewer (-) requests we want in flight than we have.
            delta = needed.get(pool, 0) - self._in_flight[(king, pool)]
            if delta > 0:
                # Under target: count the new requests as in flight and queue them.
                self._in_flight[(king, pool)] += delta
                req = FetchRequest(king, pool)
                for _ in range(delta):
                    self._to_fetch.put_nowait((*self._priority(req), req))
                launched += delta
                self._cancel[(king, pool)] = 0  # nothing to cancel
            else:
                # On or over target: queue the surplus for cancellation (0 if exact),
                # spent by _take_cancel as requests are pulled.
                self._cancel[(king, pool)] = -delta
        return launched

    # -- fetcher -------------------------------------------------------------------

    async def next_to_fetch(self) -> FetchRequest:
        """Block until a wanted request is ready; stale/cancelled ones drop here."""
        while True:
            _, _, req = await self._to_fetch.get()
            if self._is_stale(req) or self._take_cancel(req):
                self._finish(req)  # unwanted: free its slot and take the next
                continue
            return req

    async def submit_fetched(
        self,
        req: FetchRequest,
        candidate: CommitCandidate,
        *,
        fingerprint: str = "",
        fetch_seconds: float = 0.0,
        rejections: Counter[RejectReason] | None = None,
    ) -> None:
        """Hand a fetched commit to the describers (waits when they are saturated)."""
        fetched = FetchedCommit(
            req, candidate, fetch_seconds, rejections or Counter(), fingerprint
        )
        await self._to_describe.put((*self._priority(req), fetched))

    def fetch_failed(self, req: FetchRequest) -> None:
        """Keep a request alive after a failed fetch, to retry on a later pass."""
        self._to_fetch.put_nowait((*self._priority(req), req))

    # -- describer -----------------------------------------------------------------

    @asynccontextmanager
    async def lease_to_describe(self) -> AsyncIterator[FetchedCommit]:
        """Yield the next wanted fetched commit, freeing its fingerprint on exit.

        The reservation lifetime is the ``async with`` scope, so the describer
        cannot forget to release it; the slot fate (done/requeue) stays explicit.
        """
        fetched = await self._next_to_describe()
        try:
            yield fetched
        finally:
            self.release_fingerprint(fetched.fingerprint)

    async def _next_to_describe(self) -> FetchedCommit:
        """Block until a wanted fetched commit is ready; stale/cancelled ones drop here."""
        while True:
            _, _, fetched = await self._to_describe.get()
            if self._is_stale(fetched.request) or self._take_cancel(fetched.request):
                self.release_fingerprint(fetched.fingerprint)
                self._finish(fetched.request)
                continue
            return fetched

    def done(self, req: FetchRequest) -> None:
        """A task was inserted for the request: free its in-flight slot."""
        self._finish(req)

    def requeue(self, req: FetchRequest) -> None:
        """The request yielded no task (dup/failure): send it back for a fresh commit."""
        self._to_fetch.put_nowait((*self._priority(req), req))

    # -- accounting (internal + introspection) -------------------------------------

    def in_flight(self, king_id: str, pool: PoolType) -> int:
        return self._in_flight[(king_id, pool)]

    def reserve_fingerprint(self, fingerprint: str) -> bool:
        """Claim *fingerprint* as in-flight; False if a unit already holds it.

        The single fetcher gates admissions on this, so the same commit mined from
        two forks is described once rather than once per concurrent describer.
        """
        if fingerprint in self._inflight_fingerprints:
            return False
        self._inflight_fingerprints.add(fingerprint)
        return True

    def release_fingerprint(self, fingerprint: str) -> None:
        self._inflight_fingerprints.discard(fingerprint)

    def _is_stale(self, req: FetchRequest) -> bool:
        """True if the request belongs to a king that no longer reigns."""
        return req.king_id != self.current_king

    def _take_cancel(self, req: FetchRequest) -> bool:
        """Spend one queued cancellation for the request's pool; True if cancelled."""
        key = (req.king_id, req.pool)
        if self._cancel[key] > 0:
            self._cancel[key] -= 1
            return True
        return False

    def _finish(self, req: FetchRequest) -> None:
        key = (req.king_id, req.pool)
        self._in_flight[key] = max(0, self._in_flight[key] - 1)
