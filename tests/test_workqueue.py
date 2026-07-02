"""Unit tests for WorkQueue — the task-generator's in-flight accounting + queues."""

from __future__ import annotations

from tau.db import PoolDeficit
from tau.db.status import PoolType
from tau.github import CommitCandidate, CommitFile
from tau.workers.task_generator import FetchRequest, WorkQueue

P1 = PoolType.POOL_ONE
P2 = PoolType.POOL_TWO


def _deficit(pool: PoolType, deficit: int, king: str = "king-1") -> PoolDeficit:
    return PoolDeficit(king, pool, deficit)


def _candidate(commit: str = "a" * 40) -> CommitCandidate:
    return CommitCandidate(
        repo_full_name="octo/repo",
        repo_clone_url="https://github.com/octo/repo.git",
        commit_sha=commit,
        parent_sha="b" * 40,
        message="m",
        html_url="",
        author_name=None,
        event_id="",
        files=[CommitFile("a.py", "modified", 1, 1, 2, "@@")],
    )


# -- sync_to_deficits (controller side) ------------------------------------------------


async def test_sync_to_deficits_launches_deficit_and_counts_in_flight() -> None:
    wq = WorkQueue(5)
    assert wq.sync_to_deficits([_deficit(P1, 2), _deficit(P2, 1)]) == 3
    assert wq.in_flight("king-1", P1) == 2
    assert wq.in_flight("king-1", P2) == 1
    assert wq.current_king == "king-1"


async def test_sync_to_deficits_does_not_double_launch_in_flight_work() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 3)])
    assert wq.sync_to_deficits([_deficit(P1, 3)]) == 0  # same deficit, work still in flight
    assert wq.in_flight("king-1", P1) == 3


async def test_sync_to_deficits_empty_clears_king() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])
    assert wq.sync_to_deficits([]) == 0
    assert wq.current_king is None


async def test_done_frees_a_slot_so_next_poll_relaunches() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 2)])
    wq.done(FetchRequest("king-1", P1))  # one task inserted
    assert wq.sync_to_deficits([_deficit(P1, 2)]) == 1
    assert wq.in_flight("king-1", P1) == 2


# -- fetch side ---------------------------------------------------------------------


async def test_next_to_fetch_returns_a_wanted_request() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])
    assert await wq.next_to_fetch() == FetchRequest("king-1", P1)


async def test_next_to_fetch_drops_stale_king_requests() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])  # enqueues 1 for king-1
    wq.sync_to_deficits([_deficit(P1, 1, king="king-2")])  # current king → king-2, enqueues 1 more
    req = await wq.next_to_fetch()  # the king-1 request is stale and dropped
    assert req.king_id == "king-2"
    assert wq.in_flight("king-1", P1) == 0  # stale slot freed
    assert wq.in_flight("king-2", P1) == 1


async def test_next_to_fetch_honours_cancel_quota() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 3)])  # 3 enqueued, in_flight 3
    assert wq.sync_to_deficits([_deficit(P1, 1)]) == 0  # deficit dropped → cancel 2
    assert await wq.next_to_fetch() == FetchRequest("king-1", P1)  # drops 2, returns the 3rd
    assert wq.in_flight("king-1", P1) == 1


async def test_fetch_failed_keeps_the_request_alive() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])
    req = await wq.next_to_fetch()
    wq.fetch_failed(req)
    assert await wq.next_to_fetch() == req
    assert wq.in_flight("king-1", P1) == 1


# -- describe side ------------------------------------------------------------------


async def test_round_trip_fetch_to_describe_to_done() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])
    req = await wq.next_to_fetch()
    candidate = _candidate()
    await wq.submit_fetched(req, candidate)
    async with wq.lease_to_describe() as fetched:
        assert fetched.request is req and fetched.candidate is candidate
        wq.done(req)
    assert wq.in_flight("king-1", P1) == 0


async def test_requeue_sends_a_request_back_to_fetch() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])
    req = await wq.next_to_fetch()
    wq.requeue(req)
    assert await wq.next_to_fetch() == req
    assert wq.in_flight("king-1", P1) == 1  # requeue keeps it alive, doesn't free the slot


async def test_lease_to_describe_skips_a_stale_item() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])  # king-1
    stale = await wq.next_to_fetch()
    await wq.submit_fetched(stale, _candidate("a" * 40))
    wq.sync_to_deficits([_deficit(P1, 1, king="king-2")])  # king changes; the queued item is now stale
    current = await wq.next_to_fetch()
    await wq.submit_fetched(current, _candidate("c" * 40))
    async with wq.lease_to_describe() as fetched:  # drops stale king-1, returns king-2
        assert fetched.request == current
    assert wq.in_flight("king-1", P1) == 0  # stale slot freed


# -- pool priority ------------------------------------------------------------------


async def test_fetch_prioritises_pool_one_even_for_a_requeued_retry() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1), _deficit(P2, 1)])
    p1 = await wq.next_to_fetch()
    assert p1.pool == P1
    # Only the pool-two request is left queued; requeue the pool-one retry behind it.
    wq.requeue(p1)
    # Priority, not FIFO: the requeued pool-one retry still comes out before pool-two.
    assert (await wq.next_to_fetch()).pool == P1
    assert (await wq.next_to_fetch()).pool == P2


# -- in-flight fingerprint dedup ----------------------------------------------------


def test_reserve_fingerprint_is_exclusive_until_released() -> None:
    wq = WorkQueue(5)
    assert wq.reserve_fingerprint("fp-1") is True
    assert wq.reserve_fingerprint("fp-1") is False  # already held
    wq.release_fingerprint("fp-1")
    assert wq.reserve_fingerprint("fp-1") is True  # free again after release


async def test_lease_releases_a_dropped_stale_fingerprint() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1)])  # king-1
    stale = await wq.next_to_fetch()
    assert wq.reserve_fingerprint("fp-stale") is True
    await wq.submit_fetched(stale, _candidate("a" * 40), fingerprint="fp-stale")
    wq.sync_to_deficits([_deficit(P1, 1, king="king-2")])  # king-1 item is now stale
    current = await wq.next_to_fetch()
    await wq.submit_fetched(current, _candidate("c" * 40), fingerprint="fp-current")
    async with wq.lease_to_describe() as fetched:  # drops + releases the stale one
        assert fetched.request == current
    assert wq.reserve_fingerprint("fp-stale") is True  # released when its item was dropped


async def test_describe_prioritises_pool_one_over_pool_two() -> None:
    wq = WorkQueue(5)
    wq.sync_to_deficits([_deficit(P1, 1), _deficit(P2, 1)])
    p1 = await wq.next_to_fetch()
    p2 = await wq.next_to_fetch()
    # Submit pool-two first; the describers must still pull pool-one ahead of it.
    await wq.submit_fetched(p2, _candidate("c" * 40))
    await wq.submit_fetched(p1, _candidate("a" * 40))
    async with wq.lease_to_describe() as fetched:
        assert fetched.request.pool == P1
    async with wq.lease_to_describe() as fetched:
        assert fetched.request.pool == P2
