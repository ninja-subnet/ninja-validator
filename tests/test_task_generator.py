"""Tests for the task-generator pipeline coroutines, driven by fakes (no Postgres).

WorkQueue itself is covered in test_workqueue.py; here we exercise the thin
coroutines (_fetch_once / _describe_once / _make_task) against it, plus a
manual-pump end-to-end that fills a deficit and then launches nothing.
"""

from __future__ import annotations

import json
from collections import Counter

from tau.db import PoolDeficit
from tau.db.status import PoolType
from tau.github import (
    CommitCandidate,
    CommitFile,
    CommitSourceUnavailable,
    NoCommitMetCriteria,
    SampledCommit,
)
from tau.openrouter import RenderablePrompt
from tau.pools import PoolTargets
from tau.taskgen import TaskGenerationError, content_fingerprint
from tau.workers.task_generator import (
    FetchedCommit,
    FetchRequest,
    GeneratorConfig,
    WorkQueue,
    pipeline,
)

P1 = PoolType.POOL_ONE
_GOOD_JSON = json.dumps({"title": "T", "description": "D", "acceptance_criteria": ["c1"]})


def _candidate(commit: str = "a" * 40) -> CommitCandidate:
    return CommitCandidate(
        repo_full_name="octo/repo",
        repo_clone_url="https://github.com/octo/repo.git",
        commit_sha=commit,
        parent_sha="b" * 40,
        message="msg",
        html_url="",
        author_name="dev",
        event_id="",
        files=[CommitFile("a.py", "modified", 60, 50, 110, "@@ -1 +1 @@\n-old\n+new")],
    )


def _config(**kw) -> GeneratorConfig:
    return GeneratorConfig(openrouter_api_key="k", **kw)


class FakeSampler:
    """Hands out canned commits in order; raises CommitSampleError when empty."""

    def __init__(self, candidates: list[CommitCandidate]) -> None:
        self._candidates = list(candidates)
        self.calls = 0

    async def sample_commit(self, max_attempts: int = 25) -> SampledCommit:
        self.calls += 1
        if not self._candidates:
            raise NoCommitMetCriteria("exhausted")
        return SampledCommit(self._candidates.pop(0), Counter())


class FakeLLM:
    """LLMClient stand-in: returns a canned string or raises a canned exception."""

    model = "fake/model"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    async def complete_text(self, prompt: RenderablePrompt) -> str:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeDb:
    """In-memory stand-in for GeneratorDb whose deficit reflects its inserts."""

    def __init__(self, *, king: str | None = "king-1", targets: dict[PoolType, int] | None = None):
        self.king = king
        self.targets = targets or {P1: 2}
        self.inserted: dict[str, dict] = {}
        self.failures: list[dict] = []

    async def pending_pool_deficits(self, targets: PoolTargets) -> list[PoolDeficit]:
        if self.king is None:
            return []
        counts = Counter(
            (row["king_id"], PoolType(row["pool_type"])) for row in self.inserted.values()
        )
        return [
            PoolDeficit(self.king, pool, target - counts[(self.king, pool)])
            for pool, target in self.targets.items()
            if target - counts[(self.king, pool)] > 0
        ]

    async def fingerprint_exists(self, fingerprint: str) -> bool:
        return fingerprint in self.inserted

    async def insert_task_candidate(self, *, content_fingerprint: str, **kw) -> bool:
        if content_fingerprint in self.inserted:
            return False
        self.inserted[content_fingerprint] = {"content_fingerprint": content_fingerprint, **kw}
        return True

    async def record_generation_failure(self, **kw) -> None:
        self.failures.append(kw)


# -- fetcher ------------------------------------------------------------------------


async def test_fetch_once_submits_sampled_commit_to_describe() -> None:
    work = WorkQueue(5)
    work.sync_to_deficits([PoolDeficit("king-1", P1, 1)])
    await pipeline._fetch_once(FakeSampler([_candidate()]), work, FakeDb())
    async with work.lease_to_describe() as fetched:
        assert fetched.request == FetchRequest("king-1", P1)
        assert fetched.candidate.repo_full_name == "octo/repo"
        assert fetched.fingerprint  # fetcher computed it and attached it for the describer


async def test_fetch_once_skips_commit_already_in_db() -> None:
    work = WorkQueue(5)
    work.sync_to_deficits([PoolDeficit("king-1", P1, 1)])
    candidate = _candidate()
    db = FakeDb()
    db.inserted[content_fingerprint(candidate)] = {}  # already a known task
    await pipeline._fetch_once(FakeSampler([candidate]), work, db)
    assert await work.next_to_fetch() == FetchRequest("king-1", P1)  # requeued, not described
    assert work.in_flight("king-1", P1) == 1


async def test_fetch_once_skips_commit_already_in_flight() -> None:
    work = WorkQueue(5)
    work.sync_to_deficits([PoolDeficit("king-1", P1, 2)])
    db = FakeDb()
    sampler = FakeSampler([_candidate(), _candidate()])  # two copies of the same commit
    await pipeline._fetch_once(sampler, work, db)  # first reserves the fingerprint
    await pipeline._fetch_once(sampler, work, db)  # second sees it in flight -> requeued
    async with work.lease_to_describe() as first:
        assert first.candidate.commit_sha == "a" * 40
    assert await work.next_to_fetch() == FetchRequest("king-1", P1)  # only one made it through


async def test_fetch_once_requeues_on_sample_failure(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_FETCH_BACKOFF_SECONDS", 0)
    work = WorkQueue(5)
    work.sync_to_deficits([PoolDeficit("king-1", P1, 1)])
    await pipeline._fetch_once(FakeSampler([]), work, FakeDb())  # raises CommitSampleError
    assert await work.next_to_fetch() == FetchRequest("king-1", P1)  # re-queued, still alive
    assert work.in_flight("king-1", P1) == 1


class _RaisingSampler:
    """Sampler stub whose sample_commit always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def sample_commit(self, max_attempts: int = 25) -> SampledCommit:
        raise self.exc


async def test_fetch_once_screened_only_requeues_without_backoff(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(secs: float) -> None:
        slept.append(secs)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fake_sleep)
    work = WorkQueue(5)
    work.sync_to_deficits([PoolDeficit("king-1", P1, 1)])
    # Nothing met the quality bar this round -> benign churn.
    await pipeline._fetch_once(_RaisingSampler(NoCommitMetCriteria("none passed")), work, FakeDb())
    assert await work.next_to_fetch() == FetchRequest("king-1", P1)  # re-queued, alive
    assert work.in_flight("king-1", P1) == 1
    assert slept == []  # benign: no backoff


async def test_fetch_once_backs_off_on_infra_trouble(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(secs: float) -> None:
        slept.append(secs)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fake_sleep)
    work = WorkQueue(5)
    work.sync_to_deficits([PoolDeficit("king-1", P1, 1)])
    await pipeline._fetch_once(
        _RaisingSampler(CommitSourceUnavailable("rate limited")), work, FakeDb()
    )
    assert work.in_flight("king-1", P1) == 1  # still alive
    assert slept == [pipeline._FETCH_BACKOFF_SECONDS]  # source unavailable -> backed off


# -- _make_task ---------------------------------------------------------------------


def _fetched(candidate: CommitCandidate) -> FetchedCommit:
    """A fetched commit with the fingerprint the fetcher would have attached."""
    return FetchedCommit(
        FetchRequest("king-1", P1), candidate, fingerprint=content_fingerprint(candidate)
    )


async def test_make_task_inserts() -> None:
    db = FakeDb()
    inserted = await pipeline._make_task(
        _fetched(_candidate()), db, FakeLLM(_GOOD_JSON), _config(llm_attempts=1)
    )
    assert inserted is True
    assert len(db.inserted) == 1


async def test_make_task_false_on_insert_conflict() -> None:
    # Dedup moved to the fetcher; _make_task now relies on the insert's ON CONFLICT
    # as the cross-process backstop, so it still describes (calls the LLM) but the
    # conflicting insert returns False.
    candidate = _candidate()
    db = FakeDb()
    db.inserted[content_fingerprint(candidate)] = {}  # another process won the race
    llm = FakeLLM(_GOOD_JSON)
    assert await pipeline._make_task(_fetched(candidate), db, llm, _config()) is False
    assert llm.calls == 1


async def test_make_task_false_when_llm_keeps_failing() -> None:
    db = FakeDb()
    llm = FakeLLM(TaskGenerationError("no json"))
    result = await pipeline._make_task(
        _fetched(_candidate()), db, llm, _config(llm_attempts=2)
    )
    assert result is False
    assert len(db.inserted) == 0
    assert llm.calls == 2  # retried per llm_attempts
    # The abandoned commit is logged as a generation failure with the spend + reason.
    assert len(db.failures) == 1
    failure = db.failures[0]
    assert failure["commit_sha"] == "a" * 40
    assert failure["attempts"] == 2
    assert "no json" in failure["reason"]


# -- describer ----------------------------------------------------------------------


async def _seed_describe(work: WorkQueue, candidate: CommitCandidate) -> FetchRequest:
    """Drive a request through sync_to_deficits then fetch so it waits in the describe queue."""
    work.sync_to_deficits([PoolDeficit("king-1", P1, 1)])
    req = await work.next_to_fetch()
    await work.submit_fetched(req, candidate, fingerprint=content_fingerprint(candidate))
    return req


async def test_describe_once_inserts_and_marks_done() -> None:
    work = WorkQueue(5)
    await _seed_describe(work, _candidate())
    db = FakeDb()
    await pipeline._describe_once(work, db, FakeLLM(_GOOD_JSON), _config(llm_attempts=1))
    assert len(db.inserted) == 1
    assert work.in_flight("king-1", P1) == 0  # done → slot freed


async def test_describe_once_requeues_when_llm_fails() -> None:
    work = WorkQueue(5)
    req = await _seed_describe(work, _candidate())
    db = FakeDb()
    await pipeline._describe_once(work, db, FakeLLM(TaskGenerationError("x")), _config(llm_attempts=1))
    assert len(db.inserted) == 0
    assert work.in_flight("king-1", P1) == 1  # still alive
    assert await work.next_to_fetch() == req  # re-queued for a fresh commit


# -- end-to-end (manual pump) -------------------------------------------------------


async def test_pipeline_fills_deficit_then_idles() -> None:
    work = WorkQueue(5)
    db = FakeDb(targets={P1: 2})
    sampler = FakeSampler([_candidate("a" * 40), _candidate("c" * 40)])
    llm = FakeLLM(_GOOD_JSON)
    config = _config(llm_attempts=1)

    assert work.sync_to_deficits(await db.pending_pool_deficits(PoolTargets())) == 2
    await pipeline._fetch_once(sampler, work, db)
    await pipeline._fetch_once(sampler, work, db)
    await pipeline._describe_once(work, db, llm, config)
    await pipeline._describe_once(work, db, llm, config)

    assert len(db.inserted) == 2
    assert work.in_flight("king-1", P1) == 0
    # Inserts are now reflected, so the next poll's deficit is 0 → nothing launched.
    assert work.sync_to_deficits(await db.pending_pool_deficits(PoolTargets())) == 0
