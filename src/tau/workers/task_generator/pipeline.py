"""Async pipeline that fills task pools: poll -> fetch -> describe -> insert.

controller -> WorkQueue -> fetcher (1, serial) -> describers (K, concurrent) -> insert

The coroutines stay thin by driving a WorkQueue (the queues + in-flight accounting);
one fetcher serialises rate-limited GitHub, the describers run LLM calls concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from tau.axiom import get_axiom
from tau.db import GenerationMetrics, GeneratorDb
from tau.github import (
    CommitCandidate,
    CommitSampler,
    CommitSourceUnavailable,
    GitHubRequestError,
    NoCommitMetCriteria,
    RejectReason,
)
from tau.openrouter import LLMClient
from tau.pools import PoolTargets
from tau.taskgen import (
    GeneratedTask,
    TaskGenerationError,
    content_fingerprint,
    generate_task_description,
)

from .config import GeneratorConfig
from .workqueue import FetchedCommit, FetchRequest, WorkQueue

log = logging.getLogger(__name__)

# Back-off before retrying a request whose commit fetch failed, so a barren or
# rate-limited GitHub window can't spin the single fetcher in a tight loop.
_FETCH_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class DescribeOutcome:
    """Result of describing one commit: the task on success, else why it failed."""

    task: GeneratedTask | None
    attempts: int  # LLM attempts spent (the winning attempt's number on success)
    error: str | None  # last failure message; set only when task is None


# -- assembly -----------------------------------------------------------------------


async def run(
    *,
    db: GeneratorDb,
    sampler: CommitSampler,
    llm: LLMClient,
    config: GeneratorConfig,
    targets: PoolTargets,
    stop: asyncio.Event | None = None,
) -> None:
    """Run the controller + fetcher + describers until *stop* is set, then drain."""
    work = WorkQueue(config.describe_concurrency)
    workers = [
        asyncio.create_task(_controller(db, targets, work, config), name="controller"),
        asyncio.create_task(_fetcher(sampler, work, db), name="fetcher"),
    ]
    workers += [
        asyncio.create_task(_describer(work, db, llm, config), name=f"describer-{i}")
        for i in range(config.describe_concurrency)
    ]
    log.info(
        "task-generator running: %d describer(s), poll %.0fs",
        config.describe_concurrency,
        config.poll_seconds,
    )
    try:
        if stop is not None:
            await stop.wait()
        else:
            await asyncio.gather(*workers)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


# -- controller ---------------------------------------------------------------------


async def _controller(
    db: GeneratorDb, targets: PoolTargets, work: WorkQueue, config: GeneratorConfig
) -> None:
    while True:
        try:
            launched = work.sync_to_deficits(await db.pending_pool_deficits(targets))
            if launched:
                log.info(
                    "controller: launched %d task(s) for king %s",
                    launched,
                    work.current_king,
                )
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            log.exception("controller tick failed")
            get_axiom().exception(
                "task-generator", "unexpected_error", stage="controller", error=str(ex)
            )
        await asyncio.sleep(config.poll_seconds)


# -- fetcher (single, sequential) ---------------------------------------------------


async def _fetcher(sampler: CommitSampler, work: WorkQueue, db: GeneratorDb) -> None:
    while True:
        try:
            await _fetch_once(sampler, work, db)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            log.exception("fetcher iteration failed")
            get_axiom().exception(
                "task-generator", "unexpected_error", stage="fetcher", error=str(ex)
            )


async def _fetch_once(sampler: CommitSampler, work: WorkQueue, db: GeneratorDb) -> None:
    """Take the next wanted request, sample a commit, dedup it, hand it to describers."""
    req = await work.next_to_fetch()
    started = time.monotonic()
    try:
        sampled = await sampler.sample_commit()
    except NoCommitMetCriteria as exc:
        # Benign: nothing met the quality bar this round. Requeue and retry promptly
        # -- not a failure, so no warning and no backoff.
        log.debug("no commit met the quality bar this round, retrying: %s", exc)
        work.fetch_failed(req)
        return
    except (CommitSourceUnavailable, GitHubRequestError) as exc:
        # Barren / rate-limited source, or a transport failure that escaped the
        # sampler: warn and back off before retrying so we do not hammer GitHub.
        log.warning("commit fetch failed, backing off: %s", exc)
        get_axiom().warn(
            source="task-generator", event_type="fetch_failed", reason=str(exc)
        )
        await _backoff_and_requeue(req, work)
        return
    fetch_seconds = time.monotonic() - started
    candidate = sampled.candidate
    fingerprint = content_fingerprint(candidate)
    # Dedup here so a describer never spends an LLM call on a commit we already
    # have (DB) or are already describing (in-flight reserve). The king's request
    # is kept alive to sample a fresh commit instead.
    if await db.fingerprint_exists(fingerprint):
        skip_reason = "already a task"
    elif not work.reserve_fingerprint(fingerprint):
        skip_reason = "already in flight"
    else:
        skip_reason = None
    if skip_reason:
        log.debug(
            "skip %s@%s: %s", candidate.repo_full_name, candidate.commit_sha[:8], skip_reason
        )
        work.fetch_failed(req)
        return
    log.info(
        "queued commit %s@%s for description (king %s, pool %d, fetched in %.1fs, %d rejected first)",
        candidate.repo_full_name,
        candidate.commit_sha[:8],
        req.king_id,
        int(req.pool),
        fetch_seconds,
        sum(sampled.rejections.values()),
    )
    await work.submit_fetched(
        req,
        candidate,
        fingerprint=fingerprint,
        fetch_seconds=fetch_seconds,
        rejections=sampled.rejections,
    )


async def _backoff_and_requeue(req: FetchRequest, work: WorkQueue) -> None:
    """Pause (so we don't hammer GitHub), then keep the request alive for a retry."""
    await asyncio.sleep(_FETCH_BACKOFF_SECONDS)
    work.fetch_failed(req)


# -- describers (concurrent pool) ---------------------------------------------------


async def _describer(
    work: WorkQueue, db: GeneratorDb, llm: LLMClient, config: GeneratorConfig
) -> None:
    while True:
        try:
            await _describe_once(work, db, llm, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("describer iteration failed")


async def _describe_once(
    work: WorkQueue, db: GeneratorDb, llm: LLMClient, config: GeneratorConfig
) -> None:
    """Describe the next wanted commit; keep the unit alive on any failure.

    The lease frees the fetcher's fingerprint reservation on exit; here we only
    decide the unit's slot fate (done vs requeue).
    """
    async with work.lease_to_describe() as fetched:
        req = fetched.request
        try:
            inserted = await _make_task(fetched, db, llm, config)
        except Exception as ex:
            log.exception("describe unit errored; re-queueing to stay alive")
            get_axiom().exception(
                "task-generator", "unexpected_error", stage="describer", error=str(ex)
            )
            work.requeue(req)  # never drop a unit; that would leak an in-flight slot
        else:
            if inserted:
                work.done(req)
            else:
                work.requeue(req)  # unusable output / insert race: try another commit


async def _make_task(
    fetched: FetchedCommit, db: GeneratorDb, llm: LLMClient, config: GeneratorConfig
) -> bool:
    """Describe the commit and insert it (with telemetry). True iff a row was added."""
    req, candidate = fetched.request, fetched.candidate
    outcome = await _describe_with_retries(candidate, llm, config)
    if outcome.task is None:
        # The model never produced a usable task for this commit. It yields no
        # tasks row, so log the abandoned commit to keep the give-up visible.
        await db.record_generation_failure(
            king_id=req.king_id,
            pool_type=int(req.pool),
            repo_full_name=candidate.repo_full_name,
            commit_sha=candidate.commit_sha,
            model=llm.model,
            attempts=outcome.attempts,
            reason=outcome.error,
        )
        get_axiom().warn(
            source="task-generator",
            event_type="generation_failed",
            king_id=req.king_id,
            pool=req.pool.name,
            repo_full_name=candidate.repo_full_name,
            commit_sha=candidate.commit_sha,
            model=llm.model,
            attempts=outcome.attempts,
            reason=outcome.error,
        )
        return False
    task, llm_attempt = outcome.task, outcome.attempts
    rejections = fetched.rejections
    metrics = GenerationMetrics(
        model=llm.model,
        fetch_seconds=fetched.fetch_seconds,
        llm_seconds=task.elapsed_seconds,
        llm_attempt=llm_attempt,
        rejected_duplicate=rejections[RejectReason.DUPLICATE],
        rejected_structural=rejections[RejectReason.STRUCTURAL],
        rejected_quality=rejections[RejectReason.QUALITY],
        rejected_fetch_error=rejections[RejectReason.FETCH_ERROR],
    )
    task_id = f"{candidate.commit_sha[:16]}-{fetched.fingerprint[:8]}"
    inserted = await db.insert_task_candidate(
        task_id=task_id,
        king_id=req.king_id,
        pool_type=int(req.pool),
        problem_statement=task.prompt_text,
        reference_patch=candidate.combined_patch,
        repo_clone_url=candidate.repo_clone_url,
        parent_sha=candidate.parent_sha,
        commit_sha=candidate.commit_sha,
        content_fingerprint=fetched.fingerprint,
        metrics=metrics,
    )
    if inserted:
        log.info(
            "inserted task %s for king %s pool %d",
            task.title,
            req.king_id,
            int(req.pool),
        )
        get_axiom().info(
            source="task-generator",
            event_type="task_inserted",
            task_id=task_id,
            title=task.title,
            king_id=req.king_id,
            pool=req.pool.name,
            repo_full_name=candidate.repo_full_name,
            commit_sha=candidate.commit_sha,
            model=metrics.model,
            llm_attempt=metrics.llm_attempt,
            fetch_seconds=metrics.fetch_seconds,
            llm_seconds=metrics.llm_seconds,
            rejected_duplicate=metrics.rejected_duplicate,
            rejected_structural=metrics.rejected_structural,
            rejected_quality=metrics.rejected_quality,
            rejected_fetch_error=metrics.rejected_fetch_error,
        )
    return inserted  # False on a cross-process insert race: caller tries another commit


async def _describe_with_retries(
    candidate: CommitCandidate, llm: LLMClient, config: GeneratorConfig
) -> DescribeOutcome:
    """Up to llm_attempts tries on the same commit.

    Returns the task and the winning attempt number on success, else a task-less
    outcome carrying how many attempts were spent and the last failure message.
    """
    last_error = "no attempts made"
    for attempt in range(1, config.llm_attempts + 1):
        try:
            task = await asyncio.wait_for(
                generate_task_description(candidate=candidate, client=llm),
                timeout=config.llm_timeout,
            )
            return DescribeOutcome(task=task, attempts=attempt, error=None)
        except asyncio.CancelledError:
            raise
        except TaskGenerationError as exc:
            last_error = str(exc)
            log.debug(
                "describe attempt %d/%d rejected: %s", attempt, config.llm_attempts, exc
            )
        except Exception as exc:  # transport or timeout: retry, then a fresh commit
            last_error = str(exc)
            log.warning(
                "describe attempt %d/%d failed: %s", attempt, config.llm_attempts, exc
            )
    return DescribeOutcome(task=None, attempts=config.llm_attempts, error=last_error)
