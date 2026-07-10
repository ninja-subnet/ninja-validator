"""The task-solver's two-phase loop.

The scheduler keeps up to ``max_containers`` jobs running **concurrently** (one
sandbox per job), refilling free slots without waiting for stragglers:
  Phase A (qualification): CANDIDATE tasks of the reigning king → run the king's
    agent → PENDING_SCREEN or DISQUALIFIED.
  Phase B (duel solve): QUALIFIED tasks in active challenges whose king or challenger
    side lacks a fresh challenge-scoped solution → run that side's agent → store the
    solution for the judge.

Phase B jobs are gathered first so an active duel is not starved by qualification
backlog. Concurrency is safe:
each solve is fully isolated — its own proxy (OS-assigned port) and auth token and its
own work dir, all on the one shared internal sandbox network — and DB writes go through
the engine's connection pool.

Persist-vs-retry hinges on *whose* fault the failure is. A terminal run — success, or a
bad agent that crashes / returns an empty result / overspends / times out — persists a
row / status so the task is not re-selected. A miner-unrelated infrastructure fault
(``_RETRYABLE_EXIT_REASONS``: the LLM upstream, or the sandbox/docker layer) persists
nothing, leaving the task for a later tick to retry.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import docker

from tau.axiom import get_axiom
from tau.axiom.labels import Severity
from tau.db import DuelSolveJob, SolveJob, SolverDb
from tau.sandbox import (
    EXIT_AGENT_ERROR,
    EXIT_COMPLETED,
    EXIT_SANDBOX_ERROR,
    EXIT_UPSTREAM_ERROR,
    AgentRunRequest,
    AgentRunResult,
    clone_task_repo,
    run_agent_in_container,
)
from tau.sandbox.repo import CloneError

from .config import SolverConfig

log = logging.getLogger(__name__)

# Entry file every submission bundle must expose (the validator agent contract).
AGENT_ENTRYPOINT = "agent.py"

# Sleep after a tick that filled ``max_containers`` (backlog likely remains), so a
# solve backlog drains between batches without waiting the full ``poll_seconds``.
BACKLOG_POLL_SECONDS = 1.0

# Miner-unrelated infrastructure faults: the LLM upstream (unreachable / timeout / out of
# funds / rate limit / provider 5xx) or the sandbox/docker layer. A solve that ends this
# way persists nothing and is retried on a later tick. Everything else — including a bad
# agent that crashes or returns an empty result — is a terminal outcome and is saved.
_RETRYABLE_EXIT_REASONS = frozenset({EXIT_UPSTREAM_ERROR, EXIT_SANDBOX_ERROR})

JobKey = tuple[str, str, str, str]
PendingWork = tuple[JobKey, str, Callable[[], None]]


def _agent_dir(config: SolverConfig, submission_id: str) -> Path | None:
    """Resolve a submission's local bundle dir, or None if it is missing/invalid.

    A valid bundle is ``<submissions_dir>/<submission_id>/`` containing ``agent.py``.
    """
    bundle = config.submissions_dir / submission_id
    if not (bundle / AGENT_ENTRYPOINT).is_file():
        log.warning(
            "submission %s has no %s under %s; skipping",
            submission_id,
            AGENT_ENTRYPOINT,
            config.submissions_dir,
        )
        return None
    return bundle


# Severity per suspected solve-failure category (default ERROR). A model timeout is
# an expected transient (info); suspected miner code is a warning; the rest are errors.
_FAILURE_SEVERITY: dict[str, Severity] = {
    "llm_timeout": Severity.INFO,
    "llm_error": Severity.ERROR,
    "agent_error": Severity.WARNING,
    "sandbox_error": Severity.ERROR,
}


def _report_failure(
    *, phase: str, job: SolveJob | DuelSolveJob, result: AgentRunResult
) -> None:
    """Emit a granular Axiom signal for a failed solve; no-op on a clean completion.

    Routes by suspected cause so each shows up distinctly in telemetry:
      * upstream LLM-call timeout      -> llm_timeout  (info)
      * other LLM/upstream fault       -> llm_error    (error, + the upstream error)
      * suspected miner agent code     -> agent_error  (warning, + its stack trace)
      * sandbox/docker or unrecognized -> sandbox/unknown (error, generic)
    """
    reason = result.exit_reason
    if reason == EXIT_COMPLETED:
        return  # the agent ran and returned — captured by the outcome event, not here
    usage = result.usage
    if reason == EXIT_UPSTREAM_ERROR:
        timed_out = bool(usage and usage.upstream_timeout_count)
        category = "llm_timeout" if timed_out else "llm_error"
        exception = usage.last_upstream_error if usage else None
    elif reason == EXIT_AGENT_ERROR:
        category, exception = "agent_error", result.error
    elif reason == EXIT_SANDBOX_ERROR:
        category, exception = "sandbox_error", result.error
    else:  # budget trips, time/activity limits, or anything unrecognized
        category, exception = "unknown", (result.error or reason)

    get_axiom().emit(
        _FAILURE_SEVERITY.get(category, Severity.ERROR),
        "task-solver",
        "solve_job_failed",
        phase=phase,
        task_id=job.task_id,
        submission_id=job.submission_id,
        exit_reason=reason,
        category=category,
        exception=exception,
    )


def run(
    *,
    db: SolverDb,
    client: docker.DockerClient,
    config: SolverConfig,
    image_tag: str,
    stop: threading.Event,
) -> None:
    """Keep every available sandbox slot filled until *stop* is set.

    Completed keys stay excluded for one normal poll interval. Successful jobs have
    already left the DB queue; retryable infrastructure failures have not, and the
    exclusion preserves their previous retry cadence without blocking unrelated work.
    """
    cap = config.max_containers
    inflight: dict[Future[None], JobKey] = {}
    cooldown: dict[JobKey, float] = {}
    with ThreadPoolExecutor(max_workers=cap, thread_name_prefix="tau-solve") as pool:
        while not stop.is_set():
            now = time.monotonic()
            for future in [future for future in inflight if future.done()]:
                key = inflight.pop(future)
                cooldown[key] = now + config.poll_seconds
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    log.exception("solve job failed")
                    get_axiom().exception(
                        "task-solver", "solve_job_failed", details=str(exc)
                    )
            cooldown = {key: until for key, until in cooldown.items() if until > now}

            slots = cap - len(inflight)
            launched = 0
            if slots:
                try:
                    work = _pending_work(
                        db=db,
                        client=client,
                        config=config,
                        image_tag=image_tag,
                        limit=slots,
                        exclude=set(inflight.values()) | set(cooldown),
                    )
                    if work:
                        duel_count = sum(phase == "duel" for _, phase, _ in work)
                        log.info(
                            "scheduler: launched %d job(s) (%d duel + %d qualification, "
                            "%d already running, cap=%d)",
                            len(work),
                            duel_count,
                            len(work) - duel_count,
                            len(inflight),
                            cap,
                        )
                    for key, _, fn in work:
                        inflight[pool.submit(fn)] = key
                    launched = len(work)
                except Exception:  # noqa: BLE001 — one bad poll must not kill the worker
                    log.exception("solver scheduler poll failed")

            stop.wait(
                BACKLOG_POLL_SECONDS
                if inflight or launched
                else config.poll_seconds
            )


def _job_key(job: SolveJob | DuelSolveJob) -> JobKey:
    if isinstance(job, DuelSolveJob):
        return (
            "duel",
            job.task_id,
            job.submission_id,
            job.challenger_submission_id,
        )
    return ("qualification", job.task_id, job.submission_id, "")


def _pending_work(
    *,
    db: SolverDb,
    client: docker.DockerClient,
    config: SolverConfig,
    image_tag: str,
    limit: int,
    exclude: set[JobKey],
) -> list[PendingWork]:
    """Return up to *limit* fresh jobs, preserving duel-first priority."""
    duel_query_limit = limit + sum(key[0] == "duel" for key in exclude)
    duel_jobs = [
        job
        for job in db.next_duel_jobs(
            duel_query_limit,
            require_full_pool=config.require_full_pool_for_duels,
            pool_targets=config.pool_targets,
        )
        if _job_key(job) not in exclude
    ][:limit]
    remaining = limit - len(duel_jobs)
    qual_query_limit = remaining + sum(key[0] == "qualification" for key in exclude)
    qual_jobs = (
        [
            job
            for job in db.next_qualification_jobs(qual_query_limit)
            if _job_key(job) not in exclude
        ][:remaining]
        if remaining
        else []
    )
    return [
        (
            _job_key(job),
            "duel",
            partial(
                _solve_duel,
                job,
                db=db,
                client=client,
                config=config,
                image_tag=image_tag,
            ),
        )
        for job in duel_jobs
    ] + [
        (
            _job_key(job),
            "qualification",
            partial(
                _qualify,
                job,
                db=db,
                client=client,
                config=config,
                image_tag=image_tag,
            ),
        )
        for job in qual_jobs
    ]


def _qualify(
    job: SolveJob,
    *,
    db: SolverDb,
    client: docker.DockerClient,
    config: SolverConfig,
    image_tag: str,
) -> None:
    agent_dir = _agent_dir(config, job.submission_id)
    if agent_dir is None:
        return  # king bundle missing locally — leave CANDIDATE, retry next tick
    log.info(
        "qualifying task=%s with king=%s (cloning + sandboxing)…",
        job.task_id,
        job.submission_id,
    )
    try:
        result = _run(job, agent_dir, client=client, config=config, image_tag=image_tag)
    except CloneError as exc:
        db.finish_qualification(
            task_id=job.task_id,
            king_submission_id=job.submission_id,
            qualified=False,
            solution="",
        )
        log.warning(
            "qualification task=%s king=%s setup failed; marking DISQUALIFIED (%s)",
            job.task_id,
            job.submission_id,
            exc,
        )
        get_axiom().exception(
            "task-solver",
            "qualification_task_setup_failed",
            task_id=job.task_id,
            submission_id=job.submission_id,
            error=str(exc),
        )
        return
    _report_failure(phase="qualification", job=job, result=result)
    if result.exit_reason in _RETRYABLE_EXIT_REASONS:
        # Miner-unrelated infra fault (LLM upstream or sandbox/docker), not a verdict on
        # the king — persist nothing and leave the task CANDIDATE so a later tick retries
        # it (never DISQUALIFY on infrastructure).
        log.warning(
            "qualification task=%s king=%s hit %s (%s) — leaving CANDIDATE for retry",
            job.task_id,
            job.submission_id,
            result.exit_reason,
            result.error,
        )
        get_axiom().exception(
            "task-solver",
            "qualification_infra_error",
            task_id=job.task_id,
            submission_id=job.submission_id,
            exit_reason=result.exit_reason,
            error=result.error,
        )
        return
    qualified = result.success and _changed_lines(result.solution_diff) >= (
        config.qualify_min_changed_lines
    )
    db.finish_qualification(
        task_id=job.task_id,
        king_submission_id=job.submission_id,
        qualified=qualified,
        solution=result.solution_diff,
    )
    log.info(
        "qualification task=%s king=%s -> %s (exit=%s, %d changed lines)",
        job.task_id,
        job.submission_id,
        "PENDING_SCREEN" if qualified else "DISQUALIFIED",
        result.exit_reason,
        _changed_lines(result.solution_diff),
    )
    get_axiom().info(
        source="task-solver",
        event_type="qualification",
        task_id=job.task_id,
        submission_id=job.submission_id,
        exit_reason=result.exit_reason,
        viable=qualified,
        status=("pending_screen" if qualified else "disqualified"),
        elapsed_seconds=result.elapsed_seconds,
        changed_lines=_changed_lines(result.solution_diff),
    )


def _solve_duel(
    job: DuelSolveJob,
    *,
    db: SolverDb,
    client: docker.DockerClient,
    config: SolverConfig,
    image_tag: str,
) -> None:
    agent_dir = _agent_dir(config, job.submission_id)
    if agent_dir is None:
        return  # submission bundle missing locally — leave unsolved, retry next tick
    log.info(
        "solving duel task=%s challenge=%s submission=%s (cloning + sandboxing)…",
        job.task_id,
        job.challenger_submission_id,
        job.submission_id,
    )
    result = _run(job, agent_dir, client=client, config=config, image_tag=image_tag)
    _report_failure(phase="duel", job=job, result=result)
    if result.exit_reason in _RETRYABLE_EXIT_REASONS:
        # Miner-unrelated infra fault (LLM upstream or sandbox/docker) — do not persist a
        # bogus solution; leave the task unsolved so a later tick retries it. A bad agent
        # (crash / empty result) is NOT an infra fault and is saved below instead.
        log.warning(
            "duel solve task=%s challenge=%s submission=%s hit %s (%s) — leaving unsolved for retry",
            job.task_id,
            job.challenger_submission_id,
            job.submission_id,
            result.exit_reason,
            result.error,
        )
        get_axiom().exception(
            "task-solver",
            "duel_infra_error",
            task_id=job.task_id,
            challenger_submission_id=job.challenger_submission_id,
            submission_id=job.submission_id,
            exit_reason=result.exit_reason,
            error=result.error,
        )
        return
    db.save_duel_task_solution(
        task_id=job.task_id,
        challenger_submission_id=job.challenger_submission_id,
        submission_id=job.submission_id,
        solution=result.solution_diff,
        duration=result.elapsed_seconds,
        exit_reason=result.exit_reason,
        usage_summary=result.usage.to_dict() if result.usage is not None else None,
    )
    log.info(
        "duel solve task=%s challenge=%s submission=%s exit=%s success=%s",
        job.task_id,
        job.challenger_submission_id,
        job.submission_id,
        result.exit_reason,
        result.success,
    )
    get_axiom().info(
        source="task-solver",
        event_type="solution",
        task_id=job.task_id,
        challenger_submission_id=job.challenger_submission_id,
        submission_id=job.submission_id,
        exit_reason=result.exit_reason,
        success=result.success,
        elapsed_seconds=result.elapsed_seconds,
    )


def _run(
    job: SolveJob | DuelSolveJob,
    agent_dir: Path,
    *,
    client: docker.DockerClient,
    config: SolverConfig,
    image_tag: str,
) -> AgentRunResult:
    """Clone the task repo and run one sandboxed solve of *agent_dir* against it."""
    with TemporaryDirectory(prefix="tau-solve-") as tmp:
        repo_dir = clone_task_repo(
            repo_clone_url=job.repo_clone_url,
            base_commit=job.base_commit,
            token=config.github_token,
            dest=Path(tmp) / "repo",
            cache_dir=config.task_repo_cache_dir,
            fetch_concurrency=config.task_repo_fetch_concurrency,
            cache_max_entries=config.task_repo_cache_max_entries,
        )
        request = AgentRunRequest(
            task_id=job.task_id,
            problem_statement=job.problem_statement,
            repo_dir=repo_dir,
            agent_dir=agent_dir,
            budget=config.budget,
        )
        return run_agent_in_container(
            request,
            client=client,
            config=config.sandbox,
            upstream=config.upstream,
            image_tag=image_tag,
        )


def _changed_lines(diff: str) -> int:
    """Count added/removed lines in a unified diff (excluding +++/--- file headers)."""
    count = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            count += 1
        elif line.startswith("-") and not line.startswith("---"):
            count += 1
    return count
