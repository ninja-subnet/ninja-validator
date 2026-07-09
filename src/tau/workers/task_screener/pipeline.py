"""Concurrent poll/reconcile loop for qualification-time task screening."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from tau.axiom import get_axiom
from tau.db.task_screening import TaskScreenRequest, TaskScreeningDb
from tau.openrouter import LLMClient
from tau.task_screening import Candidate, ScreeningResult, Task

from .config import TaskScreenerConfig, TaskScreenMode
from .runner import ScreenRun, screen_with_fallback

log = logging.getLogger(__name__)

ScreenKey = tuple[str, str]  # (task_id, king_submission_id)


async def run_task_screener(
    *,
    db: TaskScreeningDb,
    clients: Sequence[LLMClient],
    config: TaskScreenerConfig,
    stop: asyncio.Event,
) -> None:
    """Poll until stopped and cancel calls whose task/king is no longer pending."""
    state = LoopState()
    log.info(
        "task screener running: mode %s, max king score %.3f, concurrency %d, poll %.1fs",
        config.mode,
        config.max_king_score,
        config.concurrency,
        config.poll_seconds,
    )
    try:
        while not stop.is_set():
            try:
                await _reconcile(db, clients, config, state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("task screener poll tick failed")
                get_axiom().exception(
                    "task-screener", "unexpected_error", error=str(exc)
                )
            await _sleep_until_stop(stop, config.poll_seconds)
    finally:
        for task in state.inflight.values():
            task.cancel()
        await asyncio.gather(*state.inflight.values(), return_exceptions=True)


async def _reconcile(
    db: TaskScreeningDb,
    clients: Sequence[LLMClient],
    config: TaskScreenerConfig,
    state: LoopState,
) -> None:
    inflight = state.inflight
    valid = {
        _key(req): req
        for req in await db.pending_requests(
            include_deferred=config.mode is TaskScreenMode.DISABLED
        )
    }

    obsolete: list[tuple[ScreenKey, asyncio.Task[None]]] = []
    for key, task in list(inflight.items()):
        if key not in valid:
            log.info("cancelling obsolete task screen for task %s", key[0])
            task.cancel()
            obsolete.append((key, task))

    if obsolete:
        # Cancellation callbacks are scheduled asynchronously. Await and remove the
        # exact old task now so obsolete calls cannot occupy concurrency slots for
        # another poll tick (or remove a later replacement with the same key).
        await asyncio.gather(*(task for _, task in obsolete), return_exceptions=True)
        for key, task in obsolete:
            if inflight.get(key) is task:
                inflight.pop(key, None)

    for key, request in valid.items():
        if key in inflight:
            continue
        if len(inflight) >= config.concurrency:
            break
        inflight[key] = _spawn(db, clients, config, request, inflight)


def _spawn(
    db: TaskScreeningDb,
    clients: Sequence[LLMClient],
    config: TaskScreenerConfig,
    request: TaskScreenRequest,
    inflight: dict[ScreenKey, asyncio.Task[None]],
) -> asyncio.Task[None]:
    key = _key(request)
    task = asyncio.create_task(
        _screen_and_save(db, clients, config, request),
        name=f"task-screen:{request.task_id}",
    )
    task.add_done_callback(lambda done: _on_done(key, done, inflight))
    return task


def _on_done(
    key: ScreenKey,
    task: asyncio.Task[None],
    inflight: dict[ScreenKey, asyncio.Task[None]],
) -> None:
    if inflight.get(key) is task:
        inflight.pop(key, None)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("task screen for %s failed unexpectedly: %r", key[0], exc)
        get_axiom().error(
            "task-screener",
            "task_screen_failed",
            task_id=key[0],
            king_submission_id=key[1],
            error=repr(exc),
        )


async def _screen_and_save(
    db: TaskScreeningDb,
    clients: Sequence[LLMClient],
    config: TaskScreenerConfig,
    request: TaskScreenRequest,
) -> None:
    if config.mode is TaskScreenMode.DISABLED:
        await _save_disabled(db, config, request)
        return

    run = await screen_with_fallback(
        Task(
            task_id=request.task_id,
            problem_statement=request.problem_statement,
            reference_patch=request.reference_patch,
        ),
        Candidate(
            submission_id=request.king_submission_id,
            patch=request.qualification_solution,
        ),
        clients=clients,
        attempts=config.attempts,
        total_timeout_seconds=config.total_timeout_seconds,
        per_attempt_timeout_seconds=config.timeout_seconds,
    )

    if run.result is None:
        await _save_retryable_error(db, config, request, run)
        return

    decision = _decision(
        run.result,
        max_score=config.max_king_score,
        mode=config.mode,
    )
    if decision is None:
        # Defensive fail-closed handling for an impossible/invalid core result.
        invalid = ScreenRun(
            result=None,
            attempts=run.attempts,
            duration_seconds=run.duration_seconds,
            error="task screening returned a scored result without a score",
            error_model=run.result.model,
        )
        await _save_retryable_error(db, config, request, invalid)
        return

    outcome, score, reason, rationale = decision
    saved = await db.save_decision(
        task_id=request.task_id,
        king_submission_id=request.king_submission_id,
        outcome=outcome,
        king_score=score,
        max_score=config.max_king_score,
        reason=reason,
        model=run.result.model,
        rationale=rationale,
        attempts=run.attempts,
        duration_seconds=run.duration_seconds,
    )
    if not saved:
        log.info("discarded stale task screen result for task %s", request.task_id)
        return

    log.info(
        "screened task %s -> %s score=%s max=%.3f (%d attempt(s), %.1fs)",
        request.task_id,
        outcome,
        score,
        config.max_king_score,
        run.attempts,
        run.duration_seconds,
    )
    get_axiom().info(
        source="task-screener",
        event_type="task_screen_saved",
        task_id=request.task_id,
        king_submission_id=request.king_submission_id,
        outcome=outcome,
        reason=reason,
        king_score=score,
        max_score=config.max_king_score,
        model=run.result.model,
        attempts=run.attempts,
        duration_seconds=run.duration_seconds,
        fingerprint=run.result.fingerprint,
    )


def _decision(
    result: ScreeningResult, *, max_score: float, mode: TaskScreenMode
) -> tuple[str, float | None, str, str] | None:
    if result.is_blocked:
        reason = str(result.blocked_reason or "blocked")
        rationale = result.rationale
        if result.blocked_evidence:
            rationale = (
                f"{rationale}\n{result.blocked_evidence}"
                if rationale
                else result.blocked_evidence
            )
        return "disqualified", None, reason, rationale
    if result.score is None:
        return None
    if mode is TaskScreenMode.SHADOW:
        return "qualified", result.score, "shadow_score_recorded", result.rationale
    if result.score > max_score:
        return "disqualified", result.score, "score_above_max", result.rationale
    return "qualified", result.score, "score_at_or_below_max", result.rationale


async def _save_retryable_error(
    db: TaskScreeningDb,
    config: TaskScreenerConfig,
    request: TaskScreenRequest,
    run: ScreenRun,
) -> None:
    error = run.error or "task screening failed without an error message"
    result = await db.save_error(
        task_id=request.task_id,
        king_submission_id=request.king_submission_id,
        max_score=config.max_king_score,
        model=run.error_model,
        error=error,
        attempts=run.attempts,
        duration_seconds=run.duration_seconds,
        max_failed_runs=config.max_failed_runs,
        retry_base_seconds=config.retry_base_seconds,
        retry_max_seconds=config.retry_max_seconds,
    )
    log.warning(
        "task screen error for %s (saved=%s, exhausted=%s, failed_runs=%s): %s",
        request.task_id,
        result.saved,
        result.exhausted,
        result.failed_runs,
        error,
    )
    get_axiom().warn(
        source="task-screener",
        event_type=(
            "task_screen_failed" if result.exhausted else "task_screen_retryable_error"
        ),
        task_id=request.task_id,
        king_submission_id=request.king_submission_id,
        saved=result.saved,
        exhausted=result.exhausted,
        failed_runs=result.failed_runs,
        cumulative_attempts=result.cumulative_attempts,
        next_retry_at=(
            result.next_retry_at.isoformat() if result.next_retry_at else None
        ),
        model=run.error_model,
        attempts=run.attempts,
        duration_seconds=run.duration_seconds,
        error=error,
    )


async def _save_disabled(
    db: TaskScreeningDb,
    config: TaskScreenerConfig,
    request: TaskScreenRequest,
) -> None:
    saved = await db.save_decision(
        task_id=request.task_id,
        king_submission_id=request.king_submission_id,
        outcome="qualified",
        king_score=None,
        max_score=config.max_king_score,
        reason="screening_disabled",
        model=None,
        rationale=None,
        attempts=0,
        duration_seconds=0.0,
    )
    if saved:
        log.info("task screening disabled; qualified task %s", request.task_id)
        get_axiom().info(
            source="task-screener",
            event_type="task_screen_saved",
            task_id=request.task_id,
            king_submission_id=request.king_submission_id,
            outcome="qualified",
            reason="screening_disabled",
            king_score=None,
            max_score=config.max_king_score,
            model=None,
            attempts=0,
            duration_seconds=0.0,
        )
    else:
        log.info("discarded stale disabled-mode decision for task %s", request.task_id)


async def _sleep_until_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


def _key(request: TaskScreenRequest) -> ScreenKey:
    return request.task_id, request.king_submission_id


@dataclass
class LoopState:
    inflight: dict[ScreenKey, asyncio.Task[None]] = field(default_factory=dict)
