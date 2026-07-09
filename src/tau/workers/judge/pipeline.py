"""The judge worker's poll loop.

A single loop reconciles a bounded set of in-flight judgment tasks against
`JudgeDb.pending_judge_requests()` each tick: start newly-wanted pairs (up to
`concurrency`) and cancel any in-flight judgment the latest poll no longer wants
-- challenge ended, king changed, pool advanced, task disqualified. No separate
queue: the pending query is the source of truth, so the in-flight dict is the
only state carried between ticks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from tau.axiom import get_axiom
from tau.db import JudgeDb, JudgeRequest, TaskScreenDuelComparison
from tau.judging import Solution, Task
from tau.openrouter import LLMClient

from .config import JudgeWorkerConfig
from .fallback import JudgeRun, judge_with_fallback

log = logging.getLogger(__name__)

# (task_id, king_submission_id, challenger_submission_id) -- a verdict's identity.
JudgeKey = tuple[str, str, str]


async def run_judge_worker(
    *,
    db: JudgeDb,
    clients: Sequence[LLMClient],
    config: JudgeWorkerConfig,
    stop: asyncio.Event,
) -> None:
    """Poll + reconcile until *stop* is set, then cancel any in-flight judgments."""
    state = LoopStateLogging()
    log.info(
        "judge worker running: concurrency %d, poll %.0fs",
        config.concurrency,
        config.poll_seconds,
    )
    try:
        while not stop.is_set():
            try:
                await _reconcile(db, clients, config, state)
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                log.exception("judge poll tick failed")
                get_axiom().exception("judge", "unexpected_error", error=str(ex))
            await _sleep_until_stop(stop, config.poll_seconds)
    finally:
        # Awaiting a task does not auto-cancel it, so tear the children down here.
        for task in state.inflight.values():
            task.cancel()
        await asyncio.gather(*state.inflight.values(), return_exceptions=True)


async def _reconcile(
    db: JudgeDb,
    clients: Sequence[LLMClient],
    config: JudgeWorkerConfig,
    state: LoopStateLogging,
) -> None:
    """One poll tick: cancel obsolete in-flight judgments, start newly-wanted ones."""
    inflight = state.inflight
    valid = {_key(req): req for req in await db.pending_judge_requests()}

    for key, task in list(inflight.items()):
        if key not in valid:
            state.note_obsolete(key)
            task.cancel()

    state.note_pending(valid)

    for key, req in valid.items():
        if key in inflight:
            continue  # already running; the poll keeps returning it until saved
        if len(inflight) >= config.concurrency:
            break  # at capacity; remaining wanted pairs start on a later tick
        inflight[key] = _spawn(db, clients, config, req, inflight)
        state.note_started(key, config.concurrency)

    state.note_settled(valid)


def _spawn(
    db: JudgeDb,
    clients: Sequence[LLMClient],
    config: JudgeWorkerConfig,
    req: JudgeRequest,
    inflight: dict[JudgeKey, asyncio.Task[None]],
) -> asyncio.Task[None]:
    key = _key(req)
    task = asyncio.create_task(
        _judge_and_save(db, clients, config, req), name=f"judge:{key[0]}"
    )
    task.add_done_callback(lambda t: _on_judgment_done(key, t, inflight))
    return task


def _on_judgment_done(
    key: JudgeKey,
    task: asyncio.Task[None],
    inflight: dict[JudgeKey, asyncio.Task[None]],
) -> None:
    inflight.pop(key, None)
    if task.cancelled():
        return  # obsolete (poll cancelled it) or shutdown -- nothing to record
    exc = task.exception()
    if exc is not None:
        log.error("judgment for task %s failed: %r", key[0], exc)
        get_axiom().error(
            "judge",
            "judgment_failed",
            task_id=key[0],
            king_submission_id=key[1],
            challenger_submission_id=key[2],
            error=repr(exc),
        )


async def _judge_and_save(
    db: JudgeDb,
    clients: Sequence[LLMClient],
    config: JudgeWorkerConfig,
    req: JudgeRequest,
) -> None:
    task, king_solution, challenger_solution = req
    run = await judge_with_fallback(
        task,
        king_solution,
        challenger_solution,
        clients=clients,
        attempts=config.attempts,
        total_timeout_seconds=config.total_timeout_seconds,
    )
    comparison = await db.save_judgment(
        task,
        king_solution,
        challenger_solution,
        run.judgment,
        attempts=run.attempts,
        duration_seconds=run.duration_seconds,
    )
    log.info(
        "saved judgment for task %s: winner=%s (%d attempt(s), %.1fs)",
        task.task_id,
        run.judgment.winner,
        run.attempts,
        run.duration_seconds,
    )
    _emit_judgment(task, king_solution, challenger_solution, run)
    if comparison is not None:
        _emit_task_screen_duel_comparison(comparison)


def _emit_judgment(
    task: Task, king: Solution, challenger: Solution, run: JudgeRun
) -> None:
    """Send a saved judgment to Axiom: `judgment_degraded` when the verdict is a
    neutral LLM-unavailable fallback (error set), else `judgment_saved`."""
    judgement = run.judgment
    fields: dict[str, object] = {
        "task_id": task.task_id,
        "king_submission_id": king.submission_id,
        "challenger_submission_id": challenger.submission_id,
        "winner": judgement.winner,
        "king_score": judgement.king_score,
        "challenger_score": judgement.challenger_score,
        "model": judgement.model,
        "attempts": run.attempts,
        "duration_seconds": run.duration_seconds,
    }
    if judgement.error is not None:
        get_axiom().warn(
            source="judge",
            event_type="judgment_degraded",
            error=judgement.error,
            **fields,
        )
    else:
        get_axiom().info(source="judge", event_type="judgment_saved", **fields)


def _emit_task_screen_duel_comparison(
    comparison: TaskScreenDuelComparison,
) -> None:
    """Emit non-punitive drift telemetry; downstream analysis needs many samples."""
    get_axiom().info(
        source="judge",
        event_type="task_screen_duel_comparison",
        task_id=comparison.task_id,
        king_submission_id=comparison.king_submission_id,
        challenger_submission_id=comparison.challenger_submission_id,
        screening_king_score=comparison.screening_king_score,
        duel_king_score=comparison.duel_king_score,
        duel_minus_screen_king_score_delta=(
            comparison.duel_minus_screen_king_score_delta
        ),
        screening_model=comparison.screening_model,
        duel_model=comparison.duel_model,
        qualification_patch_sha256=comparison.qualification_patch_sha256,
        duel_patch_sha256=comparison.duel_patch_sha256,
        qualification_patch_matches_duel_patch=(
            comparison.qualification_patch_matches_duel_patch
        ),
    )


async def _sleep_until_stop(stop: asyncio.Event, seconds: float) -> None:
    """Sleep up to *seconds*, waking early if *stop* is set."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


@dataclass
class LoopStateLogging:
    """In-flight judgments plus edge-trigger state for the loop's log lines."""

    inflight: dict[JudgeKey, asyncio.Task[None]] = field(default_factory=dict)
    _announced: set[JudgeKey] = field(default_factory=set)
    _idle: bool = False

    def note_obsolete(self, key: JudgeKey) -> None:
        log.info("cancelling obsolete judgment for task %s", key[0])

    def note_pending(self, valid: dict[JudgeKey, JudgeRequest]) -> None:
        """Log how many pending rounds are new since the last tick (deduped)."""
        new_rounds = [key for key in valid if key not in self._announced]
        if new_rounds:
            log.info(
                "watcher: %d new round(s) to judge (%d in flight, %d pending total)",
                len(new_rounds),
                len(self.inflight),
                len(valid),
            )
        self._announced = set(valid)

    def note_started(self, key: JudgeKey, concurrency: int) -> None:
        log.info(
            "starting judgment for task %s (%d/%d in flight)",
            key[0],
            len(self.inflight),
            concurrency,
        )

    def note_settled(self, valid: dict[JudgeKey, JudgeRequest]) -> None:
        """Log "caught up" once when nothing is pending or in flight."""
        if not valid and not self.inflight:
            if not self._idle:
                log.info("all caught up: no pending rounds, waiting for new work")
                self._idle = True
        else:
            self._idle = False


def _key(req: JudgeRequest) -> JudgeKey:
    task, king_solution, challenger_solution = req
    return (
        task.task_id,
        king_solution.submission_id,
        challenger_solution.submission_id,
    )
