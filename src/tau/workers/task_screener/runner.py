"""Retry and total-time policy around one independent task score.

Unlike duel judging, exhaustion never fabricates a neutral score: ``result`` is
``None`` and ``error`` is set so the DB layer can back off and, after a bounded
number of failed runs, disqualify the task.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from tau.openrouter import LLMClient
from tau.task_screening import Candidate, ScreeningResult, Task, score_candidate

log = logging.getLogger(__name__)


class RetryError(RuntimeError): ...


@dataclass(frozen=True, slots=True)
class ScreenRun:
    result: ScreeningResult | None
    attempts: int  # LLM calls started; a pre-LLM safety block records zero
    duration_seconds: float
    error: str | None = None
    error_model: str | None = None


class _AttemptCounter:
    """Caller-owned state that survives ``wait_for`` cancelling a hung call."""

    def __init__(self) -> None:
        self.value = 0
        self.last_model: str | None = None

    def tick(self, model: str) -> None:
        self.value += 1
        self.last_model = model

    def retract_static_block(self) -> None:
        """A pre-LLM safety block is a screen result, but not an LLM attempt."""
        self.value -= 1
        self.last_model = None


async def screen_with_fallback(
    task: Task,
    candidate: Candidate,
    *,
    clients: Sequence[LLMClient],
    attempts: int,
    total_timeout_seconds: float,
    per_attempt_timeout_seconds: float | None = None,
) -> ScreenRun:
    """Score with retries, returning an explicit retryable failure on exhaustion."""
    counter = _AttemptCounter()
    started = time.monotonic()
    result: ScreeningResult | None = None
    error: str | None = None
    try:
        result = await asyncio.wait_for(
            screen_with_retries(
                task,
                candidate,
                clients=clients,
                attempts=attempts,
                per_attempt_timeout_seconds=per_attempt_timeout_seconds,
                counter=counter,
            ),
            timeout=total_timeout_seconds,
        )
    except TimeoutError:
        error = f"task screening exceeded {total_timeout_seconds:g}s total timeout"
    except Exception as exc:
        error = f"task screening failed: {exc}"
    return ScreenRun(
        result=result,
        attempts=counter.value,
        duration_seconds=time.monotonic() - started,
        error=error,
        error_model=counter.last_model,
    )


async def screen_with_retries(
    task: Task,
    candidate: Candidate,
    *,
    clients: Sequence[LLMClient],
    attempts: int,
    per_attempt_timeout_seconds: float | None = None,
    counter: _AttemptCounter | None = None,
) -> ScreeningResult:
    last_error: str | None = None
    attempts = max(1, attempts)
    disabled_routes: set[int] = set()
    # Interleave routes by attempt. A primary timeout therefore gives the fallback
    # route a chance within the shared total budget instead of spending every retry
    # on the primary first.
    for attempt in range(1, attempts + 1):
        for route_index, client in enumerate(clients):
            if route_index in disabled_routes:
                continue
            if counter is not None:
                counter.tick(client.model)
            try:
                call = score_candidate(task, candidate, client=client)
                result = (
                    await call
                    if per_attempt_timeout_seconds is None
                    else await asyncio.wait_for(
                        call, timeout=per_attempt_timeout_seconds
                    )
                )
                if result.is_blocked and counter is not None:
                    counter.retract_static_block()
                return result
            except TimeoutError:
                if per_attempt_timeout_seconds is None:
                    detail = "timed out"
                else:
                    detail = f"timed out after {per_attempt_timeout_seconds:g}s"
                last_error = f"{client.model}: {detail}"
                log.warning(
                    "task screen attempt failed model=%s attempt=%s/%s: %s",
                    client.model,
                    attempt,
                    attempts,
                    detail,
                )
            except Exception as exc:
                last_error = f"{client.model}: {exc}"
                log.warning(
                    "task screen attempt failed model=%s attempt=%s/%s: %s",
                    client.model,
                    attempt,
                    attempts,
                    exc,
                )
                if _is_route_error(str(exc)):
                    disabled_routes.add(route_index)

    raise RetryError(last_error or "no task screening clients configured")


def _is_route_error(error: str) -> bool:
    lowered = error.lower()
    return (
        "openrouter returned no choices" in lowered
        or "provider returned error" in lowered
        or "error_code=403" in lowered
    )
