"""Judge one duel round robustly: retry across attempts and model fallbacks.

Wraps the pure `tau.judging.judge` with the worker's transport policy -- retries
per client, fall through to the next model, a total-time cap, and a neutral tie
when everything is exhausted. The judging core stays unaware of any of this.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from tau.judging import Judgment, Solution, Task, judge, neutral_judgment
from tau.openrouter import LLMClient

from .config import TOTAL_TIMEOUT_SECONDS

log = logging.getLogger(__name__)


class RetryError(RuntimeError): ...


@dataclass(frozen=True, slots=True)
class JudgeRun:
    """A completed judgment plus the worker telemetry around producing it."""

    judgment: Judgment
    attempts: int  # LLM attempts started across retries + model fallbacks
    duration_seconds: float


class _AttemptCounter:
    """Caller-owned tally so the attempt count survives a timeout cancellation."""

    def __init__(self) -> None:
        self.value = 0

    def tick(self) -> None:
        self.value += 1


async def judge_with_fallback(
    task: Task,
    king_solution: Solution,
    challenger_solution: Solution,
    *,
    clients: Sequence[LLMClient],
    attempts: int,
    total_timeout_seconds: float = TOTAL_TIMEOUT_SECONDS,
) -> JudgeRun:
    counter = _AttemptCounter()
    started = time.monotonic()
    try:
        judgment = await asyncio.wait_for(
            judge_with_retries(
                task,
                king_solution,
                challenger_solution,
                clients=clients,
                attempts=attempts,
                counter=counter,
            ),
            timeout=total_timeout_seconds,
        )
    except TimeoutError:
        judgment = neutral_judgment(
            f"LLM diff judge exceeded {total_timeout_seconds:g}s total timeout"
        )
    except Exception as exc:
        judgment = neutral_judgment(f"LLM diff judge failed: {exc}")
    return JudgeRun(judgment, counter.value, time.monotonic() - started)


async def judge_with_retries(
    task: Task,
    king_solution: Solution,
    challenger_solution: Solution,
    *,
    clients: Sequence[LLMClient],
    attempts: int,
    counter: _AttemptCounter | None = None,
) -> Judgment:
    last_error: str | None = None
    attempts = max(1, attempts)
    for client in clients:
        for attempt in range(1, attempts + 1):
            if counter is not None:
                counter.tick()
            try:
                return await judge(
                    task, king_solution, challenger_solution, client=client
                )
            except Exception as exc:
                last_error = f"{client.model}: {exc}"
                log.warning(
                    "judge attempt failed model=%s attempt=%s/%s: %s",
                    client.model,
                    attempt,
                    attempts,
                    exc,
                )
                if _is_route_error(str(exc)):
                    break

    raise RetryError(last_error)


def _is_route_error(error: str) -> bool:
    lowered = error.lower()
    return (
        "openrouter returned no choices" in lowered
        or "provider returned error" in lowered
        or "error_code=403" in lowered
    )
