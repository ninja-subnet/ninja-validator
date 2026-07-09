"""Single-candidate task scoring orchestration."""

from __future__ import annotations

from tau.openrouter import LLMClient

from .parsing import parse_score
from .prompt import build_prompt
from .types import (
    Candidate,
    ScreeningResult,
    Task,
)


async def score_candidate(
    task: Task,
    candidate: Candidate,
    *,
    client: LLMClient,
) -> ScreeningResult:
    """Score one king qualification patch with one LLM attempt.

    There is no duel judgment, fake opponent, or role blinding in this path.
    Transport and parse errors propagate so the worker can retry without
    manufacturing a score. The worker performs prompt-injection rejection once
    before entering this retry path.
    """

    raw = await client.complete_text(build_prompt(task, candidate))
    return ScreeningResult(score=parse_score(raw), model=client.model)
