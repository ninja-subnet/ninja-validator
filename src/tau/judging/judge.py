from __future__ import annotations

from tau.openrouter import LLMClient

from .blinding import blind, unblind
from .parsing import parse_verdict
from .prompt import build_prompt
from .safety import injection_verdict
from .types import Judgment, Solution, Task


async def judge(
    task: Task,
    king_solution: Solution,
    challenger_solution: Solution,
    *,
    client: LLMClient,
    seed: str | None = None,
) -> Judgment:
    """Score two solutions for a task with one LLM attempt.

    The judge owns only judging logic — blinding, prompt construction, parsing,
    unblinding. All transport/model choices (model, sampling, timeout, wire
    format) belong to the client, which the worker configures and selects; the
    judge does not catch transport/model errors, so workers decide retry and
    neutral fallback.
    """
    guarded = injection_verdict(king_solution.patch, challenger_solution.patch)
    if guarded is not None:
        return guarded

    blinded = blind(
        seed or _default_seed(task, king_solution, challenger_solution),
        king_solution.patch,
        challenger_solution.patch,
    )
    prompt = build_prompt(task, blinded.candidate_a_patch, blinded.candidate_b_patch)
    raw = await client.complete_text(prompt)
    verdict = parse_verdict(raw)
    result = unblind(blinded, verdict)
    return Judgment(
        winner=result.winner,
        king_score=result.king_score,
        challenger_score=result.challenger_score,
        rationale=verdict.rationale,
        model=client.model,
    )


def neutral_judgment(reason: str | None = None) -> Judgment:
    return Judgment(
        winner="tie",
        king_score=0.5,
        challenger_score=0.5,
        rationale="LLM diff judge unavailable; using neutral score.",
        error=reason,
    )


def _default_seed(
    task: Task,
    king_solution: Solution,
    challenger_solution: Solution,
) -> str:
    return f"{task.task_id}:{king_solution.submission_id}:{challenger_solution.submission_id}"
