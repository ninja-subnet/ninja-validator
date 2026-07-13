from __future__ import annotations

import hashlib
import json

from tau.openrouter import LLMClient
from tau.utils.seeding import stable_seed

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

    seed_text = seed or _default_seed(task, king_solution, challenger_solution)
    model_seed = stable_seed(seed_text)
    blinded = blind(seed_text, king_solution.patch, challenger_solution.patch)
    prompt = build_prompt(task, blinded.candidate_a_patch, blinded.candidate_b_patch)
    raw = await client.complete_text(prompt, seed=model_seed)
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
    return json.dumps(
        {
            "kind": "tau-judge-v1",
            "task_id": task.task_id,
            "problem_statement_sha256": _sha256(task.problem_statement),
            "reference_patch_sha256": _sha256(task.reference_patch),
            "king_patch_sha256": _sha256(king_solution.patch),
            "challenger_patch_sha256": _sha256(challenger_solution.patch),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
