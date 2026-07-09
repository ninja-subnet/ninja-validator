"""Single-candidate task scoring orchestration."""

from __future__ import annotations

import hashlib
import json

from tau.judging.safety import detect_injection
from tau.openrouter import LLMClient

from .parsing import parse_score
from .prompt import build_prompt
from .types import (
    BlockReason,
    Candidate,
    ScreeningOutcome,
    ScreeningResult,
    STATIC_PROMPT_INJECTION_MODEL,
    Task,
)

_FINGERPRINT_VERSION = 1


async def score_candidate(
    task: Task,
    candidate: Candidate,
    *,
    client: LLMClient,
) -> ScreeningResult:
    """Score one king qualification patch with one LLM attempt.

    There is no duel judgment, fake opponent, or role blinding in this path.
    Prompt-injection evidence is returned as an explicit, terminal blocked
    result without calling the LLM. Transport and parse errors intentionally
    propagate so the worker can retry without manufacturing a score.
    """

    fingerprint = screening_fingerprint(task, candidate)
    injection_evidence = detect_injection(candidate.patch)
    if injection_evidence is not None:
        return ScreeningResult(
            outcome=ScreeningOutcome.BLOCKED,
            score=None,
            rationale="Qualification patch blocked before LLM task screening.",
            model=STATIC_PROMPT_INJECTION_MODEL,
            fingerprint=fingerprint,
            blocked_reason=BlockReason.PROMPT_INJECTION,
            blocked_evidence=injection_evidence,
        )

    raw = await client.complete_text(build_prompt(task, candidate))
    parsed = parse_score(raw)
    return ScreeningResult(
        outcome=ScreeningOutcome.SCORED,
        score=parsed.score,
        rationale=parsed.rationale,
        model=client.model,
        fingerprint=fingerprint,
    )


def screening_fingerprint(task: Task, candidate: Candidate) -> str:
    """Return a stable digest of only the semantic input shown to the scorer."""

    payload = {
        "candidate_patch": candidate.patch,
        "problem_statement": task.problem_statement,
        "reference_patch": task.reference_patch,
        "version": _FINGERPRINT_VERSION,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
