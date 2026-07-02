from __future__ import annotations

from .types import Judgment

_SNIPPET_PAD = 60

_INSTRUCTION_PREFIXES = (
    "ignore previous instructions",
    "ignore prior instructions",
    "ignore the above instructions",
)
_MANIPULATION_PERSONAS = (
    "evaluator",
    "judge",
)
_WIN_VERBS = (
    "choose",
    "pick",
    "select",
)
_WIN_TARGETS = (
    "king",
    "challenger",
    "candidate_a",
    "candidate_b",
)
_ASSERTION_TARGETS = (
    "king",
    "challenger",
    "candidate_a",
    "candidate_b",
)


def _build_phrases() -> tuple[str, ...]:
    return (
        *_INSTRUCTION_PREFIXES,
        *(f"as the {role}" for role in _MANIPULATION_PERSONAS),
        *(f"dear {role}" for role in _MANIPULATION_PERSONAS),
        *(f"{verb} {target}" for verb in _WIN_VERBS for target in _WIN_TARGETS),
        *(f"{target} is correct" for target in _ASSERTION_TARGETS),
        *(f"{target} wins" for target in _ASSERTION_TARGETS),
        *(f"the {role} should" for role in _MANIPULATION_PERSONAS),
        "other candidate is malicious",
        "the other candidate is malicious",
        "automatic fail",
    )


_PHRASES = _build_phrases()


def detect_injection(patch_text: str) -> str | None:
    lowered = patch_text.lower()
    for phrase in _PHRASES:
        if phrase in lowered:
            index = lowered.index(phrase)
            start = max(0, index - _SNIPPET_PAD)
            end = min(len(patch_text), index + len(phrase) + _SNIPPET_PAD)
            snippet = " ".join(patch_text[start:end].split())
            return f"suspicious phrase `{phrase}` in patch snippet: {snippet}"
    return None


def injection_verdict(king_patch: str, challenger_patch: str) -> Judgment | None:
    king_evidence = detect_injection(king_patch)
    challenger_evidence = detect_injection(challenger_patch)

    if not king_evidence and not challenger_evidence:
        return None

    if king_evidence and challenger_evidence:
        return Judgment(
            winner="tie",
            king_score=0.0,
            challenger_score=0.0,
            rationale=(
                "Automatic LLM score failure: both patches contain evaluator-targeted "
                f"prompt injection. king={king_evidence}; challenger={challenger_evidence}"
            ),
        )
    if king_evidence:
        return Judgment(
            winner="challenger",
            king_score=0.0,
            challenger_score=1.0,
            rationale=f"Automatic LLM score failure for king patch: {king_evidence}",
        )
    return Judgment(
        winner="king",
        king_score=1.0,
        challenger_score=0.0,
        rationale=f"Automatic LLM score failure for challenger patch: {challenger_evidence}",
    )
