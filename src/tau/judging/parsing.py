"""Parse the LLM judge's raw response into a RawVerdict — pure, no IO.

Ported from validate.py: _extract_json_object, _score_0_to_1, and the payload
extraction in _parse_diff_judge_payload. The verdict is expressed in candidate
A/B terms (the LLM's vocabulary); mapping A/B back to the two solutions is the
blinding module's job, so this module stays independent of roles.

A declared winner is kept as stated; a missing or invalid one is filled from
the scores.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

CandidateWinner = Literal["candidate_a", "candidate_b", "tie"]


@dataclass(frozen=True, slots=True)
class RawVerdict:
    """The judge verdict as the LLM stated it, in candidate A/B terms.

    Scores are normalized to [0, 1] (the model is asked for 0-100).
    """

    winner: CandidateWinner
    score_a: float
    score_b: float
    rationale: str = ""


def parse_verdict(raw: str) -> RawVerdict:
    """Extract a RawVerdict from the model's raw text.

    Raises ValueError if no JSON object can be recovered, so the orchestrator can
    fall back to a neutral judgment.
    """
    payload = extract_json_object(raw)
    if payload is None:
        raise ValueError("judge did not return a JSON object")

    # Absent winner -> tie, matching the validator's payload.get("winner", "tie").
    winner_token = str(payload.get("winner", "tie")).strip().lower()
    score_a = _score_0_to_1(payload.get("candidate_a_score"))
    score_b = _score_0_to_1(payload.get("candidate_b_score"))
    if score_a is None or score_b is None:
        score_a, score_b = _scores_from_winner(winner_token)

    rationale = str(payload.get("rationale") or "").strip()
    return RawVerdict(
        winner=_declared_winner(winner_token, score_a, score_b),
        score_a=score_a,
        score_b=score_b,
        rationale=rationale,
    )


def _declared_winner(
    winner_token: str, score_a: float, score_b: float
) -> CandidateWinner:
    """The model's declared winner; an unrecognized token derives from the scores."""
    if winner_token in ("candidate_a", "candidate_b", "tie"):
        return winner_token
    if score_a > score_b:
        return "candidate_a"
    if score_b > score_a:
        return "candidate_b"
    return "tie"


def _scores_from_winner(winner: str) -> tuple[float, float]:
    if winner == "candidate_a":
        return 1.0, 0.0
    if winner == "candidate_b":
        return 0.0, 1.0
    return 0.5, 0.5


def _score_0_to_1(raw: Any) -> float | None:
    """0-100 (model's scale) -> [0, 1]; None if not a number."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return _clamp01(value / 100.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def extract_json_object(raw_output: str) -> dict[str, Any] | None:
    """Recover a JSON object from raw text, including ```json fenced blocks."""
    try:
        payload = json.loads(raw_output)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = textwrap.dedent(raw_output)
    for start in ("```json", "```"):
        if start not in fenced:
            continue
        for part in fenced.split(start)[1:]:
            body = part.split("```", 1)[0].strip()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return None
