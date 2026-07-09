"""Strict parsing for the single-candidate screening response."""

from __future__ import annotations

import math

from tau.judging.parsing import extract_json_object


def parse_score(raw: str) -> float:
    """Parse ``{score: 0-100, rationale: str}`` and normalize to ``[0, 1]``.

    Invalid or out-of-range model output raises ``ValueError``. The caller is a
    worker with retry policy; silently fabricating or clamping a low score could
    incorrectly admit an unscreened task.
    """

    payload = extract_json_object(raw)
    if payload is None:
        raise ValueError("task screener did not return a JSON object")
    if "score" not in payload:
        raise ValueError("task screener response is missing score")
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("task screener score must be a JSON number")
    numeric_score = float(score)
    if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 100.0:
        raise ValueError("task screener score must be finite and in [0, 100]")

    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("task screener rationale must be a non-empty string")
    return numeric_score / 100.0
