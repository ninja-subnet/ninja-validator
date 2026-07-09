"""Strict parsing for the single-candidate screening response."""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedScore:
    score: float
    rationale: str


def parse_score(raw: str) -> ParsedScore:
    """Parse ``{score: 0-100, rationale: str}`` and normalize to ``[0, 1]``.

    Invalid or out-of-range model output raises ``ValueError``. The caller is a
    worker with retry policy; silently fabricating or clamping a low score could
    incorrectly admit an unscreened task.
    """

    payload = _extract_json_object(raw)
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
    return ParsedScore(score=numeric_score / 100.0, rationale=rationale.strip())


def _extract_json_object(raw_output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_output)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass

    fenced = textwrap.dedent(raw_output)
    for marker in ("```json", "```"):
        if marker not in fenced:
            continue
        for part in fenced.split(marker)[1:]:
            body = part.split("```", 1)[0].strip()
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                return payload
    return None
