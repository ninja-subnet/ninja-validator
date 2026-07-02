"""Response parsing for submission security qualification."""

from __future__ import annotations

import json
import textwrap
from typing import Any

from .config import DEFAULT_SECURITY_QUALIFICATION_MODEL
from .types import SecurityQualificationResult, SecurityVerdict


def parse_security_qualification(
    raw: str,
    *,
    model: str = DEFAULT_SECURITY_QUALIFICATION_MODEL,
) -> SecurityQualificationResult:
    payload = _extract_json_object(raw)
    if payload is None:
        raise ValueError("security qualification did not return a JSON object")
    return SecurityQualificationResult(
        verdict=_coerce_verdict(payload.get("verdict")),
        overall_score=_coerce_score(payload.get("overall_score")),
        security_score=_coerce_score(payload.get("security_score")),
        summary=str(payload.get("summary") or "").strip(),
        reasons=_string_tuple(payload.get("reasons")),
        risks=_string_tuple(payload.get("risks")),
        required_changes=_string_tuple(payload.get("required_changes")),
        raw_payload=payload,
        model=model,
    )


def _extract_json_object(raw_output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_output)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = textwrap.dedent(raw_output).strip()
    if fenced.startswith("```"):
        body = fenced.strip("`").strip()
        if body.startswith("json"):
            body = body[4:].strip()
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    start = fenced.find("{")
    end = fenced.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(fenced[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_verdict(value: Any) -> SecurityVerdict:
    verdict = str(value or "fail").strip().lower()
    if verdict in {"pass", "warn", "fail"}:
        return verdict  # type: ignore[return-value]
    return "fail"


def _coerce_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)
