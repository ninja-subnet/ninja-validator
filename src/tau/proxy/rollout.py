"""Minimal rollout-event helpers vendored for the proxy's optional event sink.

The proxy can emit one ``llm_call`` event per upstream request to a caller-supplied
sink. There is no rollout *store* in this repo yet (a documented follow-up), so the
sink defaults to ``None`` and these helpers are exercised only when a caller wires
one in. Ported down to the essentials from the legacy ``tau.rollouts.schema`` /
``tau.rollouts.redaction``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    """Recursively replace any secret substring in strings with ``[redacted]``.

    Keeps event payloads safe to persist: the upstream key and per-solve token are
    passed in so they never leak into a recorded request/response body.
    """
    if not secrets:
        return value
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[redacted]")
        return redacted
    if isinstance(value, dict):
        return {key: redact_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    return value


def build_llm_event(
    *,
    method: str,
    path: str,
    request_payload: Any,
    response_payload: Any,
    status_code: int | None,
    latency_ms: int,
    request_model: str | None,
    response_model: str | None,
    usage: dict[str, Any],
    cost: float | None,
    started_at: str,
    finished_at: str,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build one ``llm_call`` rollout event (secrets redacted from the bodies)."""
    return {
        "type": "llm_call",
        "source": "tau_proxy",
        "started_at": started_at,
        "finished_at": finished_at,
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "request": redact_value(request_payload, secrets),
        "response": redact_value(response_payload, secrets),
        "usage": usage,
        "cost": cost,
        "model_requested": request_model,
        "model_effective": response_model or request_model,
    }
