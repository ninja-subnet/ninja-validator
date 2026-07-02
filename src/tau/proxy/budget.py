"""Budget enforcement + usage accounting types for the LLM proxy.

Ported verbatim (minus the legacy ``from_config``) from the old
``openrouter_proxy.py``. ``SolveBudget`` caps a single solve's spend; the proxy
mutates a ``SolveUsageSummary`` as requests flow through, and a worker reads a
snapshot afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Exit reasons surfaced when a budget trips (a solve's exit_reason mirrors these).
REQUEST_LIMIT_EXIT_REASON = "request_limit_exceeded"
TOKEN_LIMIT_EXIT_REASON = "token_limit_exceeded"
COST_LIMIT_EXIT_REASON = "cost_limit_exceeded"
PROXY_ERROR_EXIT_REASON = "proxy_error"


def _env_int(env: Mapping[str, str], key: str) -> int | None:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _env_float(env: Mapping[str, str], key: str) -> float | None:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


@dataclass(slots=True)
class SolveBudget:
    """Per-solve spend caps. ``None`` means that dimension is unbounded."""

    max_requests: int | None = None
    max_total_tokens: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_cost: float | None = None
    max_tokens_per_request: int | None = None

    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_requests,
                self.max_total_tokens,
                self.max_prompt_tokens,
                self.max_completion_tokens,
                self.max_cost,
                self.max_tokens_per_request,
            )
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SolveBudget | None:
        """Build a budget from ``SOLVER_MAX_*`` env vars; ``None`` if none are set.

        Recognises ``SOLVER_MAX_REQUESTS``, ``SOLVER_MAX_TOTAL_TOKENS``,
        ``SOLVER_MAX_PROMPT_TOKENS``, ``SOLVER_MAX_COMPLETION_TOKENS``,
        ``SOLVER_MAX_COST``, ``SOLVER_MAX_TOKENS_PER_REQUEST``.
        """
        env = os.environ if environ is None else environ
        budget = cls(
            max_requests=_env_int(env, "SOLVER_MAX_REQUESTS"),
            max_total_tokens=_env_int(env, "SOLVER_MAX_TOTAL_TOKENS"),
            max_prompt_tokens=_env_int(env, "SOLVER_MAX_PROMPT_TOKENS"),
            max_completion_tokens=_env_int(env, "SOLVER_MAX_COMPLETION_TOKENS"),
            max_cost=_env_float(env, "SOLVER_MAX_COST"),
            max_tokens_per_request=_env_int(env, "SOLVER_MAX_TOKENS_PER_REQUEST"),
        )
        return budget if budget.enabled() else None


@dataclass(slots=True)
class ProxyRequestRecord:
    """One forwarded (or rejected) request's outcome and token accounting."""

    method: str
    path: str
    status_code: int | None
    latency_ms: int
    request_model: str | None = None
    response_model: str | None = None
    generation_id: str | None = None
    first_token_latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    rejected: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "request_model": self.request_model,
            "response_model": self.response_model,
            "generation_id": self.generation_id,
            "first_token_latency_ms": self.first_token_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "rejected": self.rejected,
            "error": self.error,
        }


@dataclass(slots=True)
class SolveUsageSummary:
    """Running totals for one solve. Mutated under the proxy lock; snapshot to read."""

    request_count: int = 0
    rejected_request_count: int = 0
    first_token_count: int = 0
    success_count: int = 0
    error_count: int = 0
    # Subset of error_count attributable to the *upstream* infrastructure rather than
    # the agent: transport failures (unreachable/timeout) and infra-class statuses
    # (402 out-of-funds, 429 rate-limit, 401/403 our key, 5xx). Used to tell a
    # miner-unrelated LLM outage (retry the solve) from a bad agent (persist it).
    upstream_error_count: int = 0
    # Subset of upstream_error_count that were timeouts (connect/read timeout, or a 408/
    # 504 upstream status) — surfaced distinctly so the caller can flag LLM-call timeouts.
    upstream_timeout_count: int = 0
    # The most recent upstream error (type + message, or "HTTP <status>"), for telemetry.
    last_upstream_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    budget_exceeded_reason: str | None = None
    requests: list[ProxyRequestRecord] = field(default_factory=list)

    def snapshot(self) -> SolveUsageSummary:
        return SolveUsageSummary(
            request_count=self.request_count,
            rejected_request_count=self.rejected_request_count,
            first_token_count=self.first_token_count,
            success_count=self.success_count,
            error_count=self.error_count,
            upstream_error_count=self.upstream_error_count,
            upstream_timeout_count=self.upstream_timeout_count,
            last_upstream_error=self.last_upstream_error,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cached_tokens=self.cached_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cost=self.cost,
            budget_exceeded_reason=self.budget_exceeded_reason,
            requests=[ProxyRequestRecord(**record.to_dict()) for record in self.requests],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "rejected_request_count": self.rejected_request_count,
            "first_token_count": self.first_token_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "upstream_error_count": self.upstream_error_count,
            "upstream_timeout_count": self.upstream_timeout_count,
            "last_upstream_error": self.last_upstream_error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "budget_exceeded_reason": self.budget_exceeded_reason,
            "requests": [record.to_dict() for record in self.requests],
        }
