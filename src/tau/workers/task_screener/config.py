"""Tunable configuration for the asynchronous task-screener worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tau.task_screening import DEFAULT_SCREENING_MODEL
from tau.utils.env import env_bool, env_float, env_int, env_str

_FALLBACK_MODELS: tuple[str, ...] = (DEFAULT_SCREENING_MODEL,)
_REASONING = {"enabled": True, "exclude": True}
_DEFAULT_PROVIDER_ONLY = ("z-ai/fp8",)
_DEFAULT_FALLBACK_PROVIDER_ONLY = ("atlas-cloud/fp8",)


@dataclass(frozen=True, slots=True)
class TaskScreenerConfig:
    openrouter_api_key: str
    model: str = DEFAULT_SCREENING_MODEL
    fallback_models: tuple[str, ...] = _FALLBACK_MODELS
    provider: dict[str, Any] | None = field(default_factory=lambda: _default_provider())
    fallback_provider: dict[str, Any] | None = field(
        default_factory=lambda: _default_fallback_provider()
    )
    attempts: int = 4
    max_king_score: float = 0.70
    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 4_096
    timeout_seconds: int = 120
    total_timeout_seconds: float = 300.0
    reasoning: dict[str, Any] | None = field(default_factory=lambda: dict(_REASONING))
    concurrency: int = 5
    poll_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.openrouter_api_key:
            raise ValueError("openrouter_api_key is required")
        if not 0 <= self.max_king_score <= 1:
            raise ValueError("max_king_score must be between 0 and 1")
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        if self.total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds must be positive")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TaskScreenerConfig:
        """Build config from ``OPENROUTER_API_KEY`` + ``TAU_TASK_SCREEN_*``."""
        env = os.environ if environ is None else environ
        api_key = env_str(env, "OPENROUTER_API_KEY", "")
        if not api_key:
            raise OSError("OPENROUTER_API_KEY not set")
        d = cls(openrouter_api_key=api_key)
        return cls(
            openrouter_api_key=api_key,
            model=env_str(env, "TAU_TASK_SCREEN_MODEL", d.model),
            fallback_models=_env_csv(
                env, "TAU_TASK_SCREEN_FALLBACK_MODELS", d.fallback_models
            ),
            provider=_provider_from_env(env, default=d.provider),
            fallback_provider=_provider_from_env(
                env,
                default=d.fallback_provider,
                prefix="TAU_TASK_SCREEN_FALLBACK_PROVIDER",
            ),
            attempts=env_int(env, "TAU_TASK_SCREEN_ATTEMPTS", d.attempts),
            max_king_score=env_float(
                env, "TAU_TASK_SCREEN_MAX_KING_SCORE", d.max_king_score
            ),
            max_tokens=env_int(env, "TAU_TASK_SCREEN_MAX_TOKENS", d.max_tokens),
            timeout_seconds=env_int(
                env, "TAU_TASK_SCREEN_LLM_TIMEOUT", d.timeout_seconds
            ),
            total_timeout_seconds=env_float(
                env, "TAU_TASK_SCREEN_TOTAL_TIMEOUT", d.total_timeout_seconds
            ),
            concurrency=env_int(env, "TAU_TASK_SCREEN_CONCURRENCY", d.concurrency),
            poll_seconds=env_float(env, "TAU_TASK_SCREEN_POLL_SECONDS", d.poll_seconds),
        )


def _env_csv(
    env: Mapping[str, str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    if name not in env:
        return default
    return tuple(part.strip() for part in env[name].split(",") if part.strip())


def _default_provider() -> dict[str, Any]:
    return {"only": list(_DEFAULT_PROVIDER_ONLY), "allow_fallbacks": False}


def _default_fallback_provider() -> dict[str, Any]:
    return {
        "only": list(_DEFAULT_FALLBACK_PROVIDER_ONLY),
        "allow_fallbacks": False,
    }


def _provider_from_env(
    env: Mapping[str, str],
    *,
    default: dict[str, Any] | None,
    prefix: str = "TAU_TASK_SCREEN_PROVIDER",
) -> dict[str, Any] | None:
    provider_vars = {
        f"{prefix}_ONLY",
        f"{prefix}_ORDER",
        f"{prefix}_QUANTIZATIONS",
        f"{prefix}_ALLOW_FALLBACKS",
    }
    if not any(name in env for name in provider_vars):
        return dict(default) if default is not None else None

    provider: dict[str, Any] = {}
    only = _env_csv(env, f"{prefix}_ONLY", ())
    if only:
        provider["only"] = list(only)
    order = _env_csv(env, f"{prefix}_ORDER", ())
    if order:
        provider["order"] = list(order)
    quantizations = _env_csv(env, f"{prefix}_QUANTIZATIONS", ())
    if quantizations:
        provider["quantizations"] = list(quantizations)
    if f"{prefix}_ALLOW_FALLBACKS" in env:
        provider["allow_fallbacks"] = env_bool(env, f"{prefix}_ALLOW_FALLBACKS", True)
    return provider or None
