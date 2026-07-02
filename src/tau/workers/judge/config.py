"""Tunable configuration for the judge worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tau.judging.types import DEFAULT_JUDGE_MODEL
from tau.openrouter.dummy import DummyLLMConfig
from tau.utils.env import env_bool, env_float, env_int, env_str

_FALLBACK_MODELS = ("moonshotai/kimi-k2.6",)
_REASONING = {"enabled": True, "exclude": True}
# Cap on one round's total judging time, across retries and model fallbacks.
TOTAL_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class JudgeWorkerConfig:
    openrouter_api_key: str
    model: str = DEFAULT_JUDGE_MODEL
    fallback_models: tuple[str, ...] = _FALLBACK_MODELS
    attempts: int = 4
    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 16_000
    timeout_seconds: int = 120
    total_timeout_seconds: float = TOTAL_TIMEOUT_SECONDS
    reasoning: dict[str, Any] | None = field(default_factory=lambda: dict(_REASONING))
    # Judgments running concurrently (one LLM call each).
    concurrency: int = 5
    # Idle sleep between DB polls (seconds).
    poll_seconds: float = 10.0

    # Token-free testing: swap the real LLM for a DummyJudgeClient. The toggle is a
    # worker-level decision; the behaviour knobs sit in a nested DummyLLMConfig
    # (only consulted when use_dummy_llm is set).
    use_dummy_llm: bool = False
    dummy: DummyLLMConfig = field(default_factory=DummyLLMConfig)

    def __post_init__(self) -> None:
        if not self.use_dummy_llm and not self.openrouter_api_key:
            raise ValueError("openrouter_api_key is required unless use_dummy_llm is set")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> JudgeWorkerConfig:
        """Build a config from ``OPENROUTER_API_KEY`` + ``TAU_JUDGE_*``.

        Raises ``OSError`` if ``OPENROUTER_API_KEY`` is unset. Pass *environ* to
        read from a mapping other than ``os.environ`` (tests).
        """
        env = os.environ if environ is None else environ
        use_dummy_llm = env_bool(env, "TAU_JUDGE_USE_DUMMY_LLM", False)
        api_key = env_str(env, "OPENROUTER_API_KEY", "")
        if not api_key and not use_dummy_llm:
            raise OSError(
                "OPENROUTER_API_KEY not set "
                "(set TAU_JUDGE_USE_DUMMY_LLM=1 to run token-free)"
            )
        d = cls(openrouter_api_key=api_key, use_dummy_llm=use_dummy_llm)
        return cls(
            openrouter_api_key=api_key,
            model=env_str(env, "TAU_JUDGE_MODEL", d.model),
            attempts=env_int(env, "TAU_JUDGE_ATTEMPTS", d.attempts),
            timeout_seconds=env_int(env, "TAU_JUDGE_LLM_TIMEOUT", d.timeout_seconds),
            total_timeout_seconds=env_float(
                env, "TAU_JUDGE_TOTAL_TIMEOUT", d.total_timeout_seconds
            ),
            concurrency=env_int(env, "TAU_JUDGE_CONCURRENCY", d.concurrency),
            poll_seconds=env_float(env, "TAU_JUDGE_POLL_SECONDS", d.poll_seconds),
            use_dummy_llm=use_dummy_llm,
            dummy=DummyLLMConfig(
                avg_latency_seconds=env_float(
                    env, "TAU_JUDGE_DUMMY_AVG_LATENCY", d.dummy.avg_latency_seconds
                ),
                latency_sigma=env_float(
                    env, "TAU_JUDGE_DUMMY_LATENCY_SIGMA", d.dummy.latency_sigma
                ),
                slow_rate=env_float(env, "TAU_JUDGE_DUMMY_SLOW_RATE", d.dummy.slow_rate),
                outlier_factor=env_float(
                    env, "TAU_JUDGE_DUMMY_OUTLIER_FACTOR", d.dummy.outlier_factor
                ),
                failure_rate=env_float(
                    env, "TAU_JUDGE_DUMMY_FAILURE_RATE", d.dummy.failure_rate
                ),
            ),
        )
