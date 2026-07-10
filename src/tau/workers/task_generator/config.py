"""Tunable configuration for the task-generator worker.

Generation-local knobs only (model, timeouts, concurrency, poll cadence); pool
targets live in the shared tau.pools.PoolTargets.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from tau.openrouter.client import DEFAULT_MODEL
from tau.openrouter.dummy import DummyLLMConfig
from tau.utils.env import env_bool, env_float, env_int, env_str


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Generation knobs (see module docstring for what's deliberately absent)."""

    openrouter_api_key: str
    generator_model: str = DEFAULT_MODEL
    # Number of concurrent describer coroutines (LLM fan-out). The single fetcher
    # is the GitHub rate-limit bottleneck, so this bounds only the LLM concurrency.
    describe_concurrency: int = 5
    # LLM tries per commit before moving to a fresh commit (2 = one retry).
    llm_attempts: int = 2
    # Per-attempt LLM timeout (seconds).
    llm_timeout: float = 120.0
    # Idle sleep between controller polls (seconds).
    poll_seconds: float = 30.0
    # Real qualification candidates kept in flight while any pool is incomplete.
    qualification_inflight_target: int = 100

    # Token-free testing: swap the real LLM for tau.openrouter.DummyLLMClient. The
    # toggle is a generator-level decision; the behaviour knobs are tucked into a
    # nested DummyLLMConfig (only consulted when use_dummy_llm is set).
    use_dummy_llm: bool = False
    dummy: DummyLLMConfig = field(default_factory=DummyLLMConfig)

    def __post_init__(self) -> None:
        if not self.use_dummy_llm and not self.openrouter_api_key:
            raise ValueError("openrouter_api_key is required unless use_dummy_llm is set")
        if self.describe_concurrency < 1:
            raise ValueError("describe_concurrency must be >= 1")
        if self.llm_attempts < 1:
            raise ValueError("llm_attempts must be >= 1")
        if self.llm_timeout <= 0:
            raise ValueError("llm_timeout must be positive")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.qualification_inflight_target < 1:
            raise ValueError("qualification_inflight_target must be >= 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GeneratorConfig:
        """Build a config from ``OPENROUTER_API_KEY`` + ``TAU_GENERATOR_*``.

        Raises ``OSError`` if ``OPENROUTER_API_KEY`` is unset (the worker cannot
        generate without it). Pass *environ* to read from a mapping other than
        ``os.environ`` (tests).
        """
        env = os.environ if environ is None else environ
        use_dummy_llm = env_bool(env, "TAU_GENERATOR_USE_DUMMY_LLM", False)
        api_key = env_str(env, "OPENROUTER_API_KEY", "")
        if not api_key and not use_dummy_llm:
            raise OSError(
                "OPENROUTER_API_KEY not set "
                "(set TAU_GENERATOR_USE_DUMMY_LLM=1 to run token-free)"
            )
        d = cls(openrouter_api_key=api_key, use_dummy_llm=use_dummy_llm)
        return cls(
            openrouter_api_key=api_key,
            generator_model=env_str(env, "TAU_GENERATOR_MODEL", d.generator_model),
            describe_concurrency=env_int(
                env, "TAU_GENERATOR_DESCRIBE_CONCURRENCY", d.describe_concurrency
            ),
            llm_attempts=env_int(env, "TAU_GENERATOR_LLM_ATTEMPTS", d.llm_attempts),
            llm_timeout=env_float(
                env, "TAU_GENERATOR_LLM_TIMEOUT", d.llm_timeout
            ),
            poll_seconds=env_float(env, "TAU_GENERATOR_POLL_SECONDS", d.poll_seconds),
            qualification_inflight_target=env_int(
                env,
                "TAU_GENERATOR_QUALIFICATION_INFLIGHT_TARGET",
                d.qualification_inflight_target,
            ),
            use_dummy_llm=use_dummy_llm,
            dummy=DummyLLMConfig(
                avg_latency_seconds=env_float(
                    env, "TAU_GENERATOR_DUMMY_AVG_LATENCY", d.dummy.avg_latency_seconds
                ),
                latency_sigma=env_float(
                    env, "TAU_GENERATOR_DUMMY_LATENCY_SIGMA", d.dummy.latency_sigma
                ),
                slow_rate=env_float(
                    env, "TAU_GENERATOR_DUMMY_SLOW_RATE", d.dummy.slow_rate
                ),
                outlier_factor=env_float(
                    env, "TAU_GENERATOR_DUMMY_OUTLIER_FACTOR", d.dummy.outlier_factor
                ),
                failure_rate=env_float(
                    env, "TAU_GENERATOR_DUMMY_FAILURE_RATE", d.dummy.failure_rate
                ),
            ),
        )
