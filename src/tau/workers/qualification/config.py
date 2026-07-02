"""Tunable configuration for the submission qualification worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tau.qualification import SecurityQualificationConfig
from tau.utils.env import env_float, env_int, env_str


@dataclass(frozen=True, slots=True)
class QualificationWorkerConfig:
    openrouter_api_key: str
    submissions_dir: Path = Path("submissions")
    base_path: Path | None = None
    window_size: int = 3
    poll_seconds: float = 10.0
    llm_timeout_seconds: int = 120
    llm_max_tokens: int = 16_000
    security: SecurityQualificationConfig = SecurityQualificationConfig()

    def __post_init__(self) -> None:
        if not self.openrouter_api_key:
            raise ValueError("openrouter_api_key is required")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be positive")
        if self.llm_max_tokens <= 0:
            raise ValueError("llm_max_tokens must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> QualificationWorkerConfig:
        """Build a config from ``OPENROUTER_API_KEY`` + ``TAU_QUALIFICATION_*``."""
        env = os.environ if environ is None else environ
        api_key = env_str(env, "OPENROUTER_API_KEY", "")
        if not api_key:
            raise OSError("OPENROUTER_API_KEY not set")
        d = cls(openrouter_api_key=api_key)
        base = env_str(env, "TAU_QUALIFICATION_BASE_PATH", "")
        return cls(
            openrouter_api_key=api_key,
            submissions_dir=Path(
                env_str(env, "TAU_SUBMISSIONS_DIR", str(d.submissions_dir))
            ),
            base_path=Path(base) if base else None,
            window_size=env_int(env, "TAU_QUALIFICATION_WINDOW_SIZE", d.window_size),
            poll_seconds=env_float(
                env, "TAU_QUALIFICATION_POLL_SECONDS", d.poll_seconds
            ),
            llm_timeout_seconds=env_int(
                env, "TAU_QUALIFICATION_LLM_TIMEOUT", d.llm_timeout_seconds
            ),
            llm_max_tokens=env_int(
                env, "TAU_QUALIFICATION_MAX_TOKENS", d.llm_max_tokens
            ),
            security=SecurityQualificationConfig.from_env(env),
        )
