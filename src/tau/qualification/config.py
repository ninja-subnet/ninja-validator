"""Configuration for submission security qualification."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tau.utils.env import env_int, env_str

DEFAULT_AGENT_ENTRYPOINT = "agent.py"
DEFAULT_SECURITY_QUALIFICATION_MODEL = "google/gemini-3.1-flash-lite"


@dataclass(frozen=True, slots=True)
class SecurityQualificationConfig:
    """Prompt rendering knobs for the security qualification gate."""

    model: str = DEFAULT_SECURITY_QUALIFICATION_MODEL
    agent_entrypoint: str = DEFAULT_AGENT_ENTRYPOINT
    patch_max_chars: int = 120_000
    base_entrypoint_max_chars: int = 80_000
    submitted_entrypoint_max_chars: int = 120_000
    base_files_max_chars: int = 120_000
    submitted_files_max_chars: int = 160_000

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.agent_entrypoint.strip():
            raise ValueError("agent_entrypoint must be non-empty")
        for field_name in (
            "patch_max_chars",
            "base_entrypoint_max_chars",
            "submitted_entrypoint_max_chars",
            "base_files_max_chars",
            "submitted_files_max_chars",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> SecurityQualificationConfig:
        """Build a config from ``TAU_SECURITY_QUALIFICATION_*`` env vars."""
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            model=env_str(env, "TAU_SECURITY_QUALIFICATION_MODEL", d.model),
            agent_entrypoint=env_str(
                env, "TAU_SECURITY_QUALIFICATION_AGENT_ENTRYPOINT", d.agent_entrypoint
            ),
            patch_max_chars=env_int(
                env, "TAU_SECURITY_QUALIFICATION_PATCH_MAX_CHARS", d.patch_max_chars
            ),
            base_entrypoint_max_chars=env_int(
                env,
                "TAU_SECURITY_QUALIFICATION_BASE_ENTRYPOINT_MAX_CHARS",
                d.base_entrypoint_max_chars,
            ),
            submitted_entrypoint_max_chars=env_int(
                env,
                "TAU_SECURITY_QUALIFICATION_SUBMITTED_ENTRYPOINT_MAX_CHARS",
                d.submitted_entrypoint_max_chars,
            ),
            base_files_max_chars=env_int(
                env,
                "TAU_SECURITY_QUALIFICATION_BASE_FILES_MAX_CHARS",
                d.base_files_max_chars,
            ),
            submitted_files_max_chars=env_int(
                env,
                "TAU_SECURITY_QUALIFICATION_SUBMITTED_FILES_MAX_CHARS",
                d.submitted_files_max_chars,
            ),
        )
