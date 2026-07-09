"""Configuration for the task-screener worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from tau.utils.env import env_float, env_int, env_str
from tau.workers.judge.config import JudgeWorkerConfig


class TaskScreenMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass(frozen=True, slots=True)
class TaskScreenerConfig:
    """Screen policy plus the production judge's shared LLM configuration."""

    mode: TaskScreenMode = TaskScreenMode.SHADOW
    llm: JudgeWorkerConfig | None = None
    max_king_score: float = 0.70
    concurrency: int = 5
    poll_seconds: float = 10.0
    max_failed_runs: int = 3
    retry_base_seconds: float = 60.0
    retry_max_seconds: float = 900.0

    def __post_init__(self) -> None:
        try:
            mode = TaskScreenMode(self.mode)
        except ValueError as exc:
            choices = ", ".join(member.value for member in TaskScreenMode)
            raise ValueError(f"mode must be one of: {choices}") from exc
        object.__setattr__(self, "mode", mode)
        if mode is not TaskScreenMode.DISABLED and self.llm is None:
            raise ValueError("active task screening requires LLM configuration")
        if not 0 <= self.max_king_score <= 1:
            raise ValueError("max_king_score must be between 0 and 1")
        if self.concurrency < 1 or self.poll_seconds <= 0:
            raise ValueError("concurrency and poll_seconds must be positive")
        if self.max_failed_runs < 1:
            raise ValueError("max_failed_runs must be >= 1")
        if (
            self.retry_base_seconds <= 0
            or self.retry_max_seconds < self.retry_base_seconds
        ):
            raise ValueError("retry delays must be positive and ordered")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TaskScreenerConfig:
        env = os.environ if environ is None else environ
        defaults = cls(mode=TaskScreenMode.DISABLED)
        raw_mode = env_str(env, "TAU_TASK_SCREEN_MODE", TaskScreenMode.SHADOW).lower()
        try:
            mode = TaskScreenMode(raw_mode)
        except ValueError as exc:
            choices = ", ".join(member.value for member in TaskScreenMode)
            raise ValueError(f"TAU_TASK_SCREEN_MODE must be one of: {choices}") from exc

        llm = None
        if mode is not TaskScreenMode.DISABLED:
            # Screening deliberately shares the production duel judge's model,
            # routes, reasoning controls, retry count, and time/token budgets.
            judge_env = dict(env)
            judge_env["TAU_JUDGE_USE_DUMMY_LLM"] = "false"
            llm = JudgeWorkerConfig.from_env(judge_env)

        return cls(
            mode=mode,
            llm=llm,
            max_king_score=env_float(
                env, "TAU_TASK_SCREEN_MAX_KING_SCORE", defaults.max_king_score
            ),
            concurrency=env_int(
                env, "TAU_TASK_SCREEN_CONCURRENCY", defaults.concurrency
            ),
            poll_seconds=env_float(
                env, "TAU_TASK_SCREEN_POLL_SECONDS", defaults.poll_seconds
            ),
            max_failed_runs=env_int(
                env, "TAU_TASK_SCREEN_MAX_FAILED_RUNS", defaults.max_failed_runs
            ),
            retry_base_seconds=env_float(
                env,
                "TAU_TASK_SCREEN_RETRY_BASE_SECONDS",
                defaults.retry_base_seconds,
            ),
            retry_max_seconds=env_float(
                env, "TAU_TASK_SCREEN_RETRY_MAX_SECONDS", defaults.retry_max_seconds
            ),
        )
