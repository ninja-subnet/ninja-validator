from enum import StrEnum
from typing import Any, Literal, TypeAlias


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


EventType: TypeAlias = Literal[
    "sandbox_started",
    "qualification",
    "solution",
    "init_worker",
    "exit_worker",
    "weights_set",
    "weights_skipped",
    "challenge_opened",
    "pool_advanced",
    "king_promoted",
    "challenge_closed",
    "judgment_saved",
    "judgment_degraded",
    "task_inserted",
    "task_screen_saved",
    "submission_qualification",
    # errors
    "challenger_infra_error",
    "qualification_infra_error",
    "solve_job_failed",
    "unexpected_error",
    "weights_rejected",
    "judgment_failed",
    "generation_failed",
    "fetch_failed",
    "submission_qualification_error",
    "task_screen_failed",
    "task_screen_retryable_error",
]


Source: TypeAlias = Literal[
    "qualification",
    "task-solver",
    "weight-setter",
    "duel-resolver",
    "task-generator",
    "task-screener",
    "judge",
]


Details: TypeAlias = dict[str, Any]
