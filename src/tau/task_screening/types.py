"""Public value types for single-candidate task screening."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

DEFAULT_SCREENING_MODEL = "z-ai/glm-5.2"
STATIC_PROMPT_INJECTION_MODEL = "static/prompt-injection"


class ScreeningOutcome(StrEnum):
    """Whether a patch was scored or blocked before reaching the model."""

    SCORED = "scored"
    BLOCKED = "blocked"


class BlockReason(StrEnum):
    """Terminal reasons that prevent an LLM screening score."""

    PROMPT_INJECTION = "prompt_injection"


@dataclass(frozen=True, slots=True)
class Task:
    """The task context needed to evaluate one qualification patch."""

    task_id: str
    problem_statement: str
    reference_patch: str = ""


@dataclass(frozen=True, slots=True)
class Candidate:
    """A king qualification patch awaiting task screening."""

    submission_id: str
    patch: str


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """One terminal core result from a single screening attempt.

    ``score`` is normalized to ``[0, 1]`` for a scored response. A locally
    blocked patch deliberately has no score: assigning a high, low, or neutral
    value would conflate a security decision with the model's assessment.

    ``fingerprint`` identifies the semantic task/reference/patch content. It is
    stable across database row identifiers and can be persisted by a worker for
    idempotency and telemetry.
    """

    outcome: ScreeningOutcome
    score: float | None
    rationale: str
    model: str
    fingerprint: str
    blocked_reason: BlockReason | None = None
    blocked_evidence: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is ScreeningOutcome.SCORED:
            if self.score is None or not 0.0 <= self.score <= 1.0:
                raise ValueError("a scored screening result requires score in [0, 1]")
            if self.blocked_reason is not None or self.blocked_evidence is not None:
                raise ValueError(
                    "a scored screening result cannot have blocked details"
                )
            return

        if self.score is not None:
            raise ValueError("a blocked screening result cannot have a score")
        if self.blocked_reason is None:
            raise ValueError("a blocked screening result requires a blocked reason")
        if not self.blocked_evidence:
            raise ValueError("a blocked screening result requires evidence")

    @property
    def is_blocked(self) -> bool:
        return self.outcome is ScreeningOutcome.BLOCKED
