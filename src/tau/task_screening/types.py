"""Public value types for single-candidate task screening."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Task:
    """The task context needed to evaluate one qualification patch."""

    problem_statement: str
    reference_patch: str = ""


@dataclass(frozen=True, slots=True)
class Candidate:
    """A king qualification patch awaiting task screening."""

    patch: str


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """One normalized single-candidate score."""

    score: float
    model: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
