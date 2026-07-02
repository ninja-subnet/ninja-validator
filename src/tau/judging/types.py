from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEFAULT_JUDGE_MODEL = "google/gemini-3.1-flash-lite"


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    problem_statement: str
    reference_patch: str


@dataclass(frozen=True, slots=True)
class Solution:
    submission_id: str
    patch: str
    exit_reason: str | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Judgment:
    winner: Literal["king", "challenger", "tie"]
    king_score: float
    challenger_score: float
    rationale: str = ""
    model: str = DEFAULT_JUDGE_MODEL
    error: str | None = None
