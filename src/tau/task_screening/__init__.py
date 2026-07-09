"""Single-candidate scoring used to screen king-solvable tasks into the pool."""

from .prompt import ScreeningPrompt, build_prompt
from .scoring import score_candidate, screening_fingerprint
from .types import (
    DEFAULT_SCREENING_MODEL,
    BlockReason,
    Candidate,
    ScreeningOutcome,
    ScreeningResult,
    STATIC_PROMPT_INJECTION_MODEL,
    Task,
)

__all__ = [
    "DEFAULT_SCREENING_MODEL",
    "BlockReason",
    "Candidate",
    "ScreeningOutcome",
    "ScreeningPrompt",
    "ScreeningResult",
    "STATIC_PROMPT_INJECTION_MODEL",
    "Task",
    "build_prompt",
    "score_candidate",
    "screening_fingerprint",
]
