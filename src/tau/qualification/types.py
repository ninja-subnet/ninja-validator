"""Public value types for submission security qualification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from .config import DEFAULT_SECURITY_QUALIFICATION_MODEL

SecurityVerdict = Literal["pass", "warn", "fail"]


class QualificationOutcome(StrEnum):
    QUALIFIED = "qualified"
    NEEDS_REVIEW = "needs_review"
    DISQUALIFIED = "disqualified"


@dataclass(frozen=True, slots=True)
class SecurityQualificationInput:
    submitted_files: Mapping[str, str]
    patch: str = ""
    base_files: Mapping[str, str] | None = None
    hotkey: str | None = None
    submission_id: str | None = None
    static_findings: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SecurityQualificationResult:
    verdict: SecurityVerdict
    overall_score: int = 0
    security_score: int = 0
    summary: str = ""
    reasons: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    required_changes: tuple[str, ...] = ()
    raw_payload: dict[str, Any] = field(default_factory=dict)
    model: str = DEFAULT_SECURITY_QUALIFICATION_MODEL
