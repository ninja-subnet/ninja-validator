"""Security-only LLM qualification for submitted miner agents.

This module is the small async facade. The pure pieces live next to it in
``config``, ``prompt``, ``parse``, ``policy``, and ``types``.
"""

from __future__ import annotations

from tau.openrouter import LLMClient

from .config import (
    DEFAULT_AGENT_ENTRYPOINT,
    DEFAULT_SECURITY_QUALIFICATION_MODEL,
    SecurityQualificationConfig,
)
from .parse import parse_security_qualification
from .policy import (
    SECURITY_RISK_CATEGORIES,
    RISK_CATEGORY_ALIASES,
    qualification_outcome,
    security_failures,
    security_risk_categories,
)
from .prompt import build_security_qualification_prompt
from .system_prompt import SECURITY_QUALIFICATION_SYSTEM_PROMPT
from .types import (
    QualificationOutcome,
    SecurityQualificationInput,
    SecurityQualificationResult,
    SecurityVerdict,
)

AGENT_ENTRYPOINT = DEFAULT_AGENT_ENTRYPOINT


async def qualify_submission_security(
    qualification_input: SecurityQualificationInput,
    *,
    client: LLMClient,
    config: SecurityQualificationConfig | None = None,
) -> SecurityQualificationResult:
    prompt = build_security_qualification_prompt(qualification_input, config=config)
    raw = await client.complete_text(prompt)
    return parse_security_qualification(raw, model=client.model)


__all__ = [
    "AGENT_ENTRYPOINT",
    "DEFAULT_AGENT_ENTRYPOINT",
    "DEFAULT_SECURITY_QUALIFICATION_MODEL",
    "QualificationOutcome",
    "RISK_CATEGORY_ALIASES",
    "SECURITY_QUALIFICATION_SYSTEM_PROMPT",
    "SECURITY_RISK_CATEGORIES",
    "SecurityQualificationConfig",
    "SecurityQualificationInput",
    "SecurityQualificationResult",
    "SecurityVerdict",
    "build_security_qualification_prompt",
    "parse_security_qualification",
    "qualification_outcome",
    "qualify_submission_security",
    "security_failures",
    "security_risk_categories",
]
