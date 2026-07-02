"""Outcome policy for submission security qualification results."""

from __future__ import annotations

import re
from typing import Any

from .types import QualificationOutcome, SecurityQualificationResult


def security_risk_categories(result: SecurityQualificationResult) -> frozenset[str]:
    risks = result.raw_payload.get("risks")
    if not isinstance(risks, list):
        risks = list(result.risks)
    return frozenset(
        category
        for item in risks
        for category in [_risk_category(item)]
        if category is not None
    )


def security_failures(result: SecurityQualificationResult) -> list[str]:
    failures: list[str] = []
    if result.verdict == "fail":
        failures.append("security qualification returned verdict=fail.")
    for category in sorted(security_risk_categories(result) & SECURITY_RISK_CATEGORIES):
        failures.append(f"security qualification reported risk category: {category}.")
    return _dedupe(failures)


def qualification_outcome(result: SecurityQualificationResult) -> QualificationOutcome:
    if security_failures(result):
        return QualificationOutcome.DISQUALIFIED
    if result.verdict == "warn":
        return QualificationOutcome.NEEDS_REVIEW
    return QualificationOutcome.QUALIFIED


def _risk_category(item: Any) -> str | None:
    if isinstance(item, dict):
        return _normalize_risk_category(
            item.get("category")
            or item.get("type")
            or item.get("name")
            or item.get("risk")
        )
    if isinstance(item, str):
        return _normalize_risk_category(_risk_label_from_text(item))
    return None


def _risk_label_from_text(text: str) -> str:
    return re.split(r":|\s+(?:-|--|—)\s+", text.strip(), maxsplit=1)[0]


def _normalize_risk_category(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return RISK_CATEGORY_ALIASES.get(normalized, normalized or None)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


RISK_CATEGORY_ALIASES = {
    "container-escape": "sandbox-escape",
    "c2": "network-exfiltration",
    "command-and-control": "network-exfiltration",
    "credential-theft": "secret-theft",
    "creds-theft": "secret-theft",
    "data-exfiltration": "exfiltration",
    "destructive-host-tampering": "destructive-tampering",
    "dns-exfil": "network-exfiltration",
    "dns-exfiltration": "network-exfiltration",
    "docker-escape": "docker-sandbox-escape",
    "docker-sandbox-escape": "docker-sandbox-escape",
    "filesystem-escape": "host-filesystem-access",
    "host-escape": "sandbox-escape",
    "host-filesystem": "host-filesystem-access",
    "host-filesystem-access": "host-filesystem-access",
    "host-tampering": "destructive-tampering",
    "network-exfil": "network-exfiltration",
    "network-exfiltration": "network-exfiltration",
    "prompt-exfil": "prompt-exfiltration",
    "prompt-exfiltration": "prompt-exfiltration",
    "sandbox-escape": "sandbox-escape",
    "secret-exfiltration": "secret-theft",
    "secrets-theft": "secret-theft",
    "secret-theft": "secret-theft",
}

SECURITY_RISK_CATEGORIES = frozenset(
    {
        "cryptomining",
        "destructive-tampering",
        "docker-sandbox-escape",
        "exfiltration",
        "host-filesystem-access",
        "network-exfiltration",
        "persistence",
        "privilege-escalation",
        "prompt-exfiltration",
        "sandbox-escape",
        "secret-theft",
    }
)
