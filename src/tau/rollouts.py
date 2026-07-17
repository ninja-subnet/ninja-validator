"""Stable identities and shared types for persisted agent rollouts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

RolloutEvent = Mapping[str, Any]


def rollout_id(
    *,
    phase: str,
    task_id: str,
    submission_id: str,
    challenger_submission_id: str | None = None,
) -> str:
    """Return the deterministic identity of one terminal solve.

    Qualification has one terminal rollout per task/submission. Duel rollouts are
    additionally scoped to the challenger, matching ``DuelTaskSolution``'s cache
    identity. Length-prefixing makes the input unambiguous even when IDs contain the
    separator used in a simpler joined representation.
    """
    parts = (phase, task_id, challenger_submission_id or "", submission_id)
    encoded = b"".join(
        len(part.encode("utf-8")).to_bytes(4, "big") + part.encode("utf-8")
        for part in parts
    )
    return f"{phase}-{hashlib.sha256(encoded).hexdigest()[:32]}"
