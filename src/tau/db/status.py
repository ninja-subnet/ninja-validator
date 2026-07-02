"""Enum contracts for integer columns in the schema.

These values are persisted (``tasks.status_id`` / ``tasks.pool_type``), so they are
part of the on-disk contract — keep them stable. Shared by the DB seam and the
workers so neither side hard-codes bare integers.
"""

from __future__ import annotations

from enum import IntEnum


class TaskStatus(IntEnum):
    """Lifecycle of a ``tasks`` row (the ``status_id`` column).

    The task-generator inserts ``CANDIDATE``; the task-solver runs the king and
    flips it to ``QUALIFIED`` (king solved it → usable in a duel) or
    ``DISQUALIFIED`` (king failed → drop it).
    """

    CANDIDATE = 0
    QUALIFIED = 1
    DISQUALIFIED = 2


class PoolType(IntEnum):
    """Which duel pool a task belongs to (the ``pool_type`` column)."""

    POOL_ONE = 1
    POOL_TWO = 2


class ChallengeStatus(IntEnum):
    """Status of a ``challenges`` row (the ``status`` column).

    ``POOL_ONE``/``POOL_TWO`` name the running duel pool and MUST equal the matching
    ``PoolType`` values (queries gate active rounds on ``pool_type == status``).
    ``CLOSED`` is the only terminal state; the per-pool outcome is recorded in
    ``duel_resolutions`` (a promotion also inserts a ``kings`` row), not here.
    """

    CLOSED = 0
    POOL_ONE = 1
    POOL_TWO = 2


class DuelOutcome(IntEnum):
    """Per-pool verdict recorded in ``duel_resolutions`` (the ``outcome`` column).

    Written when the resolver concludes a pool. ``CHALLENGER_WON`` at ``POOL_TWO``
    crowns the challenger; at ``POOL_ONE`` it only advances them to ``POOL_TWO``.
    """

    KING_WON = 0
    CHALLENGER_WON = 1
    CHALLENGER_DEREGISTERED = 2


class SubmissionStatus(IntEnum):
    """Lifecycle of a ``submissions`` row (the ``status_id`` column).

    Ingestion writes ``UNVERIFIED``. The qualification worker is the gatekeeper
    that promotes a submission to ``ELIGIBLE`` or moves it to a terminal status.
    The duel-resolver only reads ``ELIGIBLE`` rows.
    """

    UNVERIFIED = 0
    ELIGIBLE = 1
    DISQUALIFIED = 2
    NEEDS_REVIEW = 3
