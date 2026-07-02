"""Database middleware for the validator workers.

The chain watcher uses the `Database` Protocol in `interface.py` (a concrete
instance from `connect()`), wrapped by `adapters.DatabaseSnapshotSink`:

    from tau.db import connect
    from tau.db.adapters import DatabaseSnapshotSink

    db = connect()                          # env-resolved DATABASE_URL
    sink = DatabaseSnapshotSink(db)         # ready for bittensor.worker.run(...)

The task-generator and judge workers use their own focused seams,
`GeneratorDb` and `JudgeDb`.
"""

from __future__ import annotations

from .adapters import DatabaseSnapshotSink
from .database import SqlDatabase, connect
from .duel_resolver import DuelResolverDb
from .engine import database_url, session_scope
from .generator import GenerationMetrics, GeneratorDb, PoolDeficit
from .judge import JudgeDb, JudgeRequest
from .qualification import QualificationCandidate, QualificationDb
from .solver import SolveJob, SolverDb
from .status import (
    ChallengeStatus,
    DuelOutcome,
    PoolType,
    SubmissionStatus,
    TaskStatus,
)
from .weight_setter import WeightSetterDb

__all__ = [
    "JudgeDb",
    "JudgeRequest",
    "QualificationCandidate",
    "QualificationDb",
    "SqlDatabase",
    "DatabaseSnapshotSink",
    "DuelResolverDb",
    "connect",
    "GenerationMetrics",
    "GeneratorDb",
    "PoolDeficit",
    "ChallengeStatus",
    "DuelOutcome",
    "SolveJob",
    "SolverDb",
    "PoolType",
    "SubmissionStatus",
    "TaskStatus",
    "WeightSetterDb",
    "database_url",
    "session_scope",
]
