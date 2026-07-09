"""Database middleware for the validator workers.

The chain watcher uses the `Database` Protocol in `interface.py` (a concrete
instance from `connect()`), wrapped by `adapters.DatabaseSnapshotSink`:

    from tau.db import connect
    from tau.db.adapters import DatabaseSnapshotSink

    db = connect()                          # env-resolved DATABASE_URL
    sink = DatabaseSnapshotSink(db)         # ready for bittensor.worker.run(...)

The task-generator, task-screener, and judge workers use their own focused seams:
`GeneratorDb`, `TaskScreeningDb`, and `JudgeDb`.
"""

from __future__ import annotations

from .adapters import DatabaseSnapshotSink
from .database import SqlDatabase, connect
from .duel_resolver import DuelResolverDb
from .engine import database_url, session_scope
from .generator import GenerationMetrics, GeneratorDb, PoolDeficit
from .judge import JudgeDb, JudgeRequest, TaskScreenDuelComparison
from .qualification import QualificationCandidate, QualificationDb
from .solver import DuelSolveJob, SolveJob, SolverDb
from .status import (
    ChallengeStatus,
    DuelOutcome,
    PoolType,
    SubmissionStatus,
    TaskStatus,
)
from .task_screening import ScreeningFailureSave, TaskScreenRequest, TaskScreeningDb
from .weight_setter import WeightSetterDb

__all__ = [
    "JudgeDb",
    "JudgeRequest",
    "TaskScreenDuelComparison",
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
    "DuelSolveJob",
    "SolveJob",
    "SolverDb",
    "PoolType",
    "SubmissionStatus",
    "TaskStatus",
    "TaskScreenRequest",
    "TaskScreeningDb",
    "ScreeningFailureSave",
    "WeightSetterDb",
    "database_url",
    "session_scope",
]
