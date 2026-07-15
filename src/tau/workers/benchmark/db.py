"""Benchmark worker's view of the database (sync, self-contained).

A tiny read-only slice: list every king so the loop can diff against what has
already been benchmarked on disk. ``king_id`` IS the submission id (1:1 FK), so it
doubles as the id of the submission whose agent gets benchmarked — the same fact
``SolverDb`` relies on. Kept sync (like ``SolverDb``): the worker's real work is
blocking subprocess/docker, so async buys nothing.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from sqlalchemy import select

from tau.db import models
from tau.db.engine import create_db_engine, session_factory, session_scope

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KingRow:
    king_id: str
    king_from: dt.datetime


class BenchmarkDb:
    """Read-only access to the ``kings`` table for the benchmark worker."""

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_db_engine(url, echo=echo)
        self._sessions = session_factory(self._engine)

    def close(self) -> None:
        self._engine.dispose()

    def all_kings(self) -> list[KingRow]:
        """Every king, oldest first. The disk marker decides what still needs running."""
        with session_scope(self._sessions) as session:
            rows = session.execute(
                select(models.King.king_id, models.King.king_from).order_by(models.King.king_from)
            ).all()
            return [KingRow(king_id=row.king_id, king_from=row.king_from) for row in rows]
