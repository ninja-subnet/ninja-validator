"""The weight-setter's database access: the rolling king window."""

from __future__ import annotations

from sqlalchemy import select

from tau.weights.types import RecentKing

from . import models
from .engine import create_db_engine, session_factory, session_scope


class WeightSetterDb:
    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self._engine = create_db_engine(url, echo=echo)
        self._sessions = session_factory(self._engine)

    def close(self) -> None:
        self._engine.dispose()

    def recent_kings(self, window: int) -> list[RecentKing]:
        """The most recent `window` kings, most-recent-first (reigning king first)."""
        if window <= 0:
            return []
        stmt = (
            select(models.King.king_id, models.Submission.hotkey)
            .join(
                models.Submission,
                models.Submission.submission_id == models.King.king_id,
            )
            .order_by(models.King.king_from.desc())
            .limit(window)
        )
        with session_scope(self._sessions) as session:
            rows = session.execute(stmt).all()
        return [RecentKing(king_id=row.king_id, hotkey=row.hotkey) for row in rows]
