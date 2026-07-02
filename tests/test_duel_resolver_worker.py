"""Tests for the duel-resolver worker: config + the apply dispatch, driven by a fake DB.

The decision logic is covered in test_duel.py and the DB seam in
test_db_duel_resolver.py; here we check that each Action maps to the right write and
that idle logging is edge-triggered.
"""

from __future__ import annotations

import logging

import pytest

from tau.db import DuelResolverDb
from tau.db.status import DuelOutcome, PoolType
from tau.duel import (
    ActiveChallenge,
    AdvancePool,
    ChallengeSnapshot,
    CloseChallenge,
    CloseReason,
    DuelScoringMethod,
    Nothing,
    OpenChallenge,
    Promote,
    Tally,
    WaitReason,
)
from tau.workers.duel_resolver import DuelResolverConfig
from tau.workers.duel_resolver.pipeline import _apply
from tau.workers.duel_resolver.telemetry import TickLog, emit_axiom

P1 = PoolType.POOL_ONE
P2 = PoolType.POOL_TWO


def _ac(pool: PoolType = P1, *, tally: Tally | None = None) -> ActiveChallenge:
    return ActiveChallenge(
        challenger_submission_id="c",
        king_submission_id="k",
        pool=pool,
        pool_target=1,
        tally=tally if tally is not None else Tally(0, 0, 0),
        challenger_registered=True,
    )


class FakeDb(DuelResolverDb):
    """A DuelResolverDb with its writes stubbed: records the call, skips the database.

    Subclasses the real seam so `_apply(db: DuelResolverDb, ...)` type-checks; the
    `__init__` deliberately skips the engine setup, so no connection is opened.
    """

    def __init__(self, *, applied: bool = True) -> None:
        self.applied = applied
        self.calls: list[tuple[object, ...]] = []

    async def open_challenge(
        self, king_id: str, challenger_submission_id: str
    ) -> bool:
        self.calls.append(("open", (king_id, challenger_submission_id)))
        return self.applied

    async def advance_pool(
        self,
        challenge: ActiveChallenge,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
    ) -> bool:
        self.calls.append(
            ("advance", challenge, scoring_method, round_win_margin, mean_score_margin)
        )
        return self.applied

    async def close_challenge(
        self,
        challenge: ActiveChallenge,
        outcome: DuelOutcome,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
    ) -> bool:
        self.calls.append(
            ("close", challenge, outcome, scoring_method, round_win_margin, mean_score_margin)
        )
        return self.applied

    async def promote(
        self,
        challenge: ActiveChallenge,
        *,
        scoring_method: DuelScoringMethod,
        round_win_margin: int,
        mean_score_margin: float,
    ) -> bool:
        self.calls.append(
            ("promote", challenge, scoring_method, round_win_margin, mean_score_margin)
        )
        return self.applied


class FakeAxiom:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def info(self, **fields: object) -> None:
        self.events.append(fields)


# -- config -------------------------------------------------------------------------


def test_config_from_env_reads_overrides() -> None:
    config = DuelResolverConfig.from_env(
        {
            "TAU_DUEL_SCORING_METHOD": "mean",
            "TAU_DUEL_ROUND_WIN_MARGIN": "2",
            "TAU_DUEL_MEAN_SCORE_MARGIN": "0.075",
            "TAU_DUEL_POLL_SECONDS": "3.5",
        }
    )
    assert config.scoring_method is DuelScoringMethod.MEAN
    assert config.round_win_margin == 2
    assert config.mean_score_margin == 0.075
    assert config.poll_seconds == 3.5


def test_config_does_not_read_legacy_win_margin_name() -> None:
    config = DuelResolverConfig.from_env({"TAU_DUEL_WIN_MARGIN": "2"})
    assert config.round_win_margin == 0


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        DuelResolverConfig(round_win_margin=-1)
    with pytest.raises(ValueError):
        DuelResolverConfig(mean_score_margin=-0.01)
    with pytest.raises(ValueError):
        DuelResolverConfig(scoring_method="nonsense")
    with pytest.raises(ValueError):
        DuelResolverConfig(poll_seconds=0)


# -- apply dispatch -----------------------------------------------------------------


async def test_apply_open_challenge() -> None:
    db = FakeDb()
    await _apply(db, OpenChallenge("k", "c"), TickLog(), config=DuelResolverConfig())
    assert db.calls == [("open", ("k", "c"))]


async def test_apply_advance_pool_forwards_resolution_config() -> None:
    db, challenge = FakeDb(), _ac(P1)
    config = DuelResolverConfig(round_win_margin=2, mean_score_margin=0.075)
    await _apply(db, AdvancePool(challenge), TickLog(), config=config)
    assert db.calls == [("advance", challenge, DuelScoringMethod.ROUND_WINS, 2, 0.075)]


async def test_apply_resolution_event_logs_scoring_config_and_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_axiom = FakeAxiom()
    monkeypatch.setitem(emit_axiom.__globals__, "get_axiom", lambda: fake_axiom)
    db = FakeDb()
    challenge = _ac(
        P1,
        tally=Tally(
            2,
            1,
            0,
            king_score_mean=0.4,
            challenger_score_mean=0.5,
            score_mean_delta=0.1,
            score_mean_rounds=3,
        ),
    )
    config = DuelResolverConfig(
        scoring_method=DuelScoringMethod.MEAN,
        round_win_margin=2,
        mean_score_margin=0.075,
    )

    await _apply(db, AdvancePool(challenge), TickLog(), config=config)

    assert fake_axiom.events == [
        {
            "source": "duel-resolver",
            "event_type": "pool_advanced",
            "scoring_method": "mean",
            "round_win_margin": 2,
            "mean_score_margin": 0.075,
            "challenger_submission_id": "c",
            "king_submission_id": "k",
            "pool": "POOL_ONE",
            "pool_target": 1,
            "wins": 2,
            "losses": 1,
            "ties": 0,
            "king_score_mean": 0.4,
            "challenger_score_mean": 0.5,
            "score_mean_delta": 0.1,
            "score_mean_rounds": 3,
        }
    ]


async def test_apply_promote() -> None:
    db, challenge = FakeDb(), _ac(P2)
    config = DuelResolverConfig(scoring_method=DuelScoringMethod.MEAN)
    await _apply(db, Promote(challenge), TickLog(), config=config)
    assert db.calls == [("promote", challenge, DuelScoringMethod.MEAN, 0, 0.05)]


async def test_apply_close_challenge_maps_king_defended() -> None:
    db, challenge = FakeDb(), _ac(P1)
    await _apply(
        db,
        CloseChallenge(challenge, CloseReason.KING_DEFENDED),
        TickLog(),
        config=DuelResolverConfig(),
    )
    assert db.calls == [
        ("close", challenge, DuelOutcome.KING_WON, DuelScoringMethod.ROUND_WINS, 0, 0.05)
    ]


async def test_apply_close_challenge_maps_deregistered() -> None:
    db, challenge = FakeDb(), _ac(P1)
    await _apply(
        db,
        CloseChallenge(challenge, CloseReason.CHALLENGER_DEREGISTERED),
        TickLog(),
        config=DuelResolverConfig(),
    )
    assert db.calls == [
        (
            "close",
            challenge,
            DuelOutcome.CHALLENGER_DEREGISTERED,
            DuelScoringMethod.ROUND_WINS,
            0,
            0.05,
        )
    ]


async def test_apply_nothing_writes_nothing() -> None:
    db = FakeDb()
    await _apply(db, Nothing(WaitReason.NO_KING), TickLog(), config=DuelResolverConfig())
    assert db.calls == []


async def test_apply_logs_skip_when_write_does_not_apply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = FakeDb(applied=False)  # the guarded write matched nothing
    with caplog.at_level(logging.INFO):
        await _apply(db, AdvancePool(_ac(P1)), TickLog(), config=DuelResolverConfig())
    assert any("skipped" in r.message for r in caplog.records)


# -- idle log (edge-triggered) ------------------------------------------------------


def test_idle_dedupes_until_reason_changes(caplog: pytest.LogCaptureFixture) -> None:
    tick = TickLog()
    with caplog.at_level(logging.INFO):
        tick.idle(WaitReason.NO_KING)
        tick.idle(WaitReason.NO_KING)  # same -> not logged again
        tick.idle(WaitReason.DUEL_IN_PROGRESS)  # changed -> logged
    messages = [r.message for r in caplog.records]
    assert sum("no_king" in m for m in messages) == 1
    assert sum("duel_in_progress" in m for m in messages) == 1


def test_an_action_rearms_the_idle_line(caplog: pytest.LogCaptureFixture) -> None:
    tick = TickLog()
    with caplog.at_level(logging.INFO):
        tick.idle(WaitReason.NO_CHALLENGER)
        tick.action(True, "opened challenge: c vs king k")  # real work between idles
        tick.idle(WaitReason.NO_CHALLENGER)  # logged again, idle line re-armed
    assert sum("no_challenger" in r.message for r in caplog.records) == 2


def test_snapshot_dataclass_is_constructible() -> None:
    # Sanity that the worker's input type is importable/usable here.
    assert ChallengeSnapshot(None, None, None).active_challenge is None
