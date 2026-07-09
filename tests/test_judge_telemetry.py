"""Unit coverage for privacy-safe task-screen/duel drift telemetry."""

from __future__ import annotations

from hashlib import sha256

import pytest

from tau.db.judge import _task_screen_duel_comparison
from tau.judging import Judgment, Solution, Task
from tau.workers.judge.config import JudgeWorkerConfig
from tau.workers.judge.fallback import JudgeRun
from tau.workers.judge.pipeline import (
    _emit_task_screen_duel_comparison,
    _judge_and_save,
)


class FakeAxiom:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def info(self, **fields: object) -> None:
        self.events.append(fields)


class FakeJudgeDb:
    def __init__(self, comparison: object) -> None:
        self.comparison = comparison
        self.saved = 0

    async def save_judgment(self, *args: object, **kwargs: object) -> object:
        self.saved += 1
        return self.comparison


def test_comparison_hashes_patches_and_computes_duel_minus_screen_delta() -> None:
    comparison = _task_screen_duel_comparison(
        task_id="task",
        king_submission_id="king",
        challenger_submission_id="challenger",
        screening_king_score=0.25,
        duel_king_score=0.80,
        screening_model="test/screener",
        duel_model="test/judge",
        qualification_patch="qualification patch",
        duel_patch="fresh duel patch",
    )

    assert comparison.duel_minus_screen_king_score_delta == pytest.approx(0.55)
    assert (
        comparison.qualification_patch_sha256
        == sha256(b"qualification patch").hexdigest()
    )
    assert comparison.duel_patch_sha256 == sha256(b"fresh duel patch").hexdigest()
    assert comparison.qualification_patch_matches_duel_patch is False
    assert "qualification patch" not in repr(comparison)
    assert "fresh duel patch" not in repr(comparison)


def test_comparison_keeps_delta_nullable_when_screening_did_not_score() -> None:
    comparison = _task_screen_duel_comparison(
        task_id="task",
        king_submission_id="king",
        challenger_submission_id="challenger",
        screening_king_score=None,
        duel_king_score=0.50,
        screening_model=None,
        duel_model="test/judge",
        qualification_patch="same patch",
        duel_patch="same patch",
    )

    assert comparison.duel_minus_screen_king_score_delta is None
    assert comparison.qualification_patch_matches_duel_patch is True


def test_comparison_event_contains_scores_hashes_and_composite_judgment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_axiom = FakeAxiom()
    monkeypatch.setitem(
        _emit_task_screen_duel_comparison.__globals__, "get_axiom", lambda: fake_axiom
    )
    comparison = _task_screen_duel_comparison(
        task_id="task",
        king_submission_id="king",
        challenger_submission_id="challenger",
        screening_king_score=0.30,
        duel_king_score=0.90,
        screening_model="test/screener",
        duel_model="test/judge",
        qualification_patch="qualification patch",
        duel_patch="fresh duel patch",
    )

    _emit_task_screen_duel_comparison(comparison)

    assert fake_axiom.events == [
        {
            "source": "judge",
            "event_type": "task_screen_duel_comparison",
            "task_id": "task",
            "king_submission_id": "king",
            "challenger_submission_id": "challenger",
            "screening_king_score": 0.30,
            "duel_king_score": 0.90,
            "duel_minus_screen_king_score_delta": pytest.approx(0.60),
            "screening_model": "test/screener",
            "duel_model": "test/judge",
            "qualification_patch_sha256": sha256(b"qualification patch").hexdigest(),
            "duel_patch_sha256": sha256(b"fresh duel patch").hexdigest(),
            "qualification_patch_matches_duel_patch": False,
        }
    ]
    assert "qualification patch" not in repr(fake_axiom.events)
    assert "fresh duel patch" not in repr(fake_axiom.events)


async def test_write_conflict_context_does_not_emit_duplicate_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_axiom = FakeAxiom()
    monkeypatch.setitem(_judge_and_save.__globals__, "get_axiom", lambda: fake_axiom)

    async def fake_judge_with_fallback(*args: object, **kwargs: object) -> JudgeRun:
        return JudgeRun(Judgment("king", 0.8, 0.2, model="test/judge"), 1, 0.1)

    monkeypatch.setitem(
        _judge_and_save.__globals__, "judge_with_fallback", fake_judge_with_fallback
    )
    db = FakeJudgeDb(comparison=None)  # INSERT conflict/no matching screen context

    await _judge_and_save(
        db,  # type: ignore[arg-type]
        [],
        JudgeWorkerConfig(openrouter_api_key="", use_dummy_llm=True),
        (
            Task("task", "problem", "reference patch"),
            Solution("king", "fresh duel patch"),
            Solution("challenger", "challenger patch"),
        ),
    )

    assert db.saved == 1
    assert [event["event_type"] for event in fake_axiom.events] == ["judgment_saved"]
