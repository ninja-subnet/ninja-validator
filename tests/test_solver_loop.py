"""Persist-vs-retry policy of the task-solver loop handlers.

The rule: a miner-unrelated infrastructure fault (``upstream_error`` from an LLM outage,
or ``sandbox_error`` from docker) persists nothing and is retried on a later tick; every
other terminal outcome — success, an empty result, or a crashing (``agent_error``) agent
— is saved instantly. Driven with a fake DB and a stubbed ``_run``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau.db import DuelSolveJob, SolveJob
from tau.sandbox import (
    EXIT_AGENT_ERROR,
    EXIT_COMPLETED,
    EXIT_SANDBOX_ERROR,
    EXIT_UPSTREAM_ERROR,
    AgentRunResult,
)
from tau.sandbox.repo import CloneError
from tau.workers.task_solver import loop as loop_mod


class _FakeDb:
    def __init__(self) -> None:
        self.qualifications: list[dict] = []
        self.duel_solutions: list[dict] = []

    def finish_qualification(self, **kw) -> None:
        self.qualifications.append(kw)

    def save_duel_task_solution(self, **kw) -> None:
        self.duel_solutions.append(kw)


def _job() -> SolveJob:
    return SolveJob(
        task_id="t1", submission_id="s1", problem_statement="p",
        repo_clone_url="u", base_commit="c",
    )


def _duel_job() -> DuelSolveJob:
    return DuelSolveJob(
        task_id="t1", submission_id="s1", challenger_submission_id="c1",
        problem_statement="p", repo_clone_url="u", base_commit="c",
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(qualify_min_changed_lines=1, submissions_dir=Path("/nonexistent"))


def _result(exit_reason: str, *, success: bool = False, diff: str = "") -> AgentRunResult:
    return AgentRunResult(
        success=success, solution_diff=diff, exit_reason=exit_reason,
        elapsed_seconds=1.0, usage=None,
        error="boom" if exit_reason != EXIT_COMPLETED else None,
    )


def _stub_run(monkeypatch, result: AgentRunResult) -> None:
    monkeypatch.setattr(loop_mod, "_agent_dir", lambda *a, **k: Path("/bundle"))
    monkeypatch.setattr(loop_mod, "_run", lambda *a, **k: result)


def _qualify(db, cfg) -> None:
    loop_mod._qualify(_job(), db=db, client=None, config=cfg, image_tag="img")


def _duel(db, cfg) -> None:
    loop_mod._solve_duel(_duel_job(), db=db, client=None, config=cfg, image_tag="img")


# Retryable infra faults persist nothing; terminal outcomes persist a row/status.
_RETRYABLE = [EXIT_UPSTREAM_ERROR, EXIT_SANDBOX_ERROR]
_TERMINAL = [
    _result(EXIT_AGENT_ERROR),  # bogus agent that crashed / emitted nothing
    _result(EXIT_COMPLETED, success=True, diff=""),  # agent returned an empty result
    _result(EXIT_COMPLETED, success=True, diff="+added\n-removed"),  # real solution
]


def test_tick_prioritizes_duel_jobs_before_qualification(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    duel_jobs = [
        _duel_job(),
        DuelSolveJob(
            task_id="t1",
            submission_id="c1",
            challenger_submission_id="c1",
            problem_statement="p",
            repo_clone_url="u",
            base_commit="c",
        ),
    ]
    qual_jobs = [
        _job(),
        SolveJob(
            task_id="t2",
            submission_id="s1",
            problem_statement="p",
            repo_clone_url="u",
            base_commit="c",
        ),
    ]

    class Db:
        duel_limit: int | None = None
        qualification_limit: int | None = None

        def next_duel_jobs(
            self,
            limit: int,
            *,
            require_full_pool: bool = False,
            pool_targets=None,
        ) -> list[DuelSolveJob]:
            _ = (require_full_pool, pool_targets)
            self.duel_limit = limit
            return duel_jobs[:limit]

        def next_qualification_jobs(self, limit: int) -> list[SolveJob]:
            self.qualification_limit = limit
            return qual_jobs[:limit]

    db = Db()
    monkeypatch.setattr(
        loop_mod, "_solve_duel", lambda job, **_kw: calls.append(("duel", job.submission_id))
    )
    monkeypatch.setattr(
        loop_mod, "_qualify", lambda job, **_kw: calls.append(("qual", job.submission_id))
    )

    ran = loop_mod._tick(
        db=db,
        client=None,
        config=SimpleNamespace(
            max_containers=3,
            require_full_pool_for_duels=False,
            pool_targets=None,
        ),
        image_tag="img",
        stop=threading.Event(),
    )

    assert ran == 3
    assert db.duel_limit == 3
    assert db.qualification_limit == 1
    assert sum(kind == "duel" for kind, _submission in calls) == 2
    assert sum(kind == "qual" for kind, _submission in calls) == 1


# -- qualification -----------------------------------------------------------
@pytest.mark.parametrize("reason", _RETRYABLE)
def test_qualification_infra_fault_persists_nothing(monkeypatch, reason) -> None:
    _stub_run(monkeypatch, _result(reason))
    db, cfg = _FakeDb(), _config()
    _qualify(db, cfg)
    assert db.qualifications == []  # left CANDIDATE for a later tick to retry


@pytest.mark.parametrize("result", _TERMINAL)
def test_qualification_terminal_outcome_persists(monkeypatch, result) -> None:
    _stub_run(monkeypatch, result)
    db, cfg = _FakeDb(), _config()
    _qualify(db, cfg)
    assert len(db.qualifications) == 1  # a verdict is recorded (QUALIFIED/DISQUALIFIED)
    assert db.qualifications[0]["exit_reason"] == result.exit_reason


def test_qualification_agent_error_disqualifies(monkeypatch) -> None:
    _stub_run(monkeypatch, _result(EXIT_AGENT_ERROR))
    db, cfg = _FakeDb(), _config()
    _qualify(db, cfg)
    assert db.qualifications[0]["qualified"] is False


# -- duel --------------------------------------------------------------------
@pytest.mark.parametrize("reason", _RETRYABLE)
def test_duel_infra_fault_persists_nothing(monkeypatch, reason) -> None:
    _stub_run(monkeypatch, _result(reason))
    db, cfg = _FakeDb(), _config()
    _duel(db, cfg)
    assert db.duel_solutions == []  # no bogus solution; retried on a later tick


@pytest.mark.parametrize("result", _TERMINAL)
def test_duel_terminal_outcome_saves_solution(monkeypatch, result) -> None:
    _stub_run(monkeypatch, result)
    db, cfg = _FakeDb(), _config()
    _duel(db, cfg)
    assert len(db.duel_solutions) == 1  # bad agents (empty/crash) are saved instantly too
    assert db.duel_solutions[0]["exit_reason"] == result.exit_reason
    assert db.duel_solutions[0]["challenger_submission_id"] == "c1"


# -- Axiom failure routing (_report_failure) ---------------------------------
class _FakeAxiom:
    def __init__(self) -> None:
        self.failures: list[dict] = []
        self.exceptions: list[dict] = []

    def emit(self, severity, source, event_type, **kw) -> None:
        self.failures.append(kw)

    def exception(self, source, event_type, **kw) -> None:
        self.exceptions.append({"source": source, "event_type": event_type, **kw})


def test_qualification_clone_error_disqualifies_task(monkeypatch) -> None:
    fake = _FakeAxiom()
    monkeypatch.setattr(loop_mod, "get_axiom", lambda: fake)
    monkeypatch.setattr(loop_mod, "_agent_dir", lambda *a, **k: Path("/bundle"))

    def raise_clone_error(*_args, **_kwargs):
        raise CloneError("git checkout timed out after 300 seconds")

    monkeypatch.setattr(loop_mod, "_run", raise_clone_error)
    db, cfg = _FakeDb(), _config()
    _qualify(db, cfg)

    assert db.qualifications == [
        {
            "task_id": "t1",
            "king_submission_id": "s1",
            "qualified": False,
            "solution": "",
            "duration": 0.0,
            "exit_reason": "task_setup_failed",
        }
    ]
    assert fake.exceptions[0]["event_type"] == "qualification_task_setup_failed"


def _usage(*, timeouts: int = 0, last: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(upstream_timeout_count=timeouts, last_upstream_error=last)


def _report(monkeypatch, result: AgentRunResult) -> _FakeAxiom:
    fake = _FakeAxiom()
    monkeypatch.setattr(loop_mod, "get_axiom", lambda: fake)
    loop_mod._report_failure(phase="challenger", job=_job(), result=result)
    return fake


def _res(reason: str, *, usage=None, error: str | None = None) -> AgentRunResult:
    return AgentRunResult(
        success=False, solution_diff="", exit_reason=reason,
        elapsed_seconds=1.0, usage=usage, error=error,
    )


def test_report_failure_llm_timeout_is_info_category(monkeypatch) -> None:
    fake = _report(monkeypatch, _res(EXIT_UPSTREAM_ERROR, usage=_usage(timeouts=1, last="ReadTimeout: x")))
    assert fake.failures[0]["category"] == "llm_timeout"
    assert fake.failures[0]["exception"] == "ReadTimeout: x"


def test_report_failure_llm_error_when_not_timeout(monkeypatch) -> None:
    fake = _report(monkeypatch, _res(EXIT_UPSTREAM_ERROR, usage=_usage(timeouts=0, last="HTTP 402")))
    assert fake.failures[0]["category"] == "llm_error"
    assert fake.failures[0]["exception"] == "HTTP 402"


def test_report_failure_agent_error_includes_traceback(monkeypatch) -> None:
    fake = _report(monkeypatch, _res(EXIT_AGENT_ERROR, error="Traceback ... ValueError"))
    assert fake.failures[0]["category"] == "agent_error"
    assert "ValueError" in fake.failures[0]["exception"]


def test_report_failure_sandbox_error(monkeypatch) -> None:
    fake = _report(monkeypatch, _res(EXIT_SANDBOX_ERROR, error="docker boom"))
    assert fake.failures[0]["category"] == "sandbox_error"


def test_report_failure_unrecognized_is_generic(monkeypatch) -> None:
    fake = _report(monkeypatch, _res("time_limit_exceeded", error=None))
    assert fake.failures[0]["category"] == "unknown"


def test_report_failure_noop_on_completion(monkeypatch) -> None:
    fake = _report(monkeypatch, _res(EXIT_COMPLETED))
    assert fake.failures == []
