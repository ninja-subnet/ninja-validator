"""Persist-vs-retry policy of the task-solver loop handlers.

The rule: a miner-unrelated infrastructure fault (``upstream_error`` from an LLM outage,
or ``sandbox_error`` from docker) persists nothing and is retried on a later tick; every
other terminal outcome — success, an empty result, or a crashing (``agent_error``) agent
— is saved instantly. Driven with a fake DB and a stubbed ``_run``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tau.db import SolveJob
from tau.sandbox import (
    EXIT_AGENT_ERROR,
    EXIT_COMPLETED,
    EXIT_SANDBOX_ERROR,
    EXIT_UPSTREAM_ERROR,
    AgentRunResult,
)
from tau.workers.task_solver import loop as loop_mod


class _FakeDb:
    def __init__(self, *, finish_saved: bool = True) -> None:
        self.qualifications: list[dict] = []
        self.solutions: list[dict] = []
        self.finish_saved = finish_saved

    def finish_qualification(self, **kw) -> bool:
        if not self.finish_saved:
            return False
        self.qualifications.append(kw)
        return True

    def save_task_solution(self, **kw) -> None:
        self.solutions.append(kw)


def _job() -> SolveJob:
    return SolveJob(
        task_id="t1", submission_id="s1", problem_statement="p",
        repo_clone_url="u", base_commit="c",
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


def _challenge(db, cfg) -> None:
    loop_mod._solve_challenger(_job(), db=db, client=None, config=cfg, image_tag="img")


# Retryable infra faults persist nothing; terminal outcomes persist a row/status.
_RETRYABLE = [EXIT_UPSTREAM_ERROR, EXIT_SANDBOX_ERROR]
_TERMINAL = [
    _result(EXIT_AGENT_ERROR),  # bogus agent that crashed / emitted nothing
    _result(EXIT_COMPLETED, success=True, diff=""),  # agent returned an empty result
    _result(EXIT_COMPLETED, success=True, diff="+added\n-removed"),  # real solution
]


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


def test_qualification_discards_result_when_already_finished(monkeypatch) -> None:
    _stub_run(monkeypatch, _result(EXIT_COMPLETED, success=True, diff="+added\n-removed"))
    db, cfg = _FakeDb(finish_saved=False), _config()
    _qualify(db, cfg)
    assert db.qualifications == []


# -- challenger --------------------------------------------------------------
@pytest.mark.parametrize("reason", _RETRYABLE)
def test_challenger_infra_fault_persists_nothing(monkeypatch, reason) -> None:
    _stub_run(monkeypatch, _result(reason))
    db, cfg = _FakeDb(), _config()
    _challenge(db, cfg)
    assert db.solutions == []  # no bogus solution; retried on a later tick


@pytest.mark.parametrize("result", _TERMINAL)
def test_challenger_terminal_outcome_saves_solution(monkeypatch, result) -> None:
    _stub_run(monkeypatch, result)
    db, cfg = _FakeDb(), _config()
    _challenge(db, cfg)
    assert len(db.solutions) == 1  # bad agents (empty/crash) are saved instantly too
    assert db.solutions[0]["exit_reason"] == result.exit_reason


# -- Axiom failure routing (_report_failure) ---------------------------------
class _FakeAxiom:
    def __init__(self) -> None:
        self.failures: list[dict] = []

    def emit(self, severity, source, event_type, **kw) -> None:
        self.failures.append(kw)


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
