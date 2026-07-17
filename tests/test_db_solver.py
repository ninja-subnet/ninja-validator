"""Integration tests for the task-solver DB seam (``SolverDb``).

Same Postgres harness as ``test_db_generator.py``: needs a reachable server via
``TAU_TEST_DATABASE_URL`` (a throwaway DB, distinct from your dev one — the schema is
dropped/recreated around each test). Skipped when the server is unreachable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import make_url, select, text

from tau.db import SolverDb, TaskStatus
from tau.db.engine import create_db_engine, session_factory, session_scope
from tau.db.models import (
    Base,
    Challenge,
    DuelTaskSolution,
    King,
    Rollout,
    Submission,
    Task,
    TaskScreening,
)
from tau.db.solver import _duel_side_order
from tau.pools import PoolTargets

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_TEST_URL = os.environ.get("TAU_TEST_DATABASE_URL")

_PARENT = "b" * 40
_COMMIT = "a" * 40


def _maintenance_url(url: str) -> str:
    return make_url(url).set(database="postgres").render_as_string(hide_password=False)


def _server_reachable(url: str | None) -> bool:
    if not url:
        return False
    try:
        engine = create_db_engine(_maintenance_url(url))
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _server_reachable(_TEST_URL),
    reason="TAU_TEST_DATABASE_URL unset or its Postgres server unreachable — skipping DB tests",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database() -> None:
    assert _TEST_URL is not None
    db_name = make_url(_TEST_URL).database
    engine = create_db_engine(_maintenance_url(_TEST_URL))
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            already = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not already:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        engine.dispose()


@pytest.fixture
def db() -> Iterator[SolverDb]:
    engine = create_db_engine(_TEST_URL)
    with engine.begin() as conn:
        # The trigram index on tasks.problem_statement needs this extension; the
        # migration/initdb create it, but create_all() here does not.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()
    solver = SolverDb(_TEST_URL)
    try:
        yield solver
    finally:
        solver.close()
        engine = create_db_engine(_TEST_URL)
        Base.metadata.drop_all(engine)
        engine.dispose()


# -- seeding helpers ---------------------------------------------------------------


def _seed(db: SolverDb, fn) -> None:  # noqa: ANN001
    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        fn(session)


def _submission(session, sub_id: str, *, block: int = 1):  # noqa: ANN001
    # The solver no longer reads agent_files (agents come from the local submissions
    # dir, keyed by submission id), so it is left null here.
    session.add(Submission(submission_id=sub_id, block=block, hotkey=f"hk-{sub_id}"))


def _king(session, king_id: str):  # noqa: ANN001
    # king_id IS the submission id (1:1 FK); king_from defaults to now() in the DB.
    session.add(King(king_id=king_id))


def _task(
    session, task_id: str, king_id: str, *, pool_type: int, status_id: int, fp: str
):  # noqa: ANN001
    session.add(
        Task(
            task_id=task_id,
            king_id=king_id,
            pool_type=pool_type,
            problem_statement=f"solve {task_id}",
            status_id=status_id,
            repo_clone_url="https://github.com/octo/repo.git",
            parent_sha=_PARENT,
            commit_sha=_COMMIT,
            reference_patch="diff",
            content_fingerprint=fp,
        )
    )


# -- phase A: qualification --------------------------------------------------------


def test_qualification_jobs_returns_candidates_of_current_king(db: SolverDb) -> None:
    def seed(s):  # noqa: ANN001
        _submission(s, "king-1")
        _king(s, "king-1")
        _task(
            s,
            "t-pool2",
            "king-1",
            pool_type=2,
            status_id=int(TaskStatus.CANDIDATE),
            fp="f2",
        )
        _task(
            s,
            "t-pool1",
            "king-1",
            pool_type=1,
            status_id=int(TaskStatus.CANDIDATE),
            fp="f1",
        )
        _task(
            s,
            "t-done",
            "king-1",
            pool_type=1,
            status_id=int(TaskStatus.QUALIFIED),
            fp="f3",
        )

    _seed(db, seed)
    jobs = db.next_qualification_jobs(10)
    ids = [j.task_id for j in jobs]
    assert set(ids) == {"t-pool1", "t-pool2"}  # only CANDIDATE tasks
    assert ids[0] == "t-pool1"  # pool_type ordering: pool 1 first
    job = jobs[0]
    assert job.submission_id == "king-1"
    assert job.base_commit == _PARENT


def test_qualification_jobs_respects_limit(db: SolverDb) -> None:
    def seed(s):  # noqa: ANN001
        _submission(s, "king-1")
        _king(s, "king-1")
        for i in range(3):
            _task(
                s,
                f"t{i}",
                "king-1",
                pool_type=1,
                status_id=int(TaskStatus.CANDIDATE),
                fp=f"f{i}",
            )

    _seed(db, seed)
    assert len(db.next_qualification_jobs(2)) == 2


def test_finish_qualification_viable_moves_to_pending_and_records_solve(
    db: SolverDb,
) -> None:
    def seed(s):  # noqa: ANN001
        _submission(s, "king-1")
        _king(s, "king-1")
        _task(
            s, "t1", "king-1", pool_type=1, status_id=int(TaskStatus.CANDIDATE), fp="f1"
        )

    _seed(db, seed)
    db.finish_qualification(
        task_id="t1",
        king_submission_id="king-1",
        qualified=True,
        solution="diff --git a/x b/x",
        duration=1.2,
        exit_reason="completed",
        success=True,
        usage_summary={"request_count": 1, "total_tokens": 9},
        rollout_events=[
            {"index": 0, "type": "llm_call", "request": {"model": "m"}}
        ],
    )
    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        task = session.get(Task, "t1")
        screening = session.get(TaskScreening, "t1")
        assert task.status_id == int(TaskStatus.PENDING_SCREEN)
        assert screening.king_submission_id == "king-1"
        assert screening.qualification_solution == "diff --git a/x b/x"
        assert screening.reason is None
        assert screening.king_score is None
        rollout = session.scalars(select(Rollout)).one()
        assert rollout.phase == "qualification"
        assert rollout.success is True
        assert rollout.usage_summary == {"request_count": 1, "total_tokens": 9}
        assert rollout.events == [
            {"index": 0, "type": "llm_call", "request": {"model": "m"}}
        ]


def test_finish_qualification_non_viable_disqualifies_without_screening(
    db: SolverDb,
) -> None:
    def seed(s):  # noqa: ANN001
        _submission(s, "king-1")
        _king(s, "king-1")
        _task(
            s, "t1", "king-1", pool_type=1, status_id=int(TaskStatus.CANDIDATE), fp="f1"
        )

    _seed(db, seed)
    db.finish_qualification(
        task_id="t1",
        king_submission_id="king-1",
        qualified=False,
        solution="",
    )
    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        assert session.get(Task, "t1").status_id == int(TaskStatus.DISQUALIFIED)
        assert session.get(TaskScreening, "t1") is None


def test_finish_qualification_ignores_stale_non_candidate_completion(
    db: SolverDb,
) -> None:
    def seed(s):  # noqa: ANN001
        _submission(s, "king-1")
        _king(s, "king-1")
        _task(
            s,
            "t1",
            "king-1",
            pool_type=1,
            status_id=int(TaskStatus.DISQUALIFIED),
            fp="f1",
        )

    _seed(db, seed)
    db.finish_qualification(
        task_id="t1",
        king_submission_id="king-1",
        qualified=True,
        solution="late patch",
    )
    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        assert session.get(Task, "t1").status_id == int(TaskStatus.DISQUALIFIED)
        assert session.get(TaskScreening, "t1") is None
        assert session.scalars(select(Rollout)).all() == []


# -- phase B: fresh duel solves ----------------------------------------------------


def _seed_active_challenge(db: SolverDb) -> None:
    def seed(s):  # noqa: ANN001
        _submission(s, "king-1")
        _submission(s, "sub-chal")
        _king(s, "king-1")
        _challenge_status = 1
        s.add(
            Challenge(
                challenger_submission_id="sub-chal",
                king_id="king-1",
                status=_challenge_status,
            )
        )
        # QUALIFIED, pool_type matches the active challenge's status (1).
        _task(
            s, "t1", "king-1", pool_type=1, status_id=int(TaskStatus.QUALIFIED), fp="f1"
        )
        # Wrong pool_type / not qualified -> must be excluded.
        _task(
            s,
            "t-wrongpool",
            "king-1",
            pool_type=2,
            status_id=int(TaskStatus.QUALIFIED),
            fp="f2",
        )
        _task(
            s,
            "t-candidate",
            "king-1",
            pool_type=1,
            status_id=int(TaskStatus.CANDIDATE),
            fp="f3",
        )
        _task(
            s,
            "t-pending-screen",
            "king-1",
            pool_type=1,
            status_id=int(TaskStatus.PENDING_SCREEN),
            fp="f-pending",
        )

    _seed(db, seed)


def _expected_duel_pair(
    task_id: str,
    *,
    king_submission_id: str = "king-1",
    challenger_submission_id: str = "sub-chal",
) -> list[tuple[str, str, str]]:
    return [
        (
            task_id,
            king_submission_id if side == "king" else challenger_submission_id,
            challenger_submission_id,
        )
        for side in _duel_side_order(
            task_id=task_id,
            king_submission_id=king_submission_id,
            challenger_submission_id=challenger_submission_id,
        )
    ]


def test_duel_jobs_returns_unsolved_king_and_challenger_for_active_challenge(
    db: SolverDb,
) -> None:
    _seed_active_challenge(db)
    jobs = db.next_duel_jobs(10)
    assert [
        (j.task_id, j.submission_id, j.challenger_submission_id) for j in jobs
    ] == _expected_duel_pair("t1")
    assert jobs[0].base_commit == _PARENT


def test_duel_jobs_can_wait_for_full_qualified_pool(db: SolverDb) -> None:
    _seed_active_challenge(db)
    targets = PoolTargets(pool_one=2, pool_two=2)

    assert db.next_duel_jobs(10, require_full_pool=True, pool_targets=targets) == []

    def add_second_qualified(s):  # noqa: ANN001
        _task(
            s, "t2", "king-1", pool_type=1, status_id=int(TaskStatus.QUALIFIED), fp="f4"
        )

    _seed(db, add_second_qualified)
    jobs = db.next_duel_jobs(10, require_full_pool=True, pool_targets=targets)
    assert [(j.task_id, j.submission_id, j.challenger_submission_id) for j in jobs] == [
        *_expected_duel_pair("t1"),
        *_expected_duel_pair("t2"),
    ]


def test_duel_jobs_preserves_complete_pairs_when_capacity_is_odd(db: SolverDb) -> None:
    _seed_active_challenge(db)

    def add_second_qualified(s):  # noqa: ANN001
        _task(
            s, "t2", "king-1", pool_type=1, status_id=int(TaskStatus.QUALIFIED), fp="f4"
        )

    _seed(db, add_second_qualified)

    jobs = db.next_duel_jobs(3)

    assert [
        (j.task_id, j.submission_id, j.challenger_submission_id) for j in jobs
    ] == _expected_duel_pair("t1")


def test_duel_jobs_excludes_each_already_solved_side(db: SolverDb) -> None:
    _seed_active_challenge(db)
    db.save_duel_task_solution(
        task_id="t1",
        challenger_submission_id="sub-chal",
        submission_id="king-1",
        solution="king diff",
        duration=1.0,
        exit_reason="completed",
    )
    jobs = db.next_duel_jobs(10)
    assert [(j.task_id, j.submission_id) for j in jobs] == [("t1", "sub-chal")]

    db.save_duel_task_solution(
        task_id="t1",
        challenger_submission_id="sub-chal",
        submission_id="sub-chal",
        solution="challenger diff",
        duration=1.0,
        exit_reason="completed",
    )
    assert db.next_duel_jobs(10) == []


def test_save_duel_task_solution_is_idempotent(db: SolverDb) -> None:
    _seed_active_challenge(db)
    for _ in range(2):
        db.save_duel_task_solution(
            task_id="t1",
            challenger_submission_id="sub-chal",
            submission_id="sub-chal",
            solution="diff",
            duration=1.0,
            exit_reason="completed",
        )
    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        rows = session.scalars(select(DuelTaskSolution)).all()
        rollout_rows = session.scalars(select(Rollout)).all()
    assert len(rows) == 1
    assert len(rollout_rows) == 1


def test_save_duel_task_solution_persists_full_rollout(db: SolverDb) -> None:
    _seed_active_challenge(db)
    db.save_duel_task_solution(
        task_id="t1",
        challenger_submission_id="sub-chal",
        submission_id="sub-chal",
        solution="diff",
        duration=1.0,
        exit_reason="completed",
        success=True,
        rollout_events=[
            {
                "index": 0,
                "type": "llm_call",
                "request": {"messages": [{"role": "user", "content": "task"}]},
                "response": {"choices": [{"message": {"content": "answer"}}]},
            }
        ],
    )
    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        rollout = session.scalars(select(Rollout)).one()

    assert rollout.phase == "duel"
    assert rollout.challenger_submission_id == "sub-chal"
    assert rollout.events[0]["request"]["messages"][0]["content"] == "task"


def test_save_duel_task_solution_keeps_only_safe_usage_summary(db: SolverDb) -> None:
    _seed_active_challenge(db)
    db.save_duel_task_solution(
        task_id="t1",
        challenger_submission_id="sub-chal",
        submission_id="sub-chal",
        solution="diff",
        duration=1.0,
        exit_reason="completed",
        usage_summary={
            "request_count": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0,
            "last_upstream_error": "internal-upstream.invalid do-not-publish marker",
            "requests": [
                {
                    "method": "POST",
                    "path": "http://internal-upstream.invalid/v1/chat/completions",
                    "status_code": 200,
                    "latency_ms": 250,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "error": "do-not-publish marker from internal-upstream.invalid",
                }
            ],
        },
    )

    with session_scope(session_factory(db._engine)) as session:  # noqa: SLF001
        row = session.scalars(select(DuelTaskSolution)).one()

    assert row.usage_summary == {
        "request_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "requests": [
            {
                "index": 0,
                "method": "POST",
                "status_code": 200,
                "latency_ms": 250,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        ],
    }
    encoded = str(row.usage_summary)
    assert "internal-upstream.invalid" not in encoded
    assert "do-not-publish marker" not in encoded
    assert "chat/completions" not in encoded
