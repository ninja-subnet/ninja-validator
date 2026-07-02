"""Schema-level checks that need no database — enum contracts + ORM table shapes."""

from __future__ import annotations

from sqlalchemy import CheckConstraint

from tau.db import ChallengeStatus, GeneratorDb, PoolType, SubmissionStatus, TaskStatus
from tau.db.models import Challenge, King, Submission, SubmissionQualification, Task


def test_task_status_values_are_stable() -> None:
    # These ints are persisted in tasks.status_id — they must not drift.
    assert (TaskStatus.CANDIDATE, TaskStatus.QUALIFIED, TaskStatus.DISQUALIFIED) == (0, 1, 2)


def test_submission_status_values_are_stable() -> None:
    assert (
        SubmissionStatus.UNVERIFIED,
        SubmissionStatus.ELIGIBLE,
        SubmissionStatus.DISQUALIFIED,
        SubmissionStatus.NEEDS_REVIEW,
    ) == (0, 1, 2, 3)


def test_pool_type_values_are_stable() -> None:
    assert (PoolType.POOL_ONE, PoolType.POOL_TWO) == (1, 2)


def test_task_has_commit_source_columns() -> None:
    cols = set(Task.__table__.columns.keys())
    assert {
        "repo_clone_url",
        "parent_sha",
        "commit_sha",
        "reference_patch",
        "content_fingerprint",
    } <= cols


def test_task_commit_columns_and_status_are_not_null() -> None:
    # Every task is commit-derived and always has a status, so these are required.
    cols = Task.__table__.columns
    for name in (
        "repo_clone_url",
        "parent_sha",
        "commit_sha",
        "reference_patch",
        "content_fingerprint",
        "status_id",
    ):
        assert cols[name].nullable is False, name


def test_task_fingerprint_index_is_unique_on_content_fingerprint() -> None:
    indexes = {index.name: index for index in Task.__table__.indexes}
    assert "uq_tasks_fingerprint" in indexes
    fingerprint_index = indexes["uq_tasks_fingerprint"]
    assert fingerprint_index.unique
    assert [c.name for c in fingerprint_index.columns] == ["content_fingerprint"]


def test_status_and_pool_have_check_constraints_matching_the_enums() -> None:
    checks = {
        c.name: str(c.sqltext)
        for c in Task.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert "ck_tasks_status_id" in checks
    assert "ck_tasks_pool_type" in checks
    # The CHECK must list exactly the enum values (derived from them, can't drift).
    assert all(str(s.value) in checks["ck_tasks_status_id"] for s in TaskStatus)
    assert all(str(p.value) in checks["ck_tasks_pool_type"] for p in PoolType)


def test_submission_status_has_check_constraint_matching_the_enum() -> None:
    checks = {
        c.name: str(c.sqltext)
        for c in Submission.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert "ck_submissions_status_id" in checks
    assert all(str(s.value) in checks["ck_submissions_status_id"] for s in SubmissionStatus)


def test_submission_agent_files_is_manifest_text_not_code_payload() -> None:
    assert Submission.__table__.c.agent_files.type.python_type is str
    assert Submission.__table__.c.agent_files.nullable is True


def test_submission_qualification_table_shape() -> None:
    assert [c.name for c in SubmissionQualification.__table__.primary_key.columns] == [
        "submission_id"
    ]
    fk = next(iter(SubmissionQualification.__table__.c.submission_id.foreign_keys))
    assert fk.column.table.name == "submissions"
    assert fk.column.name == "submission_id"
    cols = set(SubmissionQualification.__table__.columns.keys())
    assert {
        "outcome",
        "verdict",
        "overall_score",
        "security_score",
        "model",
        "summary",
        "reasons",
        "risks",
        "risk_categories",
        "failures",
        "required_changes",
        "base_files_available",
        "error",
        "duration_seconds",
        "created_at",
        "updated_at",
    } <= cols


def test_challenge_status_values_are_stable() -> None:
    # Persisted in challenges.status; the active-pool values MUST equal PoolType so
    # the judge gate (task.pool_type == challenge.status) holds.
    assert (ChallengeStatus.CLOSED, ChallengeStatus.POOL_ONE, ChallengeStatus.POOL_TWO) == (
        0,
        1,
        2,
    )
    assert (ChallengeStatus.POOL_ONE, ChallengeStatus.POOL_TWO) == (
        PoolType.POOL_ONE,
        PoolType.POOL_TWO,
    )


def test_kings_primary_key_is_king_id() -> None:
    assert [c.name for c in King.__table__.primary_key.columns] == ["king_id"]


def test_kings_king_id_is_fk_to_submissions() -> None:
    # king_id holds the king's submission_id (the natural key), no surrogate.
    fk = next(iter(King.__table__.c.king_id.foreign_keys))
    assert fk.column.table.name == "submissions"
    assert fk.column.name == "submission_id"


def test_challenge_primary_key_is_challenger_submission_id() -> None:
    # One challenge per submission, ever.
    assert [c.name for c in Challenge.__table__.primary_key.columns] == [
        "challenger_submission_id"
    ]


def test_challenges_status_has_check_constraint_matching_the_enum() -> None:
    checks = {
        c.name: str(c.sqltext)
        for c in Challenge.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert "ck_challenges_status" in checks
    assert all(str(s.value) in checks["ck_challenges_status"] for s in ChallengeStatus)


def test_challenge_king_fk_points_at_kings_king_id() -> None:
    fk = next(iter(Challenge.__table__.c.king_id.foreign_keys))
    assert fk.column.table.name == "kings"
    assert fk.column.name == "king_id"


def test_generator_db_is_importable() -> None:
    assert GeneratorDb is not None
