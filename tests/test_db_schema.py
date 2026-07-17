"""Schema-level checks that need no database — enum contracts + ORM table shapes."""

from __future__ import annotations

from sqlalchemy import CheckConstraint

from tau.db import ChallengeStatus, GeneratorDb, PoolType, SubmissionStatus, TaskStatus
from tau.db.solver import _duel_side_order
from tau.db.models import (
    Challenge,
    DuelResolution,
    DuelTaskSolution,
    Judgement,
    King,
    KingArchive,
    Rollout,
    Submission,
    SubmissionQualification,
    Task,
    TaskScreening,
)


def test_task_status_values_are_stable() -> None:
    # These ints are persisted in tasks.status_id — they must not drift.
    assert (TaskStatus.CANDIDATE, TaskStatus.QUALIFIED, TaskStatus.DISQUALIFIED) == (
        0,
        1,
        2,
    )
    assert TaskStatus.PENDING_SCREEN == 3


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
    assert all(
        str(s.value) in checks["ck_submissions_status_id"] for s in SubmissionStatus
    )


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


def test_task_screening_table_shape_and_foreign_keys() -> None:
    assert [c.name for c in TaskScreening.__table__.primary_key.columns] == ["task_id"]
    cols = TaskScreening.__table__.columns
    assert {
        "task_id",
        "king_submission_id",
        "qualification_solution",
        "king_score",
        "max_score",
        "reason",
        "model",
        "failed_runs",
        "next_retry_at",
        "created_at",
        "updated_at",
    } == set(cols.keys())
    for name in (
        "task_id",
        "king_submission_id",
        "qualification_solution",
        "failed_runs",
        "created_at",
        "updated_at",
    ):
        assert cols[name].nullable is False, name
    fk_targets = {
        (fk.parent.name, fk.column.table.name, fk.column.name)
        for fk in TaskScreening.__table__.foreign_keys
    }
    assert ("task_id", "tasks", "task_id") in fk_targets
    assert (
        "king_submission_id",
        "submissions",
        "submission_id",
    ) in fk_targets


def test_task_screening_checks_bound_scores_and_failures() -> None:
    checks = {
        c.name: str(c.sqltext)
        for c in TaskScreening.__table__.constraints
        if isinstance(c, CheckConstraint)
    }
    assert "ck_task_screenings_king_score" in checks
    assert "ck_task_screenings_max_score" in checks
    assert "ck_task_screenings_failed_runs" in checks
    assert "failed_runs >= 0" in checks["ck_task_screenings_failed_runs"]


def test_challenge_status_values_are_stable() -> None:
    # Persisted in challenges.status; the active-pool values MUST equal PoolType so
    # the judge gate (task.pool_type == challenge.status) holds.
    assert (
        ChallengeStatus.CLOSED,
        ChallengeStatus.POOL_ONE,
        ChallengeStatus.POOL_TWO,
    ) == (
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


def test_duel_task_solution_is_scoped_to_challenge() -> None:
    assert [c.name for c in DuelTaskSolution.__table__.primary_key.columns] == [
        "task_id",
        "challenger_submission_id",
        "submission_id",
    ]
    assert "usage_summary" in DuelTaskSolution.__table__.columns
    assert DuelTaskSolution.__table__.c.usage_summary.nullable is True
    fk_targets = {
        (fk.parent.name, fk.column.table.name, fk.column.name)
        for fk in DuelTaskSolution.__table__.foreign_keys
    }
    assert ("task_id", "tasks", "task_id") in fk_targets
    assert (
        "challenger_submission_id",
        "challenges",
        "challenger_submission_id",
    ) in fk_targets
    assert ("submission_id", "submissions", "submission_id") in fk_targets


def test_rollout_table_has_normalized_capture_shape() -> None:
    assert [c.name for c in Rollout.__table__.primary_key.columns] == ["rollout_id"]
    assert {
        "phase",
        "task_id",
        "submission_id",
        "challenger_submission_id",
        "success",
        "solution_diff",
        "exit_reason",
        "duration_seconds",
        "usage_summary",
        "events",
        "created_at",
    } <= set(Rollout.__table__.columns.keys())
    assert Rollout.__table__.c.events.nullable is True
    indexes = {index.name for index in Rollout.__table__.indexes}
    assert "ix_rollouts_task_order" in indexes


def test_king_archive_table_is_a_persistent_retry_queue() -> None:
    assert [c.name for c in KingArchive.__table__.primary_key.columns] == ["king_id"]
    assert {
        "promoted_to",
        "status",
        "attempts",
        "next_attempt_at",
        "last_error",
        "updated_at",
        "completed_at",
    } <= set(KingArchive.__table__.columns.keys())


def test_duel_resolution_keeps_raw_token_and_merged_scores() -> None:
    columns = DuelResolution.__table__.columns
    assert {
        "king_score_mean",
        "challenger_score_mean",
        "score_mean_delta",
        "king_total_tokens",
        "challenger_total_tokens",
        "king_token_boost",
        "challenger_token_boost",
        "king_combined_score",
        "challenger_combined_score",
        "combined_score_delta",
    } <= set(columns.keys())
    assert columns.king_total_tokens.nullable is True
    assert columns.challenger_total_tokens.nullable is True
    assert columns.king_combined_score.nullable is True
    assert columns.challenger_combined_score.nullable is True
    assert columns.combined_score_delta.nullable is True


def test_duel_side_order_is_stable_and_varies_by_task() -> None:
    first = _duel_side_order(
        task_id="task-0",
        king_submission_id="king",
        challenger_submission_id="challenger",
    )
    assert first == _duel_side_order(
        task_id="task-0",
        king_submission_id="king",
        challenger_submission_id="challenger",
    )
    orders = {
        _duel_side_order(
            task_id=f"task-{index}",
            king_submission_id="king",
            challenger_submission_id="challenger",
        )
        for index in range(32)
    }
    assert orders == {("king", "challenger"), ("challenger", "king")}


def test_judgements_no_longer_reference_task_wide_solutions() -> None:
    fk_tables = {
        fk.column.table.name
        for constraint in Judgement.__table__.foreign_key_constraints
        for fk in constraint.elements
    }
    assert "task_solutions" not in fk_tables
    assert {"tasks", "submissions", "challenges"} <= fk_tables


def test_generator_db_is_importable() -> None:
    assert GeneratorDb is not None
