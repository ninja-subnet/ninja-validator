"""SQLAlchemy ORM models — the runtime view of the validator schema.

These mirror the schema in the Alembic migrations under
`deploy/migrate/alembic/versions/`. Tests build the schema from these models
(`Base.metadata.create_all`); the migrations are the deploy path. Keep the two in sync
— a column changed here must be reflected in a migration.
"""

from __future__ import annotations

import datetime as dt
from enum import IntEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .status import ChallengeStatus, DuelOutcome, PoolType, SubmissionStatus, TaskStatus


class Base(DeclarativeBase):
    pass


def _enum_values_sql(enum: type[IntEnum]) -> str:
    """Render an IntEnum's values as a SQL ``IN`` list body, e.g. ``0, 1, 2``."""
    return ", ".join(str(member.value) for member in enum)


class Submission(Base):
    __tablename__ = "submissions"

    submission_id: Mapped[str] = mapped_column(Text, primary_key=True)
    # The chain block the submission was seen at; a bare number, not a FK.
    block: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    hotkey: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    # Manifest text only; workers load actual code from TAU_SUBMISSIONS_DIR/submission_id.
    agent_files: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    status_id: Mapped[int | None] = mapped_column(Integer)
    # First 16 hex of the agent bundle sha256, for duplicate detection.
    agent_sha256_prefix: Mapped[str | None] = mapped_column(Text, index=True)

    king: Mapped[King | None] = relationship(back_populates="submission")
    qualification: Mapped[SubmissionQualification | None] = relationship(
        back_populates="submission"
    )

    __table_args__ = (
        CheckConstraint(
            f"status_id IS NULL OR status_id IN ({_enum_values_sql(SubmissionStatus)})",
            name="ck_submissions_status_id",
        ),
    )


class SubmissionQualification(Base):
    """Security qualification result for a submission.

    One mutable row per submission. The worker may write an infrastructure error
    while keeping the submission UNVERIFIED for retry; final outcomes update the
    submission's status in the same transaction.
    """

    __tablename__ = "submission_qualifications"

    submission_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("submissions.submission_id", ondelete="CASCADE"),
        primary_key=True,
    )
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str | None] = mapped_column(Text)
    overall_score: Mapped[int | None] = mapped_column(Integer)
    security_score: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    reasons: Mapped[str | None] = mapped_column(Text)
    risks: Mapped[str | None] = mapped_column(Text)
    risk_categories: Mapped[str | None] = mapped_column(Text)
    failures: Mapped[str | None] = mapped_column(Text)
    required_changes: Mapped[str | None] = mapped_column(Text)
    base_files_available: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    submission: Mapped[Submission] = relationship(back_populates="qualification")

    __table_args__ = (
        CheckConstraint(
            "outcome IN ('qualified', 'needs_review', 'disqualified', 'error')",
            name="ck_submission_qualifications_outcome",
        ),
        CheckConstraint(
            "verdict IS NULL OR verdict IN ('pass', 'warn', 'fail')",
            name="ck_submission_qualifications_verdict",
        ),
    )


class King(Base):
    """A submission currently reigning as king."""

    __tablename__ = "kings"

    # A king is 1:1 with its submission: king_id IS the submission_id (FK to
    # submissions), used as the primary key -- no separate surrogate.
    king_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("submissions.submission_id", ondelete="CASCADE"),
        primary_key=True,
    )
    # When the submission was crowned (UTC); the DB stamps it. Indexed: ordering by it
    # picks the reigning king (see GeneratorDb/JudgeDb).
    king_from: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    submission: Mapped[Submission] = relationship(back_populates="king")
    challenges: Mapped[list[Challenge]] = relationship(back_populates="king")
    tasks: Mapped[list[Task]] = relationship(back_populates="king")


class Challenge(Base):
    """A challenger submission contesting a king."""

    __tablename__ = "challenges"

    # One challenge per submission, ever -> challenger_submission_id is the primary key.
    challenger_submission_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("submissions.submission_id", ondelete="CASCADE"),
        primary_key=True,
    )
    # The king faced; a plain attribute, not part of the key.
    king_id: Mapped[str] = mapped_column(
        Text, ForeignKey("kings.king_id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    king: Mapped[King] = relationship(back_populates="challenges")
    resolutions: Mapped[list[DuelResolution]] = relationship(back_populates="challenge")

    __table_args__ = (
        Index("ix_challenges_king_id", "king_id"),
        Index("ix_challenges_status", "status"),
        CheckConstraint(
            f"status IN ({_enum_values_sql(ChallengeStatus)})",
            name="ck_challenges_status",
        ),
    )


class DuelResolution(Base):
    """One pool's verdict within a challenge -- the durable, append-only record of
    what the resolver decided, kept with the tally and thresholds it decided on so
    outcomes stay recoverable if the decision rules change later.

    One row per pool: a challenge that advances POOL_ONE -> POOL_TWO leaves a
    POOL_ONE row behind, so a challenge's rows together are the duel's full path.
    """

    __tablename__ = "duel_resolutions"

    # The resolved challenge; challenger_submission_id is the challenge's identity.
    challenger_submission_id: Mapped[str] = mapped_column(
        Text, ForeignKey("challenges.challenger_submission_id", ondelete="CASCADE")
    )
    # Which pool this verdict is for (PoolType); each pool resolves at most once.
    pool_type: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[int] = mapped_column(Integer, nullable=False)
    # Decision inputs snapshotted at resolution, so the verdict stays auditable under
    # any future rules: the challenger-perspective tally and the thresholds applied.
    challenger_wins: Mapped[int] = mapped_column(Integer, nullable=False)
    challenger_losses: Mapped[int] = mapped_column(Integer, nullable=False)
    ties: Mapped[int] = mapped_column(Integer, nullable=False)
    best_of: Mapped[int] = mapped_column(Integer, nullable=False)
    scoring_method: Mapped[str] = mapped_column(Text, nullable=False)
    round_win_margin: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_score_margin: Mapped[float] = mapped_column(Float, nullable=False)
    king_score_mean: Mapped[float] = mapped_column(Float, nullable=False)
    challenger_score_mean: Mapped[float] = mapped_column(Float, nullable=False)
    score_mean_delta: Mapped[float] = mapped_column(Float, nullable=False)
    score_mean_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    # Token modifier inputs/results. Historical rows have unknown token totals and
    # settings; all new resolver writes fill every field explicitly.
    token_bonus_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    token_score_tolerance: Mapped[float | None] = mapped_column(Float)
    token_min_score: Mapped[float | None] = mapped_column(Float)
    token_bonus_multiplier: Mapped[float | None] = mapped_column(Float)
    king_total_tokens: Mapped[int | None] = mapped_column(BigInteger)
    challenger_total_tokens: Mapped[int | None] = mapped_column(BigInteger)
    token_comparison_rounds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    king_token_savings_mean: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0"
    )
    challenger_token_savings_mean: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0"
    )
    king_token_boost: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0"
    )
    challenger_token_boost: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0"
    )
    # Nullable during rollout so an old resolver can keep writing after the schema
    # migration. The new resolver always fills these; readers fall back to raw score.
    king_combined_score: Mapped[float | None] = mapped_column(Float)
    challenger_combined_score: Mapped[float | None] = mapped_column(Float)
    combined_score_delta: Mapped[float | None] = mapped_column(Float)
    # When the verdict was reached (UTC); inserting the row is the decision moment.
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    challenge: Mapped[Challenge] = relationship(back_populates="resolutions")

    __table_args__ = (
        PrimaryKeyConstraint("challenger_submission_id", "pool_type"),
        CheckConstraint(
            f"pool_type IN ({_enum_values_sql(PoolType)})",
            name="ck_duel_resolutions_pool_type",
        ),
        CheckConstraint(
            f"outcome IN ({_enum_values_sql(DuelOutcome)})",
            name="ck_duel_resolutions_outcome",
        ),
        CheckConstraint(
            "scoring_method IN ('round_wins', 'mean')",
            name="ck_duel_resolutions_scoring_method",
        ),
    )


class Task(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(Text, primary_key=True)
    pool_type: Mapped[int] = mapped_column(Integer, nullable=False)
    problem_statement: Mapped[str] = mapped_column(Text, nullable=False)
    # UTC wall-clock creation time (timestamptz), filled by the DB default.
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    status_id: Mapped[int] = mapped_column(Integer, nullable=False)
    king_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("kings.king_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Commit provenance + reference patch + dedup fingerprint. Every task is
    # commit-derived and needs these to be solvable, so they are required.
    repo_clone_url: Mapped[str] = mapped_column(Text, nullable=False)
    parent_sha: Mapped[str] = mapped_column(Text, nullable=False)
    commit_sha: Mapped[str] = mapped_column(Text, nullable=False)
    reference_patch: Mapped[str] = mapped_column(Text, nullable=False)
    content_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    # Write-once generation telemetry. Nullable: observability, not part of the
    # task's validity. `model` is the LLM that produced the description.
    model: Mapped[str | None] = mapped_column(Text)
    fetch_seconds: Mapped[float | None] = mapped_column(Float)
    llm_seconds: Mapped[float | None] = mapped_column(Float)
    llm_attempt: Mapped[int | None] = mapped_column(SmallInteger)
    # Commits discarded (by reason) before the winning one was found.
    rejected_duplicate: Mapped[int | None] = mapped_column(SmallInteger)
    rejected_structural: Mapped[int | None] = mapped_column(SmallInteger)
    rejected_quality: Mapped[int | None] = mapped_column(SmallInteger)
    rejected_fetch_error: Mapped[int | None] = mapped_column(SmallInteger)

    king: Mapped[King] = relationship(back_populates="tasks")
    screening: Mapped[TaskScreening | None] = relationship(back_populates="task")
    solutions: Mapped[list[TaskSolution]] = relationship(back_populates="task")
    duel_solutions: Mapped[list[DuelTaskSolution]] = relationship(back_populates="task")

    __table_args__ = (
        Index("uq_tasks_fingerprint", "content_fingerprint", unique=True),
        # Trigram index for fuzzy search over problem statements (needs pg_trgm).
        Index(
            "ix_tasks_problem_statement_trgm",
            "problem_statement",
            postgresql_using="gin",
            postgresql_ops={"problem_statement": "gin_trgm_ops"},
        ),
        CheckConstraint(
            f"status_id IN ({_enum_values_sql(TaskStatus)})", name="ck_tasks_status_id"
        ),
        CheckConstraint(
            f"pool_type IN ({_enum_values_sql(PoolType)})", name="ck_tasks_pool_type"
        ),
    )


class TaskScreening(Base):
    """A viable king qualification patch and its screening state."""

    __tablename__ = "task_screenings"

    task_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        primary_key=True,
    )
    king_submission_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    qualification_solution: Mapped[str] = mapped_column(Text, nullable=False)
    king_score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    failed_runs: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, server_default="0"
    )
    next_retry_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    task: Mapped[Task] = relationship(back_populates="screening")

    __table_args__ = (
        CheckConstraint(
            "king_score IS NULL OR (king_score >= 0 AND king_score <= 1)",
            name="ck_task_screenings_king_score",
        ),
        CheckConstraint(
            "max_score IS NULL OR (max_score >= 0 AND max_score <= 1)",
            name="ck_task_screenings_max_score",
        ),
        CheckConstraint(
            "failed_runs >= 0",
            name="ck_task_screenings_failed_runs",
        ),
    )


class TaskGenerationFailure(Base):
    """A mined commit abandoned after the LLM failed to describe it on every attempt.

    An append-only event log for observability / post-mortem -- never read by the
    pipeline. Unlike a real task this produced no `tasks` row, so it lives here
    instead of polluting that table (and its dedup fingerprint index). `king_id`
    is plain text with no foreign key, so a failure outlives the king it was
    recorded under.
    """

    __tablename__ = "task_generation_failures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    king_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    pool_type: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    repo_full_name: Mapped[str] = mapped_column(Text, nullable=False)
    commit_sha: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    # How many LLM attempts were spent before giving up, and the last failure text.
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    # UTC wall-clock failure time (timestamptz), filled by the DB default.
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TaskSolution(Base):
    """Legacy task-grain solution cache.

    Duel comparisons use ``DuelTaskSolution`` so each king-vs-challenger challenge
    gets fresh solves under the same current inference conditions.
    """

    __tablename__ = "task_solutions"

    task_id: Mapped[str] = mapped_column(
        Text, ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False
    )
    submission_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
    )
    solution: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Float)
    exit_reason: Mapped[str | None] = mapped_column(Text)

    task: Mapped[Task] = relationship(back_populates="solutions")

    __table_args__ = (
        PrimaryKeyConstraint("task_id", "submission_id"),
        Index("ix_task_solutions_submission_id", "submission_id"),
    )


class DuelTaskSolution(Base):
    """A fresh agent patch for one task within one challenge.

    ``challenger_submission_id`` is the challenge identity. The ``submission_id`` is
    the solver that produced the patch, either the challenge's king or its challenger.
    This keeps the king side from becoming a task-wide cache reused across future
    challengers.
    """

    __tablename__ = "duel_task_solutions"

    task_id: Mapped[str] = mapped_column(
        Text, ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False
    )
    challenger_submission_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("challenges.challenger_submission_id", ondelete="CASCADE"),
        nullable=False,
    )
    submission_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
    )
    solution: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Float)
    exit_reason: Mapped[str | None] = mapped_column(Text)
    usage_summary: Mapped[dict | None] = mapped_column(JSONB)

    task: Mapped[Task] = relationship(back_populates="duel_solutions")

    __table_args__ = (
        PrimaryKeyConstraint("task_id", "challenger_submission_id", "submission_id"),
        Index(
            "ix_duel_task_solutions_challenge",
            "challenger_submission_id",
            "task_id",
        ),
        Index("ix_duel_task_solutions_submission_id", "submission_id"),
    )


class Judgement(Base):
    """Pairwise LLM judgement of a king solution vs a challenger solution for a task."""

    __tablename__ = "judgements"

    task_id: Mapped[str] = mapped_column(Text, nullable=False)
    king_submission_id: Mapped[str] = mapped_column(Text, nullable=False)
    challenger_submission_id: Mapped[str] = mapped_column(Text, nullable=False)
    llm_winner: Mapped[str | None] = mapped_column(Text)
    king_score: Mapped[float | None] = mapped_column(Float)
    challenger_score: Mapped[float | None] = mapped_column(Float)
    # Telemetry. A set `error` marks a judge-failure fallback (neutral 0.5/0.5),
    # not a tie the LLM actually decided.
    model: Mapped[str | None] = mapped_column(Text)
    rationale: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int | None] = mapped_column(SmallInteger)
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint(
            "task_id", "king_submission_id", "challenger_submission_id"
        ),
        ForeignKeyConstraint(
            ["task_id"],
            ["tasks.task_id"],
            ondelete="CASCADE",
            name="fk_judgements_task",
        ),
        ForeignKeyConstraint(
            ["king_submission_id"],
            ["submissions.submission_id"],
            ondelete="CASCADE",
            name="fk_judgements_king_submission",
        ),
        ForeignKeyConstraint(
            ["challenger_submission_id"],
            ["challenges.challenger_submission_id"],
            ondelete="CASCADE",
            name="fk_judgements_challenge",
        ),
        Index("ix_judgements_king", "task_id", "king_submission_id"),
        Index("ix_judgements_challenger", "task_id", "challenger_submission_id"),
    )


class Registration(Base):
    """Per-block record of a uid's hot/cold key assignment on the subnet.

    The ERD declares no key for this entity, so a surrogate PK is added and a
    natural uniqueness constraint prevents duplicate rows per uid/block.
    """

    __tablename__ = "registrations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    uid: Mapped[int] = mapped_column(Integer, nullable=False)
    ss58_hot: Mapped[str] = mapped_column(Text, nullable=False)
    ss58_cold: Mapped[str] = mapped_column(Text, nullable=False)
    # The block the uid *registered* at (its on-chain BlockAtRegistration), not the
    # block the watcher first observed the registration — so it stays correct across
    # downtime and the initial backfill.
    block: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Full on-chain timestamp (UTC) of `block` — when the registration was true.
    block_date: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # When the worker wrote this row, set by the DB at insert time (not the chain's
    # block time — that's `block`/`block_date`). Distinguishes "true on-chain at" from
    # "observed and stored at".
    inserted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("uid", "block", name="uq_registrations_uid_block"),
        Index("ix_registrations_uid", "uid"),
        Index("ix_registrations_ss58_hot", "ss58_hot"),
    )
