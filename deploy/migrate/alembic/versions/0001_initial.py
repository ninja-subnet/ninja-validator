"""initial schema from ERD

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Self-contained: the trigram index below needs this extension. initdb also
    # creates it, but a migration should not depend on init ordering.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # --- submissions ---------------------------------------------------------
    op.create_table(
        "submissions",
        sa.Column("submission_id", sa.Text(), nullable=False),
        sa.Column("block", sa.BigInteger(), nullable=False),
        sa.Column("hotkey", sa.Text(), nullable=False),
        sa.Column("agent_files", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("status_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("submission_id"),
        sa.CheckConstraint(
            "status_id IS NULL OR status_id IN (0, 1, 2, 3)",
            name="ck_submissions_status_id",
        ),
    )
    op.create_index("ix_submissions_block", "submissions", ["block"])
    op.create_index("ix_submissions_hotkey", "submissions", ["hotkey"])

    # --- submission_qualifications ------------------------------------------
    op.create_table(
        "submission_qualifications",
        sa.Column("submission_id", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=True),
        sa.Column("overall_score", sa.Integer(), nullable=True),
        sa.Column("security_score", sa.Integer(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("reasons", sa.Text(), nullable=True),
        sa.Column("risks", sa.Text(), nullable=True),
        sa.Column("risk_categories", sa.Text(), nullable=True),
        sa.Column("failures", sa.Text(), nullable=True),
        sa.Column("required_changes", sa.Text(), nullable=True),
        sa.Column("base_files_available", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"], ["submissions.submission_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("submission_id"),
        sa.CheckConstraint(
            "outcome IN ('qualified', 'needs_review', 'disqualified', 'error')",
            name="ck_submission_qualifications_outcome",
        ),
        sa.CheckConstraint(
            "verdict IS NULL OR verdict IN ('pass', 'warn', 'fail')",
            name="ck_submission_qualifications_verdict",
        ),
    )

    # --- kings ---------------------------------------------------------------
    # A king is 1:1 with its submission: king_id IS the submission_id (FK), used as the
    # primary key -- no separate surrogate.
    op.create_table(
        "kings",
        sa.Column("king_id", sa.Text(), nullable=False),
        # When the submission was crowned (UTC, timestamptz), filled by the DB default.
        sa.Column(
            "king_from",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["king_id"], ["submissions.submission_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("king_id"),
    )
    op.create_index("ix_kings_king_from", "kings", ["king_from"])

    # --- challenges ----------------------------------------------------------
    # One challenge per submission, ever: challenger_submission_id is the primary key;
    # king_id (the king's submission_id) is a plain attribute.
    op.create_table(
        "challenges",
        sa.Column("challenger_submission_id", sa.Text(), nullable=False),
        sa.Column("king_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Integer(), server_default="0", nullable=False),
        # UTC wall-clock creation time (timestamptz), filled by the DB default.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["challenger_submission_id"],
            ["submissions.submission_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["king_id"], ["kings.king_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("challenger_submission_id"),
        sa.CheckConstraint("status IN (0, 1, 2)", name="ck_challenges_status"),
    )
    op.create_index("ix_challenges_king_id", "challenges", ["king_id"])
    op.create_index("ix_challenges_status", "challenges", ["status"])

    # --- duel_resolutions ----------------------------------------------------
    # One row per pool resolved in a challenge: the durable, append-only verdict
    # (with the tally + thresholds it was decided on) so outcomes survive later
    # changes to the decision rules. A pool resolves at most once.
    op.create_table(
        "duel_resolutions",
        sa.Column("challenger_submission_id", sa.Text(), nullable=False),
        sa.Column("pool_type", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Integer(), nullable=False),
        # Decision inputs, snapshotted so the verdict stays auditable under any rules.
        sa.Column("challenger_wins", sa.Integer(), nullable=False),
        sa.Column("challenger_losses", sa.Integer(), nullable=False),
        sa.Column("ties", sa.Integer(), nullable=False),
        sa.Column("best_of", sa.Integer(), nullable=False),
        sa.Column("scoring_method", sa.Text(), nullable=False),
        sa.Column("round_win_margin", sa.Integer(), nullable=False),
        sa.Column("mean_score_margin", sa.Float(), nullable=False),
        sa.Column("king_score_mean", sa.Float(), nullable=False),
        sa.Column("challenger_score_mean", sa.Float(), nullable=False),
        sa.Column("score_mean_delta", sa.Float(), nullable=False),
        sa.Column("score_mean_rounds", sa.Integer(), nullable=False),
        # UTC wall-clock decision time (timestamptz), filled by the DB default.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["challenger_submission_id"],
            ["challenges.challenger_submission_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("challenger_submission_id", "pool_type"),
        sa.CheckConstraint("pool_type IN (1, 2)", name="ck_duel_resolutions_pool_type"),
        sa.CheckConstraint("outcome IN (0, 1, 2)", name="ck_duel_resolutions_outcome"),
        sa.CheckConstraint(
            "scoring_method IN ('round_wins', 'mean')",
            name="ck_duel_resolutions_scoring_method",
        ),
    )

    # --- tasks ---------------------------------------------------------------
    op.create_table(
        "tasks",
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("pool_type", sa.Integer(), nullable=False),
        sa.Column("problem_statement", sa.Text(), nullable=False),
        # UTC wall-clock creation time (timestamptz), filled by the DB default.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("status_id", sa.Integer(), nullable=False),
        sa.Column("king_id", sa.Text(), nullable=False),
        # Commit provenance the solver needs to rebuild a workspace (clone parent +
        # checkout), the reference patch the judge uses as privileged context, and
        # the dedup fingerprint. Every task is commit-derived and needs these to be
        # solvable, so they are required.
        sa.Column("repo_clone_url", sa.Text(), nullable=False),
        sa.Column("parent_sha", sa.Text(), nullable=False),
        sa.Column("commit_sha", sa.Text(), nullable=False),
        sa.Column("reference_patch", sa.Text(), nullable=False),
        sa.Column("content_fingerprint", sa.Text(), nullable=False),
        # Write-once generation telemetry (nullable observability columns).
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("fetch_seconds", sa.Float(), nullable=True),
        sa.Column("llm_seconds", sa.Float(), nullable=True),
        sa.Column("llm_attempt", sa.SmallInteger(), nullable=True),
        sa.Column("rejected_duplicate", sa.SmallInteger(), nullable=True),
        sa.Column("rejected_structural", sa.SmallInteger(), nullable=True),
        sa.Column("rejected_quality", sa.SmallInteger(), nullable=True),
        sa.Column("rejected_fetch_error", sa.SmallInteger(), nullable=True),
        sa.ForeignKeyConstraint(["king_id"], ["kings.king_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("task_id"),
        # Bound the integer domains to their enums (tau.db.status).
        sa.CheckConstraint("status_id IN (0, 1, 2)", name="ck_tasks_status_id"),
        sa.CheckConstraint("pool_type IN (1, 2)", name="ck_tasks_pool_type"),
    )
    op.create_index("ix_tasks_king_id", "tasks", ["king_id"])
    # Dedup mined commits: the generator inserts ON CONFLICT DO NOTHING on this.
    op.create_index(
        "uq_tasks_fingerprint", "tasks", ["content_fingerprint"], unique=True
    )
    # Trigram index for fuzzy search over problem statements.
    op.create_index(
        "ix_tasks_problem_statement_trgm",
        "tasks",
        ["problem_statement"],
        postgresql_using="gin",
        postgresql_ops={"problem_statement": "gin_trgm_ops"},
    )

    # --- task_generation_failures --------------------------------------------
    # Append-only log of commits abandoned after the LLM failed every attempt;
    # these produce no tasks row. No FK on king_id so a failure outlives its king.
    op.create_table(
        "task_generation_failures",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("king_id", sa.Text(), nullable=False),
        sa.Column("pool_type", sa.SmallInteger(), nullable=False),
        sa.Column("repo_full_name", sa.Text(), nullable=False),
        sa.Column("commit_sha", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("attempts", sa.SmallInteger(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        # UTC wall-clock failure time (timestamptz), filled by the DB default.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_task_generation_failures_king_id", "task_generation_failures", ["king_id"]
    )

    # --- task_solutions ------------------------------------------------------
    op.create_table(
        "task_solutions",
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("submission_id", sa.Text(), nullable=False),
        sa.Column("solution", sa.Text(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.task_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["submission_id"], ["submissions.submission_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("task_id", "submission_id"),
    )
    op.create_index(
        "ix_task_solutions_submission_id", "task_solutions", ["submission_id"]
    )

    # --- judgements -----------------------------------------------------------
    op.create_table(
        "judgements",
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("king_submission_id", sa.Text(), nullable=False),
        sa.Column("challenger_submission_id", sa.Text(), nullable=False),
        sa.Column("llm_winner", sa.Text(), nullable=True),
        sa.Column("king_score", sa.Float(), nullable=True),
        sa.Column("challenger_score", sa.Float(), nullable=True),
        # Telemetry: a set `error` marks a judge-failure fallback (neutral tie).
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.SmallInteger(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        # UTC wall-clock time the verdict was persisted (timestamptz), DB default.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["task_id", "king_submission_id"],
            ["task_solutions.task_id", "task_solutions.submission_id"],
            ondelete="CASCADE",
            name="fk_judgements_king_solution",
        ),
        sa.ForeignKeyConstraint(
            ["task_id", "challenger_submission_id"],
            ["task_solutions.task_id", "task_solutions.submission_id"],
            ondelete="CASCADE",
            name="fk_judgements_challenger_solution",
        ),
        sa.PrimaryKeyConstraint(
            "task_id", "king_submission_id", "challenger_submission_id"
        ),
    )
    op.create_index(
        "ix_judgements_king", "judgements", ["task_id", "king_submission_id"]
    )
    op.create_index(
        "ix_judgements_challenger",
        "judgements",
        ["task_id", "challenger_submission_id"],
    )

    # --- registrations -------------------------------------------------------
    op.create_table(
        "registrations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("uid", sa.Integer(), nullable=False),
        sa.Column("ss58_hot", sa.Text(), nullable=False),
        sa.Column("ss58_cold", sa.Text(), nullable=False),
        sa.Column("block", sa.BigInteger(), nullable=False),
        # Full on-chain timestamp (UTC) of `block`, not just the calendar date.
        sa.Column("block_date", sa.DateTime(timezone=True), nullable=False),
        # Set by the DB at write time — when the worker stored the row, as opposed to
        # block/block_date which are when the registration was true on-chain.
        sa.Column(
            "inserted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uid", "block", name="uq_registrations_uid_block"),
    )
    op.create_index("ix_registrations_uid", "registrations", ["uid"])
    op.create_index("ix_registrations_ss58_hot", "registrations", ["ss58_hot"])

    # --- views ---------------------------------------------------------------
    # Active, unsolved tasks for in-progress challenges.
    op.execute(
        """
        CREATE VIEW v_active_unsolved_tasks AS
        SELECT t.*
        FROM challenges c
        JOIN tasks t ON c.king_id = t.king_id
        RIGHT JOIN task_solutions ts ON t.task_id = ts.task_id
        WHERE c.status IN (1, 2)
          AND t.pool_type = c.status
          AND t.status_id = 1
        """
    )

    # Submissions that are the challenger in an active challenge.
    op.execute(
        """
        CREATE VIEW v_active_challenger_submissions AS
        SELECT s.*
        FROM submissions s
        JOIN challenges c ON s.submission_id = c.challenger_submission_id
        WHERE c.status IN (1, 2)
        """
    )

    # Latest registration row per uid (highest block), newest writes first.
    op.execute(
        """
        CREATE VIEW v_current_metagraph AS
        SELECT * FROM (
            SELECT DISTINCT ON (uid) *
            FROM registrations
            ORDER BY uid, block DESC
        ) AS latest_records
        ORDER BY inserted_at DESC
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_current_metagraph")
    op.execute("DROP VIEW IF EXISTS v_active_challenger_submissions")
    op.execute("DROP VIEW IF EXISTS v_active_unsolved_tasks")
    op.drop_table("duel_resolutions")
    op.drop_table("registrations")
    op.drop_table("judgements")
    op.drop_table("task_solutions")
    op.drop_table("task_generation_failures")
    op.drop_table("tasks")
    op.drop_table("challenges")
    op.drop_table("kings")
    op.drop_table("submission_qualifications")
    op.drop_table("submissions")
