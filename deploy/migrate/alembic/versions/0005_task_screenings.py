"""add single-task difficulty screening

Create Date: 2026-07-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_task_screenings"
down_revision: str | None = "0004_agent_sha256_prefix"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_tasks_status_id", "tasks", type_="check")
    op.create_check_constraint(
        "ck_tasks_status_id", "tasks", "status_id IN (0, 1, 2, 3)"
    )

    op.create_table(
        "task_screenings",
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("king_submission_id", sa.Text(), nullable=False),
        sa.Column("qualification_solution", sa.Text(), nullable=False),
        sa.Column("king_score", sa.Float(), nullable=True),
        sa.Column("max_score", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column(
            "failed_runs",
            sa.SmallInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["task_id"], ["tasks.task_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["king_submission_id"],
            ["submissions.submission_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("task_id"),
        sa.CheckConstraint(
            "king_score IS NULL OR (king_score >= 0 AND king_score <= 1)",
            name="ck_task_screenings_king_score",
        ),
        sa.CheckConstraint(
            "max_score IS NULL OR (max_score >= 0 AND max_score <= 1)",
            name="ck_task_screenings_max_score",
        ),
        sa.CheckConstraint(
            "failed_runs >= 0",
            name="ck_task_screenings_failed_runs",
        ),
    )
    op.create_index(
        "ix_task_screenings_king_submission_id",
        "task_screenings",
        ["king_submission_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_task_screenings_king_submission_id", table_name="task_screenings")
    op.drop_table("task_screenings")

    op.drop_constraint("ck_tasks_status_id", "tasks", type_="check")
    # A pending-screen task did not exist before this migration. Returning it to
    # CANDIDATE preserves the ability of the old task-solver to process it.
    op.execute("UPDATE tasks SET status_id = 0 WHERE status_id = 3")
    op.create_check_constraint("ck_tasks_status_id", "tasks", "status_id IN (0, 1, 2)")
