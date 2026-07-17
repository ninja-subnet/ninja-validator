"""persist full redacted agent rollouts

Revision ID: 0007_rollouts
Revises: 0006_token_efficiency_scoring
Create Date: 2026-07-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_rollouts"
down_revision: str | None = "0006_token_efficiency_scoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rollouts",
        sa.Column("rollout_id", sa.Text(), nullable=False),
        sa.Column("phase", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("submission_id", sa.Text(), nullable=False),
        sa.Column("challenger_submission_id", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("solution_diff", sa.Text(), nullable=False),
        sa.Column("exit_reason", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column(
            "usage_summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "events", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "phase IN ('qualification', 'duel')", name="ck_rollouts_phase"
        ),
        sa.CheckConstraint(
            "(phase = 'qualification' AND challenger_submission_id IS NULL) OR "
            "(phase = 'duel' AND challenger_submission_id IS NOT NULL)",
            name="ck_rollouts_challenge_scope",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["tasks.task_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["submissions.submission_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["challenger_submission_id"],
            ["challenges.challenger_submission_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("rollout_id"),
    )
    op.create_index("ix_rollouts_task_id", "rollouts", ["task_id"])
    op.create_index("ix_rollouts_submission_id", "rollouts", ["submission_id"])
    op.create_index(
        "ix_rollouts_challenge",
        "rollouts",
        ["challenger_submission_id", "task_id"],
    )


def downgrade() -> None:
    op.drop_table("rollouts")
