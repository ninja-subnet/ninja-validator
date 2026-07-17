"""queue retired king archives outside the promotion transaction

Revision ID: 0008_king_archives
Revises: 0007_rollouts
Create Date: 2026-07-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_king_archives"
down_revision: str | None = "0007_rollouts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_rollouts_task_order",
        "rollouts",
        ["task_id", "rollout_id"],
    )
    op.create_table(
        "king_archives",
        sa.Column("king_id", sa.Text(), nullable=False),
        sa.Column("promoted_to", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
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
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'succeeded')",
            name="ck_king_archives_status",
        ),
        sa.ForeignKeyConstraint(["king_id"], ["kings.king_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["promoted_to"], ["kings.king_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("king_id"),
    )
    op.create_index(
        "ix_king_archives_ready",
        "king_archives",
        ["status", "next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_table("king_archives")
    op.drop_index("ix_rollouts_task_order", table_name="rollouts")
