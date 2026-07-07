"""add submissions.agent_sha256_prefix dedup key

Revision ID: 0003_agent_sha256_prefix
Revises: 0002_challenge_scoped_solutions
Create Date: 2026-07-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_agent_sha256_prefix"
down_revision: str | None = "0002_challenge_scoped_solutions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "submissions", sa.Column("agent_sha256_prefix", sa.Text(), nullable=True)
    )
    # submission_id ends with agent_sha256[:16], so backfill from its last 16 chars.
    op.execute("UPDATE submissions SET agent_sha256_prefix = right(submission_id, 16)")
    op.create_index(
        "ix_submissions_agent_sha256_prefix",
        "submissions",
        ["agent_sha256_prefix"],
    )


def downgrade() -> None:
    op.drop_index("ix_submissions_agent_sha256_prefix", table_name="submissions")
    op.drop_column("submissions", "agent_sha256_prefix")
