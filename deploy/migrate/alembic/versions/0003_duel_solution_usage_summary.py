"""store sanitized duel solve usage summaries

Revision ID: 0003_duel_solution_usage_summary
Revises: 0002_challenge_scoped_solutions
Create Date: 2026-07-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_duel_solution_usage_summary"
down_revision: str | None = "0002_challenge_scoped_solutions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "duel_task_solutions",
        sa.Column("usage_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("duel_task_solutions", "usage_summary")
