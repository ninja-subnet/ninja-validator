"""add auditable token-efficiency duel scores

Revision ID: 0006_token_efficiency_scoring
Revises: 0005_task_screenings
Create Date: 2026-07-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_token_efficiency_scoring"
down_revision: str | None = "0005_task_screenings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_bonus_enabled",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column("duel_resolutions", sa.Column("token_score_tolerance", sa.Float()))
    op.add_column("duel_resolutions", sa.Column("token_min_score", sa.Float()))
    op.add_column("duel_resolutions", sa.Column("token_bonus_multiplier", sa.Float()))
    op.add_column("duel_resolutions", sa.Column("king_total_tokens", sa.BigInteger()))
    op.add_column(
        "duel_resolutions", sa.Column("challenger_total_tokens", sa.BigInteger())
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_comparison_rounds",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    for name in (
        "king_token_savings_mean",
        "challenger_token_savings_mean",
        "king_token_boost",
        "challenger_token_boost",
    ):
        op.add_column(
            "duel_resolutions",
            sa.Column(
                name,
                sa.Float(),
                server_default=sa.text("0"),
                nullable=False,
            ),
        )

    # Old rows did not use a token modifier. Their merged scores therefore equal
    # their already-saved raw scores; token totals/settings remain unknown (NULL).
    op.add_column("duel_resolutions", sa.Column("king_combined_score", sa.Float()))
    op.add_column(
        "duel_resolutions", sa.Column("challenger_combined_score", sa.Float())
    )
    op.add_column("duel_resolutions", sa.Column("combined_score_delta", sa.Float()))
    op.execute(
        """
        UPDATE duel_resolutions
        SET king_combined_score = king_score_mean,
            challenger_combined_score = challenger_score_mean,
            combined_score_delta = score_mean_delta
        """
    )
    # Keep these nullable for rolling-deploy compatibility: an old resolver that is
    # still draining does not know about the new columns. New code always writes
    # them, and readers use the raw scores when a transition row contains NULL.


def downgrade() -> None:
    for name in (
        "combined_score_delta",
        "challenger_combined_score",
        "king_combined_score",
        "challenger_token_boost",
        "king_token_boost",
        "challenger_token_savings_mean",
        "king_token_savings_mean",
        "token_comparison_rounds",
        "challenger_total_tokens",
        "king_total_tokens",
        "token_bonus_multiplier",
        "token_min_score",
        "token_score_tolerance",
        "token_bonus_enabled",
    ):
        op.drop_column("duel_resolutions", name)
