"""add auditable duel token-efficiency scoring

Revision ID: 0006_duel_token_efficiency
Revises: 0005_task_screenings
Create Date: 2026-07-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_duel_token_efficiency"
down_revision: str | None = "0005_task_screenings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_duel_resolutions_scoring_method",
        "duel_resolutions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_duel_resolutions_scoring_method",
        "duel_resolutions",
        "scoring_method IN ('round_wins', 'mean', 'token_efficiency')",
    )
    op.add_column(
        "duel_resolutions",
        sa.Column("token_weight", sa.Float(), server_default="0", nullable=False),
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_quality_floor", sa.Float(), server_default="0.7", nullable=False
        ),
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_efficiency_clip", sa.Float(), server_default="0.5", nullable=False
        ),
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_efficiency_mean", sa.Float(), server_default="0", nullable=False
        ),
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_usage_rounds", sa.Integer(), server_default="0", nullable=False
        ),
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "token_usage_penalty_rounds",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "duel_resolutions",
        sa.Column(
            "adjusted_score_delta", sa.Float(), server_default="0", nullable=False
        ),
    )
    op.execute("UPDATE duel_resolutions SET adjusted_score_delta = score_mean_delta")
    op.create_check_constraint(
        "ck_duel_resolutions_token_weight",
        "duel_resolutions",
        "token_weight >= 0 AND token_weight <= 1",
    )
    op.create_check_constraint(
        "ck_duel_resolutions_token_quality_floor",
        "duel_resolutions",
        "token_quality_floor >= 0 AND token_quality_floor <= 1",
    )
    op.create_check_constraint(
        "ck_duel_resolutions_token_efficiency_clip",
        "duel_resolutions",
        "token_efficiency_clip > 0 AND token_efficiency_clip <= 1",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_duel_resolutions_token_efficiency_clip",
        "duel_resolutions",
        type_="check",
    )
    op.drop_constraint(
        "ck_duel_resolutions_token_quality_floor",
        "duel_resolutions",
        type_="check",
    )
    op.drop_constraint(
        "ck_duel_resolutions_token_weight",
        "duel_resolutions",
        type_="check",
    )
    for column in (
        "adjusted_score_delta",
        "token_usage_penalty_rounds",
        "token_usage_rounds",
        "token_efficiency_mean",
        "token_efficiency_clip",
        "token_quality_floor",
        "token_weight",
    ):
        op.drop_column("duel_resolutions", column)
    op.drop_constraint(
        "ck_duel_resolutions_scoring_method",
        "duel_resolutions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_duel_resolutions_scoring_method",
        "duel_resolutions",
        "scoring_method IN ('round_wins', 'mean')",
    )
