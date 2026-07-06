"""store duel solutions per challenge

Revision ID: 0002_challenge_scoped_solutions
Revises: 0001_initial
Create Date: 2026-07-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_challenge_scoped_solutions"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "duel_task_solutions",
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("challenger_submission_id", sa.Text(), nullable=False),
        sa.Column("submission_id", sa.Text(), nullable=False),
        sa.Column("solution", sa.Text(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.task_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["challenger_submission_id"],
            ["challenges.challenger_submission_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"], ["submissions.submission_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint(
            "task_id", "challenger_submission_id", "submission_id"
        ),
    )
    op.create_index(
        "ix_duel_task_solutions_challenge",
        "duel_task_solutions",
        ["challenger_submission_id", "task_id"],
    )
    op.create_index(
        "ix_duel_task_solutions_submission_id",
        "duel_task_solutions",
        ["submission_id"],
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW v_active_unsolved_tasks AS
        SELECT t.*
        FROM challenges c
        JOIN tasks t
          ON c.king_id = t.king_id
         AND t.pool_type = c.status
         AND t.status_id = 1
        LEFT JOIN duel_task_solutions ks
          ON ks.task_id = t.task_id
         AND ks.challenger_submission_id = c.challenger_submission_id
         AND ks.submission_id = c.king_id
        LEFT JOIN duel_task_solutions cs
          ON cs.task_id = t.task_id
         AND cs.challenger_submission_id = c.challenger_submission_id
         AND cs.submission_id = c.challenger_submission_id
        WHERE c.status IN (1, 2)
          AND (ks.task_id IS NULL OR cs.task_id IS NULL)
        """
    )

    # Preserve already-saved judgements by copying their historical solution inputs
    # into the new challenge-scoped table. Unjudged active rounds intentionally are
    # not copied; they will be solved fresh after this migration.
    op.execute(
        """
        INSERT INTO duel_task_solutions (
            task_id,
            challenger_submission_id,
            submission_id,
            solution,
            duration,
            exit_reason
        )
        SELECT
            j.task_id,
            j.challenger_submission_id,
            j.king_submission_id,
            ts.solution,
            ts.duration,
            ts.exit_reason
        FROM judgements j
        JOIN task_solutions ts
          ON ts.task_id = j.task_id
         AND ts.submission_id = j.king_submission_id
        ON CONFLICT DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO duel_task_solutions (
            task_id,
            challenger_submission_id,
            submission_id,
            solution,
            duration,
            exit_reason
        )
        SELECT
            j.task_id,
            j.challenger_submission_id,
            j.challenger_submission_id,
            ts.solution,
            ts.duration,
            ts.exit_reason
        FROM judgements j
        JOIN task_solutions ts
          ON ts.task_id = j.task_id
         AND ts.submission_id = j.challenger_submission_id
        ON CONFLICT DO NOTHING
        """
    )

    op.drop_constraint(
        "fk_judgements_king_solution", "judgements", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_judgements_challenger_solution", "judgements", type_="foreignkey"
    )
    op.create_foreign_key(
        "fk_judgements_task",
        "judgements",
        "tasks",
        ["task_id"],
        ["task_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_judgements_king_submission",
        "judgements",
        "submissions",
        ["king_submission_id"],
        ["submission_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_judgements_challenge",
        "judgements",
        "challenges",
        ["challenger_submission_id"],
        ["challenger_submission_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_judgements_challenge", "judgements", type_="foreignkey")
    op.drop_constraint(
        "fk_judgements_king_submission", "judgements", type_="foreignkey"
    )
    op.drop_constraint("fk_judgements_task", "judgements", type_="foreignkey")

    # Rehydrate the legacy task-grain table enough for the old judgement FKs to hold.
    op.execute(
        """
        INSERT INTO task_solutions (
            task_id,
            submission_id,
            solution,
            duration,
            exit_reason
        )
        SELECT DISTINCT ON (task_id, submission_id)
            task_id,
            submission_id,
            solution,
            duration,
            exit_reason
        FROM duel_task_solutions
        ORDER BY task_id, submission_id
        ON CONFLICT DO NOTHING
        """
    )

    op.create_foreign_key(
        "fk_judgements_king_solution",
        "judgements",
        "task_solutions",
        ["task_id", "king_submission_id"],
        ["task_id", "submission_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_judgements_challenger_solution",
        "judgements",
        "task_solutions",
        ["task_id", "challenger_submission_id"],
        ["task_id", "submission_id"],
        ondelete="CASCADE",
    )

    op.execute(
        """
        CREATE OR REPLACE VIEW v_active_unsolved_tasks AS
        SELECT t.*
        FROM challenges c
        JOIN tasks t ON c.king_id = t.king_id
        RIGHT JOIN task_solutions ts ON t.task_id = ts.task_id
        WHERE c.status IN (1, 2)
          AND t.pool_type = c.status
          AND t.status_id = 1
        """
    )

    op.drop_table("duel_task_solutions")
