"""Print the task-solver's effect on the sample rows: task status, solutions, judgements.

    uv run python examples/task_solver/show_state.py
"""

from __future__ import annotations

from sqlalchemy import select

from tau.db import TaskStatus
from tau.db.engine import create_db_engine, session_factory, session_scope
from tau.db.models import Judgment, Task, TaskSolution


def main() -> None:
    engine = create_db_engine()
    with session_scope(session_factory(engine)) as s:
        print("== tasks (sample-*) ==")
        for t in s.scalars(
            select(Task).where(Task.task_id.like("sample-%")).order_by(Task.task_id)
        ):
            print(f"  {t.task_id:16} status={TaskStatus(t.status_id).name:12} pool={t.pool_type}")

        print("\n== task_solutions (sample-*) ==")
        rows = s.scalars(
            select(TaskSolution).where(TaskSolution.task_id.like("sample-%"))
            .order_by(TaskSolution.task_id, TaskSolution.submission_id)
        ).all()
        if not rows:
            print("  (none yet)")
        for r in rows:
            diff_len = len(r.solution or "")
            print(f"  {r.task_id:16} by {r.submission_id:16} exit={r.exit_reason!s:22} "
                  f"diff_bytes={diff_len}")

        print("\n== judgements (sample-*) ==")
        jrows = s.scalars(
            select(Judgment).where(Judgment.task_id.like("sample-%"))
        ).all()
        if not jrows:
            print("  (none yet — run the judge-worker to populate)")
        for j in jrows:
            print(f"  {j.task_id:16} king={j.king_submission_id} vs "
                  f"chal={j.challenger_submission_id} winner={j.llm_winner}")
    engine.dispose()


if __name__ == "__main__":
    main()
