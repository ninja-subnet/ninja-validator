"""Seed the DB for a task-solver dry run against locally-extracted submissions.

Agents now come from a local submissions directory (one folder per submission, named
by submission id, each a bundle whose entry point is ``agent.py`` — the ninja shape).
This script discovers those folders, then inserts a reigning king (the first folder),
a challenge per remaining folder, and two CANDIDATE tasks pointing at a small local
sample repo (cloned via ``file://`` — no GitHub token needed).

Running the task-solver afterwards will:
  Phase A  qualify the two CANDIDATE tasks by running the KING bundle;
  Phase B  produce a challenger solution per (task, challenger) — a couple of duels.

    # point at your extracted submissions (folder names = submission ids)
    uv run python examples/task_solver/seed_duel.py \
        --submissions-dir /home/robert/repos/s66_repositories/submissions

    # …or scaffold two token-free sample bundles if you have none yet:
    uv run python examples/task_solver/seed_duel.py --scaffold

Idempotent: it clears its own task/challenge/king/submission rows for the discovered
ids first. Targets DATABASE_URL (or the POSTGRES_* vars), the same DB the worker uses.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from tau.db import TaskStatus
from tau.db.engine import create_db_engine, session_factory, session_scope
from tau.db.models import Challenge, King, Submission, Task

HERE = Path(__file__).resolve().parent
SAMPLE_REPO = HERE / ".sample_repos" / "calc"
POOL = 1  # pool_type; active challenges use status == POOL to match it
TASK_IDS = ("sample-task-1", "sample-task-2")
PROBLEM = (
    "The add() function in calc.py returns `a - b` instead of `a + b`. "
    "Fix it so add(2, 3) == 5."
)
# Named so the sorted order puts the king first (the first discovered folder).
SCAFFOLD_IDS = ("sample-1", "sample-2")


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout


def make_sample_repo(path: Path) -> tuple[str, str, str]:
    """Create a 2-commit repo. Returns (parent_sha, commit_sha, reference_patch)."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "sample@tau"], path)
    _git(["config", "user.name", "tau sample"], path)
    (path / "calc.py").write_text("def add(a, b):\n    return a - b  # BUG\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "calc: initial (buggy add)"], path)
    parent = _git(["rev-parse", "HEAD"], path).strip()
    (path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "calc: fix add"], path)
    commit = _git(["rev-parse", "HEAD"], path).strip()
    return parent, commit, _git(["diff", f"{parent}..{commit}"], path)


def scaffold_sample_bundles(submissions_dir: Path) -> None:
    """Write two token-free sample bundles (a noop agent.py) if they don't exist."""
    noop = (HERE / "agents" / "noop_agent.py").read_text(encoding="utf-8")
    for sub_id, marker in zip(SCAFFOLD_IDS, ("king", "challenger")):
        bundle = submissions_dir / sub_id
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "agent.py").write_text(noop.replace('MARKER = "agent"', f'MARKER = "{marker}"'))


def discover_submission_ids(submissions_dir: Path) -> list[str]:
    """Folder names under *submissions_dir* that look like agent bundles (have agent.py)."""
    if not submissions_dir.is_dir():
        return []
    return sorted(
        p.name for p in submissions_dir.iterdir() if p.is_dir() and (p / "agent.py").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions-dir", type=Path, default=Path("submissions"),
        help="root of extracted submissions (folder names = submission ids)",
    )
    parser.add_argument(
        "--scaffold", action="store_true",
        help="create two token-free sample bundles in --submissions-dir if missing",
    )
    args = parser.parse_args()
    submissions_dir = args.submissions_dir.resolve()

    if args.scaffold:
        scaffold_sample_bundles(submissions_dir)
    ids = discover_submission_ids(submissions_dir)
    if len(ids) < 2:
        raise SystemExit(
            f"need >=2 agent bundles in {submissions_dir} (found {ids}); "
            "pass --scaffold or --submissions-dir <path with submission folders>"
        )
    king_id, challenger_ids = ids[0], ids[1:]
    print(f"submissions_dir: {submissions_dir}\n  king={king_id}  challengers={challenger_ids}")

    parent_sha, commit_sha, reference_patch = make_sample_repo(SAMPLE_REPO)
    clone_url = f"file://{SAMPLE_REPO}"
    print(f"sample repo: {clone_url} (parent {parent_sha[:8]} -> fix {commit_sha[:8]})")

    engine = create_db_engine()
    with session_scope(session_factory(engine)) as s:
        for sub_id in ids:  # idempotent: cascades to king/challenges/tasks/solutions
            existing = s.get(Submission, sub_id)
            if existing is not None:
                s.delete(existing)
        s.flush()

        for sub_id in ids:
            s.add(Submission(submission_id=sub_id, block=1, hotkey=f"hk-{sub_id}"))
        s.flush()
        s.add(King(king_id=king_id))  # king_id IS the submission id; king_from defaults to now()
        s.flush()
        for chal in challenger_ids:
            s.add(Challenge(challenger_submission_id=chal, king_id=king_id, status=POOL))
        for i, task_id in enumerate(TASK_IDS):
            s.add(Task(
                task_id=task_id, king_id=king_id, pool_type=POOL,
                problem_statement=PROBLEM, status_id=int(TaskStatus.CANDIDATE),
                repo_clone_url=clone_url, parent_sha=parent_sha, commit_sha=commit_sha,
                reference_patch=reference_patch, content_fingerprint=f"sample-fp-{i}",
            ))
    engine.dispose()
    print(
        f"\nSeeded king={king_id}, challengers={challenger_ids}, tasks={list(TASK_IDS)} "
        f"(CANDIDATE, pool={POOL}).\nRun the task-solver with "
        f"TAU_SUBMISSIONS_DIR={submissions_dir}, then show_state.py."
    )


if __name__ == "__main__":
    main()
