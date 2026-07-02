"""Token-free sample agent for dry-running the task-solver.

Deterministic: it makes a small edit (so a diff is produced) WITHOUT calling the LLM,
so you can exercise the full qualification + challenger-solve pipeline at zero token
cost and without an upstream. ``seed_duel.py`` replaces ``MARKER`` per submission so
the king and each challenger produce *different* diffs (otherwise every duel ties).

Agent contract: ``solve(repo_path, issue, model, api_base, api_key) -> dict`` that
returns at least ``{"success": bool}``. The harness collects the git diff itself.
"""

MARKER = "agent"


def solve(repo_path, issue, model, api_base, api_key):  # noqa: ANN001, ANN201
    import pathlib

    first_line = (issue.strip().splitlines() or [""])[0][:120]
    out = pathlib.Path(repo_path) / "SOLUTION.md"
    out.write_text(f"# Solution by {MARKER}\n\nAddressed: {first_line}\n", encoding="utf-8")
    return {"success": True, "patch": ""}
