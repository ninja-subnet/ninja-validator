"""Host-side task-repo checkout, fed into the sandbox.

A single-commit partial clone (no full history, no blobs until needed), checked out
detached at the task's *base* commit (``parent_sha`` — the state *before* the fix).
The clone uses the GitHub token; the ``.git`` remote/logs are then stripped so the
token never travels into the sandbox when the tree is copied in.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class CloneError(RuntimeError):
    """A git step failed while materializing a task repo."""


def _git(args: list[str], *, cwd: Path | None = None, timeout: int = 300) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise CloneError(f"git {' '.join(args)} failed: {output[-500:]}")
    return result.stdout


def _authed_url(repo_clone_url: str, token: str | None) -> str:
    """Inject the token into an https GitHub URL; pass other URLs through."""
    if token and repo_clone_url.startswith("https://"):
        rest = repo_clone_url[len("https://") :]
        # Don't double-inject if credentials are already present.
        if "@" not in rest.split("/", 1)[0]:
            return f"https://x-access-token:{token}@{rest}"
    return repo_clone_url


def _sanitize_git_metadata(repo_dir: Path) -> None:
    """Drop the token-bearing remote and the reflogs before the tree leaves the host."""
    git_dir = repo_dir / ".git"
    if not git_dir.is_dir():
        return
    # Remove the origin remote (its URL holds the token). Best-effort.
    subprocess.run(
        ["git", "remote", "remove", "origin"],
        cwd=str(repo_dir), capture_output=True, text=True, check=False,
    )
    shutil.rmtree(git_dir / "logs", ignore_errors=True)
    subprocess.run(
        ["git", "reflog", "expire", "--expire=now", "--all"],
        cwd=str(repo_dir), capture_output=True, text=True, check=False,
    )


def clone_task_repo(
    *,
    repo_clone_url: str,
    base_commit: str,
    token: str | None,
    dest: Path,
) -> Path:
    """Clone *repo_clone_url* and check out *base_commit* (detached) into *dest*.

    ``base_commit`` must be a full 40-char SHA (what the generator stores as
    ``parent_sha``); a short SHA cannot be fetched as a remote ref.
    """
    if not _FULL_SHA.match(base_commit):
        raise CloneError(f"base_commit must be a full 40-char SHA, got {base_commit!r}")
    dest.mkdir(parents=True, exist_ok=True)
    url = _authed_url(repo_clone_url, token)
    # Partial, no-checkout clone: cheapest way to land a single commit's tree.
    _git(["clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
    _git(["fetch", "--depth=1", "origin", base_commit], cwd=dest)
    _git(["checkout", "--detach", "FETCH_HEAD"], cwd=dest)
    _sanitize_git_metadata(dest)
    log.debug("cloned %s @ %s into %s", repo_clone_url, base_commit[:8], dest)
    return dest
