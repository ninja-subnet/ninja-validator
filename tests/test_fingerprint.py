"""Unit tests for the commit content fingerprint (pure, no IO)."""

from __future__ import annotations

from tau.github import CommitCandidate, CommitFile
from tau.taskgen import content_fingerprint


def _candidate(
    *,
    repo: str = "octo/repo",
    commit: str = "a" * 40,
    parent: str = "b" * 40,
    patch: str = "@@ -1 +1 @@\n-old\n+new",
) -> CommitCandidate:
    return CommitCandidate(
        repo_full_name=repo,
        repo_clone_url=f"https://github.com/{repo}.git",
        commit_sha=commit,
        parent_sha=parent,
        message="msg",
        html_url="",
        author_name="dev",
        event_id="",
        files=[CommitFile("a.py", "modified", 1, 1, 2, patch)],
    )


def test_fingerprint_is_deterministic() -> None:
    assert content_fingerprint(_candidate()) == content_fingerprint(_candidate())


def test_fingerprint_changes_with_commit_sha() -> None:
    assert content_fingerprint(_candidate(commit="a" * 40)) != content_fingerprint(
        _candidate(commit="c" * 40)
    )


def test_fingerprint_changes_with_patch() -> None:
    assert content_fingerprint(_candidate(patch="@@\n-x\n+y")) != content_fingerprint(
        _candidate(patch="@@\n-p\n+q")
    )


def test_fingerprint_independent_of_repo() -> None:
    # Same commit mined from different forks must dedupe to one task.
    assert content_fingerprint(_candidate(repo="octo/repo")) == content_fingerprint(
        _candidate(repo="forker/copy")
    )


def test_fingerprint_normalizes_sha_whitespace() -> None:
    assert content_fingerprint(_candidate(commit=" " + "a" * 40 + " ")) == content_fingerprint(
        _candidate(commit="a" * 40)
    )
