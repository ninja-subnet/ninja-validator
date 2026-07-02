"""Data objects for mined GitHub commits.

A ``CommitCandidate`` is the unit the task-generator turns into a task: the repo
coordinates the solver needs to rebuild a workspace plus the per-file diffs the
LLM describes. ``from_api`` is the single place that interprets the (untrusted,
dynamically-shaped) GitHub commit payload and rejects structurally unusable
commits, so the sampler never deals in raw JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .errors import CommitRejected, RejectReason


@dataclass(slots=True)
class CommitFile:
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CommitFile:
        return cls(
            filename=str(payload.get("filename") or ""),
            status=str(payload.get("status") or ""),
            additions=int(payload.get("additions") or 0),
            deletions=int(payload.get("deletions") or 0),
            changes=int(payload.get("changes") or 0),
            patch=payload.get("patch"),
        )


@dataclass(slots=True)
class CommitCandidate:
    repo_full_name: str
    repo_clone_url: str
    commit_sha: str
    parent_sha: str
    message: str
    html_url: str
    author_name: str | None
    event_id: str
    files: list[CommitFile]
    commit_tree_sha: str | None = None

    @property
    def combined_patch(self) -> str:
        """Reassemble a single unified diff from the per-file patches.

        This is the GitHub commits-API patch: cheap but lossy (it truncates or
        omits large-file diffs). The task-generator stores it as the task's
        ``reference_patch``; the solver, which clones the parent anyway, may
        later replace it with an exact local ``git diff``.
        """
        blocks: list[str] = []
        for item in self.files:
            if not item.patch:
                continue
            blocks.append(
                "\n".join(
                    [
                        f"diff --git a/{item.filename} b/{item.filename}",
                        f"--- a/{item.filename}",
                        f"+++ b/{item.filename}",
                        item.patch,
                    ],
                ),
            )
        return "\n".join(blocks).strip()

    @property
    def short_sha(self) -> str:
        return self.commit_sha[:12]

    @property
    def changed_files(self) -> list[str]:
        return [item.filename for item in self.files]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["combined_patch"] = self.combined_patch
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CommitCandidate:
        files = payload.get("files") or []
        return cls(
            repo_full_name=str(payload.get("repo_full_name") or ""),
            repo_clone_url=str(payload.get("repo_clone_url") or ""),
            commit_sha=str(payload.get("commit_sha") or ""),
            parent_sha=str(payload.get("parent_sha") or ""),
            message=str(payload.get("message") or ""),
            html_url=str(payload.get("html_url") or ""),
            author_name=payload.get("author_name"),
            event_id=str(payload.get("event_id") or ""),
            files=[CommitFile.from_dict(item) for item in files if isinstance(item, dict)],
            commit_tree_sha=payload.get("commit_tree_sha"),
        )

    @classmethod
    def from_api(
        cls,
        payload: Mapping[str, Any],
        *,
        repo_full_name: str,
        event_id: str = "",
    ) -> CommitCandidate:
        """Build a candidate from the GitHub commit API payload.

        ``repo_full_name`` is supplied by the caller — the single-commit endpoint
        does not echo it back. Raises :class:`CommitRejected` for commits that can
        never make a good task: no parent, a merge (multi-parent, non-linear diff),
        or no changed files.
        """
        parents = payload.get("parents") or []
        if not parents:
            raise CommitRejected("commit has no parent", reason=RejectReason.STRUCTURAL)
        if len(parents) > 1:
            raise CommitRejected("merge commit skipped", reason=RejectReason.STRUCTURAL)
        parent_sha = (parents[0] or {}).get("sha")
        commit_sha = payload.get("sha")
        if not commit_sha or not parent_sha:
            raise CommitRejected("commit payload missing sha / parent sha", reason=RejectReason.STRUCTURAL)

        files = [
            CommitFile.from_dict(item)
            for item in payload.get("files") or []
            if isinstance(item, dict)
        ]
        if not files:
            raise CommitRejected("commit had no changed files", reason=RejectReason.STRUCTURAL)

        commit = payload.get("commit") or {}
        return cls(
            repo_full_name=repo_full_name,
            repo_clone_url=f"https://github.com/{repo_full_name}.git",
            commit_sha=str(commit_sha),
            parent_sha=str(parent_sha),
            message=str(commit.get("message") or "").strip(),
            html_url=str(payload.get("html_url") or ""),
            author_name=(commit.get("author") or {}).get("name"),
            event_id=event_id,
            files=files,
            commit_tree_sha=(commit.get("tree") or {}).get("sha"),
        )
