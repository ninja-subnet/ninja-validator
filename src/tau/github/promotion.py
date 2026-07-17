"""Publish a promoted local submission bundle to a GitHub branch."""

from __future__ import annotations

import ast
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .client import GitHubClient
from .errors import GitHubRequestError

AGENT_ENTRYPOINT = "agent.py"
AGENT_JSON = "agent.json"
AGENT_MANIFEST = "tau_agent_files.json"


class PromotionPublishError(RuntimeError):
    """Raised when a promoted submission could not be published."""


@dataclass(frozen=True, slots=True)
class PromotionPublishConfig:
    submissions_dir: Path
    repo: str
    branch: str = "main"

    def __post_init__(self) -> None:
        if not self.repo.strip():
            raise ValueError("repo is required")
        if not self.branch.strip():
            raise ValueError("branch is required")


@dataclass(frozen=True, slots=True)
class PublishedPromotion:
    submission_id: str
    repo: str
    branch: str
    commit_sha: str
    file_count: int


class GitHubPromotionPublisher:
    """Publish a local winner bundle as a commit on the configured GitHub branch."""

    def __init__(self, client: GitHubClient, config: PromotionPublishConfig) -> None:
        self._client = client
        self._config = config

    async def publish_submission(self, submission_id: str) -> PublishedPromotion:
        files = load_submission_files(self._config.submissions_dir, submission_id)
        error = validate_agent_py(files[AGENT_ENTRYPOINT])
        if error is not None:
            raise PromotionPublishError(
                f"submission {submission_id} {AGENT_ENTRYPOINT} is invalid: {error}"
            )

        base_head_sha = await fetch_branch_head_sha(
            self._client, repo=self._config.repo, branch=self._config.branch
        )
        commit_sha = await publish_agent_files_commit(
            self._client,
            repo=self._config.repo,
            branch=self._config.branch,
            base_head_sha=base_head_sha,
            files=files,
            message=promotion_commit_message(
                submission_id=submission_id,
                file_count=len(files),
                base_head_sha=base_head_sha,
            ),
        )
        return PublishedPromotion(
            submission_id=submission_id,
            repo=self._config.repo,
            branch=self._config.branch,
            commit_sha=commit_sha,
            file_count=len(files),
        )


def load_submission_files(submissions_dir: Path, submission_id: str) -> dict[str, str]:
    bundle = submissions_dir / submission_id
    if not bundle.is_dir():
        raise PromotionPublishError(f"submission bundle is missing: {bundle}")
    files = _load_agent_json_files(bundle) or _scan_python_files(bundle)
    if AGENT_ENTRYPOINT not in files:
        raise PromotionPublishError(
            f"submission bundle has no {AGENT_ENTRYPOINT}: {bundle}"
        )
    return files


def validate_agent_py(text: str) -> str | None:
    if not text.strip():
        return "empty file"
    if _has_unresolved_conflict_markers(text):
        return "unresolved conflict markers"
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"syntax error at line {exc.lineno}: {exc.msg}"
    has_solve = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "solve"
        for node in tree.body
    )
    if not has_solve:
        return "missing top-level solve function"
    return None


async def fetch_branch_head_sha(
    client: GitHubClient, *, repo: str, branch: str
) -> str:
    payload = await client.get_json(f"/repos/{repo}/branches/{quote(branch, safe='')}")
    commit = payload.get("commit") if isinstance(payload, dict) else None
    sha = str(commit.get("sha") or "") if isinstance(commit, dict) else ""
    if not _is_sha(sha):
        raise PromotionPublishError(f"GitHub branch {repo}:{branch} has no commit sha")
    return sha.lower()


async def publish_agent_files_commit(
    client: GitHubClient,
    *,
    repo: str,
    branch: str,
    base_head_sha: str,
    files: dict[str, str],
    message: str,
) -> str:
    if not _is_sha(base_head_sha):
        raise PromotionPublishError(f"invalid base head sha: {base_head_sha!r}")

    commit_payload = await client.get_json(f"/repos/{repo}/git/commits/{base_head_sha}")
    tree = commit_payload.get("tree") if isinstance(commit_payload, dict) else None
    base_tree_sha = str(tree.get("sha") or "") if isinstance(tree, dict) else ""
    if not _is_sha(base_tree_sha):
        raise PromotionPublishError("GitHub commit fetch did not include a base tree sha")

    publish_files = dict(files)
    publish_files[AGENT_MANIFEST] = json.dumps(sorted(files), indent=2) + "\n"
    tree_entries: list[dict[str, Any]] = [
        {"path": path, "mode": "100644", "type": "blob", "content": content}
        for path, content in sorted(publish_files.items())
    ]
    for path in await _fetch_previous_manifest_paths(client, repo=repo, ref=base_head_sha):
        if path not in publish_files:
            tree_entries.append({"path": path, "mode": "100644", "sha": None})

    tree_payload = await client.post_json(
        f"/repos/{repo}/git/trees",
        {"base_tree": base_tree_sha, "tree": tree_entries},
    )
    new_tree_sha = _payload_sha(tree_payload, "GitHub tree create")
    if new_tree_sha == base_tree_sha:
        return base_head_sha
    new_commit_payload = await client.post_json(
        f"/repos/{repo}/git/commits",
        {"message": message, "tree": new_tree_sha, "parents": [base_head_sha]},
    )
    new_commit_sha = _payload_sha(new_commit_payload, "GitHub commit create")
    await client.patch_json(
        f"/repos/{repo}/git/refs/heads/{quote(branch, safe='/')}",
        {"sha": new_commit_sha, "force": False},
    )
    return new_commit_sha


def promotion_commit_message(
    *, submission_id: str, file_count: int, base_head_sha: str
) -> str:
    return (
        f"Promote submission {submission_id} as king\n\n"
        f"Winning submission id: {submission_id}\n"
        f"Published file count: {file_count}\n"
        f"Base head before publication: {base_head_sha}"
    )


async def _fetch_previous_manifest_paths(
    client: GitHubClient, *, repo: str, ref: str
) -> list[str]:
    try:
        payload = await client.get_json(
            f"/repos/{repo}/contents/{quote(AGENT_MANIFEST, safe='/')}",
            ref=ref,
        )
    except GitHubRequestError:
        return []
    if not isinstance(payload, dict):
        return []
    encoded = str(payload.get("content") or "")
    if str(payload.get("encoding") or "").lower() != "base64":
        return []
    try:
        content = base64.b64decode(encoded.encode("ascii"), validate=False).decode(
            "utf-8"
        )
        paths = json.loads(content)
    except Exception:
        return []
    if not isinstance(paths, list):
        return []
    return sorted({str(path).strip() for path in paths if str(path).strip()})


def _load_agent_json_files(bundle: Path) -> dict[str, str] | None:
    path = bundle / AGENT_JSON
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw_files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(raw_files, dict):
        return None
    files = {
        str(name): str(content)
        for name, content in sorted(raw_files.items(), key=lambda item: str(item[0]))
        if str(name).endswith(".py")
    }
    return files or None


def _scan_python_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for file_path in sorted(root.rglob("*.py")):
        if file_path.is_symlink() or not file_path.is_file():
            continue
        relative = file_path.relative_to(root)
        if any(part == "__pycache__" or part.startswith(".") for part in relative.parts):
            continue
        files[relative.as_posix()] = file_path.read_text(
            encoding="utf-8", errors="replace"
        )
    return files


def _has_unresolved_conflict_markers(text: str) -> bool:
    saw_start = False
    saw_separator = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<<<<<<<"):
            saw_start = True
        elif saw_start and stripped.startswith("======="):
            saw_separator = True
        elif saw_start and saw_separator and stripped.startswith(">>>>>>>"):
            return True
    return False


def _payload_sha(payload: Any, context: str) -> str:
    sha = str(payload.get("sha") or "") if isinstance(payload, dict) else ""
    if not _is_sha(sha):
        raise PromotionPublishError(f"{context} did not return a commit sha")
    return sha.lower()


def _is_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None
