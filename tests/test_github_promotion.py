from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from tau.github.promotion import GitHubPromotionPublisher, PromotionPublishConfig

BASE = "a" * 40
TREE = "b" * 40
NEW_TREE = "c" * 40
NEW_COMMIT = "d" * 40


class FakeGitHubClient:
    def __init__(self, *, new_tree: str = NEW_TREE) -> None:
        self.new_tree = new_tree
        self.posts: list[tuple[str, Any]] = []
        self.patches: list[tuple[str, Any]] = []

    async def get_json(self, path: str, **params: Any) -> Any:
        if path == "/repos/octo/published/branches/main":
            return {"commit": {"sha": BASE}}
        if path == f"/repos/octo/published/git/commits/{BASE}":
            return {"tree": {"sha": TREE}}
        if path == "/repos/octo/published/contents/tau_agent_files.json":
            assert params == {"ref": BASE}
            content = base64.b64encode(
                json.dumps(["agent.py", "old.py"]).encode()
            ).decode()
            return {"encoding": "base64", "content": content}
        raise AssertionError(f"unexpected GET {path}")

    async def post_json(self, path: str, payload: Any) -> Any:
        self.posts.append((path, payload))
        if path == "/repos/octo/published/git/trees":
            return {"sha": self.new_tree}
        if path == "/repos/octo/published/git/commits":
            return {"sha": NEW_COMMIT}
        raise AssertionError(f"unexpected POST {path}")

    async def patch_json(self, path: str, payload: Any) -> Any:
        self.patches.append((path, payload))
        return {"sha": NEW_COMMIT}


async def test_publisher_commits_submission_bundle_and_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "submissions" / "winner"
    bundle.mkdir(parents=True)
    (bundle / "agent.py").write_text(
        "def solve(repo_path, issue, model, api_base, api_key):\n    return {}\n"
    )
    (bundle / "helper.py").write_text("VALUE = 1\n")
    client = FakeGitHubClient()
    publisher = GitHubPromotionPublisher(
        client,
        PromotionPublishConfig(
            submissions_dir=tmp_path / "submissions", repo="octo/published"
        ),
    )

    published = await publisher.publish_submission("winner")

    assert published.commit_sha == NEW_COMMIT
    tree_entries = client.posts[0][1]["tree"]
    paths = {entry["path"] for entry in tree_entries}
    assert {"agent.py", "helper.py", "tau_agent_files.json", "old.py"} <= paths
    assert next(entry for entry in tree_entries if entry["path"] == "old.py")[
        "sha"
    ] is None
    assert client.patches == [
        (
            "/repos/octo/published/git/refs/heads/main",
            {"sha": NEW_COMMIT, "force": False},
        )
    ]


async def test_publisher_skips_commit_when_submission_tree_is_unchanged(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "submissions" / "winner"
    bundle.mkdir(parents=True)
    (bundle / "agent.py").write_text(
        "def solve(repo_path, issue, model, api_base, api_key):\n    return {}\n"
    )
    client = FakeGitHubClient(new_tree=TREE)
    publisher = GitHubPromotionPublisher(
        client,
        PromotionPublishConfig(
            submissions_dir=tmp_path / "submissions", repo="octo/published"
        ),
    )

    published = await publisher.publish_submission("winner")

    assert published.commit_sha == BASE
    assert [path for path, _ in client.posts] == [
        "/repos/octo/published/git/trees"
    ]
    assert client.patches == []
