from __future__ import annotations

import asyncio
import random
from collections.abc import Callable

import httpx
import pytest

from tau.github import (
    CommitCandidate,
    CommitFile,
    CommitRejected,
    CommitSampleError,
    CommitSampler,
    GitHubClient,
    GitHubConfig,
    GitHubRequestError,
    GitHubTokenRotator,
)

_CONFIG = GitHubConfig()


# -- fixtures --------------------------------------------------------------------


def _file(
    filename: str,
    *,
    status: str = "modified",
    additions: int = 0,
    deletions: int = 0,
    patch: str | None = "@@ -1 +1 @@\n-old\n+new",
) -> CommitFile:
    return CommitFile(
        filename=filename,
        status=status,
        additions=additions,
        deletions=deletions,
        changes=additions + deletions,
        patch=patch,
    )


def _candidate(files: list[CommitFile]) -> CommitCandidate:
    return CommitCandidate(
        repo_full_name="octo/repo",
        repo_clone_url="https://github.com/octo/repo.git",
        commit_sha="a" * 40,
        parent_sha="b" * 40,
        message="fix things",
        html_url="https://github.com/octo/repo/commit/aaa",
        author_name="dev",
        event_id="",
        files=files,
    )


# A well-formed GitHub commit API payload: single parent, one modified code file
# with > 100 changed lines -> passes the quality gate.
GOOD_COMMIT = {
    "sha": "a" * 40,
    "parents": [{"sha": "b" * 40}],
    "html_url": "https://github.com/octo/repo/commit/aaa",
    "commit": {"message": "fix things", "author": {"name": "dev"}, "tree": {"sha": "c" * 40}},
    "files": [
        {
            "filename": "src/app.py",
            "status": "modified",
            "additions": 80,
            "deletions": 40,
            "changes": 120,
            "patch": "@@ -1 +1 @@\n-old\n+new",
        }
    ],
}

SEARCH_HIT = {"items": [{"repository": {"full_name": "octo/repo"}, "sha": "a" * 40}]}


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> GitHubClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url="https://api.github.com", transport=transport)
    return GitHubClient(http=http, **kwargs)


# -- CommitCandidate parsing / combined_patch -----------------------------------


def test_from_dict_parses_recorded_payload() -> None:
    payload = {
        "repo_full_name": "octo/repo",
        "repo_clone_url": "https://github.com/octo/repo.git",
        "commit_sha": "c" * 40,
        "parent_sha": "d" * 40,
        "message": "do the thing",
        "html_url": "https://github.com/octo/repo/commit/ccc",
        "author_name": "dev",
        "event_id": "evt-1",
        "commit_tree_sha": "e" * 40,
        "files": [
            {
                "filename": "src/app.py",
                "status": "modified",
                "additions": 60,
                "deletions": 50,
                "changes": 110,
                "patch": "@@ -1 +1 @@\n-old\n+new",
            },
            "not-a-dict-should-be-skipped",
        ],
    }
    candidate = CommitCandidate.from_dict(payload)
    assert candidate.repo_full_name == "octo/repo"
    assert candidate.commit_sha == "c" * 40
    assert candidate.parent_sha == "d" * 40
    assert candidate.short_sha == "c" * 12
    assert candidate.changed_files == ["src/app.py"]
    assert len(candidate.files) == 1  # the non-dict entry is dropped


def test_commit_file_from_dict_defaults() -> None:
    cf = CommitFile.from_dict({"filename": "x.py"})
    assert cf.status == ""
    assert cf.additions == 0 and cf.deletions == 0 and cf.changes == 0
    assert cf.patch is None


def test_combined_patch_builds_diff_blocks_and_skips_empty_patches() -> None:
    candidate = _candidate(
        [
            _file("src/app.py", patch="@@ -1 +1 @@\n-old\n+new"),
            _file("docs/readme.md", patch=None),  # no patch -> skipped
        ]
    )
    combined = candidate.combined_patch
    assert "diff --git a/src/app.py b/src/app.py" in combined
    assert "--- a/src/app.py" in combined
    assert "+++ b/src/app.py" in combined
    assert "docs/readme.md" not in combined


def test_to_dict_includes_combined_patch() -> None:
    candidate = _candidate([_file("src/app.py", additions=60, deletions=50)])
    data = candidate.to_dict()
    assert data["combined_patch"] == candidate.combined_patch
    assert data["repo_full_name"] == "octo/repo"


# -- CommitCandidate.from_api ----------------------------------------------------


def test_from_api_builds_candidate() -> None:
    candidate = CommitCandidate.from_api(GOOD_COMMIT, repo_full_name="octo/repo", event_id="e1")
    assert candidate.repo_clone_url == "https://github.com/octo/repo.git"
    assert candidate.commit_sha == "a" * 40
    assert candidate.parent_sha == "b" * 40
    assert candidate.event_id == "e1"
    assert candidate.commit_tree_sha == "c" * 40
    assert candidate.changed_files == ["src/app.py"]


def test_from_api_rejects_no_parent() -> None:
    with pytest.raises(CommitRejected):
        CommitCandidate.from_api({**GOOD_COMMIT, "parents": []}, repo_full_name="octo/repo")


def test_from_api_rejects_merge_commit() -> None:
    payload = {**GOOD_COMMIT, "parents": [{"sha": "1"}, {"sha": "2"}]}
    with pytest.raises(CommitRejected):
        CommitCandidate.from_api(payload, repo_full_name="octo/repo")


def test_from_api_rejects_no_files() -> None:
    with pytest.raises(CommitRejected):
        CommitCandidate.from_api({**GOOD_COMMIT, "files": []}, repo_full_name="octo/repo")


# -- quality gate ----------------------------------------------------------------


def test_quality_check_accepts_modified_code_commit() -> None:
    candidate = _candidate([_file("src/app.py", status="modified", additions=80, deletions=40)])
    assert CommitSampler._quality_check(candidate, _CONFIG) is None


def test_quality_check_rejects_when_no_code_files() -> None:
    candidate = _candidate([_file("README.md", status="modified", additions=200, deletions=0)])
    reason = CommitSampler._quality_check(candidate, _CONFIG)
    assert reason is not None and "code file" in reason


def test_quality_check_rejects_lockfiles_as_non_code() -> None:
    candidate = _candidate([_file("poetry.lock", status="modified", additions=500, deletions=0)])
    assert CommitSampler._quality_check(candidate, _CONFIG) is not None


def test_quality_check_rejects_too_few_changed_lines() -> None:
    candidate = _candidate([_file("src/app.py", status="modified", additions=40, deletions=10)])
    reason = CommitSampler._quality_check(candidate, _CONFIG)
    assert reason is not None and "code lines changed" in reason


def test_quality_check_rejects_when_no_modified_files() -> None:
    candidate = _candidate([_file("src/new.py", status="added", additions=150, deletions=0)])
    reason = CommitSampler._quality_check(candidate, _CONFIG)
    assert reason is not None and "modified code files" in reason


# -- GitHubClient (transport) ----------------------------------------------------


def test_client_get_json_returns_payload() -> None:
    client = _client(lambda req: httpx.Response(200, json={"ok": True}))
    assert asyncio.run(client.get_json("/anything")) == {"ok": True}


def test_client_cools_token_on_401_then_raises() -> None:
    rotator = GitHubTokenRotator(["t1", "t2"])
    client = _client(
        lambda req: httpx.Response(401, json={"message": "bad credentials"}),
        token_rotator=rotator,
    )
    with pytest.raises(GitHubRequestError):
        asyncio.run(client.get_json("/x"))
    # The 401'd token is cooled (transient), NOT permanently disabled: one remains.
    assert rotator.available_count == 1


def test_client_cools_token_on_rate_limit() -> None:
    rotator = GitHubTokenRotator(["t1", "t2"])
    client = _client(
        lambda req: httpx.Response(
            403, headers={"retry-after": "30"}, json={"message": "rate limit exceeded"}
        ),
        token_rotator=rotator,
    )
    with pytest.raises(GitHubRequestError):
        asyncio.run(client.get_json("/x"))
    assert rotator.available_count == 1  # the used token is now cooling down


def test_client_raises_request_error_on_server_error() -> None:
    client = _client(lambda req: httpx.Response(500, json={}))
    with pytest.raises(GitHubRequestError):
        asyncio.run(client.get_json("/x"))


def test_client_wraps_invalid_json_as_request_error() -> None:
    # A 200 with a non-JSON body must surface as GitHubRequestError, not a raw
    # json.JSONDecodeError that escapes the abstraction.
    client = _client(lambda req: httpx.Response(200, content=b"<html>not json</html>"))
    with pytest.raises(GitHubRequestError):
        asyncio.run(client.get_json("/x"))


# -- CommitSampler (policy) over a mock transport --------------------------------


def test_sample_commit_happy_path_via_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/commits":
            return httpx.Response(200, json=SEARCH_HIT)
        if path.startswith("/repos/") and "/commits/" in path:
            return httpx.Response(200, json=GOOD_COMMIT)
        return httpx.Response(404, json={})

    sampler = CommitSampler(rng=random.Random(0), client=_client(handler))
    candidate = asyncio.run(sampler.sample_commit()).candidate
    assert candidate.repo_full_name == "octo/repo"
    assert candidate.commit_sha == "a" * 40
    assert candidate.repo_clone_url == "https://github.com/octo/repo.git"
    assert candidate.combined_patch  # reference_patch source is non-empty


def test_sample_commit_exhausts_budget_on_merge_commits() -> None:
    merge_commit = {**GOOD_COMMIT, "parents": [{"sha": "1"}, {"sha": "2"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/commits":
            return httpx.Response(200, json=SEARCH_HIT)
        if path == "/events":
            return httpx.Response(200, json=[])
        if path.startswith("/repos/"):
            return httpx.Response(200, json=merge_commit)
        return httpx.Response(404, json={})

    sampler = CommitSampler(rng=random.Random(0), client=_client(handler))
    with pytest.raises(CommitSampleError) as excinfo:
        asyncio.run(sampler.sample_commit(max_attempts=3))
    # The give-up carries the per-reason tally for the exhausted round.
    assert sum(excinfo.value.rejections.values()) >= 1


def test_search_backs_off_when_all_results_already_rejected() -> None:
    # If a search page returns only commits already in the reject cache, the
    # refill must set a cooldown rather than re-searching on the next attempt.
    calls = {"search": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/commits":
            calls["search"] += 1
            return httpx.Response(200, json=SEARCH_HIT)
        return httpx.Response(404, json={})

    sampler = CommitSampler(rng=random.Random(0), client=_client(handler))
    sampler._reject_add("octo/repo", "a" * 40, "seen before")

    assert asyncio.run(sampler._next_from_search()) is None  # rejected hit filtered out
    assert calls["search"] == 1
    assert asyncio.run(sampler._next_from_search()) is None  # within cooldown -> no new search
    assert calls["search"] == 1  # search was NOT hammered


def test_sample_commit_falls_back_to_firehose_when_search_empty() -> None:
    event = {
        "type": "PushEvent",
        "id": "evt-1",
        "repo": {"name": "octo/repo"},
        "payload": {"commits": [{"sha": "a" * 40, "modified": ["src/app.py"]}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/commits":
            return httpx.Response(200, json={"items": []})  # search yields nothing
        if path == "/events":
            return httpx.Response(200, json=[event])
        if path.startswith("/repos/"):
            return httpx.Response(200, json=GOOD_COMMIT)
        return httpx.Response(404, json={})

    sampler = CommitSampler(rng=random.Random(0), client=_client(handler))
    candidate = asyncio.run(sampler.sample_commit()).candidate
    assert candidate.repo_full_name == "octo/repo"
    assert candidate.event_id == "evt-1"


# -- token rotator ---------------------------------------------------------------


def test_token_rotator_round_robins() -> None:
    rotator = GitHubTokenRotator(["t1", "t2", "t3"])
    assert [asyncio.run(rotator.get_token()) for _ in range(4)] == ["t1", "t2", "t3", "t1"]


def test_token_rotator_cools_unauthorized_token() -> None:
    rotator = GitHubTokenRotator(["t1", "t2"])
    rotator.mark_unauthorized("t1")  # transient cooldown, not a permanent disable
    assert rotator.available_count == 1
    assert {asyncio.run(rotator.get_token()) for _ in range(4)} == {"t2"}


def test_token_rotator_waits_then_returns_when_all_cooling() -> None:
    rotator = GitHubTokenRotator(["t1"])
    rotator.mark_rate_limited("t1", cooldown_seconds=0.01)
    # The only token is cooling -> get_token sleeps until it frees, then returns it
    # (rather than ever handing back an empty/anonymous token).
    assert asyncio.run(rotator.get_token()) == "t1"


def test_token_rotator_from_env_reads_tokens(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKENS", "a, b ,c")
    rotator = GitHubTokenRotator.from_env()
    assert rotator is not None and rotator.size == 3


def test_token_rotator_from_env_falls_back_to_single(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKENS", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "solo")
    rotator = GitHubTokenRotator.from_env()
    assert rotator is not None and rotator.size == 1


def test_token_rotator_from_env_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKENS", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert GitHubTokenRotator.from_env() is None


# -- config ----------------------------------------------------------------------


def test_config_defaults_match_tuned_values() -> None:
    config = GitHubConfig()
    assert config.min_code_changed_lines == 100
    assert config.min_code_files == 1
    assert config.history_sample_min_days == 30
    assert config.history_sample_max_days == 1095
    assert config.buffer_low_watermark == 50


def test_config_from_env_overrides_from_explicit_mapping() -> None:
    config = GitHubConfig.from_env(
        {"TAU_GITHUB_MIN_CODE_CHANGED_LINES": "250", "TAU_GITHUB_HTTP_TIMEOUT": "12.5"}
    )
    assert config.min_code_changed_lines == 250
    assert config.http_timeout == 12.5
    assert config.min_code_files == 1  # unset keys keep defaults
    assert config.history_sample_max_days == 1095


def test_config_from_env_ignores_unparseable_values() -> None:
    config = GitHubConfig.from_env({"TAU_GITHUB_MIN_CODE_FILES": "not-an-int"})
    assert config.min_code_files == GitHubConfig().min_code_files


def test_config_rejects_inverted_day_window() -> None:
    with pytest.raises(ValueError):
        GitHubConfig(history_sample_min_days=100, history_sample_max_days=10)


def test_config_rejects_inverted_watermarks() -> None:
    with pytest.raises(ValueError):
        GitHubConfig(buffer_low_watermark=500, buffer_high_watermark=50)
