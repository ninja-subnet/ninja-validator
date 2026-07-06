"""Unit tests for the sandbox helpers that need no Docker daemon.

The container lifecycle itself is exercised end-to-end (see the plan's verification),
but the pure logic — result parsing, the context tar, clone-URL auth injection, the
harness source, and the diff-line counter — is tested here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tau.sandbox.harness import HARNESS_SCRIPT, RESULT_SENTINEL
from tau.sandbox.network import _OWN_CONTAINER_ID
from tau.sandbox.repo import CloneError, _authed_url, _git
from tau.sandbox.runner import _parse_result, _prepare_workdir
from tau.sandbox.types import AgentRunRequest
from tau.workers.task_solver.loop import _changed_lines


def test_parse_result_reads_sentinel_line_amid_noise() -> None:
    output = (
        b"agent chatter\n"
        b'{"not": "the result"}\n'
        + RESULT_SENTINEL.encode()
        + b'{"ok": true, "success": true, "patch": "diff"}\n'
    )
    payload = _parse_result(output)
    assert payload == {"ok": True, "success": True, "patch": "diff"}


def test_parse_result_handles_missing_and_garbage() -> None:
    assert _parse_result(b"") is None
    assert _parse_result(b"no sentinel here") is None
    assert _parse_result((RESULT_SENTINEL + "{bad json").encode()) is None


def test_authed_url_injects_token_for_https_github() -> None:
    assert (
        _authed_url("https://github.com/octo/repo.git", "ght")
        == "https://x-access-token:ght@github.com/octo/repo.git"
    )


def test_authed_url_passes_through_without_token_or_credentials() -> None:
    # No token -> unchanged.
    assert _authed_url("https://github.com/octo/repo.git", None) == (
        "https://github.com/octo/repo.git"
    )
    # Already has credentials -> not double-injected.
    url = "https://user:pw@github.com/octo/repo.git"
    assert _authed_url(url, "ght") == url


def test_git_timeout_is_clone_error_and_redacts_auth(monkeypatch) -> None:
    def timeout(*args, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("tau.sandbox.repo.subprocess.run", timeout)
    with pytest.raises(CloneError) as exc:
        _git(
            [
                "clone",
                "https://x-access-token:supersecret@github.com/octo/repo.git",
                "/tmp/repo",
            ],
            timeout=1,
        )

    message = str(exc.value)
    assert "timed out after 1 seconds" in message
    assert "supersecret" not in message
    assert "https://<redacted>@github.com/octo/repo.git" in message


def test_prepare_workdir_lays_out_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n")
    # A multi-file agent bundle: agent.py entry + an agent/ package (the ninja shape).
    bundle = tmp_path / "submission"
    (bundle / "agent").mkdir(parents=True)
    (bundle / "agent.py").write_text("def solve(**kw):\n    return {'success': True}\n")
    (bundle / "agent" / "__init__.py").write_text("")
    req = AgentRunRequest(
        task_id="t1", problem_statement="fix the bug", repo_dir=repo, agent_dir=bundle
    )

    workdir = _prepare_workdir(req)
    try:
        assert (workdir / "repo" / "main.py").is_file()
        assert (workdir / "agent" / "agent.py").is_file()  # bundle entry (-> /work/agent/agent.py)
        assert (workdir / "agent" / "agent" / "__init__.py").is_file()  # package preserved
        assert (workdir / "task.txt").read_text() == "fix the bug"
        assert (workdir / "harness.py").is_file()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_prepare_workdir_preserves_repo_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("VALUE = 1\n")
    (repo / "linked.py").symlink_to("target.py")
    (repo / "dangling.py").symlink_to("missing.py")

    bundle = tmp_path / "submission"
    bundle.mkdir()
    (bundle / "agent.py").write_text("def solve(**kw):\n    return {'success': True}\n")
    req = AgentRunRequest(
        task_id="t1", problem_statement="fix the bug", repo_dir=repo, agent_dir=bundle
    )

    workdir = _prepare_workdir(req)
    try:
        copied_link = workdir / "repo" / "linked.py"
        copied_dangling = workdir / "repo" / "dangling.py"

        assert copied_link.is_symlink()
        assert copied_link.readlink() == Path("target.py")
        assert copied_dangling.is_symlink()
        assert copied_dangling.readlink() == Path("missing.py")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_harness_script_is_valid_python() -> None:
    compile(HARNESS_SCRIPT, "harness.py", "exec")


def test_own_container_id_regex_matches_only_containers_path() -> None:
    # The own container id is the source of the /etc/hostname bind mount.
    cid = "a" * 64
    line = f"600 500 0:50 / /etc/hostname ... /var/lib/docker/containers/{cid}/hostname"
    assert _OWN_CONTAINER_ID.search(line).group(1) == cid
    # A bare overlay-layer hash (NOT under /containers/) must not match — this is the
    # host-host false-positive the regex was tightened to avoid.
    overlay = f"/var/lib/docker/overlay2/{'b' * 64}/merged"
    assert _OWN_CONTAINER_ID.search(overlay) is None


def test_changed_lines_counts_additions_and_removals_only() -> None:
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
        " unchanged\n"
        "+brand new\n"
    )
    # Two '+' (excluding +++) and one '-' (excluding ---) = 3.
    assert _changed_lines(diff) == 3
    assert _changed_lines("") == 0
