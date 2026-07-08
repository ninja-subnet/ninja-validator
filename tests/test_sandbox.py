"""Unit tests for the sandbox helpers that need no Docker daemon.

The container lifecycle itself is exercised end-to-end (see the plan's verification),
but the pure logic — result parsing, the context tar, clone-URL auth injection, the
harness source, and the diff-line counter — is tested here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tau.sandbox.config import SandboxConfig
from tau.sandbox.harness import HARNESS_SCRIPT, RESULT_SENTINEL
from tau.sandbox.image import _build_context, _sortdir_source, image_tag
from tau.sandbox.network import _OWN_CONTAINER_ID
from tau.sandbox.repo import CloneError, _authed_url, _git
from tau.sandbox.runner import (
    _deterministic_agent_env,
    _parse_result,
    _prepare_workdir,
    _task_sampling_params,
    _task_seed,
)
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


def test_task_seed_is_stable_unique_and_json_safe() -> None:
    first = _task_seed("task-alpha")
    again = _task_seed("task-alpha")
    second = _task_seed("task-beta")

    assert first == again
    assert first != second
    assert 0 <= first < 2**53


def test_task_sampling_params_lock_validator_defaults() -> None:
    params = _task_sampling_params("task-alpha")

    assert params == {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": _task_seed("task-alpha"),
    }


def test_deterministic_agent_env_includes_sortdir_preload(monkeypatch) -> None:
    monkeypatch.delenv("TAU_DISABLE_SORTDIR", raising=False)

    env = _deterministic_agent_env()

    assert env["LD_PRELOAD"] == "/opt/tau/libsortdir.so"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["TZ"] == "UTC"
    assert env["HOME"] == "/tmp"
    assert env["TMPDIR"] == "/tmp"
    assert env["LANG"] == "C.UTF-8"
    assert env["LC_ALL"] == "C.UTF-8"


def test_deterministic_agent_env_can_disable_sortdir(monkeypatch) -> None:
    monkeypatch.setenv("TAU_DISABLE_SORTDIR", "1")

    assert "LD_PRELOAD" not in _deterministic_agent_env()


def test_sandbox_image_build_context_contains_sortdir() -> None:
    context = _build_context()

    with tarfile.open(fileobj=context, mode="r") as tar:
        names = sorted(tar.getnames())
        dockerfile = tar.extractfile("Dockerfile").read().decode("utf-8")
        sortdir = tar.extractfile("sortdir.c").read().decode("utf-8")

    assert names == ["Dockerfile", "sortdir.c"]
    assert "COPY sortdir.c /opt/tau/sortdir.c" in dockerfile
    assert "libsortdir.so" in dockerfile
    assert sortdir == _sortdir_source()
    assert "readdir64" in sortdir


def test_sandbox_image_tag_tracks_sortdir_source(monkeypatch) -> None:
    import tau.sandbox.image as image

    config = SandboxConfig(image_name="tau-test")
    monkeypatch.setattr(image, "_sortdir_source", lambda: "source one")
    first = image_tag(config)
    monkeypatch.setattr(image, "_sortdir_source", lambda: "source two")
    second = image_tag(config)

    assert first.startswith("tau-test:")
    assert len(first.removeprefix("tau-test:")) == 16
    assert first != second


def test_sortdir_shim_sorts_and_rewinds_when_compiled(tmp_path: Path) -> None:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc unavailable")

    lib = tmp_path / "libsortdir.so"
    subprocess.run(
        [
            gcc,
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(lib),
            str(Path(__file__).parents[1] / "src" / "tau" / "sandbox" / "sortdir.c"),
            "-ldl",
            "-lpthread",
        ],
        check=True,
    )

    entries = tmp_path / "entries"
    entries.mkdir()
    for name in ("z", "a", "m"):
        (entries / name).touch()

    helper = tmp_path / "check_sortdir.c"
    helper.write_text(
        r'''
#include <dirent.h>
#include <stdio.h>

static void print_rest(DIR *d) {
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        printf("%s ", e->d_name);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    DIR *d = opendir(argv[1]);
    if (d == NULL) return 1;

    struct dirent *first = readdir(d);
    printf("first=%s\n", first ? first->d_name : "NULL");

    long mark = telldir(d);
    struct dirent *second = readdir(d);
    printf("second=%s\n", second ? second->d_name : "NULL");

    seekdir(d, mark);
    struct dirent *again = readdir(d);
    printf("again=%s\n", again ? again->d_name : "NULL");

    rewinddir(d);
    printf("rewound=");
    print_rest(d);

    closedir(d);
    return 0;
}
''',
        encoding="utf-8",
    )
    helper_bin = tmp_path / "check_sortdir"
    subprocess.run([gcc, "-O2", "-o", str(helper_bin), str(helper)], check=True)

    output = subprocess.check_output(
        [str(helper_bin), str(entries)],
        env={**os.environ, "LD_PRELOAD": str(lib)},
        text=True,
    )

    assert output.splitlines() == [
        "first=.",
        "second=..",
        "again=..",
        "rewound=. .. a m z ",
    ]


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
