"""The in-container harness: load the agent, run it, emit a result line.

A slim reimplementation of the legacy harness (the old subprocess-monkeypatching
event capture is dropped — usage/rollouts are observed at the proxy instead). It
runs as ``python3 /work/harness.py`` inside the sandbox and:

  1. imports the agent file (must define ``solve(repo_path, issue, model, api_base,
     api_key) -> dict`` returning at least ``success``);
  2. calls it with the proxy's base URL + per-solve token;
  3. computes the real ``git diff`` of the working tree (tracked + untracked) — the
     authoritative patch, since an agent may edit files without returning a patch;
  4. prints exactly one ``__TAU_RESULT__{...json...}`` line on stdout.

The runner scans stdout for that sentinel, so arbitrary agent chatter is ignored.
"""

from __future__ import annotations

# Container-side layout (all under the writable /work tmpfs).
CONTAINER_WORK = "/work"
CONTAINER_REPO = "/work/repo"
CONTAINER_AGENT = "/work/agent/agent.py"
CONTAINER_PROMPT = "/work/task.txt"
CONTAINER_HARNESS = "/work/harness.py"

# Sentinel prefixing the harness's single JSON result line.
RESULT_SENTINEL = "__TAU_RESULT__"

HARNESS_SCRIPT = '''\
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

RESULT_SENTINEL = "__TAU_RESULT__"


def _load_agent(path):
    agent_dir = str(path.parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("submitted_agent", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import agent file: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_agent"] = module
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise RuntimeError(
            "Agent file must define solve(repo_path, issue, model, api_base, api_key)"
        )
    return solve


def _git(args, cwd):
    return subprocess.run(
        ["git", "-c", "safe.directory=*", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60, check=False,
    )


def _repo_diff(repo_dir):
    diff = _git(["diff", "--binary", "--", "."], repo_dir).stdout or ""
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"], repo_dir).stdout or ""
    for rel in [item for item in untracked.split("\\0") if item]:
        file_diff = _git(["diff", "--binary", "--no-index", "--", "/dev/null", rel], repo_dir)
        if file_diff.returncode in (0, 1):
            diff += file_diff.stdout or ""
    return diff


def _required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError("%s is required by the sandbox harness" % name)
    return value


def main():
    try:
        agent_file = Path(_required_env("TAU_AGENT_FILE"))
        repo_dir = Path(_required_env("TAU_REPO_DIR"))
        issue = Path(_required_env("TAU_PROMPT_FILE")).read_text(encoding="utf-8")
        model = _required_env("AGENT_MODEL")
        api_base = _required_env("OPENAI_BASE_URL")
        api_key = _required_env("OPENAI_API_KEY")
        solve = _load_agent(agent_file)
        result = solve(
            repo_path=str(repo_dir), issue=issue, model=model,
            api_base=api_base, api_key=api_key,
        )
        if not isinstance(result, dict):
            raise RuntimeError("solve() must return a dict, got %s" % type(result).__name__)
        # The git diff is authoritative; fall back to a patch the agent returned.
        diff = _repo_diff(repo_dir)
        if not diff.strip() and isinstance(result.get("patch"), str):
            diff = result["patch"]
        payload = {"ok": True, "success": bool(result.get("success")), "patch": diff}
    except Exception:
        payload = {"ok": False, "success": False, "patch": "", "error": traceback.format_exc()}
    sys.stdout.write(RESULT_SENTINEL + json.dumps(payload) + "\\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
'''
