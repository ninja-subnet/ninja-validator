#!/usr/bin/env python
"""Generate one task (cached) and run the task-solver sandbox on it, end to end.

A standalone harness for testing the solver on a SINGLE task -- no database, no
king/challenge, no migration, no worker loop. It only uses the DB-free solve seam
(``tau.sandbox.run_agent_in_container``) plus the task-generator's pure
description path. Steps:

  1. load a cached task, or generate one (sample a GitHub commit -> describe it
     with the LLM) and cache it to JSON so re-runs skip GitHub + the LLM;
  2. clone the task repo at its parent commit (the state before the fix);
  3. run your agent bundle in the hardened sandbox;
  4. print the result (success, exit_reason, token usage, diff).

Edit the consts below (at least ``AGENT_DIR``), then:

    uv run python scripts/solve_one.py             # reuse cached task, or generate one
    uv run python scripts/solve_one.py --regen     # ignore the cache; generate fresh
    uv run python scripts/solve_one.py --task-only # generate (or show cached) task; no solve
    uv run python scripts/solve_one.py --trace     # stream the agent's LLM calls live + save them

With --trace, every LLM call the agent makes is printed to stdout as it happens and
appended to TRACE_FILE (one JSON event per call) -- the agent's full trajectory,
observed at the proxy, with no agent-side logging.

Needs (in .env): OPENROUTER_API_KEY, GITHUB_TOKEN(S), a running Docker daemon, and
SOLVER_MODEL for the solve. Set SOLVER_MAX_COST / SOLVER_MAX_TOTAL_TOKENS to cap spend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO

import docker
import dotenv

from tau.github import CommitSampler, GitHubClient, GitHubConfig, GitHubTokenRotator
from tau.openrouter import OpenRouterClient
from tau.openrouter.client import DEFAULT_MODEL
from tau.proxy import LLMProxy, SolveBudget, UpstreamTarget
from tau.sandbox import (
    AgentRunRequest,
    AgentRunResult,
    SandboxConfig,
    clone_task_repo,
    ensure_sandbox_image,
    run_agent_in_container,
)
from tau.sandbox import runner as sandbox_runner
from tau.taskgen import generate_task_description
from tau.workers.task_solver.config import SolverConfig

# --- configuration (edit these) ----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Your agent bundle: a directory whose entry point is ``agent.py`` defining
#   solve(repo_path, issue, model, api_base, api_key) -> dict   (returns {"success": bool}).
AGENT_DIR = ROOT / "submissions/5GTEST0000000000-2582c482a563785f"

# Where the generated task is cached so re-runs skip the GitHub + LLM calls.
TASK_CACHE = ROOT / ".cache" / "task_02.json"

# Where --trace writes the agent's full LLM trajectory (one JSON event per call).
TRACE_FILE = ROOT / ".cache" / "trace.jsonl"

# Env var naming the model that DESCRIBES the mined commit into a task. The SOLVE
# model is separate -- the proxy enforces SandboxConfig.model (env SOLVER_MODEL).
GENERATOR_MODEL_ENV = "TAU_GENERATOR_MODEL"
GENERATOR_LLM_TIMEOUT_SECONDS = 120
MAX_SAMPLE_ATTEMPTS = 25

log = logging.getLogger("solve_one")

# --trace runtime state. The proxy's event sink fires on a proxy server thread, so the
# shared counter + file handle are guarded by a lock.
_TRACE_LOCK = threading.Lock()
_trace_fh: TextIO | None = None
_trace_count = 0


# --- entry point --------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    dotenv.load_dotenv(ROOT / ".env")
    _configure_logging()

    if args.task_only:
        # Generate (or load) and view the task only -- no agent, no solve.
        _view_task(_get_or_generate_task(regen=args.regen))
        return

    _check_agent_dir()
    if args.trace:
        _enable_tracing()
    try:
        task = _get_or_generate_task(regen=args.regen)
        result = _solve(task)
        _report(task, result)
    finally:
        _close_trace()


# --- task: load from cache or generate one ------------------------------------------
def _get_or_generate_task(*, regen: bool) -> dict:
    """Return a task dict, from the cache when present (and not ``--regen``)."""
    if TASK_CACHE.is_file() and not regen:
        log.info("using cached task: %s", TASK_CACHE)
        return json.loads(TASK_CACHE.read_text(encoding="utf-8"))

    log.info(
        "generating a task via the task-generator path (sample commit -> describe)"
    )
    task = asyncio.run(_generate_task())
    TASK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TASK_CACHE.write_text(json.dumps(task, indent=2, sort_keys=True), encoding="utf-8")
    log.info("cached generated task -> %s", TASK_CACHE)
    return task


async def _generate_task() -> dict:
    """Sample one GitHub commit and describe it into a task (no DB writes)."""
    github_config = GitHubConfig.from_env()
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get(GENERATOR_MODEL_ENV) or DEFAULT_MODEL

    async with GitHubClient.create(
        token_rotator=GitHubTokenRotator.from_env(), timeout=github_config.http_timeout
    ) as client:
        sampler = CommitSampler(
            rng=random.Random(), client=client, config=github_config
        )
        log.info("sampling a commit (up to %d attempts)...", MAX_SAMPLE_ATTEMPTS)
        candidate = (
            await sampler.sample_commit(max_attempts=MAX_SAMPLE_ATTEMPTS)
        ).candidate
        log.info("sampled %s@%s", candidate.repo_full_name, candidate.commit_sha[:8])

        async with OpenRouterClient(
            api_key, model=model, timeout=GENERATOR_LLM_TIMEOUT_SECONDS
        ) as llm:
            log.info("describing the commit into a task with %s...", model)
            generated = await generate_task_description(candidate=candidate, client=llm)

    log.info("generated task: %r", generated.title)
    return {
        "task_id": candidate.commit_sha[:16],
        "repo_full_name": candidate.repo_full_name,
        "repo_clone_url": candidate.repo_clone_url,
        "parent_sha": candidate.parent_sha,
        "commit_sha": candidate.commit_sha,
        "problem_statement": generated.prompt_text,
    }


# --- solve: run the agent in the sandbox --------------------------------------------
def _solve(task: dict) -> AgentRunResult:
    """Clone the repo and run one sandboxed solve of ``AGENT_DIR`` against the task."""
    client = docker.from_env()
    config = SandboxConfig.from_env()
    upstream = UpstreamTarget.from_env()
    budget = SolveBudget.from_env()

    log.info(
        "ensuring sandbox image (provider=%s, solve model=%s)...",
        upstream.name,
        config.model,
    )
    image_tag = ensure_sandbox_image(client, config)
    try:
        with TemporaryDirectory(prefix="solve-one-") as tmp:
            log.info(
                "cloning %s @ %s...", task["repo_clone_url"], task["parent_sha"][:8]
            )
            solver_cfg = SolverConfig.from_env()
            repo_dir = clone_task_repo(
                repo_clone_url=task["repo_clone_url"],
                base_commit=task["parent_sha"],
                token=solver_cfg.github_token,
                dest=Path(tmp) / "repo",
                cache_dir=solver_cfg.task_repo_cache_dir,
                fetch_concurrency=solver_cfg.task_repo_fetch_concurrency,
                cache_max_entries=solver_cfg.task_repo_cache_max_entries,
            )
            log.info("running agent %r in the sandbox...", AGENT_DIR.name)
            return run_agent_in_container(
                AgentRunRequest(
                    task_id=task["task_id"],
                    problem_statement=task["problem_statement"],
                    repo_dir=repo_dir,
                    agent_dir=AGENT_DIR,
                    budget=budget,
                ),
                client=client,
                config=config,
                upstream=upstream,
                image_tag=image_tag,
            )
    finally:
        client.close()


# --- tracing (--trace): stream + save the agent's LLM trajectory --------------------
def _enable_tracing() -> None:
    """Patch the sandbox proxy so every agent LLM call is streamed + saved.

    The agent reaches the model only through the per-solve proxy, which can emit a
    captured ``llm_call`` event per request. ``runner.py`` builds that proxy without a
    sink, so we swap the name it constructs (``runner.LLMProxy``) for a wrapper that
    wires our sink in. No agent-side logging needed -- the proxy sees everything.
    """
    global _trace_fh, _trace_count
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _trace_fh = TRACE_FILE.open("w", encoding="utf-8")
    _trace_count = 0

    def _traced_proxy(*args: Any, **kwargs: Any) -> LLMProxy:
        kwargs.setdefault("rollout_event_sink", _on_llm_call)
        kwargs.setdefault(
            "rollout_capture_bodies", True
        )  # keep request + response bodies
        return LLMProxy(*args, **kwargs)

    sandbox_runner.LLMProxy = _traced_proxy
    log.info("tracing on: streaming steps to stdout, saving events -> %s", TRACE_FILE)


def _close_trace() -> None:
    global _trace_fh
    if _trace_fh is not None:
        _trace_fh.close()
        _trace_fh = None
        log.info("trace saved -> %s (%d call(s))", TRACE_FILE, _trace_count)


def _on_llm_call(event: dict[str, Any]) -> None:
    """Proxy event sink (runs on a proxy server thread): save + print one LLM call."""
    global _trace_count
    with _TRACE_LOCK:
        _trace_count += 1
        if _trace_fh is not None:
            _trace_fh.write(json.dumps(event) + "\n")
            _trace_fh.flush()
        _print_step(_trace_count, event)


def _print_step(n: int, event: dict[str, Any]) -> None:
    """Print one agent step: the model's reply (its reasoning + any tool calls)."""
    message = ((event.get("response") or {}).get("choices") or [{}])[0].get(
        "message"
    ) or {}
    content = (message.get("content") or "").strip()
    tool_calls = message.get("tool_calls") or []
    usage = event.get("usage") or {}
    print(
        f"\n--- step {n}  {event.get('latency_ms')}ms  "
        f"{usage.get('total_tokens') or '?'} tok  ${event.get('cost') or 0:.4f} ---",
        flush=True,
    )
    if content:
        print(content, flush=True)
    for call in tool_calls:
        fn = call.get("function") or {}
        print(f"  > tool {fn.get('name')}: {fn.get('arguments', '')}", flush=True)
    if not content and not tool_calls:
        print("  (no response body captured -- streamed response?)", flush=True)


# --- helpers ------------------------------------------------------------------------
def _github_token() -> str | None:
    """First token from ``GITHUB_TOKENS`` (comma-separated) or ``GITHUB_TOKEN``."""
    tokens = [
        t.strip() for t in os.environ.get("GITHUB_TOKENS", "").split(",") if t.strip()
    ]
    if tokens:
        return tokens[0]
    single = os.environ.get("GITHUB_TOKEN")
    return single.strip() if single and single.strip() else None


def _check_agent_dir() -> None:
    entry = AGENT_DIR / "agent.py"
    if not entry.is_file():
        raise SystemExit(
            f"agent bundle not found -- set AGENT_DIR to a directory containing agent.py "
            f"(looked for {entry})"
        )


def _report(task: dict, result: AgentRunResult) -> None:
    diff = result.solution_diff or ""
    print("\n=== solve result ===")
    print(f"task:        {task['task_id']}  ({task['repo_full_name']})")
    print(f"success:     {result.success}")
    print(f"exit_reason: {result.exit_reason}")
    print(f"elapsed:     {result.elapsed_seconds:.1f}s")
    if result.usage is not None:
        u = result.usage
        print(
            f"llm:         {u.request_count} request(s), {u.total_tokens} tokens, cost {u.cost}"
        )
        if u.budget_exceeded_reason:
            print(f"budget:      tripped ({u.budget_exceeded_reason})")
    if result.error:
        print(f"error:       {result.error}")
    print(f"diff:        {_changed_lines(diff)} changed line(s), {len(diff)} chars")
    print("\n--- diff (first 100 lines) ---")
    print("\n".join(diff.splitlines()[:100]) or "(empty)")


def _view_task(task: dict[str, Any]) -> None:
    """Print a generated/cached task in a human-readable form (no solve)."""
    print("\n=== task ===")
    print(f"task_id:    {task['task_id']}")
    print(f"repo:       {task['repo_full_name']}")
    print(f"clone url:  {task['repo_clone_url']}")
    print(f"parent sha: {task['parent_sha']}")
    print(f"commit sha: {task['commit_sha']}")
    print(f"cached at:  {TASK_CACHE}")
    print("\n--- problem statement ---")
    print(task["problem_statement"])


def _changed_lines(diff: str) -> int:
    return sum(
        1
        for ln in diff.splitlines()
        if (ln.startswith("+") and not ln.startswith("+++"))
        or (ln.startswith("-") and not ln.startswith("---"))
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore", "urllib3", "docker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one task and solve it in the sandbox (no DB)."
    )
    parser.add_argument(
        "--regen", action="store_true", help="ignore the cache; generate a fresh task"
    )
    parser.add_argument(
        "--task-only",
        action="store_true",
        help="generate (or show cached) the task and print it; do not run the solver",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="stream the agent's LLM calls to stdout in real time and save them to TRACE_FILE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
