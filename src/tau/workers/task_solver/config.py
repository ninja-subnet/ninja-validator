"""Configuration for the task-solver worker.

Loop knobs live here; the heavier collaborators (sandbox limits, upstream selection,
per-solve budget) are the existing config objects from ``tau.sandbox`` / ``tau.proxy``,
composed in ``from_env`` so the worker has one place to build everything.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tau.pools import PoolTargets
from tau.proxy import SolveBudget, UpstreamTarget
from tau.sandbox import SandboxConfig
from tau.utils.env import env_bool, env_float, env_int, env_str


def _task_repo_cache_dir(env: Mapping[str, str]) -> Path | None:
    if "TAU_TASK_REPO_CACHE_DIR" in env:
        raw = env_str(env, "TAU_TASK_REPO_CACHE_DIR", "")
        return Path(raw) if raw else None
    work_root = env_str(env, "TAU_SANDBOX_WORK_ROOT", "")
    return Path(work_root) / "task-repo-cache" if work_root else None


def _github_token(env: Mapping[str, str]) -> str | None:
    """First token from ``GITHUB_TOKENS`` (comma-separated) or ``GITHUB_TOKEN``."""
    raw = env.get("GITHUB_TOKENS", "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if tokens:
        return tokens[0]
    single = env.get("GITHUB_TOKEN")
    return single.strip() if single and single.strip() else None


@dataclass(frozen=True, slots=True)
class SolverConfig:
    upstream: UpstreamTarget
    sandbox: SandboxConfig
    # Root holding the extracted miner submissions; each agent bundle lives in a
    # subfolder named by its submission id. The worker resolves an agent as
    # ``submissions_dir / submission_id`` (entry: ``agent.py``).
    submissions_dir: Path = Path("submissions")
    budget: SolveBudget | None = None
    github_token: str | None = None
    # Host-side cache of checked-out task repos keyed by repo URL + base commit.
    # Cuts the GitHub clone/fetch burst when both duel sides solve the same task
    # and when later challengers reuse the active task pool.
    task_repo_cache_dir: Path | None = None
    # Fresh cache misses still hit GitHub; cap those remote fetches while allowing
    # already-cached local copies to fan out freely.
    task_repo_fetch_concurrency: int = 8
    # LRU cap on cached checkouts (0 = unbounded). Sized to hold an active king's
    # full task pool plus the previous pool while it drains.
    task_repo_cache_max_entries: int = 256
    # Max sandboxes launched per loop tick (across both phases). Sequential.
    max_containers: int = 4
    poll_seconds: float = 30.0
    # When a tick fills ``max_containers``, sleep this many seconds before the
    # next tick instead of ``poll_seconds`` so duel solve backlogs drain faster.
    backlog_poll_seconds: float = 1.0
    # The king must produce at least this many changed diff lines to QUALIFY a task.
    qualify_min_changed_lines: int = 1
    # When true, duel solves wait until the active pool reaches its target of
    # QUALIFIED tasks.
    require_full_pool_for_duels: bool = False
    pool_targets: PoolTargets = field(default_factory=PoolTargets)

    def __post_init__(self) -> None:
        if self.max_containers < 1:
            raise ValueError("max_containers must be >= 1")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.backlog_poll_seconds <= 0:
            raise ValueError("backlog_poll_seconds must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SolverConfig:
        env = os.environ if environ is None else environ
        d = cls(upstream=UpstreamTarget.from_env(env), sandbox=SandboxConfig.from_env(env))
        return cls(
            upstream=d.upstream,
            sandbox=d.sandbox,
            submissions_dir=Path(env_str(env, "TAU_SUBMISSIONS_DIR", str(d.submissions_dir))),
            budget=SolveBudget.from_env(env),
            github_token=_github_token(env),
            task_repo_cache_dir=_task_repo_cache_dir(env),
            task_repo_fetch_concurrency=env_int(
                env,
                "TAU_TASK_REPO_FETCH_CONCURRENCY",
                d.task_repo_fetch_concurrency,
            ),
            task_repo_cache_max_entries=env_int(
                env,
                "TAU_TASK_REPO_CACHE_MAX_ENTRIES",
                d.task_repo_cache_max_entries,
            ),
            max_containers=env_int(env, "MAX_CONTAINERS", d.max_containers),
            poll_seconds=env_float(env, "TAU_SOLVER_POLL_SECONDS", d.poll_seconds),
            backlog_poll_seconds=env_float(
                env,
                "TAU_SOLVER_BACKLOG_POLL_SECONDS",
                d.backlog_poll_seconds,
            ),
            qualify_min_changed_lines=env_int(
                env, "TAU_SOLVER_QUALIFY_MIN_CHANGED_LINES", d.qualify_min_changed_lines
            ),
            require_full_pool_for_duels=env_bool(
                env,
                "TAU_SOLVER_REQUIRE_FULL_POOL_FOR_DUELS",
                d.require_full_pool_for_duels,
            ),
            pool_targets=PoolTargets.from_env(env),
        )
