"""Data contracts and exit reasons for the sandbox-execution interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tau.proxy import SolveBudget, SolveUsageSummary

# Why a solve ended. The budget reasons mirror tau.proxy's so a budget trip flows
# straight through to the persisted task_solutions.exit_reason.
EXIT_COMPLETED = "completed"
EXIT_TIME_LIMIT = "time_limit_exceeded"
EXIT_NO_ACTIVITY = "no_activity_timeout"
EXIT_AGENT_ERROR = "agent_error"
EXIT_SANDBOX_ERROR = "sandbox_error"
EXIT_SANDBOX_VIOLATION = "sandbox_violation"
# The LLM upstream failed for reasons unrelated to the agent (unreachable, timeout, out
# of funds, rate limited, provider 5xx). The solve is not the miner's fault, so callers
# retry it rather than persisting a result. Detected via the proxy's upstream_error_count.
EXIT_UPSTREAM_ERROR = "upstream_error"


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    """One unit of sandboxed work: run the *agent_dir* bundle on *problem_statement*.

    *repo_dir* is an already-cloned, checked-out working tree on the host (see
    ``tau.sandbox.repo.clone_task_repo``). *agent_dir* is the miner submission bundle
    root — a directory whose entry point is ``agent.py`` (and which may contain a
    supporting package, e.g. ``agent/``), matching the validator's agent contract.
    Both are bind-mounted into the sandbox. The high-level knobs default from
    ``SandboxConfig`` / the container, so a caller usually passes only the first four.
    """

    task_id: str
    problem_statement: str
    repo_dir: Path
    agent_dir: Path
    model: str | None = None
    timeout_seconds: int | None = None
    budget: SolveBudget | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Outcome of one sandboxed solve."""

    success: bool
    solution_diff: str
    exit_reason: str
    elapsed_seconds: float
    usage: SolveUsageSummary | None = None
    error: str | None = None
