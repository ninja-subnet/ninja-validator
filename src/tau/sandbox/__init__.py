"""Reusable sandboxed agent execution.

Run one agent on one task inside a hardened, internet-less container that reaches an
LLM only through ``tau.proxy``. The same seam serves the task-solver's qualification
and challenger-solve phases (and any future caller):

    from tau.sandbox import (
        AgentRunRequest, SandboxConfig, clone_task_repo,
        ensure_sandbox_image, run_agent_in_container,
    )
    from tau.proxy import UpstreamTarget

    client = docker.from_env()
    config = SandboxConfig.from_env()
    upstream = UpstreamTarget.from_env()
    tag = ensure_sandbox_image(client, config)
    repo = clone_task_repo(repo_clone_url=..., base_commit=..., token=..., dest=...)
    result = run_agent_in_container(
        AgentRunRequest(task_id, problem_statement, repo, agent_py),
        client=client, config=config, upstream=upstream, image_tag=tag,
    )
"""

from __future__ import annotations

from .config import SandboxConfig
from .image import ensure_sandbox_image, image_tag
from .repo import CloneError, clone_task_repo
from .runner import run_agent_in_container
from .types import (
    EXIT_AGENT_ERROR,
    EXIT_COMPLETED,
    EXIT_NO_ACTIVITY,
    EXIT_SANDBOX_ERROR,
    EXIT_SANDBOX_VIOLATION,
    EXIT_TIME_LIMIT,
    EXIT_UPSTREAM_ERROR,
    AgentRunRequest,
    AgentRunResult,
)

__all__ = [
    "SandboxConfig",
    "AgentRunRequest",
    "AgentRunResult",
    "run_agent_in_container",
    "ensure_sandbox_image",
    "image_tag",
    "clone_task_repo",
    "CloneError",
    "EXIT_COMPLETED",
    "EXIT_TIME_LIMIT",
    "EXIT_NO_ACTIVITY",
    "EXIT_AGENT_ERROR",
    "EXIT_SANDBOX_ERROR",
    "EXIT_SANDBOX_VIOLATION",
    "EXIT_UPSTREAM_ERROR",
]
