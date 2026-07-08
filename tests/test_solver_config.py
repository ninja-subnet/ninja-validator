from pathlib import Path

import pytest

from tau.workers.task_solver.config import SolverConfig, _task_repo_cache_dir


def test_task_repo_cache_defaults_under_sandbox_work_root() -> None:
    assert _task_repo_cache_dir({"TAU_SANDBOX_WORK_ROOT": "/var/lib/tau/work"}) == Path(
        "/var/lib/tau/work/task-repo-cache"
    )


def test_task_repo_cache_explicit_override_and_disable() -> None:
    assert _task_repo_cache_dir({"TAU_TASK_REPO_CACHE_DIR": "/cache/repos"}) == Path(
        "/cache/repos"
    )
    assert _task_repo_cache_dir({"TAU_TASK_REPO_CACHE_DIR": ""}) is None


def test_task_repo_fetch_concurrency_default() -> None:
    assert SolverConfig(
        upstream=None,  # type: ignore[arg-type]
        sandbox=None,  # type: ignore[arg-type]
    ).task_repo_fetch_concurrency == 8


def test_solver_config_reads_backlog_poll_seconds() -> None:
    cfg = SolverConfig.from_env(
        {
            "TAU_SOLVER_BACKLOG_POLL_SECONDS": "2.5",
            "OPENROUTER_API_KEY": "k",
            "LLM_PROVIDER": "custom",
            "LLM_UPSTREAM_BASE_URL": "http://127.0.0.1:8000/v1",
            "SOLVER_MODEL": "test/model",
        }
    )
    assert cfg.backlog_poll_seconds == 2.5


def test_solver_config_rejects_non_positive_poll_seconds() -> None:
    with pytest.raises(ValueError, match="poll_seconds must be positive"):
        SolverConfig(
            upstream=None,  # type: ignore[arg-type]
            sandbox=None,  # type: ignore[arg-type]
            poll_seconds=0,
        )


def test_solver_config_rejects_non_positive_backlog_poll_seconds() -> None:
    with pytest.raises(ValueError, match="backlog_poll_seconds must be positive"):
        SolverConfig(
            upstream=None,  # type: ignore[arg-type]
            sandbox=None,  # type: ignore[arg-type]
            backlog_poll_seconds=0,
        )
