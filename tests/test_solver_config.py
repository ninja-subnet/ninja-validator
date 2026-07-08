from pathlib import Path

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
