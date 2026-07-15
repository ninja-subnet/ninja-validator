"""The benchmark worker's poll loop.

Each tick lists every king and, for any king with no completion marker on disk,
runs its submitted agent against SWE-bench Pro by invoking the benchmark suite's
single-agent entry point (``run_agent_benchmark.py``). Results are archived into a
dedicated per-king folder and a ``benchmark.json`` marker records the outcome.

Restart safety is entirely the on-disk marker + the suite's own per-instance resume:
a king interrupted mid-benchmark has no ``done`` marker, so it is re-invoked next tick
and the suite continues from the instances it already finished (it skips instance ids
already in ``preds.json`` and eval results already produced) rather than restarting
from scratch. Detection matches every other worker — level-triggered polling of
``kings`` (no DB trigger), so a stale read self-heals on the next tick.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

from .config import AGENT_ENTRYPOINT, BenchmarkConfig
from .db import BenchmarkDb, KingRow

log = logging.getLogger(__name__)

# Small artifacts copied from the benchmark run into the dedicated per-king folder.
_ARCHIVE_FILES = (
    "summary.md",
    "cost_summary.json",
    "costs.jsonl",
    "eval_results.json",
    # Baseline comparison (suite pipeline step 6): honest, CI-aware report vs the
    # published Qwen3.6-27B score, with the comparability ledger.
    "comparison.md",
    "comparison.json",
)
_MARKER = "benchmark.json"


def run(*, db: BenchmarkDb, config: BenchmarkConfig, stop: threading.Event) -> None:
    """Run ticks until *stop* is set, sleeping ``poll_seconds`` between them."""
    while not stop.is_set():
        try:
            _tick(db=db, config=config, stop=stop)
        except Exception:  # noqa: BLE001 — one bad tick must not kill the worker
            log.exception("benchmark tick failed")
        stop.wait(config.poll_seconds)


def _tick(*, db: BenchmarkDb, config: BenchmarkConfig, stop: threading.Event) -> None:
    kings = db.all_kings()
    pending = [k for k in kings if not _is_done(config, k.king_id)]
    if not pending:
        log.debug("benchmark: %d king(s), all benchmarked", len(kings))
        return
    log.info("benchmark: %d king(s) pending of %d total", len(pending), len(kings))
    for king in pending:
        if stop.is_set():
            return
        _benchmark_king(config=config, king=king)


def _is_done(config: BenchmarkConfig, king_id: str) -> bool:
    marker = config.results_dir / king_id / _MARKER
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def _agent_dir(config: BenchmarkConfig, king_id: str) -> Path | None:
    """Resolve a king's submission bundle, or None if missing/invalid.

    A valid bundle is ``<submissions_dir>/<king_id>/`` containing ``agent.py``
    (same contract as the task-solver's ``_agent_dir``).
    """
    bundle = (config.submissions_dir / king_id).resolve()
    if not (bundle / AGENT_ENTRYPOINT).is_file():
        log.warning("king %s has no %s under %s; skipping", king_id, AGENT_ENTRYPOINT, config.submissions_dir)
        return None
    return bundle


def _benchmark_king(*, config: BenchmarkConfig, king: KingRow) -> None:
    king_id = king.king_id
    agent_dir = _agent_dir(config, king_id)
    if agent_dir is None:
        return
    runner = config.bench_repo_dir / config.runner_script
    if not config.bench_venv_python.is_file():
        log.error("benchmark suite venv not found at %s; cannot benchmark king %s",
                  config.bench_venv_python, king_id)
        return
    if not runner.is_file():
        log.error("benchmark runner %s not found; cannot benchmark king %s", runner, king_id)
        return
    if not config.openrouter_api_key:
        log.error("OPENROUTER_API_KEY not set; cannot benchmark king %s", king_id)
        return

    # Delegate to the suite's single-agent entry point. run_name = king_id so the suite
    # archives to <bench_repo>/results/<king_id>/, which we then copy into our dedicated
    # folder. Sampling is pinned EXPLICITLY per flag (highest precedence in the suite:
    # agent-config `sampling:` < BENCH_* env < CLI) so every king runs the same
    # baseline-comparable config regardless of this process's environment.
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = config.openrouter_api_key
    # The suite's scripts re-exec each other via their repo .venv by default;
    # BENCH_PYTHON pins them to the same interpreter we invoke (essential in the
    # containerized worker, where the repo's host-built .venv is unusable).
    env["BENCH_PYTHON"] = str(config.bench_venv_python)
    cmd = [
        str(config.bench_venv_python), config.runner_script,
        "--agent-dir", str(agent_dir),
        "--name", king_id,
        "--model", config.model,
        "--slice", config.slice_spec,
        "--workers", str(config.agent_workers),
        "--temperature", str(config.temperature),
        "--top-p", str(config.top_p),
    ]
    if config.sampling_json:
        cmd += ["--sampling-json", config.sampling_json]
    log.info("benchmarking king %s (agent=%s, model=%s, slice=%s)",
             king_id, agent_dir, config.model, config.slice_spec or "full")

    try:
        subprocess.run(cmd, cwd=str(config.bench_repo_dir), env=env,
                       timeout=config.bench_timeout_seconds, check=True)
    except subprocess.TimeoutExpired:
        log.warning("king %s benchmark exceeded %ds; will resume next tick",
                    king_id, config.bench_timeout_seconds)
        return
    except subprocess.CalledProcessError as exc:
        log.error("king %s benchmark failed (exit %s); will retry next tick", king_id, exc.returncode)
        return

    _archive_and_mark(config=config, king=king, agent_dir=agent_dir,
                      dest_dir=config.results_dir / king_id)


def _archive_and_mark(*, config: BenchmarkConfig, king: KingRow, agent_dir: Path, dest_dir: Path) -> None:
    """Copy the small summary artifacts into the dedicated folder + write the marker."""
    src = config.bench_repo_dir / "results" / king.king_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in _ARCHIVE_FILES:
        src_file, dest_file = src / name, dest_dir / name
        if src_file.is_file() and src_file.resolve() != dest_file.resolve():
            shutil.copy2(src_file, dest_file)

    summary: dict = {}
    if (dest_dir / "cost_summary.json").is_file():
        try:
            summary = json.loads((dest_dir / "cost_summary.json").read_text())
        except (json.JSONDecodeError, OSError):
            summary = {}
    comparison: dict = {}
    if (dest_dir / "comparison.json").is_file():
        try:
            comparison = json.loads((dest_dir / "comparison.json").read_text())
        except (json.JSONDecodeError, OSError):
            comparison = {}

    marker = {
        "king_id": king.king_id,
        "king_from": king.king_from.isoformat() if isinstance(king.king_from, dt.datetime) else str(king.king_from),
        "run_name": king.king_id,
        "agent_dir": str(agent_dir),
        "model": config.model,
        "sampling": {"temperature": config.temperature, "top_p": config.top_p,
                     "sampling_json": config.sampling_json or None},
        "slice": config.slice_spec,
        "instances": summary.get("instances"),
        "resolved": summary.get("resolved"),
        "resolve_rate": summary.get("resolve_rate"),
        "total_cost_usd": summary.get("total_cost_usd"),
        # Headline of the suite's baseline comparison (comparison.json), if produced.
        "resolve_rate_ci95_pct": comparison.get("resolve_rate_ci95_pct"),
        "baseline": comparison.get("baseline"),
        "baseline_verdict": comparison.get("verdict"),
        "status": "done",
    }
    (dest_dir / _MARKER).write_text(json.dumps(marker, indent=2))
    log.info("king %s benchmarked: resolved=%s rate=%s cost=$%s -> %s",
             king.king_id, marker["resolved"], marker["resolve_rate"],
             marker["total_cost_usd"], dest_dir)
