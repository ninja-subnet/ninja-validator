"""Configuration for the benchmark worker.

The worker watches the ``kings`` table and, for each king not yet benchmarked,
runs the king's submitted agent against SWE-bench Pro by shelling out to the
benchmark suite's single-agent entry point (``run_agent_benchmark.py`` in the
``ninja-benchmark--swe-bench-controller-2`` repo). That script generates the per-run
config, pins sampling, runs the full pipeline (agent -> official Scale eval -> cost
report), and archives ``results/<name>/``; this config just tells the worker where
the suite lives and what predefined parameters to benchmark each king with.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tau.utils.env import env_float, env_int, env_str

# Entry file every submission bundle must expose (the validator agent contract).
AGENT_ENTRYPOINT = "agent.py"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    # Root holding extracted miner submissions; a king's bundle is
    # ``submissions_dir / king_id`` (king_id IS the submission id), entry ``agent.py``.
    submissions_dir: Path = Path("submissions")
    # Dedicated folder where this worker saves per-king results + the completion marker.
    results_dir: Path = Path("benchmark_results")
    # The benchmark suite checkout (ninja-benchmark--swe-bench-controller-2): contains
    # run_agent_benchmark.py + its .venv. Override with TAU_BENCH_REPO_DIR.
    bench_repo_dir: Path = Path("/root/subnet66/benchmark/ninja-benchmark--swe-bench-controller-2")
    # Suite entry point that benchmarks one agent from a path.
    runner_script: str = "run_agent_benchmark.py"
    # Predefined benchmark parameters. Model + sampling mirror the published
    # Qwen3.6-27B SWE-bench Pro baseline (temp=1.0, top_p=0.95, thinking mode) so
    # per-king results are comparable to it; the suite injects these on every LLM
    # call via its usage proxy.
    model: str = "qwen/qwen3.6-27b"
    temperature: float = 1.0
    top_p: float = 0.95
    # Extra sampling params forwarded verbatim to the suite's --sampling-json
    # (empty = don't pass). Reasoning must be enabled explicitly: sampling knobs
    # alone cannot turn thinking mode on. env_str treats a blank env var as unset,
    # so to disable the default set TAU_BENCH_SAMPLING_JSON='{}' (a no-op merge).
    sampling_json: str = '{"reasoning": {"enabled": true}}'
    slice_spec: str = "0:50"
    agent_workers: int = 4
    # OpenRouter key passed through to bench.py's environment.
    openrouter_api_key: str = ""
    # How long a single king's full benchmark may run before we give up this tick.
    bench_timeout_seconds: int = 6 * 60 * 60
    poll_seconds: float = 60.0
    # Python interpreter to run the suite with. Empty = the suite's own
    # <bench_repo_dir>/.venv (host runs). The containerized worker sets
    # TAU_BENCH_VENV_PYTHON to its bootstrap venv (deploy/worker/
    # benchmark-entrypoint.sh) because a host-built .venv's symlinks do not
    # resolve inside the worker image.
    venv_python_override: str = ""

    @property
    def bench_venv_python(self) -> Path:
        if self.venv_python_override:
            return Path(self.venv_python_override)
        return self.bench_repo_dir / ".venv" / "bin" / "python"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BenchmarkConfig:
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            submissions_dir=Path(env_str(env, "TAU_SUBMISSIONS_DIR", str(d.submissions_dir))),
            results_dir=Path(env_str(env, "TAU_BENCHMARK_RESULTS_DIR", str(d.results_dir))),
            bench_repo_dir=Path(env_str(env, "TAU_BENCH_REPO_DIR", str(d.bench_repo_dir))),
            runner_script=env_str(env, "TAU_BENCH_RUNNER_SCRIPT", d.runner_script),
            model=env_str(env, "TAU_BENCH_MODEL", d.model),
            temperature=env_float(env, "TAU_BENCH_TEMPERATURE", d.temperature),
            top_p=env_float(env, "TAU_BENCH_TOP_P", d.top_p),
            sampling_json=env_str(env, "TAU_BENCH_SAMPLING_JSON", d.sampling_json),
            slice_spec=env_str(env, "TAU_BENCH_SLICE", d.slice_spec),
            agent_workers=env_int(env, "TAU_BENCH_WORKERS", d.agent_workers),
            openrouter_api_key=env_str(env, "OPENROUTER_API_KEY", d.openrouter_api_key),
            bench_timeout_seconds=env_int(env, "TAU_BENCH_TIMEOUT_SECONDS", d.bench_timeout_seconds),
            poll_seconds=env_float(env, "TAU_BENCHMARK_POLL_SECONDS", d.poll_seconds),
            venv_python_override=env_str(env, "TAU_BENCH_VENV_PYTHON", d.venv_python_override),
        )
