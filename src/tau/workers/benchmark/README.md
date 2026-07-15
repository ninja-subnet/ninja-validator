# benchmark-worker

Watches the `kings` table and automatically benchmarks each **new king's submitted
agent** against SWE-bench Pro, saving a score + cost to a dedicated per-king folder.
Restart-safe: on startup (and every tick) it skips kings already benchmarked and
resumes any that were interrupted.

## How it works

Level-triggered polling (same detection as every other worker — no DB trigger):

1. `BenchmarkDb.all_kings()` lists every king (`king_id`, which IS the submission id).
2. A king needs benchmarking if `<results_dir>/<king_id>/benchmark.json` is absent or
   not `status: done`.
3. For each pending king: resolve `<submissions_dir>/<king_id>/agent.py`, then shell
   out to the benchmark suite's single-agent entry point:
   `run_agent_benchmark.py --agent-dir <bundle> --name <king_id> --model … --slice … --workers …`.
4. The suite runs the king's real `agent.py` (`solve()` contract) inside the SWE-bench
   Pro sweap images, scores with the official Scale evaluator, reports cost, and
   archives to `<bench_repo>/results/<king_id>/`.
5. The worker copies the small summary artifacts into `<results_dir>/<king_id>/` and
   writes the `benchmark.json` completion marker.

**Restart safety** is the on-disk marker + the suite's per-instance resume: an
interrupted king has no `done` marker, so it is re-invoked and continues from the
instances already finished (skips ids in `preds.json`, reuses existing eval results).

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `DATABASE_URL` (or `POSTGRES_*`) | — | the validator DB holding `kings` |
| `TAU_SUBMISSIONS_DIR` | `submissions` | root of extracted submission bundles (prod: `…/netuid-66/private-submissions`) |
| `TAU_BENCHMARK_RESULTS_DIR` | `benchmark_results` | dedicated per-king results folder |
| `TAU_BENCH_REPO_DIR` | `/root/subnet66/benchmark/ninja-benchmark--swe-bench-controller-2` | benchmark suite checkout (has `run_agent_benchmark.py` + `.venv`) |
| `TAU_BENCH_RUNNER_SCRIPT` | `run_agent_benchmark.py` | suite entry point (rel. to `TAU_BENCH_REPO_DIR`) |
| `TAU_BENCH_MODEL` | `qwen/qwen3-coder` | model the king's agent must use |
| `TAU_BENCH_SLICE` | `0:50` | instance slice per king (`""` = full 731) |
| `TAU_BENCH_WORKERS` | `4` | concurrent agent containers |
| `TAU_BENCH_TIMEOUT_SECONDS` | `21600` | max wall-clock for one king's benchmark per tick |
| `OPENROUTER_API_KEY` | — | forwarded to the suite (usage proxy → OpenRouter) |
| `TAU_BENCHMARK_POLL_SECONDS` | `60` | poll interval |
| `BENCH_*` (e.g. `BENCH_TEMPERATURE`, `BENCH_SEED`, `BENCH_SAMPLING_JSON`) | — | passed through to pin sampling (see below) |

## Run

```bash
uv run benchmark-worker
```

Prerequisites: the benchmark suite set up at `TAU_BENCH_REPO_DIR` (`.venv` +
submodules; see its `README.md`/`BENCHMARK.md`), the docker daemon available, and
`OPENROUTER_API_KEY` set. See the commented `benchmark-worker` block in `compose.yaml`
for the containerized shape.

## Sampling (comparability)

Sampling is fixed per run so scores are reproducible and comparable to public
SWE-bench Pro results. The suite resolves it from its agent config's `sampling:` block
(default `temperature: 0.0, top_p: 1.0`), overridden by `BENCH_*` env vars. The worker
passes its whole environment through, so exporting e.g. `BENCH_TEMPERATURE=0.2` /
`BENCH_SEED=0` / `BENCH_SAMPLING_JSON='{"top_k":40}'` for the worker process applies to
every king. Supported knobs: `temperature, top_p, top_k, min_p, top_a,
frequency_penalty, presence_penalty, repetition_penalty, seed, max_tokens, logprobs,
top_logprobs, logit_bias`.

Two things to keep in mind:
- **Model**: set `TAU_BENCH_MODEL` to the subnet's live `SOLVER_MODEL` if you want the
  benchmark to use the exact model the validator enforces (default `qwen/qwen3-coder`).
- **Signal**: this is an *objective* SWE-bench Pro test-resolution score — a complement
  to, not a reproduction of, the subnet's blinded LLM-judge ranking.

## Output (per king)

```
<results_dir>/<king_id>/
├── benchmark.json      # completion marker: resolved, resolve_rate, cost, status
├── summary.md          # headline numbers
├── cost_summary.json   # full cost/resolve breakdown + full-run projection
├── costs.jsonl         # per-instance tokens/USD/exit status
└── eval_results.json   # {instance_id: resolved} from the official evaluator
```

The full run (preds, trajectories, per-run config) lives under the suite checkout:
`<bench_repo>/swebench-results/<king_id>/`, `<bench_repo>/results/<king_id>/`, and the
generated config `<bench_repo>/bench_configs/generated/<king_id>.yaml`.
