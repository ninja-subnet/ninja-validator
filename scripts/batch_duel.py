#!/usr/bin/env python3
"""DB-free king vs challenger batch eval (no Postgres, no worker loop).

Loads (or generates) a fixed task pool, solves both agent bundles with concurrent
sandboxes, then judges head-to-head with the same GLM diff judge stack as production
(``judge_with_fallback``, ``TAU_JUDGE_CONCURRENCY``).

Complements:
  - ``scripts/solve_one.py`` for single-task debugging
  - ``examples/task_solver/`` for the full DB-backed worker dry run

Typical usage::

    uv run python scripts/batch_duel.py \\
      --king /path/to/default-king \\
      --challenger /path/to/my-agent \\
      --out-dir /tmp/batch_my-agent \\
      --task-dir .cache/batch_tasks_50 \\
      --king-patch-dir /tmp/batch_baseline/patch_cache \\
      --skip-reference

Resume: re-run the same command; cached solves and verdicts in ``out-dir`` are skipped.

Needs (in ``.env``): ``OPENROUTER_API_KEY`` (judge), ``GITHUB_TOKEN`` (task gen / clones),
Docker, and solver upstream vars (``LLM_PROVIDER``, ``SOLVER_MODEL``, …) for solves.
Uses the same task-repo checkout cache as the task-solver worker when
``TAU_TASK_REPO_CACHE_DIR`` / ``TAU_SANDBOX_WORK_ROOT`` are set (see ``.env.example``).
Set ``TAU_JUDGE_USE_DUMMY_LLM=1`` for token-free judge smoke tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

import docker
import dotenv

from tau.github import CommitSampler, GitHubClient, GitHubConfig, GitHubTokenRotator
from tau.judging.types import Solution, Task
from tau.openrouter import OpenRouterClient
from tau.openrouter.client import DEFAULT_MODEL
from tau.proxy import SolveBudget, UpstreamTarget
from tau.sandbox import (
    AgentRunRequest,
    SandboxConfig,
    clone_task_repo,
    ensure_sandbox_image,
    run_agent_in_container,
)
from tau.taskgen import generate_task_description
from tau.workers.judge import judge_with_fallback
from tau.workers.judge.config import JudgeWorkerConfig
from tau.workers.judge.main import _build_judge_clients
from tau.workers.task_solver.config import SolverConfig

ROOT = Path(__file__).resolve().parent.parent
GENERATOR_MODEL_ENV = "TAU_GENERATOR_MODEL"
GENERATOR_LLM_TIMEOUT_SECONDS = 120
MAX_SAMPLE_ATTEMPTS = 25
MAX_TASK_GEN_RETRIES = 40

log = logging.getLogger("batch_duel")
_io_lock = Lock()


@dataclass
class SolveSession:
    """One Docker client + sandbox image tag shared across concurrent solves."""

    client: docker.DockerClient
    config: SandboxConfig
    upstream: UpstreamTarget
    budget: SolveBudget
    image_tag: str
    github_token: str | None
    task_repo_cache_dir: Path | None
    task_repo_fetch_concurrency: int
    task_repo_cache_max_entries: int

    @classmethod
    def open(cls) -> SolveSession:
        solver_cfg = SolverConfig.from_env()
        client = docker.from_env()
        config = SandboxConfig.from_env()
        upstream = UpstreamTarget.from_env()
        budget = SolveBudget.from_env()
        image_tag = ensure_sandbox_image(client, config)
        return cls(
            client=client,
            config=config,
            upstream=upstream,
            budget=budget,
            image_tag=image_tag,
            github_token=solver_cfg.github_token,
            task_repo_cache_dir=solver_cfg.task_repo_cache_dir,
            task_repo_fetch_concurrency=solver_cfg.task_repo_fetch_concurrency,
            task_repo_cache_max_entries=solver_cfg.task_repo_cache_max_entries,
        )

    def close(self) -> None:
        self.client.close()

    def solve_patch(self, task: dict, agent_dir: Path) -> str:
        with TemporaryDirectory(prefix="batch-duel-") as tmp:
            repo_dir = clone_task_repo(
                repo_clone_url=task["repo_clone_url"],
                base_commit=task["parent_sha"],
                token=self.github_token,
                dest=Path(tmp) / "repo",
                cache_dir=self.task_repo_cache_dir,
                fetch_concurrency=self.task_repo_fetch_concurrency,
                cache_max_entries=self.task_repo_cache_max_entries,
            )
            result = run_agent_in_container(
                AgentRunRequest(
                    task_id=task["task_id"],
                    problem_statement=task["problem_statement"],
                    repo_dir=repo_dir,
                    agent_dir=agent_dir,
                    budget=self.budget,
                ),
                client=self.client,
                config=self.config,
                upstream=self.upstream,
                image_tag=self.image_tag,
            )
        return result.solution_diff or ""


def patch_path(agent_dir: Path, task: dict, patch_dir: Path) -> Path:
    return patch_dir / f"{agent_dir.name}_{task['task_id']}.diff"


def _append_jsonl(path: Path, row: dict) -> None:
    with _io_lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def load_done_eval(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["task_id"], row["agent"]))
    return done


def load_done_judge(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        json.loads(line)["task_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_judge_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_results(rows: list[dict], *, king_name: str, challenger_name: str) -> dict[str, Any]:
    if not rows:
        return {
            "tasks": 0,
            "king": king_name,
            "challenger": challenger_name,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "mean_king_score": 0.0,
            "mean_challenger_score": 0.0,
            "mean_margin": 0.0,
        }
    mk = sum(r["king_score"] for r in rows) / len(rows)
    mc = sum(r["challenger_score"] for r in rows) / len(rows)
    return {
        "tasks": len(rows),
        "king": king_name,
        "challenger": challenger_name,
        "wins": sum(1 for r in rows if r["winner"] == "challenger"),
        "losses": sum(1 for r in rows if r["winner"] == "king"),
        "ties": sum(1 for r in rows if r["winner"] == "tie"),
        "mean_king_score": round(mk, 4),
        "mean_challenger_score": round(mc, 4),
        "mean_margin": round(mc - mk, 4),
    }


def solve_and_cache(
    session: SolveSession,
    task: dict,
    agent_dir: Path,
    patch_dir: Path,
    *,
    refresh: bool,
    alt_patch_dir: Path | None,
) -> tuple[str, dict]:
    path = patch_path(agent_dir, task, patch_dir)
    if not refresh and path.is_file():
        diff = path.read_text(encoding="utf-8")
        return diff, {"agent": agent_dir.name, "cached": True, "diff_chars": len(diff)}
    if not refresh and alt_patch_dir is not None:
        alt = patch_path(agent_dir, task, alt_patch_dir)
        if alt.is_file():
            diff = alt.read_text(encoding="utf-8")
            path.write_text(diff, encoding="utf-8")
            return diff, {
                "agent": agent_dir.name,
                "cached": True,
                "borrowed_from": str(alt_patch_dir),
                "diff_chars": len(diff),
            }
    log.info("solving %s on %s...", agent_dir.name, task["task_id"][:8])
    diff = session.solve_patch(task, agent_dir)
    path.write_text(diff, encoding="utf-8")
    return diff, {
        "agent": agent_dir.name,
        "cached": False,
        "diff_chars": len(diff),
        "has_patch": bool(diff.strip()),
    }


def _solve_job(
    session: SolveSession,
    task: dict,
    agent_dir: Path,
    patch_dir: Path,
    eval_out: Path,
    *,
    refresh: bool,
    alt_patch_dir: Path | None,
) -> tuple[str, str, dict]:
    _, meta = solve_and_cache(
        session,
        task,
        agent_dir,
        patch_dir,
        refresh=refresh,
        alt_patch_dir=alt_patch_dir,
    )
    row = {"task_id": task["task_id"], "title": task.get("title", ""), **meta}
    _append_jsonl(eval_out, row)
    return task["task_id"], agent_dir.name, meta


def _run_concurrent_solves(
    session: SolveSession,
    tasks: list[dict],
    agent_dir: Path,
    patch_dir: Path,
    eval_out: Path,
    done_eval: set[tuple[str, str]],
    *,
    concurrency: int,
    refresh: bool,
    alt_patch_dir: Path | None,
) -> None:
    pending = [
        task
        for task in tasks
        if (task["task_id"], agent_dir.name) not in done_eval or refresh
    ]
    if not pending:
        return
    log.info(
        "solving %d tasks with %s (concurrency=%d)...",
        len(pending),
        agent_dir.name,
        concurrency,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _solve_job,
                session,
                task,
                agent_dir,
                patch_dir,
                eval_out,
                refresh=refresh,
                alt_patch_dir=alt_patch_dir,
            ): task
            for task in pending
        }
        for fut in as_completed(futures):
            tid, agent_name, meta = fut.result()
            done_eval.add((tid, agent_name))
            log.info(
                "solved %s on %s cached=%s chars=%s",
                agent_name,
                tid[:8],
                meta.get("cached"),
                meta.get("diff_chars"),
            )


async def _generate_task() -> dict:
    github_config = GitHubConfig.from_env()
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get(GENERATOR_MODEL_ENV) or DEFAULT_MODEL
    async with GitHubClient.create(
        token_rotator=GitHubTokenRotator.from_env(), timeout=github_config.http_timeout
    ) as client:
        sampler = CommitSampler(rng=random.Random(), client=client, config=github_config)
        candidate = (await sampler.sample_commit(max_attempts=MAX_SAMPLE_ATTEMPTS)).candidate
        async with OpenRouterClient(
            api_key, model=model, timeout=GENERATOR_LLM_TIMEOUT_SECONDS
        ) as llm:
            generated = await generate_task_description(candidate=candidate, client=llm)
    return {
        "task_id": candidate.commit_sha[:16],
        "repo_full_name": candidate.repo_full_name,
        "repo_clone_url": candidate.repo_clone_url,
        "parent_sha": candidate.parent_sha,
        "commit_sha": candidate.commit_sha,
        "problem_statement": generated.prompt_text,
        "title": generated.title,
    }


async def ensure_tasks(args: argparse.Namespace) -> list[dict]:
    args.task_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    for index in range(1, args.count + 1):
        task_path = args.task_dir / f"task_{index:02d}.json"
        if task_path.is_file() and not args.regen_tasks:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            log.info("loaded task %d: %s", index, task.get("title", task["task_id"]))
        else:
            for attempt in range(1, MAX_TASK_GEN_RETRIES + 1):
                try:
                    log.info("generating task %d/%d (try %d)...", index, args.count, attempt)
                    task = await _generate_task()
                    break
                except Exception as exc:
                    log.warning("task %d generation failed: %s", index, exc)
                    if attempt == MAX_TASK_GEN_RETRIES:
                        raise
            task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
        tasks.append(task)
    return tasks


def _verdict_row(
    task: dict,
    *,
    king_name: str,
    challenger_name: str,
    king_submission_id: str,
    challenger_submission_id: str,
    king_patch: str,
    challenger_patch: str,
    run,
) -> dict:
    judgment = run.judgment
    margin = judgment.challenger_score - judgment.king_score
    return {
        "task_id": task["task_id"],
        "title": task.get("title", ""),
        "repo": task.get("repo_full_name", ""),
        "king_agent": king_name,
        "challenger_agent": challenger_name,
        "king_submission_id": king_submission_id,
        "challenger_submission_id": challenger_submission_id,
        "winner": judgment.winner,
        "king_score": judgment.king_score,
        "challenger_score": judgment.challenger_score,
        "margin": round(margin, 4),
        "rationale": judgment.rationale,
        "model": judgment.model,
        "error": judgment.error,
        "judge_attempts": run.attempts,
        "judge_duration_seconds": round(run.duration_seconds, 3),
        "king_has_patch": bool(king_patch.strip()),
        "challenger_has_patch": bool(challenger_patch.strip()),
        "king_patch_chars": len(king_patch),
        "challenger_patch_chars": len(challenger_patch),
    }


async def _judge_one(
    task: dict,
    *,
    king_name: str,
    challenger_name: str,
    king_submission_id: str,
    challenger_submission_id: str,
    king_patch: str,
    challenger_patch: str,
    clients: list,
    judge_cfg: JudgeWorkerConfig,
) -> dict:
    task_obj = Task(
        task_id=task["task_id"],
        problem_statement=task["problem_statement"],
        reference_patch="",
    )
    run = await judge_with_fallback(
        task_obj,
        Solution(submission_id=king_submission_id, patch=king_patch),
        Solution(submission_id=challenger_submission_id, patch=challenger_patch),
        clients=clients,
        attempts=judge_cfg.attempts,
        total_timeout_seconds=judge_cfg.total_timeout_seconds,
    )
    return _verdict_row(
        task,
        king_name=king_name,
        challenger_name=challenger_name,
        king_submission_id=king_submission_id,
        challenger_submission_id=challenger_submission_id,
        king_patch=king_patch,
        challenger_patch=challenger_patch,
        run=run,
    )


async def run_parallel_judges(
    tasks: list[dict],
    *,
    king_dir: Path,
    challenger_dir: Path,
    patch_dir: Path,
    judge_out: Path,
    judge_cfg: JudgeWorkerConfig,
    judge_concurrency: int,
    done_judge: set[str],
    refresh_judge: bool,
) -> None:
    king_submission_id = king_dir.name
    challenger_submission_id = challenger_dir.name
    pending = [
        task
        for task in tasks
        if task["task_id"] not in done_judge or refresh_judge
    ]
    if not pending:
        return

    log.info("judging %d tasks (concurrency=%d)...", len(pending), judge_concurrency)
    sem = asyncio.Semaphore(judge_concurrency)

    async with AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(client)
            for client in _build_judge_clients(judge_cfg)
        ]

        async def _run(task: dict) -> None:
            async with sem:
                tid = task["task_id"]
                king_patch = patch_path(king_dir, task, patch_dir).read_text(encoding="utf-8")
                challenger_patch = patch_path(challenger_dir, task, patch_dir).read_text(
                    encoding="utf-8"
                )
                log.info(
                    "judging %s king=%d chars challenger=%d chars",
                    tid[:8],
                    len(king_patch),
                    len(challenger_patch),
                )
                verdict = await _judge_one(
                    task,
                    king_name=king_dir.name,
                    challenger_name=challenger_dir.name,
                    king_submission_id=king_submission_id,
                    challenger_submission_id=challenger_submission_id,
                    king_patch=king_patch,
                    challenger_patch=challenger_patch,
                    clients=clients,
                    judge_cfg=judge_cfg,
                )
                _append_jsonl(judge_out, verdict)
                log.info(
                    "verdict %s winner=%s margin=%.3f",
                    tid[:8],
                    verdict["winner"],
                    verdict["margin"],
                )

        await asyncio.gather(*[_run(task) for task in pending])


async def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    judge_cfg = JudgeWorkerConfig.from_env()
    king_dir = Path(args.king).resolve()
    challenger_dir = Path(args.challenger).resolve()
    for agent_dir in (king_dir, challenger_dir):
        if not (agent_dir / "agent.py").is_file():
            raise SystemExit(f"missing agent.py in {agent_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.patch_dir.mkdir(parents=True, exist_ok=True)
    eval_out = args.out_dir / "batch_eval_results.jsonl"
    judge_out = args.out_dir / "batch_judge_results.jsonl"
    summary_out = args.out_dir / "summary.json"

    done_eval = load_done_eval(eval_out)
    done_judge: set[str] = set()
    if args.refresh_judge:
        judge_out.write_text("", encoding="utf-8")
    else:
        done_judge = load_done_judge(judge_out)

    tasks = await ensure_tasks(args)

    session = SolveSession.open()
    try:
        _run_concurrent_solves(
            session,
            tasks,
            king_dir,
            args.patch_dir,
            eval_out,
            done_eval,
            concurrency=args.concurrency,
            refresh=args.refresh_solves,
            alt_patch_dir=args.king_patch_dir,
        )
        _run_concurrent_solves(
            session,
            tasks,
            challenger_dir,
            args.patch_dir,
            eval_out,
            done_eval,
            concurrency=args.concurrency,
            refresh=args.refresh_solves,
            alt_patch_dir=None,
        )
    finally:
        session.close()

    await run_parallel_judges(
        tasks,
        king_dir=king_dir,
        challenger_dir=challenger_dir,
        patch_dir=args.patch_dir,
        judge_out=judge_out,
        judge_cfg=judge_cfg,
        judge_concurrency=args.judge_concurrency,
        done_judge=done_judge,
        refresh_judge=args.refresh_judge,
    )
    by_task_id = {row["task_id"]: row for row in load_judge_results(judge_out)}
    judge_results = [by_task_id[task["task_id"]] for task in tasks if task["task_id"] in by_task_id]

    summary = summarize_results(judge_results, king_name=king_dir.name, challenger_name=challenger_dir.name)
    summary["solve_concurrency"] = args.concurrency
    summary["judge_concurrency"] = args.judge_concurrency
    summary["duration_seconds"] = round(time.monotonic() - started, 1)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== batch duel summary ===")
    print(f"tasks: {summary['tasks']}  solve_concurrency: {args.concurrency}")
    print(f"judge_concurrency: {args.judge_concurrency}  duration: {summary['duration_seconds']}s")
    if summary["tasks"]:
        print(f"king: {king_dir.name}  challenger: {challenger_dir.name}")
        print(
            f"W/L/T: {summary['wins']}/{summary['losses']}/{summary['ties']}  "
            f"mean margin={summary['mean_margin']:+.3f}"
        )
    print(f"summary: {summary_out}")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "docker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB-free king vs challenger batch duel eval.")
    parser.add_argument("--king", required=True, help="Path to king agent bundle directory.")
    parser.add_argument("--challenger", required=True, help="Path to challenger agent bundle directory.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".cache" / "batch_duel")
    parser.add_argument("--task-dir", type=Path, default=ROOT / ".cache" / "batch_tasks_50")
    parser.add_argument("--patch-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("BATCH_CONCURRENCY", "25")),
        help="Max concurrent sandbox solves.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=int(os.environ.get("TAU_JUDGE_CONCURRENCY", "5")),
        help="Max concurrent GLM judge calls (matches prod judge worker default).",
    )
    parser.add_argument("--king-patch-dir", type=Path, default=None, help="Reuse king patches from another run.")
    parser.add_argument("--regen-tasks", action="store_true")
    parser.add_argument("--refresh-solves", action="store_true")
    parser.add_argument("--refresh-judge", action="store_true")
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Accepted for CLI compatibility; reference patches are not fetched.",
    )
    args = parser.parse_args(argv)
    if args.patch_dir is None:
        args.patch_dir = args.out_dir / "patch_cache"
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.judge_concurrency < 1:
        raise SystemExit("--judge-concurrency must be >= 1")
    if args.skip_reference:
        log.debug("--skip-reference is a no-op; batch_duel never fetches reference patches")
    return args


def main(argv: list[str] | None = None) -> None:
    dotenv.load_dotenv(ROOT / ".env")
    _configure_logging()
    args = parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
