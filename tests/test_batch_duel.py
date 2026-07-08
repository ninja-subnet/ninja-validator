from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "batch_duel.py"


def _load_batch_duel():
    spec = importlib.util.spec_from_file_location("batch_duel", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="batch_duel")
def fixture_batch_duel():
    return _load_batch_duel()


def test_patch_path_uses_agent_name_and_task_id(batch_duel) -> None:
    agent = Path("/tmp/agents/challenger-a")
    task = {"task_id": "abc123"}
    path = batch_duel.patch_path(agent, task, Path("/tmp/out"))
    assert path.name == "challenger-a_abc123.diff"


def test_summarize_results_computes_mean_margin(batch_duel) -> None:
    rows = [
        {"winner": "challenger", "king_score": 0.4, "challenger_score": 0.7},
        {"winner": "king", "king_score": 0.8, "challenger_score": 0.5},
    ]
    summary = batch_duel.summarize_results(rows, king_name="king", challenger_name="chal")
    assert summary["tasks"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["mean_margin"] == pytest.approx(0.0)


def test_load_done_judge_reads_task_ids(batch_duel, tmp_path: Path) -> None:
    judge_out = tmp_path / "batch_judge_results.jsonl"
    judge_out.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "t1", "winner": "king"}),
                json.dumps({"task_id": "t2", "winner": "tie"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert batch_duel.load_done_judge(judge_out) == {"t1", "t2"}


@pytest.mark.asyncio
async def test_run_parallel_judges_respects_concurrency(batch_duel, tmp_path: Path, monkeypatch) -> None:
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_judge_one(task, **kwargs):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        run = type("Run", (), {})()
        run.judgment = type(
            "J",
            (),
            {
                "winner": "tie",
                "king_score": 0.5,
                "challenger_score": 0.5,
                "rationale": "ok",
                "model": "dummy/test",
                "error": None,
            },
        )()
        run.attempts = 1
        run.duration_seconds = 0.01
        return batch_duel._verdict_row(
            task,
            king_name="king",
            challenger_name="chal",
            king_submission_id="king",
            challenger_submission_id="chal",
            king_patch="",
            challenger_patch="",
            run=run,
        )

    monkeypatch.setattr(batch_duel, "_judge_one", fake_judge_one)
    monkeypatch.setattr(
        batch_duel,
        "_build_judge_clients",
        lambda _cfg: [],
    )

    king_dir = tmp_path / "king"
    chal_dir = tmp_path / "chal"
    patch_dir = tmp_path / "patches"
    king_dir.mkdir()
    chal_dir.mkdir()
    patch_dir.mkdir()
    (king_dir / "agent.py").write_text("# king\n", encoding="utf-8")
    (chal_dir / "agent.py").write_text("# chal\n", encoding="utf-8")

    tasks = [{"task_id": f"t{i}", "problem_statement": "fix", "repo_full_name": "r"} for i in range(6)]
    for task in tasks:
        (patch_dir / f"king_{task['task_id']}.diff").write_text("patch", encoding="utf-8")
        (patch_dir / f"chal_{task['task_id']}.diff").write_text("patch", encoding="utf-8")

    judge_out = tmp_path / "batch_judge_results.jsonl"
    cfg = batch_duel.JudgeWorkerConfig.from_env(
        {"OPENROUTER_API_KEY": "k", "TAU_JUDGE_USE_DUMMY_LLM": "1"}
    )

    await batch_duel.run_parallel_judges(
        tasks,
        king_dir=king_dir,
        challenger_dir=chal_dir,
        patch_dir=patch_dir,
        judge_out=judge_out,
        judge_cfg=cfg,
        judge_concurrency=2,
        done_judge=set(),
        refresh_judge=False,
    )

    assert peak <= 2
    assert len(batch_duel.load_judge_results(judge_out)) == 6


@pytest.mark.asyncio
async def test_run_end_to_end_judge_only_with_dummy_llm(
    batch_duel, tmp_path: Path, monkeypatch
) -> None:
    class FakeSession:
        def close(self) -> None:
            return None

    monkeypatch.setattr(batch_duel.SolveSession, "open", lambda: FakeSession())
    monkeypatch.setattr(
        batch_duel,
        "_run_concurrent_solves",
        lambda *args, **kwargs: None,
    )

    king_dir = tmp_path / "default-king"
    chal_dir = tmp_path / "challenger-b"
    patch_dir = tmp_path / "patch_cache"
    out_dir = tmp_path / "out"
    task_dir = tmp_path / "tasks"
    for path in (king_dir, chal_dir, patch_dir, out_dir, task_dir):
        path.mkdir()
    (king_dir / "agent.py").write_text("# king\n", encoding="utf-8")
    (chal_dir / "agent.py").write_text("# chal\n", encoding="utf-8")

    task = {
        "task_id": "task001",
        "title": "demo",
        "repo_full_name": "org/repo",
        "problem_statement": "Fix the bug.",
    }
    (task_dir / "task_01.json").write_text(json.dumps(task), encoding="utf-8")
    (patch_dir / "default-king_task001.diff").write_text("+fix\n", encoding="utf-8")
    (patch_dir / "challenger-b_task001.diff").write_text("+fix2\n", encoding="utf-8")

    eval_out = out_dir / "batch_eval_results.jsonl"
    eval_out.write_text(
        json.dumps({"task_id": "task001", "agent": "default-king", "cached": True}) + "\n"
        + json.dumps({"task_id": "task001", "agent": "challenger-b", "cached": True}) + "\n",
        encoding="utf-8",
    )

    argv = [
        "--king",
        str(king_dir),
        "--challenger",
        str(chal_dir),
        "--out-dir",
        str(out_dir),
        "--task-dir",
        str(task_dir),
        "--patch-dir",
        str(patch_dir),
        "--count",
        "1",
        "--concurrency",
        "1",
        "--judge-concurrency",
        "1",
    ]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("TAU_JUDGE_USE_DUMMY_LLM", "1")

    await batch_duel.run(batch_duel.parse_args(argv))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["tasks"] == 1
    assert (out_dir / "batch_judge_results.jsonl").is_file()
