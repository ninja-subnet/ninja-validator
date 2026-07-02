"""Tests for the token-free dummy LLM clients + their DummyLLMConfig wiring."""

from __future__ import annotations

import pytest

from tau.github import CommitCandidate, CommitFile
from tau.judging.parsing import parse_verdict
from tau.openrouter import DummyLLMConfig, TextPrompt
from tau.taskgen import TaskGenerationError, generate_task_description
from tau.workers.judge import DummyJudgeClient
from tau.workers.task_generator import DummyTaskClient, GeneratorConfig

# Shaped like build_generation_prompt's output so the "Commit message:" title lift fires.
_PROMPT = TextPrompt("Repository: octo/repo\nCommit message: Fix the thing\nDiff:\n@@")


def _candidate() -> CommitCandidate:
    return CommitCandidate(
        repo_full_name="octo/repo",
        repo_clone_url="https://github.com/octo/repo.git",
        commit_sha="a" * 40,
        parent_sha="b" * 40,
        message="Fix the thing",
        html_url="",
        author_name="dev",
        event_id="",
        files=[CommitFile("a.py", "modified", 5, 5, 10, "@@ -1 +1 @@\n-x\n+y")],
    )


# -- output -------------------------------------------------------------------------


async def test_complete_text_returns_parseable_task() -> None:
    client = DummyTaskClient(config=DummyLLMConfig(avg_latency_seconds=0.0, seed=1))
    raw = await client.complete_text(_PROMPT)
    assert "Fix the thing" in raw  # title lifted from the commit message line


async def test_generate_task_description_with_dummy() -> None:
    client = DummyTaskClient(config=DummyLLMConfig(avg_latency_seconds=0.0, seed=1))
    task = await generate_task_description(candidate=_candidate(), client=client)
    assert task.title == "Fix the thing"
    assert task.description and task.acceptance_criteria
    assert task.elapsed_seconds >= 0.0


async def test_failure_rate_one_raises_task_generation_error() -> None:
    client = DummyTaskClient(
        config=DummyLLMConfig(avg_latency_seconds=0.0, failure_rate=1.0, seed=1)
    )
    with pytest.raises(TaskGenerationError):
        await generate_task_description(candidate=_candidate(), client=client)


# -- latency ------------------------------------------------------------------------


async def test_slow_outlier_sleeps_past_a_short_wait() -> None:
    import asyncio

    # slow_rate=1 -> the outlier path sleeps in [timeout, timeout*factor] = [0.1, 0.3],
    # well past a 0.02s wait_for, which must therefore time out.
    client = DummyTaskClient(
        timeout=0.1, config=DummyLLMConfig(slow_rate=1.0, outlier_factor=3.0, seed=1)
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(client.complete_text(_PROMPT), timeout=0.02)


# -- judge dummy --------------------------------------------------------------------


async def test_judge_dummy_returns_parseable_verdict() -> None:
    client = DummyJudgeClient(config=DummyLLMConfig(avg_latency_seconds=0.0, seed=1))
    raw = await client.complete_text(TextPrompt("judge these two patches"))
    verdict = parse_verdict(raw)  # raises if not a well-formed verdict
    assert verdict.winner in {"candidate_a", "candidate_b", "tie"}
    assert 0.0 <= verdict.score_a <= 1.0
    assert 0.0 <= verdict.score_b <= 1.0


async def test_judge_dummy_failure_returns_unparseable_output() -> None:
    client = DummyJudgeClient(
        config=DummyLLMConfig(avg_latency_seconds=0.0, failure_rate=1.0, seed=1)
    )
    raw = await client.complete_text(TextPrompt("judge these"))
    with pytest.raises(ValueError):
        parse_verdict(raw)  # no JSON object -> the judge's retry/neutral path


def test_dummy_config_validates_ranges() -> None:
    with pytest.raises(ValueError):
        DummyLLMConfig(slow_rate=1.5)
    with pytest.raises(ValueError):
        DummyLLMConfig(outlier_factor=0.5)
    with pytest.raises(ValueError):
        DummyLLMConfig(failure_rate=-0.1)


# -- config wiring ------------------------------------------------------------------


def test_generator_config_dummy_from_env_needs_no_api_key() -> None:
    config = GeneratorConfig.from_env(
        {
            "TAU_GENERATOR_USE_DUMMY_LLM": "1",
            "TAU_GENERATOR_DUMMY_AVG_LATENCY": "3",
            "TAU_GENERATOR_DUMMY_SLOW_RATE": "0.1",
            "TAU_GENERATOR_DUMMY_FAILURE_RATE": "0.25",
        }
    )
    assert config.use_dummy_llm is True
    assert config.openrouter_api_key == ""  # not required in dummy mode
    assert config.dummy.avg_latency_seconds == 3.0
    assert config.dummy.slow_rate == 0.1
    assert config.dummy.failure_rate == 0.25


def test_generator_config_requires_api_key_without_dummy() -> None:
    with pytest.raises(OSError):
        GeneratorConfig.from_env({})
