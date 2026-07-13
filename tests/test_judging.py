from __future__ import annotations

import asyncio
import json

import httpx

from tau.workers.judge import judge_with_fallback
from tau.workers.judge.config import JudgeWorkerConfig

from tau.judging import Solution, Task, judge
from tau.judging.parsing import parse_verdict
from tau.judging.prompt import build_prompt
from tau.openrouter import OpenRouterClient, RenderablePrompt, TextPrompt
from tau.workers.judge.main import _build_judge_clients


class FakeClient:
    def __init__(self, response: str | Exception, *, model: str = "test/model") -> None:
        self.response = response
        self.model = model
        self.prompt: RenderablePrompt | None = None
        self.seed: int | None = None
        self.calls = 0

    async def complete_text(
        self, prompt: RenderablePrompt, *, seed: int | None = None
    ) -> str:
        self.prompt = prompt
        self.seed = seed
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class HangingClient:
    model = "test/model"

    def __init__(self) -> None:
        self.prompt: RenderablePrompt | None = None
        self.seed: int | None = None
        self.calls = 0

    async def complete_text(
        self, prompt: RenderablePrompt, *, seed: int | None = None
    ) -> str:
        self.prompt = prompt
        self.seed = seed
        self.calls += 1
        await asyncio.sleep(1)
        raise AssertionError("unreachable after timeout")


def test_judge_config_defaults_to_glm_pinned_endpoint() -> None:
    config = JudgeWorkerConfig.from_env({"OPENROUTER_API_KEY": "k"})

    assert config.model == "z-ai/glm-5.2"
    assert config.fallback_models == ("z-ai/glm-5.2",)
    assert config.provider == {
        "only": ["z-ai/fp8"],
        "allow_fallbacks": False,
    }
    assert config.fallback_provider == {
        "only": ["atlas-cloud/fp8"],
        "allow_fallbacks": False,
    }
    assert config.max_tokens == 32_000


def test_judge_config_reads_openrouter_provider_routing() -> None:
    config = JudgeWorkerConfig.from_env(
        {
            "OPENROUTER_API_KEY": "k",
            "TAU_JUDGE_MODEL": "z-ai/glm-5.2",
            "TAU_JUDGE_FALLBACK_MODELS": "z-ai/glm-5.2",
            "TAU_JUDGE_PROVIDER_ONLY": "z-ai/fp8",
            "TAU_JUDGE_PROVIDER_ALLOW_FALLBACKS": "false",
            "TAU_JUDGE_FALLBACK_PROVIDER_ONLY": "atlas-cloud/fp8",
            "TAU_JUDGE_FALLBACK_PROVIDER_ALLOW_FALLBACKS": "false",
            "TAU_JUDGE_MAX_TOKENS": "32000",
        }
    )
    assert config.model == "z-ai/glm-5.2"
    assert config.fallback_models == ("z-ai/glm-5.2",)
    assert config.provider == {
        "only": ["z-ai/fp8"],
        "allow_fallbacks": False,
    }
    assert config.fallback_provider == {
        "only": ["atlas-cloud/fp8"],
        "allow_fallbacks": False,
    }
    assert config.max_tokens == 32_000


def test_judge_clients_use_distinct_primary_and_fallback_routes() -> None:
    config = JudgeWorkerConfig.from_env({"OPENROUTER_API_KEY": "k"})

    clients = _build_judge_clients(config)

    assert len(clients) == 2
    assert clients[0]._provider == {"only": ["z-ai/fp8"], "allow_fallbacks": False}
    assert clients[1]._provider == {
        "only": ["atlas-cloud/fp8"],
        "allow_fallbacks": False,
    }
    assert clients[0]._max_tokens == 32_000
    assert clients[1]._max_tokens == 32_000
    assert clients[0]._reasoning == {"enabled": True, "exclude": True}
    assert clients[1]._reasoning == {"enabled": True, "exclude": True}


async def test_openrouter_client_sends_provider_routing() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "model": "z-ai/glm-5.2",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenRouterClient(
            "k",
            model="z-ai/glm-5.2",
            provider={
                "only": ["z-ai/fp8"],
                "allow_fallbacks": False,
            },
            client=http,
        )
        await client.complete_text(TextPrompt("judge"), seed=12345)

    assert captured["payload"]["provider"] == {
        "only": ["z-ai/fp8"],
        "allow_fallbacks": False,
    }
    assert captured["payload"]["seed"] == 12345


def test_judge_blinds_prompt_and_stamps_client_model() -> None:
    client = FakeClient(
        json.dumps(
            {
                "winner": "candidate_a",
                "candidate_a_score": 10,
                "candidate_b_score": 90,
                "rationale": "candidate_b is better",
            }
        )
    )
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger = Solution(submission_id="challenger", patch="challenger patch")

    judgment = asyncio.run(
        judge(task, king, challenger, client=client, seed="stable-test-seed")
    )

    # The judge handed the client a prompt and a validator-owned model seed.
    assert client.prompt is not None
    assert client.seed is not None
    assert 0 <= client.seed < 2**53
    assert judgment.model == "test/model"
    assert judgment.king_score != judgment.challenger_score
    assert judgment.winner in {"king", "challenger"}
    assert judgment.error is None


def test_judge_default_seed_is_content_based_not_submission_based() -> None:
    response = json.dumps(
        {
            "winner": "tie",
            "candidate_a_score": 50,
            "candidate_b_score": 50,
            "rationale": "same pair",
        }
    )
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger_a = Solution(submission_id="challenger-a", patch="challenger patch")
    challenger_b = Solution(submission_id="challenger-b", patch="challenger patch")
    first = FakeClient(response)
    second = FakeClient(response)

    asyncio.run(judge(task, king, challenger_a, client=first))
    asyncio.run(judge(task, king, challenger_b, client=second))

    assert first.seed == second.seed
    assert first.prompt is not None
    assert second.prompt is not None
    assert first.prompt.as_text() == second.prompt.as_text()


def test_judge_default_seed_changes_when_patch_content_changes() -> None:
    response = json.dumps(
        {
            "winner": "tie",
            "candidate_a_score": 50,
            "candidate_b_score": 50,
            "rationale": "same pair",
        }
    )
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger_a = Solution(submission_id="challenger", patch="challenger patch")
    challenger_b = Solution(submission_id="challenger", patch="different patch")
    first = FakeClient(response)
    second = FakeClient(response)

    asyncio.run(judge(task, king, challenger_a, client=first))
    asyncio.run(judge(task, king, challenger_b, client=second))

    assert first.seed != second.seed


def test_worker_converts_exhausted_judge_failures_to_neutral() -> None:
    client = FakeClient(RuntimeError("provider unavailable"))
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger = Solution(submission_id="challenger", patch="challenger patch")

    run = asyncio.run(
        judge_with_fallback(
            task,
            king,
            challenger,
            clients=[client],
            attempts=2,
        )
    )

    assert run.judgment.winner == "tie"
    assert run.judgment.king_score == 0.5
    assert run.judgment.challenger_score == 0.5
    assert run.judgment.error is not None
    assert "provider unavailable" in run.judgment.error
    assert run.attempts == 2  # one client x two attempts
    assert run.duration_seconds >= 0.0


def test_worker_total_timeout_returns_timeout_neutral_without_retrying_forever() -> (
    None
):
    client = HangingClient()
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger = Solution(submission_id="challenger", patch="challenger patch")

    run = asyncio.run(
        judge_with_fallback(
            task,
            king,
            challenger,
            clients=[client],
            attempts=100,
            total_timeout_seconds=0.01,
        )
    )

    assert client.calls == 1
    assert run.judgment.winner == "tie"
    assert run.judgment.king_score == 0.5
    assert run.judgment.challenger_score == 0.5
    assert run.judgment.error == "LLM diff judge exceeded 0.01s total timeout"
    # the attempt count survives the timeout cancellation (one tick before the hang)
    assert run.attempts == 1


def test_worker_route_error_skips_remaining_attempts_for_model() -> None:
    primary = FakeClient(
        RuntimeError("OpenRouter returned no choices (error_code=403)"),
        model="primary/model",
    )
    fallback = FakeClient(
        json.dumps(
            {
                "winner": "candidate_b",
                "candidate_a_score": 30,
                "candidate_b_score": 80,
                "rationale": "fallback judged successfully",
            }
        ),
        model="fallback/model",
    )
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger = Solution(submission_id="challenger", patch="challenger patch")

    run = asyncio.run(
        judge_with_fallback(
            task,
            king,
            challenger,
            clients=[primary, fallback],
            attempts=4,
        )
    )

    assert primary.calls == 1
    assert fallback.calls == 1
    assert run.judgment.model == "fallback/model"
    assert run.judgment.error is None
    # primary's route error breaks to the fallback after one try: 1 + 1
    assert run.attempts == 2


def test_judge_keeps_declared_tie_despite_score_gap() -> None:
    client = FakeClient(
        json.dumps(
            {
                "winner": "tie",
                "candidate_a_score": 80,
                "candidate_b_score": 20,
                "rationale": "tie call",
            }
        )
    )
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="ref")
    king = Solution(submission_id="king", patch="king patch")
    challenger = Solution(submission_id="challenger", patch="challenger patch")

    judgment = asyncio.run(
        judge(task, king, challenger, client=client, seed="stable-test-seed")
    )

    assert judgment.winner == "tie"
    assert judgment.king_score != judgment.challenger_score


def test_parse_verdict_treats_missing_winner_as_tie() -> None:
    verdict = parse_verdict('{"candidate_a_score": 90, "candidate_b_score": 10}')
    assert verdict.winner == "tie"


def test_parse_verdict_derives_winner_for_invalid_token() -> None:
    verdict = parse_verdict(
        '{"winner": "nonsense", "candidate_a_score": 10, "candidate_b_score": 90}'
    )
    assert verdict.winner == "candidate_b"


def test_parse_verdict_preserves_explicit_tie() -> None:
    verdict = parse_verdict(
        '{"winner": "tie", "candidate_a_score": 90, "candidate_b_score": 10}'
    )
    assert verdict.winner == "tie"


def test_build_prompt_summarizes_reference_without_leaking_code() -> None:
    reference = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 42  # SECRET_ANSWER_TOKEN\n"
    )
    task = Task(task_id="task-1", problem_statement="Fix foo", reference_patch=reference)

    text = build_prompt(task, "candidate a patch", "candidate b patch").as_text()

    assert "SECRET_ANSWER_TOKEN" not in text
    assert "NON-CANDIDATE REFERENCE SUMMARY" in text
    assert "foo.py" in text


def test_build_prompt_marks_whitespace_only_patch_as_no_changes() -> None:
    task = Task(task_id="task-1", problem_statement="Fix it", reference_patch="")

    text = build_prompt(task, "   \n  ", "real patch").as_text()

    assert "(no changes)" in text
