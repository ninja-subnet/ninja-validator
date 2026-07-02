from __future__ import annotations

import asyncio
import json

from tau.workers.judge import judge_with_fallback

from tau.judging import Solution, Task, judge
from tau.judging.parsing import parse_verdict
from tau.judging.prompt import build_prompt
from tau.openrouter import RenderablePrompt


class FakeClient:
    def __init__(self, response: str | Exception, *, model: str = "test/model") -> None:
        self.response = response
        self.model = model
        self.prompt: RenderablePrompt | None = None
        self.calls = 0

    async def complete_text(self, prompt: RenderablePrompt) -> str:
        self.prompt = prompt
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class HangingClient:
    model = "test/model"

    def __init__(self) -> None:
        self.prompt: RenderablePrompt | None = None
        self.calls = 0

    async def complete_text(self, prompt: RenderablePrompt) -> str:
        self.prompt = prompt
        self.calls += 1
        await asyncio.sleep(1)
        raise AssertionError("unreachable after timeout")


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

    # the judge handed the client a prompt; it never threaded transport params
    assert client.prompt is not None
    assert judgment.model == "test/model"
    assert judgment.king_score != judgment.challenger_score
    assert judgment.winner in {"king", "challenger"}
    assert judgment.error is None


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
