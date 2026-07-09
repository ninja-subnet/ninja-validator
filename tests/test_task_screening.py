from __future__ import annotations

import asyncio
import json

import pytest

from tau.openrouter import RenderablePrompt
from tau.task_screening import (
    Candidate,
    ScreeningResult,
    Task,
    build_prompt,
    score_candidate,
)
from tau.task_screening.parsing import parse_score


class FakeClient:
    def __init__(
        self, response: str | Exception, *, model: str = "test/screener"
    ) -> None:
        self.response = response
        self.model = model
        self.prompt: RenderablePrompt | None = None
        self.calls = 0

    async def complete_text(self, prompt: RenderablePrompt) -> str:
        self.calls += 1
        self.prompt = prompt
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _task(**overrides: str) -> Task:
    values = {
        "problem_statement": "Make the parser accept hexadecimal numbers.",
        "reference_patch": "",
    }
    values.update(overrides)
    return Task(**values)


def _candidate(**overrides: str) -> Candidate:
    values = {
        "patch": "diff --git a/parser.py b/parser.py\n+accept_hex = True\n",
    }
    values.update(overrides)
    return Candidate(**values)


def test_score_candidate_scores_one_patch_and_normalizes_result() -> None:
    client = FakeClient(json.dumps({"score": 70, "rationale": "About 70% complete."}))

    result = asyncio.run(score_candidate(_task(), _candidate(), client=client))

    assert result == ScreeningResult(
        score=0.7,
        model="test/screener",
    )
    assert client.calls == 1
    assert client.prompt is not None


def test_score_candidate_propagates_transport_errors_for_worker_retry() -> None:
    error = RuntimeError("provider unavailable")
    client = FakeClient(error)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(score_candidate(_task(), _candidate(), client=client))

    assert caught.value is error
    assert client.calls == 1


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", "did not return a JSON object"),
        ('{"rationale": "missing"}', "missing score"),
        ('{"score": "70", "rationale": "wrong type"}', "JSON number"),
        ('{"score": true, "rationale": "bool is not a score"}', "JSON number"),
        ('{"score": -1, "rationale": "too low"}', r"in \[0, 100\]"),
        ('{"score": 101, "rationale": "too high"}', r"in \[0, 100\]"),
        ('{"score": NaN, "rationale": "not finite"}', "finite"),
        ('{"score": 70}', "rationale"),
        ('{"score": 70, "rationale": "  "}', "rationale"),
    ],
)
def test_malformed_model_outputs_raise_for_worker_retry(raw: str, message: str) -> None:
    client = FakeClient(raw)

    with pytest.raises(ValueError, match=message):
        asyncio.run(score_candidate(_task(), _candidate(), client=client))


@pytest.mark.parametrize(
    ("raw_score", "normalized"), [(0, 0.0), (70.5, 0.705), (100, 1.0)]
)
def test_parse_score_normalizes_valid_boundaries(
    raw_score: int | float, normalized: float
) -> None:
    assert (
        parse_score(json.dumps({"score": raw_score, "rationale": " coverage "}))
        == normalized
    )


def test_parse_score_accepts_a_json_fence_but_no_fabricated_defaults() -> None:
    assert parse_score('```json\n{"score": 42, "rationale": "partial"}\n```') == 0.42


def test_prompt_is_single_candidate_and_has_no_duel_vocabulary() -> None:
    prompt = build_prompt(_task(), _candidate()).as_text()
    lowered = prompt.lower()

    assert "make the parser accept hexadecimal numbers" in lowered
    assert "accept_hex = true" in lowered
    assert "qualification_patch" in lowered
    assert "candidate_a" not in lowered
    assert "candidate_b" not in lowered
    assert "challenger" not in lowered
    assert '"winner"' not in lowered


def test_prompt_summarizes_reference_without_leaking_answer_lines() -> None:
    reference = (
        "diff --git a/parser.py b/parser.py\n"
        "--- a/parser.py\n"
        "+++ b/parser.py\n"
        "@@ -1,2 +1,3 @@ def parse(value):\n"
        "-    return int(value)\n"
        "+    SECRET_REFERENCE_ANSWER = int(value, 0)\n"
        "+    return SECRET_REFERENCE_ANSWER\n"
    )

    prompt = build_prompt(_task(reference_patch=reference), _candidate()).as_text()

    assert "SECRET_REFERENCE_ANSWER" not in prompt
    assert "NON-SOLUTION REFERENCE SUMMARY" in prompt
    assert "parser.py (+2/-1)" in prompt
    assert "@@ -1,2 +1,3 @@" in prompt
    assert "def parse(value):" not in prompt


def test_prompt_preserves_a_quoted_reference_path_with_spaces() -> None:
    reference = (
        'diff --git "a/package/with space.py" "b/package/with space.py"\n'
        '--- "a/package/with space.py"\n'
        '+++ "b/package/with space.py"\n'
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    prompt = build_prompt(_task(reference_patch=reference), _candidate()).as_text()

    assert "package/with space.py (+1/-1)" in prompt


def test_prompt_marks_an_empty_patch_as_no_changes() -> None:
    text = build_prompt(_task(), _candidate(patch=" \n\t ")).as_text()

    assert '"qualification_patch": "(no changes)"' in text


def test_prompt_truncates_oversized_untrusted_input_from_the_middle() -> None:
    long_task = "TASK_HEAD" + ("x" * 21_000) + "TASK_TAIL"
    long_patch = "PATCH_HEAD" + ("y" * 61_000) + "PATCH_TAIL"

    text = build_prompt(
        _task(problem_statement=long_task), _candidate(patch=long_patch)
    ).as_text()

    assert "TASK_HEAD" in text and "TASK_TAIL" in text
    assert "PATCH_HEAD" in text and "PATCH_TAIL" in text
    assert text.count("...[truncated for diff judge]...") == 2
