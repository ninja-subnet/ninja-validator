from __future__ import annotations

import json

import pytest

from tau.openrouter import RenderablePrompt
from tau.qualification import (
    QualificationOutcome,
    SecurityQualificationConfig,
    SecurityQualificationInput,
    build_security_qualification_prompt,
    parse_security_qualification,
    qualification_outcome,
    qualify_submission_security,
    security_failures,
    security_risk_categories,
)


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


def _input(**overrides: object) -> SecurityQualificationInput:
    values = {
        "hotkey": "hk-test",
        "submission_id": "sub-test",
        "patch": "diff --git a/agent.py b/agent.py\n+print('hello')",
        "base_files": {
            "agent.py": "def solve(**kwargs):\n    return {'success': False}\n"
        },
        "submitted_files": {
            "agent.py": "def solve(**kwargs):\n    return {'success': True}\n"
        },
        "static_findings": {"findings": ["scope guard passed"]},
        "metadata": {"source": "private"},
    }
    values.update(overrides)
    return SecurityQualificationInput(**values)  # type: ignore[arg-type]


def _payload(**overrides: object) -> str:
    payload = {
        "verdict": "pass",
        "overall_score": 88,
        "security_score": 94,
        "summary": "Small safe change.",
        "reasons": ["No suspicious IO."],
        "risks": [],
        "required_changes": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_prompt_contains_system_untrusted_wrapper_and_submission_data() -> None:
    prompt = build_security_qualification_prompt(_input())

    assert prompt.system is not None
    assert "CI security reviewer" in prompt.system
    text = prompt.as_text()
    assert "<submission_data>" in text
    assert "</submission_data>" in text
    assert '"hotkey": "hk-test"' in text
    assert '"submission_id": "sub-test"' in text
    assert '"source": "private"' in text
    assert '"status": "modified"' in text
    assert "scope guard passed" in text


def test_prompt_truncates_large_patch_and_files_deterministically() -> None:
    prompt = build_security_qualification_prompt(
        _input(
            patch="x" * 120_010,
            submitted_files={
                "b.py": "b" * 100_000,
                "agent.py": "a" * 100_000,
            },
        )
    )
    text = prompt.as_text()
    data = text.split("<submission_data>\n", 1)[1].split("\n</submission_data>", 1)[0]
    payload = json.loads(data)

    assert len(payload["patch"]) == 120_000
    assert list(payload["submitted_files"]) == ["agent.py", "b.py"]
    assert len(payload["submitted_files"]["agent.py"]) == 100_000
    assert len(payload["submitted_files"]["b.py"]) == 60_000


def test_prompt_uses_security_qualification_config() -> None:
    prompt = build_security_qualification_prompt(
        _input(
            submitted_files={
                "main.py": "m" * 10,
                "agent.py": "a" * 10,
            },
        ),
        config=SecurityQualificationConfig(
            agent_entrypoint="main.py",
            patch_max_chars=4,
            submitted_entrypoint_max_chars=3,
            submitted_files_max_chars=5,
        ),
    )
    text = prompt.as_text()
    data = text.split("<submission_data>\n", 1)[1].split("\n</submission_data>", 1)[0]
    payload = json.loads(data)

    assert payload["patch"] == "diff"
    assert payload["submitted_agent_py"] == "mmm"
    assert payload["submitted_files"] == {"agent.py": "aaaaa", "main.py": ""}


def test_security_qualification_config_from_env() -> None:
    config = SecurityQualificationConfig.from_env(
        {
            "TAU_SECURITY_QUALIFICATION_MODEL": "model/security",
            "TAU_SECURITY_QUALIFICATION_AGENT_ENTRYPOINT": "main.py",
            "TAU_SECURITY_QUALIFICATION_PATCH_MAX_CHARS": "7",
            "TAU_SECURITY_QUALIFICATION_BASE_ENTRYPOINT_MAX_CHARS": "8",
            "TAU_SECURITY_QUALIFICATION_SUBMITTED_ENTRYPOINT_MAX_CHARS": "9",
            "TAU_SECURITY_QUALIFICATION_BASE_FILES_MAX_CHARS": "10",
            "TAU_SECURITY_QUALIFICATION_SUBMITTED_FILES_MAX_CHARS": "11",
        }
    )

    assert config == SecurityQualificationConfig(
        model="model/security",
        agent_entrypoint="main.py",
        patch_max_chars=7,
        base_entrypoint_max_chars=8,
        submitted_entrypoint_max_chars=9,
        base_files_max_chars=10,
        submitted_files_max_chars=11,
    )


def test_parse_plain_json_clamps_scores_and_defaults_lists() -> None:
    result = parse_security_qualification(
        json.dumps(
            {
                "verdict": "pass",
                "overall_score": 200,
                "security_score": -5,
                "summary": "Clean.",
            }
        ),
        model="model/a",
    )

    assert result.verdict == "pass"
    assert result.overall_score == 100
    assert result.security_score == 0
    assert result.reasons == ()
    assert result.risks == ()
    assert result.required_changes == ()
    assert result.model == "model/a"


def test_parse_fenced_json() -> None:
    result = parse_security_qualification(
        f"```json\n{_payload(summary='Fenced.')}\n```"
    )

    assert result.summary == "Fenced."
    assert result.verdict == "pass"


def test_risk_normalization_uses_production_aliases() -> None:
    result = parse_security_qualification(
        _payload(
            risks=[
                "container-escape: tries namespace abuse",
                "credential-theft: reads a token",
                "c2: sends callbacks",
            ]
        )
    )

    assert security_risk_categories(result) == frozenset(
        {"sandbox-escape", "secret-theft", "network-exfiltration"}
    )
    assert security_failures(result) == [
        "security qualification reported risk category: network-exfiltration.",
        "security qualification reported risk category: sandbox-escape.",
        "security qualification reported risk category: secret-theft.",
    ]


def test_outcome_qualified_for_clean_pass() -> None:
    result = parse_security_qualification(_payload(verdict="pass", risks=[]))

    assert qualification_outcome(result) is QualificationOutcome.QUALIFIED


def test_outcome_needs_review_for_warning() -> None:
    result = parse_security_qualification(_payload(verdict="warn", risks=[]))

    assert qualification_outcome(result) is QualificationOutcome.NEEDS_REVIEW


def test_outcome_disqualified_for_explicit_failure() -> None:
    result = parse_security_qualification(_payload(verdict="fail", risks=[]))

    assert qualification_outcome(result) is QualificationOutcome.DISQUALIFIED
    assert security_failures(result) == [
        "security qualification returned verdict=fail."
    ]


def test_outcome_disqualified_for_security_risk_even_when_verdict_passes() -> None:
    result = parse_security_qualification(
        _payload(verdict="pass", risks=[{"category": "host-filesystem"}])
    )

    assert qualification_outcome(result) is QualificationOutcome.DISQUALIFIED
    assert security_risk_categories(result) == frozenset({"host-filesystem-access"})


async def test_async_qualification_wrapper_calls_client_once_and_records_model() -> (
    None
):
    client = FakeClient(_payload(summary="From client."), model="model/from-client")

    result = await qualify_submission_security(_input(), client=client)

    assert client.calls == 1
    assert client.prompt is not None
    assert client.prompt.system is not None
    assert result.summary == "From client."
    assert result.model == "model/from-client"


async def test_async_qualification_wrapper_propagates_client_errors() -> None:
    client = FakeClient(RuntimeError("upstream unavailable"))

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await qualify_submission_security(_input(), client=client)
