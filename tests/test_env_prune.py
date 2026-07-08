from pathlib import Path

from tau.proxy.env_prune import disabled_file_from_env_text, prune_env_text


def test_prune_env_text_removes_disabled_url_from_upstream_list() -> None:
    result = prune_env_text(
        (
            "LLM_PROVIDER=custom\n"
            "LLM_UPSTREAM_BASE_URLS=http://gpu1:8000/v1,http://gpu2:8000/v1 # prod\n"
        ),
        {"http://gpu2:8000"},
    )

    assert result.changed is True
    assert result.removed == {"LLM_UPSTREAM_BASE_URLS": ["http://gpu2:8000/v1"]}
    assert (
        result.text
        == "LLM_PROVIDER=custom\n"
        "LLM_UPSTREAM_BASE_URLS=http://gpu1:8000/v1 # prod\n"
    )


def test_prune_env_text_preserves_quotes_and_comments() -> None:
    result = prune_env_text(
        'NINJA_INFERENCE_BASE_URLS="http://gpu1:8000/v1,http://gpu2:8000/v1" # prod\n',
        {"http://gpu1:8000/v1"},
    )

    assert (
        result.text
        == 'NINJA_INFERENCE_BASE_URLS="http://gpu2:8000/v1" # prod\n'
    )


def test_prune_env_text_does_not_remove_last_endpoint() -> None:
    original = "LLM_UPSTREAM_BASE_URLS=http://gpu1:8000/v1\n"
    result = prune_env_text(original, {"http://gpu1:8000"})

    assert result.changed is False
    assert result.text == original
    assert result.skipped_all == {"LLM_UPSTREAM_BASE_URLS": ["http://gpu1:8000/v1"]}


def test_disabled_file_from_env_text_reads_path() -> None:
    path = disabled_file_from_env_text(
        'TAU_SOLVER_DISABLED_UPSTREAMS_FILE="/var/lib/tau/disabled-upstreams.txt"\n'
    )

    assert path == Path("/var/lib/tau/disabled-upstreams.txt")
