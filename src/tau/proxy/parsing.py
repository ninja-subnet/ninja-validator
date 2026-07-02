"""Pure JSON / SSE / OpenAI-payload parsing helpers used across the proxy.

No state, no I/O — extracted so both the upstream client (streaming) and the
server (request-record building, budget estimation) share one copy. Ported from
the legacy ``openrouter_proxy.py`` module-level helpers.
"""

from __future__ import annotations

import json
from typing import Any

_ESTIMATED_CHARS_PER_TOKEN = 3
_ESTIMATED_MESSAGE_OVERHEAD_TOKENS = 8
_ESTIMATED_TOOL_OVERHEAD_TOKENS = 24


def loads_json_bytes(raw_body: bytes | None) -> Any:
    if not raw_body:
        return None
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def loads_json_text(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def sse_data_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def should_stream_chat_completion(command: str, request_path: str, payload: Any) -> bool:
    """Whether to stream from upstream to measure first-token latency.

    Miner agents get a normal non-streaming response; the proxy streams from
    upstream only to time the first token, then returns a single assembled body.
    """
    if command != "POST" or request_path != "/v1/chat/completions":
        return False
    if not isinstance(payload, dict):
        return False
    return not bool(payload.get("stream"))


def extract_request_model(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return str(model) if isinstance(model, str) else None


def extract_response_model(payload: Any) -> str | None:
    return extract_request_model(payload)


def extract_generation_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    generation_id = payload.get("id")
    return str(generation_id) if isinstance(generation_id, str) else None


def _usage(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _prompt_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _completion_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def extract_prompt_tokens(payload: Any) -> int | None:
    usage = _usage(payload)
    return _prompt_tokens_from_usage(usage) if usage else None


def extract_completion_tokens(payload: Any) -> int | None:
    usage = _usage(payload)
    return _completion_tokens_from_usage(usage) if usage else None


def extract_total_tokens(payload: Any) -> int | None:
    usage = _usage(payload)
    if not usage:
        return None
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int):
        return total_tokens
    prompt_tokens = _prompt_tokens_from_usage(usage)
    completion_tokens = _completion_tokens_from_usage(usage)
    if prompt_tokens is not None or completion_tokens is not None:
        return int(prompt_tokens or 0) + int(completion_tokens or 0)
    return None


def extract_cached_tokens(payload: Any) -> int | None:
    usage = _usage(payload)
    if not usage:
        return None
    cache_read = usage.get("cache_read_input_tokens")
    if isinstance(cache_read, int):
        return cache_read
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    return value if isinstance(value, int) else None


def extract_cache_write_tokens(payload: Any) -> int | None:
    usage = _usage(payload)
    if not usage:
        return None
    cache_creation = usage.get("cache_creation_input_tokens")
    if isinstance(cache_creation, int):
        return cache_creation
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cache_write_tokens")
    return value if isinstance(value, int) else None


def extract_reasoning_tokens(payload: Any) -> int | None:
    usage = _usage(payload)
    if not usage:
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def extract_cost(payload: Any) -> float | None:
    usage = _usage(payload)
    if not usage:
        return None
    value = usage.get("cost")
    return float(value) if isinstance(value, (int, float)) else None


def extract_response_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        return str(message) if message else None
    if isinstance(error, str):
        return error
    message = payload.get("message")
    return str(message) if message else None


def request_payload_has_messages(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    messages = payload.get("messages")
    return isinstance(messages, list) and len(messages) > 0


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    message_count = 0
    total_chars = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        message_count = len(messages)
        total_chars += sum(_estimate_content_chars(message) for message in messages)
    tools = payload.get("tools")
    if isinstance(tools, list):
        total_chars += sum(len(json.dumps(tool, sort_keys=True)) for tool in tools)
    response_format = payload.get("response_format")
    if response_format is not None:
        total_chars += len(json.dumps(response_format, sort_keys=True))
    prompt_tokens = (total_chars + (_ESTIMATED_CHARS_PER_TOKEN - 1)) // _ESTIMATED_CHARS_PER_TOKEN
    prompt_tokens += message_count * _ESTIMATED_MESSAGE_OVERHEAD_TOKENS
    if isinstance(tools, list):
        prompt_tokens += len(tools) * _ESTIMATED_TOOL_OVERHEAD_TOKENS
    return max(prompt_tokens, 1)


def _estimate_content_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (int, float, bool)):
        return len(str(value))
    if isinstance(value, dict):
        return sum(len(str(key)) + _estimate_content_chars(item) for key, item in value.items())
    if isinstance(value, list):
        return sum(_estimate_content_chars(item) for item in value)
    return len(str(value))


def extract_requested_max_output_tokens(payload: dict[str, Any]) -> int | None:
    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return max(value, 0)
    return None


def set_requested_max_output_tokens(payload: dict[str, Any], value: int) -> None:
    clamped = max(value, 0)
    existing_max_tokens = payload.get("max_tokens")
    payload["max_tokens"] = (
        min(existing_max_tokens, clamped) if isinstance(existing_max_tokens, int) else clamped
    )
    existing_max_completion = payload.get("max_completion_tokens")
    if isinstance(existing_max_completion, int):
        payload["max_completion_tokens"] = min(existing_max_completion, clamped)
