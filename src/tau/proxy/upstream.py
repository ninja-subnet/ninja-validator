"""Upstream transport for the proxy: a plain forwarder + a record/replay cache.

``HttpxUpstreamClient`` forwards a prepared request to the configured upstream. For
non-streaming chat completions it streams from upstream anyway (to time the first
token) and reassembles a single non-streamed body for the agent.
``CachedUpstreamClient`` wraps any upstream with disk record/replay.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .cache import CacheMissError, DiskCache, get_dict, json_sha256
from .parsing import (
    loads_json_bytes,
    loads_json_text,
    should_stream_chat_completion,
    sse_data_from_line,
)

# Default transport timeouts; the read timeout (how long one LLM call may take) is
# overridable via the proxy (TAU_PROXY_REQUEST_TIMEOUT_SECONDS -> LLMProxy).
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)


@dataclass(slots=True)
class UpstreamResponse:
    body: bytes
    payload: Any
    status: int
    headers: httpx.Headers
    first_token_latency_ms: int | None


class UpstreamClient(ABC):
    @abstractmethod
    def fetch(
        self,
        *,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        prepared_payload: Any,
        request_path: str,
        start: float,
    ) -> UpstreamResponse: ...


class HttpxUpstreamClient(UpstreamClient):
    def __init__(
        self,
        on_first_token: Callable[[], None] | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._on_first_token = on_first_token
        self._timeout = timeout or _UPSTREAM_TIMEOUT

    def fetch(
        self,
        *,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        prepared_payload: Any,
        request_path: str,
        start: float,
    ) -> UpstreamResponse:
        with httpx.Client(timeout=self._timeout) as client:
            if should_stream_chat_completion(command, request_path, prepared_payload):
                return self._fetch_streamed(
                    client=client,
                    url=url,
                    headers=headers,
                    payload=prepared_payload,
                    start=start,
                )
            response = client.request(command, url, headers=headers, content=body)
            return UpstreamResponse(
                body=response.content,
                payload=loads_json_bytes(response.content),
                status=response.status_code,
                headers=response.headers,
                first_token_latency_ms=None,
            )

    def _fetch_streamed(
        self,
        *,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        start: float,
    ) -> UpstreamResponse:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_options = dict(stream_payload.get("stream_options") or {})
        stream_options["include_usage"] = True
        stream_payload["stream_options"] = stream_options

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        role = "assistant"
        response_id: str | None = None
        response_model: str | None = None
        created: int | None = None
        finish_reason: str | None = None
        native_finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        first_token_latency_ms: int | None = None

        stream_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        stream_headers["Content-Type"] = "application/json"

        with client.stream("POST", url, headers=stream_headers, json=stream_payload) as response:
            if response.status_code >= 400:
                response_body = response.read()
                return UpstreamResponse(
                    body=response_body,
                    payload=loads_json_bytes(response_body),
                    status=response.status_code,
                    headers=response.headers,
                    first_token_latency_ms=None,
                )
            for line in response.iter_lines():
                data = sse_data_from_line(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                chunk = loads_json_text(data)
                if not isinstance(chunk, dict):
                    continue
                response_id = str(chunk.get("id") or response_id or "")
                response_model = str(chunk.get("model") or response_model or "")
                if chunk.get("created") is not None:
                    try:
                        created = int(chunk["created"])
                    except (TypeError, ValueError):
                        pass
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = get_dict(choice, "delta")
                    message = get_dict(choice, "message")
                    if delta.get("role"):
                        role = str(delta["role"])
                    elif message.get("role"):
                        role = str(message["role"])
                    token_seen = False
                    content = delta.get("content", message.get("content"))
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        token_seen = True
                    reasoning = (
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or message.get("reasoning")
                        or message.get("reasoning_content")
                    )
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                        token_seen = True
                    if delta.get("tool_calls") or message.get("tool_calls"):
                        token_seen = True
                    if token_seen and first_token_latency_ms is None:
                        first_token_latency_ms = int((time.monotonic() - start) * 1000)
                        if self._on_first_token is not None:
                            self._on_first_token()
                    finish_reason = choice.get("finish_reason") or finish_reason
                    native_finish_reason = (
                        choice.get("native_finish_reason") or native_finish_reason
                    )

        built_payload: dict[str, Any] = {
            "id": response_id or f"chatcmpl-proxy-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": response_model or str(payload.get("model") or ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": role, "content": "".join(content_parts)},
                    "finish_reason": finish_reason,
                },
            ],
            "usage": usage or {},
        }
        if native_finish_reason is not None:
            built_payload["choices"][0]["native_finish_reason"] = native_finish_reason
        if reasoning_parts:
            built_payload["choices"][0]["message"]["reasoning"] = "".join(reasoning_parts)
        built_body = json.dumps(built_payload).encode("utf-8")
        return UpstreamResponse(
            body=built_body,
            payload=built_payload,
            status=200,
            headers=httpx.Headers({"Content-Type": "application/json"}),
            first_token_latency_ms=first_token_latency_ms,
        )


class CachedUpstreamClient(UpstreamClient):
    """Disk record+replay around an inner client.

    With ``inner=None`` this is replay-only: a miss raises ``CacheMissError``. With
    an inner client a miss is fetched and recorded so the next call hits the cache.
    """

    def __init__(self, cache_dir: Path, inner: UpstreamClient | None = None) -> None:
        self._cache = DiskCache(cache_dir)
        self._inner = inner

    def fetch(
        self,
        *,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        prepared_payload: Any,
        request_path: str,
        start: float,
    ) -> UpstreamResponse:
        if isinstance(prepared_payload, dict):
            key = json_sha256({"path": request_path, "body": prepared_payload})
            entry = self._cache.read(key)
            if entry is not None:
                cached_payload = entry.get("body") or {}
                return UpstreamResponse(
                    body=json.dumps(cached_payload).encode("utf-8"),
                    payload=cached_payload,
                    status=int(entry.get("status_code", 200)),
                    headers=httpx.Headers({"Content-Type": "application/json"}),
                    first_token_latency_ms=None,
                )
            if self._inner is None:
                raise CacheMissError(key)
        elif self._inner is None:
            raise CacheMissError("(non-dict payload)")

        response = self._inner.fetch(
            command=command,
            url=url,
            headers=headers,
            body=body,
            prepared_payload=prepared_payload,
            request_path=request_path,
            start=start,
        )
        if (
            isinstance(prepared_payload, dict)
            and isinstance(response.payload, dict)
            and response.status < 400
        ):
            self._cache.write(key, {"status_code": response.status, "body": response.payload})
        return response
