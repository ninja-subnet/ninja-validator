"""Async OpenRouter LLM client — httpx calls to the chat-completions API.

Clean-room port of src/openrouter_client.py (HttpxLLMClient + complete_text) plus
the LLMRequest shape and normalize_base_url from src/tau/io/openrouter.py and
get_dict from src/tau/utils.py — no imports from the legacy `tau` package.

Async so one judge worker can run many LLM calls concurrently (asyncio.gather)
and reuse connections via a persistent AsyncClient. Caching (replay/record) is
intentionally NOT ported here; it layers on as a decorator if/when needed.

Env:
  OPENROUTER_UPSTREAM_BASE_URL / OPENROUTER_BASE_URL — override the base URL.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


class RenderablePrompt(Protocol):
    """A prompt that can render itself flat or as multi-part content.

    The transport owns this interface (dependency inversion): callers hand over a
    semantic prompt and never encode wire format, and the concrete client decides
    which rendering a given model gets. judging.JudgePrompt satisfies it
    structurally; TextPrompt wraps a bare string.
    """

    @property
    def system(self) -> str | None: ...
    def as_text(self) -> str: ...
    def as_content(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class TextPrompt:
    """Minimal RenderablePrompt for a bare string (both renderings are the text)."""

    text: str
    system: str | None = None

    def as_text(self) -> str:
        return self.text

    def as_content(self) -> list[dict[str, Any]]:
        return [{"type": "text", "text": self.text}]


class LLMClient(Protocol):
    """The minimal async transport interface the judge depends on.

    A client is a configured model endpoint: it owns the model and sampling
    settings, so callers pass only a prompt. OpenRouterClient satisfies it
    structurally; tests pass any object with a matching `model` and async
    `complete_text` — no subclassing, no bridge.
    """

    @property
    def model(self) -> str: ...
    async def complete_text(self, prompt: RenderablePrompt) -> str: ...


class OpenRouterClient:
    """Async client for OpenRouter's chat-completions endpoint.

    Holds a persistent ``httpx.AsyncClient`` so concurrent calls pool connections.
    Use as an async context manager, or call ``aclose()`` when done:

        async with OpenRouterClient(api_key, model="anthropic/claude-sonnet-4.6") as client:
            text = await client.complete_text(prompt)

    Model and sampling settings are fixed per client. Pass your own ``client`` to
    share a connection pool across several configured clients; then closing it is
    your job.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        reasoning: dict[str, Any] | None = None,
        timeout: int = 120,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._reasoning = reasoning
        self._timeout = timeout
        self._client = client if client is not None else httpx.AsyncClient()
        self._owns_client = client is None

    @property
    def model(self) -> str:
        return _resolve_model(self._model)

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete_text(self, prompt: RenderablePrompt) -> str:
        content = (
            prompt.as_content()
            if _supports_structured(self._model)
            else prompt.as_text()
        )
        payload: dict[str, Any] = {
            "model": _resolve_model(self._model),
            "messages": _build_messages(system=prompt.system, content=content),
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._reasoning is not None:
            payload["reasoning"] = self._reasoning

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "tau2",
        }
        log.debug("OpenRouter call model=%s timeout=%ss", payload["model"], self._timeout)
        response = await self._client.post(
            _openrouter_url(), headers=headers, json=payload, timeout=self._timeout
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(_no_choices_error(data))
        message = choices[0].get("message") or {}
        text = _extract_text(message.get("content"))
        if not text.strip():
            raise RuntimeError(_empty_content_error(data))
        return text


async def complete_text(
    *,
    prompt: RenderablePrompt | str,
    model: str | None,
    timeout: int,
    openrouter_api_key: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    reasoning: dict[str, Any] | None = None,
) -> str:
    """One-shot async completion (creates and closes its own client).

    For many calls, instantiate one OpenRouterClient and reuse it instead. A bare
    string is wrapped as a TextPrompt; the client picks the wire format.
    """
    renderable = prompt if not isinstance(prompt, str) else TextPrompt(prompt)
    async with OpenRouterClient(
        openrouter_api_key,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
        timeout=timeout,
    ) as client:
        return await client.complete_text(renderable)


def normalize_base_url(raw: str | None) -> str:
    """Normalize an OpenRouter base URL to the form WITHOUT a /v1 suffix.

    Accepts a bare base, base + /v1, or base + /v1/chat/completions.
    """
    base = (raw or "https://openrouter.ai/api").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base[: -len("/v1/chat/completions")]
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def _openrouter_url() -> str:
    return (
        normalize_base_url(
            os.environ.get("OPENROUTER_UPSTREAM_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL"),
        )
        + "/v1/chat/completions"
    )


def _resolve_model(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model


def _supports_structured(model: str | None) -> bool:
    """Whether a model accepts multi-part content + cache_control vs flat text.

    Transport-private capability rule: anthropic-family models support the
    structured, cacheable encoding; everything else gets flat text.
    """
    return _resolve_model(model).startswith("anthropic/")


def _build_messages(
    *,
    system: str | None,
    content: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _get_dict(data: dict[Any, Any], key: Any) -> dict[Any, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _no_choices_error(data: dict[str, Any]) -> str:
    error = _get_dict(data, "error")
    return (
        "OpenRouter returned no choices "
        f"(error_code={error.get('code')!r}, "
        f"error_message={_truncate_error_text(error.get('message'))!r}, "
        f"response_keys={sorted(data.keys())})"
    )


def _truncate_error_text(raw: Any, limit: int = 240) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _empty_content_error(data: dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = _get_dict(data, "usage")
    completion_details = _get_dict(usage, "completion_tokens_details")
    return (
        "OpenRouter returned empty content "
        f"(finish_reason={choice.get('finish_reason')!r}, "
        f"native_finish_reason={choice.get('native_finish_reason')!r}, "
        f"message_keys={sorted(message.keys())}, "
        f"completion_tokens={usage.get('completion_tokens')!r}, "
        f"reasoning_tokens={completion_details.get('reasoning_tokens')!r})"
    )
