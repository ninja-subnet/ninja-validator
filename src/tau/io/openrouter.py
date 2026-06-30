from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau.utils import DiskCache, json_sha256

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRequest:
    prompt: str | list[dict[str, Any]]
    model: str | None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    reasoning: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None

    def cache_key(self) -> str:
        return json_sha256({
            "prompt": self.prompt,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "reasoning": self.reasoning,
            "cache_control": self.cache_control,
            "provider": self.provider,
        })


class LLMClient(ABC):
    @abstractmethod
    def complete_text(self, request: LLMRequest, *, timeout: int) -> str: ...


class CacheMissError(RuntimeError):
    def __init__(self, key: str) -> None:
        super().__init__(f"LLM cache miss for key {key!r} and no inner client configured")
        self.key = key


class CachedLLMClient(LLMClient):
    """Replays cached responses from disk; optionally delegates to an inner client on miss.

    Each cached response is stored as a JSON file under cache_dir, named by
    LLMRequest.cache_key() (SHA-256 of request content; timeout excluded).

    With inner=None the client operates in pure replay mode and raises
    CacheMissError on a miss. With an inner client it records the response to
    disk on miss so the next call is served from cache.
    """

    def __init__(self, cache_dir: Path, inner: LLMClient | None = None) -> None:
        self._cache = DiskCache(cache_dir)
        self._inner = inner

    def complete_text(self, request: LLMRequest, *, timeout: int) -> str:
        key = request.cache_key()
        entry = self._cache.read(key)
        if entry is not None:
            log.debug("LLM cache hit key=%s", key)
            return str(entry["response"])

        if self._inner is None:
            raise CacheMissError(key)

        log.debug("LLM cache miss key=%s — delegating to inner client", key)
        result = self._inner.complete_text(request, timeout=timeout)
        self._cache.write(key, {"response": result})
        return result


class MockLLMClient(LLMClient):
    """Returns a fixed or computed response for every call. Useful in tests."""

    def __init__(self, response: str | Callable[[LLMRequest], str] = "") -> None:
        self._response = response

    def complete_text(self, request: LLMRequest, *, timeout: int) -> str:
        if callable(self._response):
            return self._response(request)
        return self._response


def normalize_base_url(raw: str | None) -> str:
    """Normalize an OpenRouter base URL to the form without a /v1 suffix.

    Accepts any of: bare base, base + /v1, base + /v1/chat/completions.
    The proxy appends handler.path (e.g. /v1/chat/completions) directly;
    the client appends /v1/chat/completions explicitly.
    """
    base = (raw or "https://openrouter.ai/api").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base[: -len("/v1/chat/completions")]
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base
