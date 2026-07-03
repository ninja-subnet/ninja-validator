"""Pluggable upstream selection for the LLM proxy.

The only thing that varies between providers is *where to forward* and *which key
to inject* — both OpenRouter and our own backend speak the same OpenAI-compatible
API (``/v1/chat/completions``, ``/v1/messages``). So provider choice is one small
config object, switched by the ``LLM_PROVIDER`` env var, not a code branch.

The injected key never leaves the host: the proxy adds it as the upstream
``Authorization`` while the sandboxed agent only ever sees the per-solve token.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Reuse the one shared helper from the async client (no /v1 suffix on the base).
from tau.openrouter.client import normalize_base_url

OPENROUTER = "openrouter"
NINJA = "ninja"
CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class UpstreamTarget:
    """Where the proxy forwards and the key it injects (kept off the sandbox)."""

    name: str  # "openrouter" | "ninja" | "custom" — for logging/telemetry
    base_url: str  # normalized, no /v1 suffix; the proxy appends the request path
    api_key: str  # injected as `Authorization: Bearer ...`
    base_urls: tuple[str, ...] = ()  # all normalized upstream bases, round-robined

    def __post_init__(self) -> None:
        raw_urls = self.base_urls or _split_urls(self.base_url)
        base_urls = _normalize_base_urls(raw_urls)
        if not base_urls:
            raise ValueError("at least one upstream base URL is required")
        object.__setattr__(self, "base_urls", base_urls)
        object.__setattr__(self, "base_url", base_urls[0])

    @property
    def endpoint_count(self) -> int:
        return len(self.base_urls)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> UpstreamTarget:
        """Resolve the upstream from ``LLM_PROVIDER`` (default ``openrouter``).

        - ``openrouter`` → ``OPENROUTER_UPSTREAM_BASE_URLS`` /
          ``OPENROUTER_UPSTREAM_BASE_URL`` / ``OPENROUTER_BASE_URL`` +
          ``OPENROUTER_API_KEY``
        - ``ninja`` (our local inference) → ``NINJA_INFERENCE_BASE_URLS`` /
          ``NINJA_INFERENCE_BASE_URL``
          + ``NINJA_INFERENCE_API_KEY``
        - ``custom`` → ``LLM_UPSTREAM_BASE_URLS`` / ``LLM_UPSTREAM_BASE_URL`` +
          ``LLM_UPSTREAM_API_KEY``

        Raises ``OSError`` if the selected provider's required vars are missing.
        """
        env = os.environ if environ is None else environ
        provider = (env.get("LLM_PROVIDER") or OPENROUTER).strip().lower()

        if provider == OPENROUTER:
            base_urls = _base_urls_from_env(
                env,
                "OPENROUTER_UPSTREAM_BASE_URLS",
                "OPENROUTER_UPSTREAM_BASE_URL",
                "OPENROUTER_BASE_URL",
            ) or (normalize_base_url(None),)
            key = env.get("OPENROUTER_API_KEY", "")
            if not key:
                raise OSError("OPENROUTER_API_KEY not set (LLM_PROVIDER=openrouter)")
            return cls(
                name=OPENROUTER, base_url=base_urls[0], api_key=key, base_urls=base_urls
            )

        if provider == NINJA:
            base_urls = _base_urls_from_env(
                env, "NINJA_INFERENCE_BASE_URLS", "NINJA_INFERENCE_BASE_URL"
            )
            key = env.get("NINJA_INFERENCE_API_KEY", "")
            if not base_urls:
                raise OSError("NINJA_INFERENCE_BASE_URL not set (LLM_PROVIDER=ninja)")
            return cls(name=NINJA, base_url=base_urls[0], api_key=key, base_urls=base_urls)

        if provider == CUSTOM:
            base_urls = _base_urls_from_env(
                env, "LLM_UPSTREAM_BASE_URLS", "LLM_UPSTREAM_BASE_URL"
            )
            key = env.get("LLM_UPSTREAM_API_KEY", "")
            if not base_urls:
                raise OSError("LLM_UPSTREAM_BASE_URL not set (LLM_PROVIDER=custom)")
            return cls(name=CUSTOM, base_url=base_urls[0], api_key=key, base_urls=base_urls)

        raise OSError(
            f"unknown LLM_PROVIDER {provider!r} (expected openrouter | ninja | custom)"
        )


def _split_urls(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _base_urls_from_env(
    env: Mapping[str, str], plural_key: str, *fallback_keys: str
) -> tuple[str, ...]:
    urls = _split_urls(env.get(plural_key))
    if urls:
        return _normalize_base_urls(urls)
    for key in fallback_keys:
        urls = _split_urls(env.get(key))
        if urls:
            return _normalize_base_urls(urls)
    return ()


def _normalize_base_urls(raw_urls: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_url in raw_urls:
        url = normalize_base_url(raw_url)
        if url not in normalized:
            normalized.append(url)
    return tuple(normalized)
