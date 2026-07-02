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

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> UpstreamTarget:
        """Resolve the upstream from ``LLM_PROVIDER`` (default ``openrouter``).

        - ``openrouter`` → ``OPENROUTER_UPSTREAM_BASE_URL`` / ``OPENROUTER_BASE_URL``
          + ``OPENROUTER_API_KEY``
        - ``ninja`` (our local inference) → ``NINJA_INFERENCE_BASE_URL``
          + ``NINJA_INFERENCE_API_KEY``
        - ``custom`` → ``LLM_UPSTREAM_BASE_URL`` + ``LLM_UPSTREAM_API_KEY``

        Raises ``OSError`` if the selected provider's required vars are missing.
        """
        env = os.environ if environ is None else environ
        provider = (env.get("LLM_PROVIDER") or OPENROUTER).strip().lower()

        if provider == OPENROUTER:
            base = env.get("OPENROUTER_UPSTREAM_BASE_URL") or env.get("OPENROUTER_BASE_URL")
            key = env.get("OPENROUTER_API_KEY", "")
            if not key:
                raise OSError("OPENROUTER_API_KEY not set (LLM_PROVIDER=openrouter)")
            return cls(name=OPENROUTER, base_url=normalize_base_url(base), api_key=key)

        if provider == NINJA:
            base = env.get("NINJA_INFERENCE_BASE_URL")
            key = env.get("NINJA_INFERENCE_API_KEY", "")
            if not base:
                raise OSError("NINJA_INFERENCE_BASE_URL not set (LLM_PROVIDER=ninja)")
            return cls(name=NINJA, base_url=normalize_base_url(base), api_key=key)

        if provider == CUSTOM:
            base = env.get("LLM_UPSTREAM_BASE_URL")
            key = env.get("LLM_UPSTREAM_API_KEY", "")
            if not base:
                raise OSError("LLM_UPSTREAM_BASE_URL not set (LLM_PROVIDER=custom)")
            return cls(name=CUSTOM, base_url=normalize_base_url(base), api_key=key)

        raise OSError(
            f"unknown LLM_PROVIDER {provider!r} (expected openrouter | ninja | custom)"
        )
