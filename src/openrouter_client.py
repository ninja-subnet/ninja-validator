from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

import tau.utils
from tau.io.chat_completion import assistant_text_from_payload, empty_content_error
from tau.io.openrouter import CachedLLMClient, LLMClient, LLMRequest, normalize_base_url
from tau.io.upstream_request_policy import DEFAULT_RATE_LIMIT_RETRIES

log = logging.getLogger("swe-eval.openrouter_client")

_DEFAULT_MODEL = "google/gemini-3.1-flash-lite"


class HttpxLLMClient(LLMClient):
    """Calls OpenRouter directly via httpx."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete_text(self, request: LLMRequest, *, timeout: int) -> str:
        url, api_key, self_hosted = _route_request(request.model, self._api_key)
        payload: dict[str, Any] = {
            "model": _resolve_model(request.model),
            "messages": _build_messages(system_prompt=request.system_prompt, prompt=request.prompt),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.reasoning is not None:
            payload["reasoning"] = request.reasoning
        if request.cache_control is not None:
            payload["cache_control"] = request.cache_control
        if self_hosted:
            # Self-hosted Qwen endpoints default to thinking ON; disable it for
            # determinism + speed (override via SOLVER_CHAT_TEMPLATE_KWARGS).
            payload["chat_template_kwargs"] = _self_hosted_template_kwargs()
        else:
            # Provider routing is OpenRouter-only; a self-hosted vLLM endpoint
            # rejects/ignores it.
            provider = request.provider or _provider_preferences_from_env()
            if provider is not None:
                payload["provider"] = provider
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "swe-eval",
        }
        log.debug("Calling model=%s self_hosted=%s timeout=%ss", payload["model"], self_hosted, timeout)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(_no_choices_error(data))
        text = assistant_text_from_payload(data)
        if not text.strip():
            raise RuntimeError(empty_content_error(data))
        return text


def _build_client(api_key: str) -> LLMClient:
    """Build an LLMClient based on environment variables.

    LLM_REPLAY_DIR  — replay-only mode: serve from cache, raise CacheMissError on miss.
    LLM_CACHE_DIR   — record+replay mode: delegate to OpenRouter on miss and save the result.
    (unset)         — call OpenRouter directly with no caching.
    """
    from pathlib import Path

    replay_dir = os.environ.get("LLM_REPLAY_DIR")
    if replay_dir:
        return CachedLLMClient(Path(replay_dir), inner=None)
    cache_dir = os.environ.get("LLM_CACHE_DIR")
    if cache_dir:
        return CachedLLMClient(Path(cache_dir), inner=HttpxLLMClient(api_key))
    return HttpxLLMClient(api_key)


def complete_text(
    *,
    prompt: str | list[dict[str, Any]],
    model: str | None,
    timeout: int,
    openrouter_api_key: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    reasoning: dict[str, Any] | None = None,
    cache_control: dict[str, Any] | None = None,
    provider: dict[str, Any] | None = None,
    rate_limit_retries: int = 1,
) -> str:
    request = LLMRequest(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning=reasoning,
        cache_control=cache_control,
        provider=provider,
    )
    client = _build_client(openrouter_api_key)
    max_rate_limit_retries = max(1, int(rate_limit_retries))
    if max_rate_limit_retries <= 1:
        return client.complete_text(request, timeout=timeout)

    last_error: BaseException | None = None
    for rate_attempt in range(max_rate_limit_retries):
        try:
            return client.complete_text(request, timeout=timeout)
        except httpx.HTTPStatusError as exc:
            if not is_retryable_openrouter_rate_limit(exc):
                raise
            last_error = exc
            if rate_attempt + 1 >= max_rate_limit_retries:
                raise
            backoff = openrouter_rate_limit_backoff_seconds(exc.response, rate_attempt)
            log.info(
                "Retrying OpenRouter rate limit for model=%s (attempt %s/%s) after %.2fs",
                _resolve_model(model),
                rate_attempt + 2,
                max_rate_limit_retries,
                backoff,
            )
            time.sleep(backoff)
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenRouter rate limit retries exhausted without a response")


def is_retryable_openrouter_rate_limit(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code != 429:
        return False
    error_text = _openrouter_error_text(exc.response).lower()
    return (
        "rate limit" in error_text
        or "too many requests" in error_text
        or "high demand" in error_text
        or not error_text
    )


def openrouter_rate_limit_backoff_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(30.0, max(0.25, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, 2.0 * (1.5 ** attempt))


def resolve_rate_limit_retries(value: int | None) -> int:
    if value is not None:
        return max(1, int(value))
    return DEFAULT_RATE_LIMIT_RETRIES


def _openrouter_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text
    if not isinstance(payload, dict):
        return response.text
    error = tau.utils.get_dict(payload, "error")
    message = error.get("message")
    return str(message) if message else ""


def _openrouter_url() -> str:
    return (
        normalize_base_url(
            os.environ.get("OPENROUTER_UPSTREAM_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL"),
        )
        + "/v1/chat/completions"
    )


def _self_hosted_template_kwargs() -> dict[str, Any]:
    raw = os.environ.get("SOLVER_CHAT_TEMPLATE_KWARGS")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            log.warning("Ignoring invalid SOLVER_CHAT_TEMPLATE_KWARGS: %r", raw)
    return {"enable_thinking": False}


def _route_request(model: str | None, default_key: str) -> tuple[str, str, bool]:
    """Pick (chat-completions URL, api_key, is_self_hosted) for a model.

    The model named by SELF_HOSTED_MODEL routes to the self-hosted solver
    endpoint (SOLVER_UPSTREAM_BASE_URL + SOLVER_UPSTREAM_API_KEY); every other
    model (e.g. the glm-5.2 judges) goes to OpenRouter with the default key. This
    lets generator/eval share the self-hosted endpoint while judges use OpenRouter.
    """
    self_hosted_model = os.environ.get("SELF_HOSTED_MODEL")
    base = os.environ.get("SOLVER_UPSTREAM_BASE_URL")
    if self_hosted_model and base and model and _resolve_model(model) == _resolve_model(self_hosted_model):
        url = normalize_base_url(base) + "/v1/chat/completions"
        return url, (os.environ.get("SOLVER_UPSTREAM_API_KEY") or default_key), True
    return _openrouter_url(), default_key, False


def _resolve_model(model: str | None) -> str:
    if not model:
        return _DEFAULT_MODEL
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model


def _provider_preferences_from_env() -> dict[str, Any] | None:
    only_raw = os.environ.get("OPENROUTER_PROVIDER_ONLY") or os.environ.get("SOLVER_PROVIDER_ONLY")
    only = [part.strip() for part in (only_raw or "").split(",") if part.strip()]
    allow_fallbacks_raw = os.environ.get("OPENROUTER_PROVIDER_ALLOW_FALLBACKS")
    if allow_fallbacks_raw is None:
        allow_fallbacks_raw = os.environ.get("SOLVER_PROVIDER_ALLOW_FALLBACKS")
    provider: dict[str, Any] = {}
    if only:
        provider["only"] = only
    if allow_fallbacks_raw is not None:
        provider["allow_fallbacks"] = allow_fallbacks_raw.strip().lower() in {"1", "true", "yes", "on"}
    return provider or None


def _build_messages(
    *,
    system_prompt: str | None,
    prompt: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _no_choices_error(data: dict[str, Any]) -> str:
    error = tau.utils.get_dict(data, "error")
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
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
