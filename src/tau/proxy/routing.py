"""Smart upstream routing for multi-endpoint inference backends.

The provider-side prompt/KV cache lives inside one backend process, so routing every
request round-robin can destroy cache locality. This module keeps the balancing
decision at the solve/conversation level: choose one endpoint, keep the proxy sticky
to it, and remember prompt-prefix affinity for future similar solves.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .cache import json_sha256
from tau.openrouter.client import normalize_base_url

_PREFIX_MESSAGE_ROLES = frozenset({"system", "developer", "user"})
_MAX_PREFIX_CHARS = 16_000
INFRA_UPSTREAM_STATUSES = frozenset({401, 402, 403, 408, 429})
_DISABLED_UPSTREAMS_FILE_ENV = "TAU_SOLVER_DISABLED_UPSTREAMS_FILE"
_PERMANENT_DISABLE_FAILURES = 4

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DisabledUpstreamStore:
    """Newline-delimited disabled upstream list, normalized without a /v1 suffix."""

    path: Path

    def load(self) -> set[str]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return set()
        except OSError:
            log.exception("failed to read disabled upstreams file %s", self.path)
            return set()
        disabled: set[str] = set()
        for line in raw.splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                disabled.add(normalize_base_url(value))
        return disabled

    def add(self, base_url: str) -> None:
        disabled = self.load()
        normalized = normalize_base_url(base_url)
        if normalized in disabled:
            return
        disabled.add(normalized)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                "\n".join(sorted(disabled)) + "\n",
                encoding="utf-8",
            )
        except OSError:
            log.exception("failed to write disabled upstreams file %s", self.path)


def disabled_upstream_store_from_env(
    environ: Mapping[str, str] | None = None,
) -> DisabledUpstreamStore | None:
    env = os.environ if environ is None else environ
    raw = (env.get(_DISABLED_UPSTREAMS_FILE_ENV) or "").strip()
    return DisabledUpstreamStore(Path(raw)) if raw else None


def filter_disabled_upstream_urls(
    base_urls: tuple[str, ...],
    *,
    store: DisabledUpstreamStore | None,
) -> tuple[str, ...]:
    if store is None:
        return base_urls
    disabled = store.load()
    if not disabled:
        return base_urls
    return tuple(url for url in base_urls if normalize_base_url(url) not in disabled)


@dataclass(slots=True)
class _EndpointState:
    in_flight: int = 0
    failures: int = 0
    cooldown_until: float = 0.0


@dataclass(slots=True)
class _AffinityEntry:
    base_url: str
    expires_at: float


class SmartUpstreamRouter:
    """Thread-safe sticky router with endpoint health and prompt-prefix affinity."""

    def __init__(
        self,
        *,
        affinity_ttl_seconds: float = 60 * 60,
        cooldown_seconds: float = 60,
        max_affinities: int = 4096,
        affinity_load_slack: int = 1,
        disabled_store: DisabledUpstreamStore | None = None,
        permanent_disable_failures: int = _PERMANENT_DISABLE_FAILURES,
    ) -> None:
        self.affinity_ttl_seconds = affinity_ttl_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_affinities = max_affinities
        self.affinity_load_slack = affinity_load_slack
        self.disabled_store = disabled_store
        self.permanent_disable_failures = permanent_disable_failures
        self._lock = Lock()
        self._states: dict[str, _EndpointState] = {}
        self._affinities: dict[str, _AffinityEntry] = {}
        self._permanently_disabled: set[str] = set()
        self._cursor = 0

    def reset(self) -> None:
        """Clear mutable state. Intended for tests and controlled restarts."""
        with self._lock:
            self._states.clear()
            self._affinities.clear()
            self._permanently_disabled.clear()
            self.disabled_store = None
            self._cursor = 0

    def configure_disabled_store(self, store: DisabledUpstreamStore | None) -> None:
        with self._lock:
            self.disabled_store = store
            self._permanently_disabled = set()

    def acquire(
        self, base_urls: tuple[str, ...], affinity_key: str | None = None
    ) -> str:
        if not base_urls:
            raise ValueError("at least one upstream base URL is required")
        now = time.monotonic()
        with self._lock:
            self._expire_affinities_locked(now)
            enabled_urls = self._enabled_base_urls_locked(base_urls)
            candidates = [
                url for url in enabled_urls if self._state_for(url).cooldown_until <= now
            ] or enabled_urls
            preferred = self._preferred_locked(
                affinity_key=affinity_key,
                base_urls=base_urls,
                candidates=candidates,
                now=now,
            )
            if preferred is not None:
                selected = preferred
            else:
                selected = self._least_loaded_locked(candidates)
            self._state_for(selected).in_flight += 1
            return selected

    def remember_affinity(self, affinity_key: str | None, base_url: str) -> None:
        if not affinity_key:
            return
        now = time.monotonic()
        with self._lock:
            self._remember_affinity_locked(affinity_key, base_url, now)

    def release(self, base_url: str) -> None:
        with self._lock:
            state = self._state_for(base_url)
            state.in_flight = max(0, state.in_flight - 1)

    def record_result(
        self,
        base_url: str,
        *,
        status_code: int | None,
        error: str | None,
        base_urls: tuple[str, ...] | None = None,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            state = self._state_for(base_url)
            if is_upstream_infra_failure(status_code=status_code, error=error):
                state.failures += 1
                state.cooldown_until = now + (
                    self.cooldown_seconds * min(state.failures, 4)
                )
                if state.failures >= self.permanent_disable_failures:
                    self._disable_permanently_locked(base_url, base_urls)
            elif status_code is not None and status_code < 400 and error is None:
                state.failures = 0
                state.cooldown_until = 0.0

    def _preferred_locked(
        self,
        *,
        affinity_key: str | None,
        base_urls: tuple[str, ...],
        candidates: list[str],
        now: float,
    ) -> str | None:
        if not affinity_key:
            return None
        entry = self._affinities.get(affinity_key)
        if (
            entry is None
            or entry.expires_at <= now
            or entry.base_url not in base_urls
            or entry.base_url not in candidates
        ):
            return None
        min_in_flight = min(self._state_for(url).in_flight for url in candidates)
        preferred_state = self._state_for(entry.base_url)
        if preferred_state.in_flight <= min_in_flight + self.affinity_load_slack:
            entry.expires_at = now + self.affinity_ttl_seconds
            return entry.base_url
        return None

    def _least_loaded_locked(self, candidates: list[str]) -> str:
        min_in_flight = min(self._state_for(url).in_flight for url in candidates)
        least_loaded = [
            url for url in candidates if self._state_for(url).in_flight == min_in_flight
        ]
        selected = least_loaded[self._cursor % len(least_loaded)]
        self._cursor += 1
        return selected

    def _remember_affinity_locked(
        self, affinity_key: str, base_url: str, now: float
    ) -> None:
        if len(self._affinities) >= self.max_affinities:
            oldest_key = min(
                self._affinities,
                key=lambda key: self._affinities[key].expires_at,
            )
            self._affinities.pop(oldest_key, None)
        self._affinities[affinity_key] = _AffinityEntry(
            base_url=base_url,
            expires_at=now + self.affinity_ttl_seconds,
        )

    def _expire_affinities_locked(self, now: float) -> None:
        expired = [
            key for key, entry in self._affinities.items() if entry.expires_at <= now
        ]
        for key in expired:
            self._affinities.pop(key, None)

    def _enabled_base_urls_locked(self, base_urls: tuple[str, ...]) -> list[str]:
        disabled = self._disabled_locked()
        enabled = [url for url in base_urls if url not in disabled]
        return enabled or list(base_urls)

    def _disabled_locked(self) -> set[str]:
        disabled = set(self._permanently_disabled)
        if self.disabled_store is not None:
            disabled.update(self.disabled_store.load())
        return disabled

    def _disable_permanently_locked(
        self, base_url: str, base_urls: tuple[str, ...] | None
    ) -> None:
        if self.disabled_store is None:
            return
        configured_urls = base_urls or tuple(self._states)
        currently_enabled = [
            url for url in configured_urls if url not in self._disabled_locked()
        ]
        if base_url in currently_enabled and len(currently_enabled) <= 1:
            log.warning(
                "not permanently disabling upstream %s; it is the last enabled endpoint",
                base_url,
            )
            return
        self._permanently_disabled.add(base_url)
        self._affinities = {
            key: entry
            for key, entry in self._affinities.items()
            if entry.base_url != base_url
        }
        self.disabled_store.add(base_url)
        log.error(
            "permanently disabled upstream %s after %d infra failures; "
            "remove it from %s and restart to re-enable",
            base_url,
            self.permanent_disable_failures,
            self.disabled_store.path,
        )

    def _state_for(self, base_url: str) -> _EndpointState:
        state = self._states.get(base_url)
        if state is None:
            state = _EndpointState()
            self._states[base_url] = state
        return state


def request_affinity_key(payload: Any, request_path: str) -> str | None:
    """Return a stable prompt-prefix key for cache-aware routing."""
    if not isinstance(payload, dict):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    prefix_messages = _prefix_messages(messages)
    if not prefix_messages:
        return None
    return json_sha256(
        {
            "path": request_path,
            "model": payload.get("model"),
            "tools": payload.get("tools"),
            "response_format": payload.get("response_format"),
            "messages": prefix_messages,
        }
    )


def _prefix_messages(messages: list[Any]) -> list[Any]:
    prefix: list[Any] = []
    char_budget = _MAX_PREFIX_CHARS
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in _PREFIX_MESSAGE_ROLES:
            break
        content = _truncate_content(message.get("content"), char_budget)
        compact = {"role": role, "content": content}
        prefix.append(compact)
        char_budget -= len(str(content))
        if role == "user" or char_budget <= 0:
            break
    return prefix


def _truncate_content(value: Any, char_budget: int) -> Any:
    if isinstance(value, str) and len(value) > char_budget:
        return value[:char_budget]
    return value


def is_upstream_infra_failure(*, status_code: int | None, error: str | None) -> bool:
    if status_code is None:
        return error is not None
    return status_code >= 500 or status_code in INFRA_UPSTREAM_STATUSES


SMART_UPSTREAM_ROUTER = SmartUpstreamRouter()
