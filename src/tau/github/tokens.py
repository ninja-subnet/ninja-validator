"""Thread-safe round-robin over GitHub PATs.

Clean-room port of the monolith's ``GitHubTokenRotator``, hardened for a
long-running worker. Every token carries a cooldown; ``get_token`` skips a token
while it is cooling and, when *all* tokens are cooling, waits for the earliest to
free up (the wait state) instead of falling back to anonymous requests.

Cooldowns are driven by what GitHub actually reports: a rate-limited token sleeps
until its reset (``x-ratelimit-reset`` / ``Retry-After``), and a token that returns
a 401 sleeps a short while in case the 401 was a transient "bad credentials" blip
(GitHub does this under load). Tokens are NEVER disabled permanently -- a genuinely
dead token simply keeps re-cooling, so a real auth problem stays visible without
crippling the worker for the rest of its life. ``from_env`` reads ``GITHUB_TOKENS``
(comma-separated) or ``GITHUB_TOKEN``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

# Fallback cooldowns (seconds) for when GitHub does not tell us how long to wait.
_RATE_LIMIT_COOLDOWN = 60.0
# A 401 is treated as transient: cool the token down briefly rather than disabling
# it for the process lifetime. If the token really is dead it just re-cools.
_UNAUTHORIZED_COOLDOWN = 120.0


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]


class GitHubTokenRotator:
    """Round-robin over GitHub PATs with cooldown-based rate-limit / 401 handling."""

    def __init__(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("GitHubTokenRotator requires at least one token")
        self._tokens = list(tokens)
        self._index = 0
        self._lock = threading.Lock()
        # token index -> monotonic time it becomes usable again.
        self._cooldowns: dict[int, float] = {}
        log.info("Token rotator initialised with %d token(s)", len(self._tokens))

    @classmethod
    def from_env(cls) -> GitHubTokenRotator | None:
        """Build a rotator from ``GITHUB_TOKENS`` or ``GITHUB_TOKEN``; None if unset."""
        raw = os.environ.get("GITHUB_TOKENS", "")
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens:
            single = os.environ.get("GITHUB_TOKEN")
            if single and single.strip():
                tokens = [single.strip()]
        return cls(tokens) if tokens else None

    @property
    def size(self) -> int:
        return len(self._tokens)

    @property
    def available_count(self) -> int:
        """How many tokens are not currently on cooldown."""
        now = time.monotonic()
        with self._lock:
            return sum(
                1 for i in range(len(self._tokens)) if self._cooldowns.get(i, 0.0) <= now
            )

    async def get_token(self) -> str:
        """Return the next token that is not on cooldown.

        If every token is cooling, sleep until the earliest one frees up (yielding
        to the event loop, not blocking it), then retry -- so a fully rate-limited
        or transiently-unauthorized fleet self-heals rather than going anonymous.
        """
        while True:
            with self._lock:
                n = len(self._tokens)
                now = time.monotonic()
                for _ in range(n):
                    idx = self._index % n
                    self._index += 1
                    ready_at = self._cooldowns.get(idx)
                    if ready_at is None or ready_at <= now:
                        self._cooldowns.pop(idx, None)  # cooldown elapsed
                        return self._tokens[idx]
                # Every token is cooling: wait for the earliest to free up.
                wait = max(0.0, min(self._cooldowns.values()) - now)
            log.warning(
                "All %d GitHub token(s) cooling down; sleeping %.0fs for the next to free up",
                n,
                wait,
            )
            await asyncio.sleep(wait)

    def mark_rate_limited(self, token: str, *, cooldown_seconds: float | None = None) -> None:
        """Cool *token* down after a rate-limit response.

        *cooldown_seconds* should come straight from the response headers
        (``Retry-After`` or ``x-ratelimit-reset``); when GitHub did not say, falls
        back to a fixed default.
        """
        seconds = (
            cooldown_seconds
            if cooldown_seconds is not None and cooldown_seconds > 0
            else _RATE_LIMIT_COOLDOWN
        )
        with self._lock:
            try:
                idx = self._tokens.index(token)
            except ValueError:
                return
            self._cooldowns[idx] = time.monotonic() + seconds
        log.info("GitHub token #%d rate-limited; cooling down %.0fs", idx + 1, seconds)

    def mark_unauthorized(self, token: str) -> None:
        """Cool *token* down briefly after a 401, treating it as a transient blip.

        Deliberately NOT a permanent disable: GitHub returns spurious 401s under
        load, and a valid token would otherwise be lost for the worker's lifetime.
        """
        with self._lock:
            try:
                idx = self._tokens.index(token)
            except ValueError:
                return
            self._cooldowns[idx] = time.monotonic() + _UNAUTHORIZED_COOLDOWN
            fingerprint = _token_fingerprint(token)
        log.warning(
            "GitHub token #%d (%s) returned HTTP 401; cooling down %.0fs (transient?), "
            "not disabling",
            idx + 1,
            fingerprint,
            _UNAUTHORIZED_COOLDOWN,
        )
