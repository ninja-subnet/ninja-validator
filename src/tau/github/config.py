"""Tunable configuration for commit sampling.

``GitHubConfig`` gathers every knob the sampler reads into one immutable object
with literal defaults. Importing this module touches nothing in the environment;
call :meth:`GitHubConfig.from_env` explicitly to fold in ``TAU_GITHUB_*`` overrides.
Invariants are checked in ``__post_init__`` so a bad config fails at construction
rather than deep inside a sampling loop.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tau.utils.env import env_float, env_int, env_str


@dataclass(frozen=True, slots=True)
class GitHubConfig:
    """Sampling knobs. All defaults match the legacy validator's tuned values."""

    # HTTP
    http_timeout: float = 30.0

    # Random historical-day commit search. The /events firehose only exposes the
    # last few minutes of activity, so workers keep resampling the same just-pushed
    # HEADs -> duplicate tasks. Sampling a commit from a random day in
    # [min_days, max_days] back via the search API spreads tasks across dates/repos
    # and sharply cuts dupes.
    history_sample_min_days: int = 30
    history_sample_max_days: int = 1095
    # Per-page size for commit search; search results are capped at 1000 total
    # (page * per_page), so the random page is bounded accordingly.
    search_per_page: int = 100
    search_query_suffix: str = "merge:false"

    # Search buffer: refill once it drops below the low watermark; cap its size at
    # twice the high watermark so a burst of hits cannot grow it without bound.
    buffer_low_watermark: int = 50
    buffer_high_watermark: int = 500
    search_fail_cooldown_seconds: float = 10.0

    # Recent-events firehose (fallback when search is unavailable / rate limited).
    event_pages_to_fetch: int = 4
    recent_events_cache_ttl_seconds: float = 60.0

    # Quality gate: accept only modification-heavy code commits.
    min_code_changed_lines: int = 100
    min_code_files: int = 1

    # Upper bound on the in-process reject cache; cleared wholesale on overflow so
    # a long-running sampler cannot leak memory (re-encountering a bad commit just
    # re-screens it once).
    reject_cache_max_size: int = 10_000

    def __post_init__(self) -> None:
        if self.http_timeout <= 0:
            raise ValueError("http_timeout must be positive")
        if not 0 <= self.history_sample_min_days <= self.history_sample_max_days:
            raise ValueError(
                "require 0 <= history_sample_min_days <= history_sample_max_days"
            )
        if not 1 <= self.search_per_page <= 100:
            raise ValueError("search_per_page must be in [1, 100]")
        if not 0 <= self.buffer_low_watermark <= self.buffer_high_watermark:
            raise ValueError(
                "require 0 <= buffer_low_watermark <= buffer_high_watermark"
            )
        if self.search_fail_cooldown_seconds < 0:
            raise ValueError("search_fail_cooldown_seconds must be non-negative")
        if self.event_pages_to_fetch < 1:
            raise ValueError("event_pages_to_fetch must be >= 1")
        if self.recent_events_cache_ttl_seconds < 0:
            raise ValueError("recent_events_cache_ttl_seconds must be non-negative")
        if self.min_code_files < 1:
            raise ValueError("min_code_files must be >= 1")
        if self.min_code_changed_lines < 0:
            raise ValueError("min_code_changed_lines must be non-negative")
        if self.reject_cache_max_size < 1:
            raise ValueError("reject_cache_max_size must be >= 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GitHubConfig:
        """Build a config from ``TAU_GITHUB_*`` env vars, falling back to defaults.

        Pass *environ* to read from a mapping other than ``os.environ`` (tests).
        """
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            http_timeout=env_float(env, "TAU_GITHUB_HTTP_TIMEOUT", d.http_timeout),
            history_sample_min_days=env_int(
                env, "TAU_GITHUB_HISTORY_SAMPLE_MIN_DAYS", d.history_sample_min_days
            ),
            history_sample_max_days=env_int(
                env, "TAU_GITHUB_HISTORY_SAMPLE_MAX_DAYS", d.history_sample_max_days
            ),
            search_per_page=env_int(env, "TAU_GITHUB_SEARCH_PER_PAGE", d.search_per_page),
            search_query_suffix=env_str(
                env, "TAU_GITHUB_SEARCH_QUERY_SUFFIX", d.search_query_suffix
            ),
            buffer_low_watermark=env_int(
                env, "TAU_GITHUB_BUFFER_LOW_WATERMARK", d.buffer_low_watermark
            ),
            buffer_high_watermark=env_int(
                env, "TAU_GITHUB_BUFFER_HIGH_WATERMARK", d.buffer_high_watermark
            ),
            search_fail_cooldown_seconds=env_float(
                env, "TAU_GITHUB_SEARCH_FAIL_COOLDOWN_SECONDS", d.search_fail_cooldown_seconds
            ),
            event_pages_to_fetch=env_int(
                env, "TAU_GITHUB_EVENT_PAGES", d.event_pages_to_fetch
            ),
            recent_events_cache_ttl_seconds=env_float(
                env,
                "TAU_GITHUB_RECENT_EVENTS_CACHE_TTL_SECONDS",
                d.recent_events_cache_ttl_seconds,
            ),
            min_code_changed_lines=env_int(
                env, "TAU_GITHUB_MIN_CODE_CHANGED_LINES", d.min_code_changed_lines
            ),
            min_code_files=env_int(env, "TAU_GITHUB_MIN_CODE_FILES", d.min_code_files),
            reject_cache_max_size=env_int(
                env, "TAU_GITHUB_REJECT_CACHE_MAX_SIZE", d.reject_cache_max_size
            ),
        )
