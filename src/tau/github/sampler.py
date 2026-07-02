"""Sample a usable random commit from public GitHub history.

``CommitSampler`` is the task-generator's commit *source* (named to avoid
confusion with subnet miners, who upload agents). It decides WHICH commit becomes
a task; all GitHub HTTP lives in :class:`tau.github.client.GitHubClient`, which is
injected. Behaviour preserved from the monolith: prefer commits from a random
historical day via the commit-search API (spreads tasks across dates/repos and
cuts duplicates), fall back to the recent-events firehose when search is
unavailable, and accept only modification-heavy code commits.

Tunables live in :class:`tau.github.config.GitHubConfig`. The reject cache is an
in-process (bounded) set and the search buffer / firehose cache are per-instance,
refilled inline — no module globals, no background thread (each worker is one
process with one sampling loop).
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from .client import GitHubClient
from .config import GitHubConfig
from .errors import (
    CommitRejected,
    CommitSourceUnavailable,
    GitHubRequestError,
    NoCommitMetCriteria,
    RejectReason,
)
from .types import CommitCandidate

log = logging.getLogger(__name__)


class _CommitRef(NamedTuple):
    """A (repo, sha) pick to fetch, plus an optional firehose event id."""

    repo_full_name: str
    commit_sha: str
    event_id: str = ""


class SampledCommit(NamedTuple):
    """A usable commit plus how many were discarded (by reason) before it."""

    candidate: CommitCandidate
    rejections: Counter[RejectReason]


# fmt: off
_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp",
        ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
        ".bash", ".zsh", ".pl", ".pm", ".r", ".lua", ".ex", ".exs", ".erl",
        ".hs", ".ml", ".mli", ".clj", ".cljs", ".vue", ".svelte", ".dart",
        ".zig", ".nim", ".cr", ".v", ".sql", ".m", ".mm",
    }
)

_SKIP_FILENAMES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "cargo.lock",
        "poetry.lock", "pipfile.lock", "gemfile.lock", "composer.lock",
        "go.sum", "flake.lock",
    }
)
# fmt: on


def _is_code_file(filename: str) -> bool:
    """Return True if the file has a recognized code extension."""
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in _CODE_EXTENSIONS)


def _is_lockfile(filename: str) -> bool:
    """Return True if the file is a lockfile or auto-generated."""
    base = filename.rsplit("/", 1)[-1].lower()
    return base in _SKIP_FILENAMES


def _event_commit_changed_files(commit: dict[str, Any]) -> list[str]:
    return [
        str(filename)
        for key in ("modified", "added", "removed")
        for filename in commit.get(key, [])
        if filename
    ]


def _event_commit_has_code_change_hint(commit: dict[str, Any]) -> bool:
    return any(
        _is_code_file(filename) and not _is_lockfile(filename)
        for filename in _event_commit_changed_files(commit)
    )


def _push_event_commits_with_code_hints(event: dict[str, Any]) -> list[dict[str, Any]]:
    commits = event.get("payload", {}).get("commits", [])
    if not isinstance(commits, list):
        return []
    hinted = [
        commit
        for commit in commits
        if isinstance(commit, dict) and _event_commit_has_code_change_hint(commit)
    ]
    return hinted or [commit for commit in commits if isinstance(commit, dict)]


def _commit_search_query(day: str, suffix: str) -> str:
    query = f"committer-date:{day}"
    suffix = suffix.strip()
    if suffix:
        query = f"{query} {suffix}"
    return query


def _reject_cache_key(repo_full_name: str, commit_sha: str) -> str:
    return f"{repo_full_name}:{commit_sha}"


def _extract_available_pages(link_header: str) -> list[int]:
    pages: set[int] = {1}
    for part in link_header.split(","):
        if "page=" not in part:
            continue
        fragment = part.split("page=", 1)[1]
        digits: list[str] = []
        for char in fragment:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            pages.add(int("".join(digits)))
    return sorted(pages)


class CommitSampler:
    """Sample random recent commits from public GitHub history."""

    def __init__(
        self,
        *,
        rng: random.Random,
        client: GitHubClient,
        config: GitHubConfig | None = None,
    ) -> None:
        self._rng = rng
        self._client = client
        self._config = config if config is not None else GitHubConfig()
        self._reject_keys: set[str] = set()
        self._date_commit_buffer: list[tuple[str, str]] = []
        self._date_commit_search_cooldown_until = 0.0
        self._recent_events_cache: tuple[float, list[dict[str, Any]]] | None = None

    # -- public API --------------------------------------------------------------

    async def sample_commit(self, max_attempts: int = 25) -> SampledCommit:
        last_error: str | None = None
        rejections: Counter[RejectReason] = Counter()
        saw_no_source = False
        for attempt in range(1, max_attempts + 1):
            log.debug("Sampling attempt %s/%s", attempt, max_attempts)
            ref = await self._next_commit_ref()
            if ref is None:
                # No commit to evaluate (search empty + no firehose) — not a
                # rejection, but a sign the source is barren or rate-limited.
                saw_no_source = True
                last_error = "no commit source available (search empty, no recent push events)"
                log.debug(last_error)
                continue
            if self._reject_contains(ref.repo_full_name, ref.commit_sha):
                last_error = f"cached reject for {ref.repo_full_name}@{ref.commit_sha}"
                log.debug(last_error)
                rejections[RejectReason.DUPLICATE] += 1
                continue
            try:
                candidate = await self._fetch_candidate(ref)
                self._screen(candidate)
            except CommitRejected as exc:
                last_error = str(exc)
                log.debug("Discarding %s@%s: %s", ref.repo_full_name, ref.commit_sha, exc)
                self._reject_add(ref.repo_full_name, ref.commit_sha, last_error)
                rejections[exc.reason] += 1  # STRUCTURAL or QUALITY
                continue
            except GitHubRequestError as exc:
                last_error = str(exc)
                log.debug("Fetch failed %s@%s: %s", ref.repo_full_name, ref.commit_sha, exc)
                self._reject_add(ref.repo_full_name, ref.commit_sha, last_error)
                rejections[RejectReason.FETCH_ERROR] += 1
                continue

            log.debug(
                "Sampled commit repo=%s sha=%s (%d rejected first)",
                candidate.repo_full_name,
                candidate.commit_sha,
                sum(rejections.values()),
            )
            return SampledCommit(candidate, rejections)
        # Out of attempts. Infra trouble (a fetch erroring, or the source going
        # empty / rate-limited) tells the fetcher to back off; a round that only
        # screened candidates out is benign churn it should retry promptly.
        infra = saw_no_source or rejections[RejectReason.FETCH_ERROR] > 0
        error_cls = CommitSourceUnavailable if infra else NoCommitMetCriteria
        raise error_cls(
            last_error or "could not sample a usable GitHub commit",
            rejections=rejections,
        )

    # -- fetch + screen ----------------------------------------------------------

    async def _fetch_candidate(self, ref: _CommitRef) -> CommitCandidate:
        payload = await self._client.get_json(
            f"/repos/{ref.repo_full_name}/commits/{ref.commit_sha}"
        )
        return CommitCandidate.from_api(
            payload, repo_full_name=ref.repo_full_name, event_id=ref.event_id
        )

    def _screen(self, candidate: CommitCandidate) -> None:
        """Raise CommitRejected unless *candidate* passes the quality gate."""
        if not candidate.combined_patch:
            raise CommitRejected(
                "commit had no textual patch content", reason=RejectReason.QUALITY
            )
        quality_reason = self._quality_check(candidate, self._config)
        if quality_reason:
            raise CommitRejected(quality_reason, reason=RejectReason.QUALITY)

    @staticmethod
    def _quality_check(candidate: CommitCandidate, config: GitHubConfig) -> str | None:
        """Return a rejection reason, or None if the commit is acceptable."""
        code_files = [
            f
            for f in candidate.files
            if f.patch and _is_code_file(f.filename) and not _is_lockfile(f.filename)
        ]
        if len(code_files) < config.min_code_files:
            return (
                f"Only {len(code_files)} code file(s), need {config.min_code_files}; "
                f"files: {[f.filename for f in candidate.files]}"
            )

        code_changed = sum(f.additions + f.deletions for f in code_files)
        if code_changed < config.min_code_changed_lines:
            return f"Only {code_changed} code lines changed, need {config.min_code_changed_lines}"

        # Prefer modification-heavy commits: agents produce more similar patches
        # when editing existing code vs writing entirely new files.
        modified_files = [f for f in code_files if f.status == "modified"]
        if not modified_files:
            return (
                f"No modified code files (all {len(code_files)} are added/removed); "
                "need at least 1 modified file for meaningful comparison"
            )

        return None

    # -- reject cache (in-memory, bounded) --------------------------------------

    def _reject_contains(self, repo_full_name: str, commit_sha: str) -> bool:
        return _reject_cache_key(repo_full_name, commit_sha) in self._reject_keys

    def _reject_add(self, repo_full_name: str, commit_sha: str, reason: str) -> None:
        if len(self._reject_keys) >= self._config.reject_cache_max_size:
            self._reject_keys.clear()
        self._reject_keys.add(_reject_cache_key(repo_full_name, commit_sha))
        log.debug("Rejecting %s@%s: %s", repo_full_name, commit_sha, reason)

    # -- commit source: search buffer, then firehose ----------------------------

    async def _next_commit_ref(self) -> _CommitRef | None:
        return await self._next_from_search() or await self._next_from_firehose()

    async def _next_from_search(self) -> _CommitRef | None:
        """Pop the next non-rejected (repo, sha) from the search buffer, refilling when low."""
        await self._refill_date_commit_buffer_if_needed()
        while self._date_commit_buffer:
            repo_full_name, commit_sha = self._date_commit_buffer.pop()
            if self._reject_contains(repo_full_name, commit_sha):
                continue
            return _CommitRef(repo_full_name, commit_sha)
        return None

    async def _refill_date_commit_buffer_if_needed(self) -> None:
        config = self._config
        if len(self._date_commit_buffer) >= config.buffer_low_watermark:
            return
        if time.monotonic() < self._date_commit_search_cooldown_until:
            return
        # Drop already-rejected commits before buffering. A page that yields no
        # *fresh* commits (empty, or all rejected/duplicate) counts as a failed
        # refill and triggers the cooldown — otherwise the buffer drains to empty
        # every attempt and we hammer the rate-limited search API.
        fresh = [
            (repo, sha)
            for repo, sha in await self._search_commits_for_random_day()
            if not self._reject_contains(repo, sha)
        ]
        if not fresh:
            self._date_commit_search_cooldown_until = (
                time.monotonic() + config.search_fail_cooldown_seconds
            )
            return
        self._date_commit_buffer.extend(fresh)
        overflow = len(self._date_commit_buffer) - (config.buffer_high_watermark * 2)
        if overflow > 0:
            del self._date_commit_buffer[:overflow]

    async def _search_commits_for_random_day(self) -> list[tuple[str, str]]:
        """One commit-search call for a random day; returns all (repo, sha) hits."""
        config = self._config
        days_back = self._rng.randint(
            config.history_sample_min_days, config.history_sample_max_days
        )
        day = (datetime.now(tz=UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        per_page = config.search_per_page
        # Search results are capped at 1000 total (page * per_page).
        max_page = max(1, 1000 // per_page)
        page = self._rng.randint(1, max_page)
        try:
            payload = await self._client.get_json(
                "/search/commits",
                q=_commit_search_query(day, config.search_query_suffix),
                sort="committer-date",
                order=self._rng.choice(["asc", "desc"]),
                per_page=per_page,
                page=page,
            )
        except GitHubRequestError as exc:
            log.debug("Commit search failed for day %s: %s", day, exc)
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not items:
            return []
        results: list[tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            repo = (item.get("repository") or {}).get("full_name")
            sha = item.get("sha")
            if repo and sha:
                results.append((str(repo), str(sha)))
        log.debug("Date-search day %s page %s -> %d commits", day, page, len(results))
        return results

    # -- recent-events firehose (fallback) --------------------------------------

    async def _next_from_firehose(self) -> _CommitRef | None:
        events = await self._recent_push_events()
        if not events:
            return None
        event = self._rng.choice(events)
        repo = (event.get("repo") or {}).get("name")
        sha = self._pick_random_commit_sha(event)
        if not repo or not sha:
            return None
        return _CommitRef(str(repo), str(sha), str(event.get("id", "")))

    async def _recent_push_events(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self._recent_events_cache is not None:
            cached_at, events = self._recent_events_cache
            if now - cached_at < self._config.recent_events_cache_ttl_seconds:
                return list(events)
        events = await self._fetch_recent_push_events()
        self._recent_events_cache = (time.monotonic(), list(events))
        return list(events)

    async def _fetch_recent_push_events(self) -> list[dict[str, Any]]:
        log.debug("Fetching recent public events page 1")
        try:
            response, payload = await self._client.get_with_response(
                "/events", page=1, per_page=30
            )
        except GitHubRequestError as exc:
            log.debug("Recent events fetch failed: %s", exc)
            return []
        events: list[dict[str, Any]] = payload if isinstance(payload, list) else []
        pages = _extract_available_pages(response.headers.get("link", ""))
        valid_pages = [page for page in pages if page > 1]
        if valid_pages:
            self._rng.shuffle(valid_pages)
            for page in valid_pages[: max(0, self._config.event_pages_to_fetch - 1)]:
                log.debug("Fetching additional public events page %s", page)
                try:
                    page_payload = await self._client.get_json(
                        "/events", page=page, per_page=30
                    )
                except GitHubRequestError as exc:
                    log.debug("Failed to fetch events page %s: %s", page, exc)
                    break
                if isinstance(page_payload, list):
                    events.extend(page_payload)
        return [
            event
            for event in events
            if isinstance(event, dict) and event.get("type") == "PushEvent"
        ]

    def _pick_random_commit_sha(self, event: dict[str, Any]) -> str | None:
        commits = _push_event_commits_with_code_hints(event)
        if not commits:
            head_sha = (event.get("payload") or {}).get("head")
            return str(head_sha) if head_sha else None
        commit = self._rng.choice(commits)
        sha = commit.get("sha")
        return str(sha) if sha else None
