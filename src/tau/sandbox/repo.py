"""Host-side task-repo checkout, fed into the sandbox.

A single-commit partial clone (no full history, no blobs until needed), checked out
detached at the task's *base* commit (``parent_sha`` — the state *before* the fix).
The clone uses the GitHub token; the ``.git`` remote/logs are then stripped so the
token never travels into the sandbox when the tree is copied in.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_AUTHED_URL = re.compile(r"https://[^/@\s]+:[^/@\s]+@")
# Keyed by the limit value: the cap is only global across callers that share one
# configured limit (true today — the task solver is the sole caller). Two different
# limits would get two independent semaphores.
_FETCH_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_FETCH_SEMAPHORES_LOCK = threading.Lock()
# How many times the ensure-then-copy dance retries when eviction or a corrupt
# entry squeezes between the exclusive and shared phases before giving up on the
# cache and cloning directly.
_CACHE_ATTEMPTS = 3


class CloneError(RuntimeError):
    """A git step failed while materializing a task repo."""


def _redact_auth(text: str) -> str:
    return _AUTHED_URL.sub("https://<redacted>@", text)


def _git_command(args: list[str]) -> str:
    return "git " + " ".join(_redact_auth(arg) for arg in args)


def _git(args: list[str], *, cwd: Path | None = None, timeout: int = 300) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CloneError(f"{_git_command(args)} timed out after {timeout} seconds") from exc
    if result.returncode != 0:
        output = _redact_auth(((result.stdout or "") + (result.stderr or "")).strip())
        raise CloneError(f"{_git_command(args)} failed: {output[-500:]}")
    return result.stdout


def _authed_url(repo_clone_url: str, token: str | None) -> str:
    """Inject the token into an https GitHub URL; pass other URLs through."""
    if token and repo_clone_url.startswith("https://"):
        rest = repo_clone_url[len("https://") :]
        # Don't double-inject if credentials are already present.
        if "@" not in rest.split("/", 1)[0]:
            return f"https://x-access-token:{token}@{rest}"
    return repo_clone_url


def _sanitize_git_metadata(repo_dir: Path) -> None:
    """Drop the token-bearing remote and the reflogs before the tree leaves the host."""
    git_dir = repo_dir / ".git"
    if not git_dir.is_dir():
        return
    # Remove the origin remote (its URL holds the token). Best-effort.
    subprocess.run(
        ["git", "remote", "remove", "origin"],
        cwd=str(repo_dir), capture_output=True, text=True, check=False,
    )
    shutil.rmtree(git_dir / "logs", ignore_errors=True)
    subprocess.run(
        ["git", "reflog", "expire", "--expire=now", "--all"],
        cwd=str(repo_dir), capture_output=True, text=True, check=False,
    )


def _cache_key(repo_clone_url: str, base_commit: str) -> str:
    raw = f"{repo_clone_url}\0{base_commit}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _flock(path: Path, mode: int):
    """Hold a blocking flock of *mode* (LOCK_EX / LOCK_SH) on *path*.

    Each call opens its own fd, so flock excludes both other processes and other
    threads of this process.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def _remote_fetch_slot(limit: int | None):
    if limit is None or limit <= 0:
        yield
        return
    with _FETCH_SEMAPHORES_LOCK:
        semaphore = _FETCH_SEMAPHORES.setdefault(limit, threading.BoundedSemaphore(limit))
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _clone_uncached(
    *,
    repo_clone_url: str,
    base_commit: str,
    token: str | None,
    dest: Path,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    url = _authed_url(repo_clone_url, token)
    # Partial, no-checkout clone: cheapest way to land a single commit's tree.
    _git(["clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
    _git(["fetch", "--depth=1", "origin", base_commit], cwd=dest)
    _git(["checkout", "--detach", "FETCH_HEAD"], cwd=dest)
    _sanitize_git_metadata(dest)
    return dest


def _cache_ready(entry: Path, ready: Path, base_commit: str) -> bool:
    if not (entry.is_dir() and (entry / ".git").is_dir()):
        return False
    try:
        # The marker records the commit it was populated for; a mismatch means a
        # corrupt or hand-edited cache, treated as a miss so it self-heals.
        return ready.read_text(encoding="utf-8").strip() == base_commit
    except OSError:
        return False


def _populate_cache_entry(
    *,
    repo_clone_url: str,
    base_commit: str,
    token: str | None,
    cache_dir: Path,
    entry: Path,
    ready: Path,
) -> None:
    # Any surviving .tmp for this key is an orphan from a hard-killed populate:
    # the caller holds the key's exclusive lock, so no live populate owns one.
    for stale in cache_dir.glob(f".{entry.name}.*.tmp"):
        shutil.rmtree(stale, ignore_errors=True)
    tmp = cache_dir / f".{entry.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        _clone_uncached(
            repo_clone_url=repo_clone_url,
            base_commit=base_commit,
            token=token,
            dest=tmp,
        )
        shutil.rmtree(entry, ignore_errors=True)
        tmp.rename(entry)
        ready.write_text(base_commit + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _copy_cached_entry(entry: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(entry, dest, symlinks=True, ignore_dangling_symlinks=True)


def _ready_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # deleted underneath us — sort as oldest, the loop re-checks
        return 0.0


def _evict_excess_entries(
    cache_dir: Path, max_entries: int | None, *, keep_key: str
) -> None:
    """Drop the oldest cache entries beyond *max_entries* (LRU by populate time).

    Runs after a miss populates a new entry — the only moment the cache grows.
    Each victim is deleted under a non-blocking exclusive flock of its key lock;
    solves copy entries out under a shared flock, so an entry mid-copy simply
    fails the try-lock and survives this round. Best-effort: an over-budget cache
    trims again on the next miss. The tiny ``.lock`` files are left behind.
    """
    if not max_entries or max_entries <= 0:
        return
    ready_files = sorted(cache_dir.glob("*.ready"), key=_ready_mtime)
    excess = len(ready_files) - max_entries
    for ready in ready_files:
        if excess <= 0:
            return
        key = ready.name.removesuffix(".ready")
        if key == keep_key:
            continue
        try:
            with (cache_dir / f"{key}.lock").open("a+", encoding="utf-8") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue  # in use (being copied or repopulated) — skip
                try:
                    # Marker first: a partial delete reads as a miss and self-heals.
                    ready.unlink(missing_ok=True)
                    shutil.rmtree(cache_dir / key, ignore_errors=True)
                    excess -= 1
                    log.debug("evicted task repo cache entry %s", key)
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            continue


def clone_task_repo(
    *,
    repo_clone_url: str,
    base_commit: str,
    token: str | None,
    dest: Path,
    cache_dir: Path | None = None,
    fetch_concurrency: int | None = None,
    cache_max_entries: int | None = None,
) -> Path:
    """Clone *repo_clone_url* and check out *base_commit* (detached) into *dest*.

    ``base_commit`` must be a full 40-char SHA (what the generator stores as
    ``parent_sha``); a short SHA cannot be fetched as a remote ref.
    """
    if not _FULL_SHA.match(base_commit):
        raise CloneError(f"base_commit must be a full 40-char SHA, got {base_commit!r}")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(repo_clone_url, base_commit)
        entry = cache_dir / key
        ready = cache_dir / f"{key}.ready"
        lock_path = cache_dir / f"{key}.lock"
        # Two phases per attempt: ensure the entry exists (exclusive lock), then
        # copy it out (shared lock, so concurrent solves overlap while eviction's
        # try-lock can't delete an entry mid-copy). An eviction squeezing between
        # the phases just fails the ready re-check and the next pass repopulates.
        for _ in range(_CACHE_ATTEMPTS):
            with _flock(lock_path, fcntl.LOCK_EX):
                if not _cache_ready(entry, ready, base_commit):
                    log.info(
                        "task repo cache miss for %s @ %s; fetching once",
                        repo_clone_url,
                        base_commit[:8],
                    )
                    with _remote_fetch_slot(fetch_concurrency):
                        _populate_cache_entry(
                            repo_clone_url=repo_clone_url,
                            base_commit=base_commit,
                            token=token,
                            cache_dir=cache_dir,
                            entry=entry,
                            ready=ready,
                        )
                    _evict_excess_entries(cache_dir, cache_max_entries, keep_key=key)
            with _flock(lock_path, fcntl.LOCK_SH):
                if _cache_ready(entry, ready, base_commit):
                    _copy_cached_entry(entry, dest)
                    log.debug(
                        "copied cached task repo %s @ %s into %s",
                        repo_clone_url,
                        base_commit[:8],
                        dest,
                    )
                    return dest
        log.warning(
            "task repo cache entry for %s @ %s kept disappearing; cloning uncached",
            repo_clone_url,
            base_commit[:8],
        )
        with _remote_fetch_slot(fetch_concurrency):
            _clone_uncached(
                repo_clone_url=repo_clone_url,
                base_commit=base_commit,
                token=token,
                dest=dest,
            )
        return dest

    _clone_uncached(
        repo_clone_url=repo_clone_url,
        base_commit=base_commit,
        token=token,
        dest=dest,
    )
    log.debug("cloned %s @ %s into %s", repo_clone_url, base_commit[:8], dest)
    return dest
