#!/usr/bin/env python3
"""Check the GitHub REST rate-limit quota for each token configured in .env.

Reads ``GITHUB_TOKENS`` (comma-separated) or ``GITHUB_TOKEN`` from the project
``.env``, then queries ``https://api.github.com/rate_limit`` for each one. That
endpoint reports the remaining quota per resource and is free (it does not itself
consume quota). Token values are never printed -- only a short sha256 fingerprint,
which matches the worker's ``token #N (...)`` log lines so you can correlate.

Usage:
    uv run python scripts/check_github_quota.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

_API = "https://api.github.com/rate_limit"
# Resources the task-generator's sampler actually uses: commit search + the core
# pool (commit fetches and the events firehose).
_SHOWN_RESOURCES = ("core", "search")


def _fingerprint(token: str) -> str:
    """Matches GitHubTokenRotator._token_fingerprint, so output lines up with logs."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]


def _tokens_from_env() -> list[str]:
    """GITHUB_TOKENS (comma-separated), else GITHUB_TOKEN -- same as the rotator."""
    raw = os.environ.get("GITHUB_TOKENS", "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        single = os.environ.get("GITHUB_TOKEN")
        if single and single.strip():
            tokens = [single.strip()]
    return tokens


def _reset_in(reset_epoch: float) -> str:
    secs = max(0, int(reset_epoch - time.time()))
    minutes, seconds = divmod(secs, 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    tokens = _tokens_from_env()
    if not tokens:
        print("No GITHUB_TOKENS / GITHUB_TOKEN found in .env", file=sys.stderr)
        return 1

    print(f"Checking {len(tokens)} GitHub token(s) via {_API}\n")
    headers_base = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for i, token in enumerate(tokens, start=1):
        fp = _fingerprint(token)
        try:
            resp = httpx.get(
                _API, headers={**headers_base, "Authorization": f"Bearer {token}"}, timeout=15
            )
        except httpx.HTTPError as exc:
            print(f"#{i}  fp={fp}  request failed: {exc}\n")
            continue

        if resp.status_code != 200:
            body = resp.text[:120].replace("\n", " ")
            print(f"#{i}  fp={fp}  HTTP {resp.status_code} (likely invalid/expired): {body}\n")
            continue

        resources = resp.json().get("resources", {})
        print(f"#{i}  fp={fp}  HTTP 200")
        for name in _SHOWN_RESOURCES:
            r = resources.get(name)
            if not r:
                continue
            print(
                f"      {name:8s} {r['remaining']:>5}/{r['limit']:<5}"
                f"  resets in {_reset_in(r['reset'])}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
