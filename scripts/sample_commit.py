#!/usr/bin/env python
"""Sample one GitHub commit candidate with ``tau.github.CommitSampler``.

This is the raw material the task-generator turns into a task. It does NOT write
a task description (that's ``tau.taskgen``, not built yet) — it just mines a
usable commit and prints it, so you can eyeball what the sampler produces.

Usage:
    uv run python scripts/sample_commit.py
    uv run python scripts/sample_commit.py --json
    uv run python scripts/sample_commit.py --patch --max-attempts 50
    uv run python scripts/sample_commit.py --seed 0 -v

Set ``GITHUB_TOKENS`` (comma-separated) or ``GITHUB_TOKEN`` to avoid the very low
unauthenticated GitHub rate limits — sampling leans on the commit-search API,
which is ~10 req/min unauthenticated and will 403 quickly without a token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

from dotenv import load_dotenv

from tau.github import (
    CommitCandidate,
    CommitSampleError,
    CommitSampler,
    GitHubClient,
    GitHubConfig,
    GitHubTokenRotator,
)

log = logging.getLogger("sample_commit")

# Project-root .env, resolved from this file so it loads regardless of cwd.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-attempts", type=int, default=25, help="mining attempts before giving up"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed the RNG for a reproducible pick"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full candidate as JSON"
    )
    parser.add_argument(
        "--patch", action="store_true", help="also print the full combined patch"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser


def _print_summary(candidate: CommitCandidate) -> None:
    patch = candidate.combined_patch
    message_first_line = next(iter(candidate.message.splitlines()), "")
    print(f"repo:       {candidate.repo_full_name}")
    print(f"clone url:  {candidate.repo_clone_url}")
    print(f"commit:     {candidate.commit_sha}")
    print(f"parent:     {candidate.parent_sha}")
    print(f"author:     {candidate.author_name or '(unknown)'}")
    print(f"url:        {candidate.html_url}")
    print(f"message:    {message_first_line}")
    print(f"files ({len(candidate.changed_files)}):")
    for name in candidate.changed_files:
        print(f"  - {name}")
    print(f"patch:      {len(patch)} chars / {patch.count(chr(0x0A)) + 1} lines")


async def _sample(args: argparse.Namespace) -> int:
    rotator = GitHubTokenRotator.from_env()
    if rotator is None:
        log.warning(
            "No GITHUB_TOKENS / GITHUB_TOKEN set — unauthenticated limits are very "
            "low; sampling may 403. Export a PAT for reliable results."
        )
    config = GitHubConfig.from_env()
    rng = random.Random(args.seed)

    async with GitHubClient.create(
        token_rotator=rotator, timeout=config.http_timeout
    ) as client:
        sampler = CommitSampler(rng=rng, client=client, config=config)
        try:
            candidate = (await sampler.sample_commit(max_attempts=args.max_attempts)).candidate
        except CommitSampleError as exc:
            log.error("Could not sample a usable commit: %s", exc)
            return 1

    if args.json:
        print(json.dumps(candidate.to_dict(), indent=2, sort_keys=True))
    else:
        _print_summary(candidate)
    if args.patch and not args.json:
        print("\n--- combined patch ---")
        print(candidate.combined_patch)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load .env before any from_env() reads os.environ. Existing env vars win.
    if load_dotenv(ENV_FILE):
        log.debug("Loaded environment from %s", ENV_FILE)

    return asyncio.run(_sample(args))


if __name__ == "__main__":
    raise SystemExit(main())
