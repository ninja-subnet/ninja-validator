"""Prune disabled solver upstream endpoints from a dotenv file.

The solver itself writes flapping endpoints to ``TAU_SOLVER_DISABLED_UPSTREAMS_FILE``.
This module is the explicit, operator-run step that reflects that state back into
``.env`` when desired.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tau.openrouter.client import normalize_base_url
from tau.proxy.routing import DisabledUpstreamStore

DISABLED_UPSTREAMS_FILE_ENV = "TAU_SOLVER_DISABLED_UPSTREAMS_FILE"
UPSTREAM_LIST_KEYS = (
    "OPENROUTER_UPSTREAM_BASE_URLS",
    "NINJA_INFERENCE_BASE_URLS",
    "LLM_UPSTREAM_BASE_URLS",
)

_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*)"
    r"(?P<value>.*?)(?P<newline>\r?\n?)$"
)


@dataclass(slots=True)
class PruneResult:
    text: str
    removed: dict[str, list[str]] = field(default_factory=dict)
    skipped_all: dict[str, list[str]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def prune_env_text(env_text: str, disabled_urls: set[str]) -> PruneResult:
    disabled = {normalize_base_url(url) for url in disabled_urls}
    result = PruneResult(text="")
    out: list[str] = []
    for line in env_text.splitlines(keepends=True):
        rewritten, removed, skipped_key = _prune_line(line, disabled)
        out.append(rewritten)
        if removed:
            result.removed[removed[0]] = removed[1]
        if skipped_key:
            result.skipped_all[skipped_key[0]] = skipped_key[1]
    result.text = "".join(out)
    return result


def disabled_file_from_env_text(env_text: str) -> Path | None:
    for line in env_text.splitlines():
        parsed = _parse_assignment(line)
        if parsed is None:
            continue
        key, value = parsed
        if key == DISABLED_UPSTREAMS_FILE_ENV and value.strip():
            return Path(value.strip())
    raw = os.environ.get(DISABLED_UPSTREAMS_FILE_ENV, "").strip()
    return Path(raw) if raw else None


def _prune_line(
    line: str, disabled: set[str]
) -> tuple[str, tuple[str, list[str]] | None, tuple[str, list[str]] | None]:
    match = _ASSIGNMENT_RE.match(line)
    if match is None:
        return line, None, None
    key = match.group("key")
    if key not in UPSTREAM_LIST_KEYS:
        return line, None, None

    value_part, suffix = _split_value_suffix(match.group("value"))
    quote, raw_value = _strip_outer_quote(value_part.strip())
    urls = [url.strip() for url in raw_value.split(",") if url.strip()]
    if not urls:
        return line, None, None

    kept = [url for url in urls if normalize_base_url(url) not in disabled]
    removed = [url for url in urls if normalize_base_url(url) in disabled]
    if not removed:
        return line, None, None
    if not kept:
        return line, None, (key, removed)

    new_value = ",".join(kept)
    if quote:
        new_value = f"{quote}{new_value}{quote}"
    rewritten = f"{match.group('prefix')}{new_value}{suffix}{match.group('newline')}"
    return rewritten, (key, removed), None


def _parse_assignment(line: str) -> tuple[str, str] | None:
    match = _ASSIGNMENT_RE.match(line)
    if match is None:
        return None
    value_part, _suffix = _split_value_suffix(match.group("value"))
    _quote, raw_value = _strip_outer_quote(value_part.strip())
    return match.group("key"), raw_value


def _split_value_suffix(value: str) -> tuple[str, str]:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                suffix_start = index - 1 if index > 0 else index
                return value[:index].rstrip(), value[suffix_start:]
    return value.rstrip(), ""


def _strip_outer_quote(value: str) -> tuple[str, str]:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[0], value[1:-1]
    return "", value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove disabled solver upstream URLs from .env endpoint lists.",
    )
    parser.add_argument(
        "env_file",
        nargs="?",
        default=".env",
        type=Path,
        help="dotenv file to update (default: .env)",
    )
    parser.add_argument(
        "disabled_file",
        nargs="?",
        type=Path,
        help=(
            "disabled upstream file; defaults to TAU_SOLVER_DISABLED_UPSTREAMS_FILE "
            "from the dotenv file"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would change without writing the dotenv file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        env_text = args.env_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"could not read {args.env_file}: {exc}", file=sys.stderr)
        return 1

    disabled_file = args.disabled_file or disabled_file_from_env_text(env_text)
    if disabled_file is None:
        print(
            f"{DISABLED_UPSTREAMS_FILE_ENV} is not set and no disabled file was passed",
            file=sys.stderr,
        )
        return 1

    disabled_urls = DisabledUpstreamStore(disabled_file).load()
    if not disabled_urls:
        print(f"no disabled upstreams found in {disabled_file}")
        return 0

    result = prune_env_text(env_text, disabled_urls)
    if not result.changed:
        print(f"no matching disabled upstream URLs found in {args.env_file}")
    else:
        for key, removed in result.removed.items():
            print(f"{key}: removed {', '.join(removed)}")
        if not args.dry_run:
            args.env_file.write_text(result.text, encoding="utf-8")

    for key, skipped in result.skipped_all.items():
        print(
            f"{key}: left unchanged because pruning would remove every endpoint "
            f"({', '.join(skipped)})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
