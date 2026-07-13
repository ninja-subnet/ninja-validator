"""Deployment configuration regression tests."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_VOLUME = (
    "${TAU_SUBMISSIONS_DIR:-/srv/tau/submissions}:"
    "${TAU_SUBMISSIONS_DIR:-/srv/tau/submissions}:ro"
)


def test_duel_resolver_can_read_submission_bundles() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))

    resolver_volumes = compose["services"]["duel-resolver"].get("volumes", [])

    assert SUBMISSIONS_VOLUME in resolver_volumes
