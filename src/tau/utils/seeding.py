"""Stable JSON-safe integer seeds shared by validator-controlled LLM calls."""

from __future__ import annotations

import hashlib

STABLE_SEED_BITS = 53
STABLE_SEED_MASK = (1 << STABLE_SEED_BITS) - 1


def stable_seed(material: str) -> int:
    """Return a deterministic 53-bit seed safe for JSON number transports."""
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & STABLE_SEED_MASK
