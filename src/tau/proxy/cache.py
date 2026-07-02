"""Disk-backed JSON cache + small JSON helpers used by the proxy.

Vendored from the legacy ``tau.utils`` (``json_sha256``, ``get_dict``, ``DiskCache``)
and ``tau.io.openrouter`` (``CacheMissError``) so ``tau.proxy`` stays self-contained
and does not drag in the old package.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def json_sha256(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the JSON-serialized payload, stable across key order.

    Used as the deterministic cache key for record/replay.
    """
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_dict(data: dict[Any, Any], key: Any) -> dict[Any, Any]:
    """Return ``data[key]`` if it is a dict, else an empty dict."""
    value = data.get(key)
    return value if isinstance(value, dict) else {}


class CacheMissError(RuntimeError):
    """Raised by a replay-only cache when a request is not on disk."""

    def __init__(self, key: str) -> None:
        super().__init__(f"proxy cache miss for key {key!r} and no inner client configured")
        self.key = key


class DiskCache:
    """JSON-file cache keyed by an arbitrary string.

    Each entry is ``cache_dir/<key>.json`` holding any JSON-serializable dict.
    Callers choose their own key and value shape; this only does the disk I/O.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def read(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001 — a corrupt entry must not crash the proxy
            log.warning("Failed to read cache entry %s: %s", path, exc)
            return None

    def write(self, key: str, data: dict[str, Any]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to write cache entry %s: %s", path, exc)
