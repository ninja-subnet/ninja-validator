"""Cross-process read/modify/write helpers for validator state.json."""

from __future__ import annotations

import fcntl
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from workspace import write_json

log = logging.getLogger("swe-eval.validator-state-io")

_PRIVATE_SUBMISSION_SOURCE = "private"


def validator_state_lock_path(state_path: Path) -> Path:
    return state_path.parent / ".state.json.lock"


@contextmanager
def validator_state_lock(state_path: Path) -> Iterator[None]:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = validator_state_lock_path(state_path)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_state_payload(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid validator state file: {state_path}")
    return payload


def private_submission_validator_queue_entry(
    *,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    registration_block: int,
    uid: int,
    accepted_at: str | None = None,
    agent_username: str | None = None,
    coldkey: str | None = None,
    coldkey_signature: str | None = None,
) -> dict[str, Any]:
    sha256 = agent_sha256.lower()
    entry: dict[str, Any] = {
        "hotkey": hotkey,
        "uid": int(uid),
        "repo_full_name": f"private-submission/{submission_id}",
        "repo_url": f"private-submission://{submission_id}",
        "commit_sha": sha256,
        "commitment": f"private-submission:{submission_id}:{sha256}",
        "commitment_block": int(registration_block),
        "source": _PRIVATE_SUBMISSION_SOURCE,
        "accepted_at": accepted_at or datetime.now(UTC).isoformat(),
    }
    if agent_username:
        entry["agent_username"] = agent_username
    if coldkey:
        entry["coldkey"] = coldkey
    if coldkey_signature:
        entry["coldkey_signature"] = coldkey_signature
    return entry


def _queue_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    queue = state.get("queue", [])
    if not isinstance(queue, list):
        return []
    return [item for item in queue if isinstance(item, dict)]


def _active_duel_hotkeys(state: dict[str, Any]) -> set[str]:
    active_duel = state.get("active_duel")
    if not isinstance(active_duel, dict):
        return set()
    hotkeys: set[str] = set()
    for key in ("king", "challenger"):
        item = active_duel.get(key)
        if isinstance(item, dict) and item.get("hotkey"):
            hotkeys.add(str(item["hotkey"]))
    return hotkeys


def _submission_sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    try:
        block = int(entry.get("commitment_block", 0))
    except (TypeError, ValueError):
        block = 0
    try:
        uid = int(entry.get("uid", 0))
    except (TypeError, ValueError):
        uid = 0
    return block, uid, str(entry.get("hotkey") or "")


def _record_commitment_acceptance(state: dict[str, Any], submission: dict[str, Any]) -> None:
    hotkey = str(submission["hotkey"])
    commitment = str(submission["commitment"])
    commitment_block = int(submission["commitment_block"])
    locked = state.setdefault("locked_commitments", {})
    if not isinstance(locked, dict):
        locked = {}
        state["locked_commitments"] = locked
    locked[hotkey] = commitment
    blocks = state.setdefault("commitment_blocks_by_hotkey", {})
    if not isinstance(blocks, dict):
        blocks = {}
        state["commitment_blocks_by_hotkey"] = blocks
    blocks[hotkey] = commitment_block
    seen = state.setdefault("seen_hotkeys", [])
    if not isinstance(seen, list):
        seen = []
        state["seen_hotkeys"] = seen
    if hotkey not in seen:
        seen.append(hotkey)


def _submission_current_for_registration(
    submission: dict[str, Any],
    registration_block: int,
) -> bool:
    try:
        return int(submission.get("commitment_block")) >= int(registration_block)
    except (TypeError, ValueError):
        return False


def _state_has_current_submission_for_hotkey(
    state: dict[str, Any],
    *,
    hotkey: str,
    registration_block: int,
) -> bool:
    current_king = state.get("current_king")
    if (
        isinstance(current_king, dict)
        and str(current_king.get("hotkey") or "") == hotkey
        and _submission_current_for_registration(current_king, registration_block)
    ):
        return True

    for key in ("queue", "recent_kings"):
        entries = state.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if (
                isinstance(entry, dict)
                and str(entry.get("hotkey") or "") == hotkey
                and _submission_current_for_registration(entry, registration_block)
            ):
                return True

    active_duel = state.get("active_duel")
    if isinstance(active_duel, dict):
        for key in ("king", "challenger"):
            entry = active_duel.get(key)
            if (
                isinstance(entry, dict)
                and str(entry.get("hotkey") or "") == hotkey
                and _submission_current_for_registration(entry, registration_block)
            ):
                return True
    return False


def _state_has_spent_marker_for_hotkey(state: dict[str, Any], hotkey: str) -> bool:
    locked = state.get("locked_commitments", {})
    if isinstance(locked, dict) and hotkey in locked:
        return True
    for key in ("seen_hotkeys", "retired_hotkeys", "disqualified_hotkeys"):
        values = state.get(key, [])
        if isinstance(values, list) and any(str(value) == hotkey for value in values):
            return True
    return False


def _clear_stale_spent_state_for_reregistered_hotkey(
    state: dict[str, Any],
    *,
    hotkey: str,
    registration_block: int,
) -> None:
    blocks = state.get("commitment_blocks_by_hotkey", {})
    if not isinstance(blocks, dict):
        return
    prior_block = blocks.get(hotkey)
    try:
        prior_block_int = int(prior_block) if prior_block is not None else None
    except (TypeError, ValueError):
        prior_block_int = None
    if prior_block_int is None:
        if (
            not _state_has_spent_marker_for_hotkey(state, hotkey)
            or _state_has_current_submission_for_hotkey(
                state,
                hotkey=hotkey,
                registration_block=registration_block,
            )
        ):
            return
    elif prior_block_int >= int(registration_block):
        return

    locked = state.get("locked_commitments", {})
    if isinstance(locked, dict):
        locked.pop(hotkey, None)
    blocks.pop(hotkey, None)
    for key in ("seen_hotkeys", "retired_hotkeys", "disqualified_hotkeys"):
        values = state.get(key, [])
        if isinstance(values, list):
            state[key] = [value for value in values if str(value) != hotkey]


def enqueue_private_submission_in_state(
    *,
    state_path: Path,
    submission: dict[str, Any],
    queue_size_limit: int | None = None,
) -> bool:
    """Atomically append an accepted private submission to state.json."""
    hotkey = str(submission.get("hotkey") or "")
    commitment = str(submission.get("commitment") or "")
    if not hotkey or not commitment:
        return False

    with validator_state_lock(state_path):
        state = _load_state_payload(state_path)
        try:
            registration_block = int(submission["commitment_block"])
        except (KeyError, TypeError, ValueError):
            return False
        _clear_stale_spent_state_for_reregistered_hotkey(
            state,
            hotkey=hotkey,
            registration_block=registration_block,
        )
        queue = _queue_entries(state)
        if any(str(item.get("commitment") or "") == commitment for item in queue):
            return False
        if any(str(item.get("hotkey") or "") == hotkey for item in queue):
            log.info(
                "Skipping atomic queue enqueue for hotkey %s: already queued",
                hotkey,
            )
            return False
        if hotkey in _active_duel_hotkeys(state):
            log.info(
                "Skipping atomic queue enqueue for %s: hotkey is in active duel",
                commitment,
            )
            return False
        locked = state.get("locked_commitments", {})
        if isinstance(locked, dict):
            existing = locked.get(hotkey)
            if existing is not None and str(existing) != commitment:
                log.warning(
                    "Skipping atomic queue enqueue for hotkey %s: locked commitment %s != %s",
                    hotkey,
                    existing,
                    commitment,
                )
                return False
        current_king = state.get("current_king")
        if isinstance(current_king, dict) and str(current_king.get("hotkey") or "") == hotkey:
            log.info(
                "Skipping atomic queue enqueue for hotkey %s: hotkey is current king",
                hotkey,
            )
            return False
        if queue_size_limit is not None and len(queue) >= int(queue_size_limit):
            log.info(
                "Skipping atomic queue enqueue for %s: queue size limit %s reached",
                commitment,
                queue_size_limit,
            )
            return False

        _record_commitment_acceptance(state, submission)
        queue.append(submission)
        queue.sort(key=_submission_sort_key)
        state["queue"] = queue
        write_json(state_path, state)
        log.info("Atomically enqueued private submission %s for hotkey %s", commitment, hotkey)
        return True
