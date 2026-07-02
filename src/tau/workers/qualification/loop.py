"""Sequential poll loop for the submission qualification worker."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path

from tau.axiom import get_axiom
from tau.db import QualificationDb
from tau.openrouter import LLMClient
from tau.qualification import (
    SecurityQualificationInput,
    qualification_outcome,
    qualify_submission_security,
    security_failures,
)

from .config import QualificationWorkerConfig

log = logging.getLogger(__name__)

AGENT_ENTRYPOINT = "agent.py"
AGENT_JSON = "agent.json"


async def run(
    *,
    db: QualificationDb,
    client: LLMClient,
    config: QualificationWorkerConfig,
    stop: asyncio.Event,
) -> None:
    """Poll for one candidate at a time until *stop* is set."""
    log.info(
        "qualification worker running: window=%d poll %.0fs submissions_dir=%s",
        config.window_size,
        config.poll_seconds,
        config.submissions_dir,
    )
    while not stop.is_set():
        try:
            await run_once(db=db, client=client, config=config)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            log.exception("qualification tick failed")
            get_axiom().exception("qualification", "unexpected_error", error=str(ex))
        await _sleep_until_stop(stop, config.poll_seconds)


async def run_once(
    *,
    db: QualificationDb,
    client: LLMClient,
    config: QualificationWorkerConfig,
) -> bool:
    """Process at most one candidate. Returns whether work was attempted."""
    candidate = await db.next_candidate(window_size=config.window_size)
    if candidate is None:
        log.debug("qualification tick: no candidate in queue head")
        return False

    started = time.monotonic()
    base_files = _load_base_files(config.base_path)
    base_files_available = base_files is not None
    try:
        submitted_files = _load_submission_files(
            config.submissions_dir, candidate.submission_id
        )
        patch = (
            _files_patch(base_files=base_files, submitted_files=submitted_files)
            if base_files is not None
            else ""
        )
        result = await qualify_submission_security(
            SecurityQualificationInput(
                submitted_files=submitted_files,
                base_files=base_files,
                patch=patch,
                hotkey=candidate.hotkey,
                submission_id=candidate.submission_id,
                metadata={
                    "block": candidate.block,
                    "agent_files": candidate.agent_files,
                    "base_files_available": base_files_available,
                },
            ),
            client=client,
            config=config.security,
        )
        outcome = qualification_outcome(result)
        failures = security_failures(result)
        saved = await db.save_qualification(
            submission_id=candidate.submission_id,
            result=result,
            outcome=outcome,
            base_files_available=base_files_available,
            failures=failures,
            duration_seconds=time.monotonic() - started,
        )
        if saved:
            log.info(
                "qualified submission %s -> %s verdict=%s model=%s",
                candidate.submission_id,
                outcome.value,
                result.verdict,
                result.model,
            )
            get_axiom().info(
                source="qualification",
                event_type="submission_qualification",
                submission_id=candidate.submission_id,
                hotkey=candidate.hotkey,
                outcome=outcome.value,
                verdict=result.verdict,
                model=result.model,
                base_files_available=base_files_available,
                failures=failures,
            )
        else:
            log.info(
                "qualification result for %s skipped: status changed",
                candidate.submission_id,
            )
        return True
    except Exception as ex:
        saved = await db.save_error(
            submission_id=candidate.submission_id,
            error=str(ex),
            base_files_available=base_files_available,
            model=getattr(client, "model", None),
            duration_seconds=time.monotonic() - started,
        )
        log.warning(
            "qualification error for submission %s: %s",
            candidate.submission_id,
            ex,
        )
        get_axiom().exception(
            "qualification",
            "submission_qualification_error",
            submission_id=candidate.submission_id,
            hotkey=candidate.hotkey,
            saved=saved,
            error=str(ex),
        )
        return True


def _load_submission_files(submissions_dir: Path, submission_id: str) -> dict[str, str]:
    bundle = submissions_dir / submission_id
    if not bundle.is_dir():
        raise FileNotFoundError(f"submission bundle is missing: {bundle}")
    files = _load_agent_json_files(bundle) or _scan_python_files(bundle)
    if AGENT_ENTRYPOINT not in files:
        raise FileNotFoundError(f"submission bundle has no {AGENT_ENTRYPOINT}: {bundle}")
    return files


def _load_base_files(base_path: Path | None) -> dict[str, str] | None:
    if base_path is None:
        return None
    try:
        return _read_agent_tree(base_path.expanduser())
    except Exception as ex:
        log.warning("base files unavailable at %s: %s", base_path, ex)
        return None


def _load_agent_json_files(bundle: Path) -> dict[str, str] | None:
    path = bundle / AGENT_JSON
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return None
    result = {
        str(name): str(content)
        for name, content in sorted(files.items(), key=lambda item: str(item[0]))
        if str(name).endswith(".py")
    }
    return result or None


def _read_agent_tree(path: Path) -> dict[str, str]:
    if path.is_file():
        if path.is_symlink():
            raise ValueError(f"agent file is a symlink: {path}")
        return {AGENT_ENTRYPOINT: path.read_text(encoding="utf-8", errors="replace")}
    if not path.is_dir():
        raise FileNotFoundError(f"agent path does not exist: {path}")
    files = _scan_python_files(path)
    if AGENT_ENTRYPOINT not in files:
        raise FileNotFoundError(f"agent tree has no {AGENT_ENTRYPOINT}: {path}")
    return files


def _scan_python_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for file_path in sorted(root.rglob("*.py")):
        if file_path.is_symlink() or not file_path.is_file():
            continue
        relative = file_path.relative_to(root)
        if any(part == "__pycache__" or part.startswith(".") for part in relative.parts):
            continue
        files[relative.as_posix()] = file_path.read_text(
            encoding="utf-8", errors="replace"
        )
    return files


def _files_patch(
    *,
    base_files: Mapping[str, str],
    submitted_files: Mapping[str, str],
) -> str:
    chunks: list[str] = []
    for path in sorted(set(base_files) | set(submitted_files)):
        base_text = base_files.get(path)
        submitted_text = submitted_files.get(path)
        chunks.append(
            "".join(
                difflib.unified_diff(
                    (base_text or "").splitlines(keepends=True),
                    (submitted_text or "").splitlines(keepends=True),
                    fromfile=f"a/{path}" if base_text is not None else "/dev/null",
                    tofile=f"b/{path}" if submitted_text is not None else "/dev/null",
                )
            )
        )
    return "".join(chunks)


async def _sleep_until_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass
