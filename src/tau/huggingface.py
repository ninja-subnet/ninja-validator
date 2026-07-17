"""Hugging Face dataset publication for retired-king tasks and agent rollouts.

The module keeps ``huggingface_hub`` optional: production/CLI wiring injects its
``HfApi`` and ``CommitOperationAdd`` objects, while schema and publisher tests need
no network dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import re
import shutil
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3
_PREVIEW_CHARS = 6_000

# Explicit schemas keep every retired-king Parquet shard compatible even when one
# shard has no events or a nullable column happens to contain only NULL values.
_VIEWER_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "tasks": (
        ("schema_version", "int"),
        ("task_id", "str"),
        ("task_owner_king_id", "str"),
        ("pool", "str"),
        ("pool_id", "int"),
        ("status", "str"),
        ("status_id", "int"),
        ("problem_statement", "str"),
        ("repo_clone_url", "str"),
        ("parent_sha", "str"),
        ("commit_sha", "str"),
        ("reference_patch_preview", "str"),
        ("reference_patch_chars", "int"),
        ("content_fingerprint", "str"),
        ("created_at", "str"),
        ("generation_model", "str"),
        ("generation_fetch_seconds", "float"),
        ("generation_llm_seconds", "float"),
        ("generation_llm_attempt", "int"),
        ("generation_rejected_duplicate", "int"),
        ("generation_rejected_structural", "int"),
        ("generation_rejected_quality", "int"),
        ("generation_rejected_fetch_error", "int"),
        ("screening_king_score", "float"),
        ("screening_max_score", "float"),
        ("screening_reason", "str"),
        ("screening_model", "str"),
        ("screening_failed_runs", "int"),
        ("screening_created_at", "str"),
        ("screening_updated_at", "str"),
    ),
    "rollouts": (
        ("schema_version", "int"),
        ("rollout_id", "str"),
        ("phase", "str"),
        ("task_id", "str"),
        ("task_owner_king_id", "str"),
        ("challenge_id", "str"),
        ("challenge_king_id", "str"),
        ("submission_id", "str"),
        ("role", "str"),
        ("success", "bool"),
        ("solution_preview", "str"),
        ("solution_chars", "int"),
        ("exit_reason", "str"),
        ("duration_seconds", "float"),
        ("capture_available", "bool"),
        ("event_count", "int"),
        ("models", "str"),
        ("created_at", "str"),
        ("usage_request_count", "int"),
        ("usage_rejected_request_count", "int"),
        ("usage_success_count", "int"),
        ("usage_error_count", "int"),
        ("usage_upstream_error_count", "int"),
        ("usage_upstream_timeout_count", "int"),
        ("usage_prompt_tokens", "int"),
        ("usage_completion_tokens", "int"),
        ("usage_total_tokens", "int"),
        ("usage_cached_tokens", "int"),
        ("usage_cache_write_tokens", "int"),
        ("usage_reasoning_tokens", "int"),
        ("usage_cost", "float"),
        ("usage_budget_exceeded_reason", "str"),
        ("judgement_winner", "str"),
        ("judgement_king_score", "float"),
        ("judgement_challenger_score", "float"),
        ("judgement_model", "str"),
        ("judgement_rationale", "str"),
        ("judgement_error", "str"),
        ("judgement_attempts", "int"),
        ("judgement_duration_seconds", "float"),
        ("judgement_created_at", "str"),
    ),
    "events": (
        ("schema_version", "int"),
        ("event_id", "str"),
        ("rollout_id", "str"),
        ("task_id", "str"),
        ("task_owner_king_id", "str"),
        ("phase", "str"),
        ("role", "str"),
        ("submission_id", "str"),
        ("challenge_id", "str"),
        ("event_index", "int"),
        ("event_type", "str"),
        ("source", "str"),
        ("started_at", "str"),
        ("finished_at", "str"),
        ("method", "str"),
        ("path", "str"),
        ("status_code", "int"),
        ("latency_ms", "int"),
        ("model_requested", "str"),
        ("model_effective", "str"),
        ("cost", "float"),
        ("prompt_tokens", "int"),
        ("completion_tokens", "int"),
        ("total_tokens", "int"),
        ("cached_tokens", "int"),
        ("cache_write_tokens", "int"),
        ("reasoning_tokens", "int"),
        ("request_preview", "str"),
        ("request_chars", "int"),
        ("request_truncated", "bool"),
        ("response_preview", "str"),
        ("response_chars", "int"),
        ("response_truncated", "bool"),
    ),
    "payloads": (
        ("schema_version", "int"),
        ("payload_id", "str"),
        ("event_id", "str"),
        ("rollout_id", "str"),
        ("task_id", "str"),
        ("task_owner_king_id", "str"),
        ("phase", "str"),
        ("role", "str"),
        ("submission_id", "str"),
        ("challenge_id", "str"),
        ("event_index", "int"),
        ("direction", "str"),
        ("chunk_index", "int"),
        ("chunk_count", "int"),
        ("content_offset", "int"),
        ("content", "str"),
        ("total_chars", "int"),
        ("content_sha256", "str"),
        ("encoding", "str"),
    ),
}

DATASET_CARD = """---
configs:
- config_name: tasks
  data_files:
  - split: train
    path: "data/tasks/*.parquet"
- config_name: rollouts
  default: true
  data_files:
  - split: train
    path: "data/rollouts/*.parquet"
- config_name: events
  data_files:
  - split: train
    path: "data/events/*.parquet"
- config_name: payloads
  data_files:
  - split: train
    path: "data/payloads/*.parquet"
---

# Tau retired-king tasks and rollouts

This dataset is written by the Tau validator when a challenger becomes king.

- `tasks` contains one viewer-friendly row per generated task.
- `rollouts` contains one viewer-friendly row per terminal qualification or duel
  solve and is the default table shown on the dataset page.
- `events` contains one flattened row per redacted proxy-observed LLM call.
- `payloads` contains complete solution diffs plus request and response bodies split
  into bounded, ordered chunks. Join on `rollout_id`/`event_id`, filter by
  `direction`, then order by `chunk_index` to reconstruct a value without truncation.
- `task_id`, `rollout_id`, and `event_id` join the tables. `task_owner_king_id`
  groups each retired king's immutable shard.

The tables deliberately use bounded previews for patches and LLM payloads so the
Hub viewer remains responsive. Lossless normalized JSONL is retained as
gzip-compressed shards under `data/raw/` for training and custom processing.

The `events` subset is intentionally absent until the first post-capture king with
stored LLM calls is retired; historical solves did not retain those call bodies.

Load any current table independently:

```python
from datasets import load_dataset

tasks = load_dataset("REPO_ID", "tasks", split="train")
rollouts = load_dataset("REPO_ID", "rollouts", split="train")
events = load_dataset("REPO_ID", "events", split="train")
payloads = load_dataset("REPO_ID", "payloads", split="train")
```

Rows with `capture_available=false` predate full rollout capture. Their solution and
usage summary are retained, but historical request/response bodies cannot be
reconstructed.
"""

_EVENTS_CONFIG_BLOCK = """- config_name: events
  data_files:
  - split: train
    path: "data/events/*.parquet"
"""
_EVENTS_DESCRIPTION = (
    "- `events` contains one flattened row per redacted proxy-observed LLM call.\n"
)
_EVENTS_LOAD = 'events = load_dataset("REPO_ID", "events", split="train")\n'


@dataclass(frozen=True, slots=True)
class KingDatasetSnapshot:
    """JSON-ready normalized rows for one king's complete task history."""

    king_id: str
    tasks: tuple[dict[str, Any], ...]
    rollouts: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class HFDatasetConfig:
    repo_id: str
    revision: str = "main"
    private: bool = True
    staging_dir: Path = Path("/var/lib/tau/hf-staging")
    batch_size: int = 25
    shard_size_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("Hugging Face dataset repo must not be blank")
        if not self.revision.strip():
            raise ValueError("Hugging Face dataset revision must not be blank")
        if self.batch_size <= 0:
            raise ValueError("Hugging Face export batch size must be positive")
        if self.shard_size_bytes <= 0:
            raise ValueError("Hugging Face shard size must be positive")


class KingDatasetSource(Protocol):
    async def export_king_tasks(self, king_id: str) -> tuple[dict[str, Any], ...]: ...

    def stream_king_rollouts(
        self, king_id: str, *, batch_size: int
    ) -> AsyncIterator[tuple[dict[str, Any], ...]]: ...


class HFApi(Protocol):
    def create_repo(self, **kwargs: Any) -> Any: ...

    def create_commit(self, **kwargs: Any) -> Any: ...

    def list_repo_files(self, **kwargs: Any) -> list[str]: ...


class HFDatasetPublisher:
    """Stream one king into bounded local shards, then publish one Hub commit."""

    def __init__(
        self,
        source: KingDatasetSource,
        *,
        api: HFApi,
        operation_factory: Callable[..., Any],
        config: HFDatasetConfig,
    ) -> None:
        self._source = source
        self._api = api
        self._operation_factory = operation_factory
        self._config = config

    async def publish_retired_king(self, king_id: str, promoted_to: str | None) -> Any:
        component = _safe_component(king_id)
        root = self._config.staging_dir / component
        await asyncio.to_thread(_prepare_staging_dir, root)
        await asyncio.to_thread(
            self._api.create_repo,
            repo_id=self._config.repo_id,
            repo_type="dataset",
            private=self._config.private,
            exist_ok=True,
        )
        remote_files = await asyncio.to_thread(
            self._api.list_repo_files,
            repo_id=self._config.repo_id,
            repo_type="dataset",
            revision=self._config.revision,
        )
        stager = _DatasetStager(
            root,
            component=component,
            shard_size_bytes=self._config.shard_size_bytes,
        )
        try:
            tasks = await self._source.export_king_tasks(king_id)
            await asyncio.to_thread(stager.write_tasks, tasks)
            async for batch in self._source.stream_king_rollouts(
                king_id, batch_size=self._config.batch_size
            ):
                await asyncio.to_thread(stager.write_rollouts, batch)
            staged = await asyncio.to_thread(
                stager.finish,
                king_id=king_id,
                promoted_to=promoted_to,
                repo_id=self._config.repo_id,
                remote_has_events=any(
                    path.startswith("data/events/") and path.endswith(".parquet")
                    for path in remote_files
                ),
            )
            result = await asyncio.to_thread(self._publish, staged)
        except BaseException:
            await asyncio.to_thread(stager.close)
            raise
        else:
            await asyncio.to_thread(shutil.rmtree, root, True)
            return result

    def _publish(self, staged: StagedKingDataset) -> Any:
        operations = [
            self._operation_factory(
                path_in_repo=path.relative_to(staged.root).as_posix(),
                path_or_fileobj=path,
            )
            for path in sorted(staged.root.rglob("*"))
            if path.is_file()
        ]
        return self._api.create_commit(
            repo_id=self._config.repo_id,
            repo_type="dataset",
            revision=self._config.revision,
            operations=operations,
            commit_message=(
                f"Archive retired king {staged.king_id} "
                f"({staged.task_count} tasks, {staged.rollout_count} rollouts)"
            ),
        )


@dataclass(frozen=True, slots=True)
class StagedKingDataset:
    root: Path
    king_id: str
    task_count: int
    rollout_count: int
    event_count: int
    payload_chunk_count: int


def _prepare_staging_dir(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)


def _arrow_schema(schema_name: str) -> Any:
    import pyarrow as pa

    type_map = {
        "bool": pa.bool_(),
        "float": pa.float64(),
        "int": pa.int64(),
        "str": pa.string(),
    }
    return pa.schema(
        [(name, type_map[kind]) for name, kind in _VIEWER_SCHEMAS[schema_name]]
    )


class _ParquetShardWriter:
    def __init__(
        self,
        directory: Path,
        *,
        component: str,
        schema_name: str,
        max_bytes: int,
    ) -> None:
        self._directory = directory
        self._component = component
        self._schema_name = schema_name
        self._schema = _arrow_schema(schema_name)
        self._max_bytes = max_bytes
        self._part = 0
        self._sink: Any | None = None
        self._writer: Any | None = None

    def write(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        import pyarrow as pa

        if self._writer is None:
            self._open()
        table = pa.Table.from_pylist(list(rows), schema=self._schema)
        self._writer.write_table(table)
        if self._sink.tell() >= self._max_bytes:
            self._close_part()

    def close(self) -> None:
        self._close_part()

    def _open(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{self._component}-{self._part:05d}.parquet"
        self._sink = pa.OSFile(str(path), "wb")
        self._writer = pq.ParquetWriter(
            self._sink,
            self._schema,
            compression="zstd",
            write_page_index=True,
        )
        self._part += 1

    def _close_part(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None


class _JsonlShardWriter:
    def __init__(self, directory: Path, *, component: str, max_bytes: int) -> None:
        self._directory = directory
        self._component = component
        self._max_bytes = max_bytes
        self._part = 0
        self._size = 0
        self._handle: Any | None = None

    def write(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            line = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            size = len(line.encode("utf-8"))
            if (
                self._handle is not None
                and self._size
                and self._size + size > self._max_bytes
            ):
                self._close_part()
            if self._handle is None:
                self._open()
            self._handle.write(line)
            self._size += size

    def close(self) -> None:
        self._close_part()

    def _open(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{self._component}-{self._part:05d}.jsonl.gz"
        self._handle = gzip.open(path, "wt", encoding="utf-8")
        self._part += 1
        self._size = 0

    def _close_part(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._size = 0


class _DatasetStager:
    def __init__(self, root: Path, *, component: str, shard_size_bytes: int) -> None:
        self._root = root
        self._component = component
        self._task_count = 0
        self._rollout_count = 0
        self._event_count = 0
        self._payload_count = 0
        self._closed = False
        self._rollouts = _ParquetShardWriter(
            root / "data/rollouts",
            component=component,
            schema_name="rollouts",
            max_bytes=shard_size_bytes,
        )
        self._events = _ParquetShardWriter(
            root / "data/events",
            component=component,
            schema_name="events",
            max_bytes=shard_size_bytes,
        )
        self._payloads = _ParquetShardWriter(
            root / "data/payloads",
            component=component,
            schema_name="payloads",
            max_bytes=shard_size_bytes,
        )
        self._raw_rollouts = _JsonlShardWriter(
            root / "data/raw/rollouts",
            component=component,
            max_bytes=shard_size_bytes,
        )

    def write_tasks(self, rows: Sequence[dict[str, Any]]) -> None:
        self._task_count = len(rows)
        _write_parquet_file(
            self._root / f"data/tasks/{self._component}.parquet",
            _task_viewer_rows(rows),
            "tasks",
        )
        _write_jsonl_gzip(
            self._root / f"data/raw/tasks/{self._component}.jsonl.gz", rows
        )

    def write_rollouts(self, rows: Sequence[dict[str, Any]]) -> None:
        rollout_rows, event_rows, payload_rows = _rollout_viewer_rows(rows)
        self._rollouts.write(rollout_rows)
        self._events.write(event_rows)
        self._payloads.write(payload_rows)
        self._raw_rollouts.write(rows)
        self._rollout_count += len(rollout_rows)
        self._event_count += len(event_rows)
        self._payload_count += len(payload_rows)

    def finish(
        self,
        *,
        king_id: str,
        promoted_to: str | None,
        repo_id: str,
        remote_has_events: bool,
    ) -> StagedKingDataset:
        self.close()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "retired_king_id": king_id,
            "promoted_to": promoted_to,
            "task_count": self._task_count,
            "rollout_count": self._rollout_count,
            "event_count": self._event_count,
            "payload_chunk_count": self._payload_count,
        }
        card = _dataset_card(
            repo_id, include_events=bool(self._event_count) or remote_has_events
        )
        (self._root / "README.md").write_text(card, encoding="utf-8")
        manifest_path = self._root / f"data/promotions/{self._component}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return StagedKingDataset(
            root=self._root,
            king_id=king_id,
            task_count=self._task_count,
            rollout_count=self._rollout_count,
            event_count=self._event_count,
            payload_chunk_count=self._payload_count,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._rollouts.close()
        self._events.close()
        self._payloads.close()
        self._raw_rollouts.close()
        self._closed = True


def _dataset_card(repo_id: str, *, include_events: bool) -> str:
    card = DATASET_CARD
    if not include_events:
        card = card.replace(_EVENTS_CONFIG_BLOCK, "")
        card = card.replace(_EVENTS_DESCRIPTION, "")
        card = card.replace(_EVENTS_LOAD, "")
    return card.replace("REPO_ID", repo_id)


def _write_jsonl_gzip(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_parquet_file(
    path: Path, rows: Sequence[dict[str, Any]], schema_name: str
) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema(schema_name)
    import pyarrow as pa

    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd", write_page_index=True)


def _preview(value: Any) -> tuple[str | None, int | None, bool]:
    if value is None:
        return None, None, False
    text, _ = _payload_text(value)
    return text[:_PREVIEW_CHARS], len(text), len(text) > _PREVIEW_CHARS


def _payload_text(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, "text"
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "json",
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _task_viewer_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    for row in rows:
        generation = _mapping(row.get("generation"))
        screening = _mapping(row.get("screening"))
        patch_preview, patch_chars, _ = _preview(row.get("reference_patch"))
        output.append(
            {
                "schema_version": row.get("schema_version"),
                "task_id": row.get("task_id"),
                "task_owner_king_id": row.get("task_owner_king_id"),
                "pool": row.get("pool"),
                "pool_id": row.get("pool_id"),
                "status": row.get("status"),
                "status_id": row.get("status_id"),
                "problem_statement": row.get("problem_statement"),
                "repo_clone_url": row.get("repo_clone_url"),
                "parent_sha": row.get("parent_sha"),
                "commit_sha": row.get("commit_sha"),
                "reference_patch_preview": patch_preview,
                "reference_patch_chars": patch_chars,
                "content_fingerprint": row.get("content_fingerprint"),
                "created_at": row.get("created_at"),
                "generation_model": generation.get("model"),
                "generation_fetch_seconds": generation.get("fetch_seconds"),
                "generation_llm_seconds": generation.get("llm_seconds"),
                "generation_llm_attempt": generation.get("llm_attempt"),
                "generation_rejected_duplicate": generation.get("rejected_duplicate"),
                "generation_rejected_structural": generation.get("rejected_structural"),
                "generation_rejected_quality": generation.get("rejected_quality"),
                "generation_rejected_fetch_error": generation.get(
                    "rejected_fetch_error"
                ),
                "screening_king_score": screening.get("king_score"),
                "screening_max_score": screening.get("max_score"),
                "screening_reason": screening.get("reason"),
                "screening_model": screening.get("model"),
                "screening_failed_runs": screening.get("failed_runs"),
                "screening_created_at": screening.get("created_at"),
                "screening_updated_at": screening.get("updated_at"),
            }
        )
    return tuple(output)


def _rollout_viewer_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    rollouts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for row in rows:
        usage = _mapping(row.get("usage"))
        judgement = _mapping(row.get("judgement"))
        raw_events = row.get("events")
        event_values = raw_events if isinstance(raw_events, list) else []
        solution_preview, solution_chars, _ = _preview(row.get("solution_diff"))
        models = sorted(
            {
                str(event.get("model_effective"))
                for event in event_values
                if isinstance(event, dict) and event.get("model_effective")
            }
        )
        rollouts.append(
            {
                "schema_version": row.get("schema_version"),
                "rollout_id": row.get("rollout_id"),
                "phase": row.get("phase"),
                "task_id": row.get("task_id"),
                "task_owner_king_id": row.get("task_owner_king_id"),
                "challenge_id": row.get("challenge_id"),
                "challenge_king_id": row.get("challenge_king_id"),
                "submission_id": row.get("submission_id"),
                "role": row.get("role"),
                "success": row.get("success"),
                "solution_preview": solution_preview,
                "solution_chars": solution_chars,
                "exit_reason": row.get("exit_reason"),
                "duration_seconds": row.get("duration_seconds"),
                "capture_available": row.get("capture_available"),
                "event_count": len(event_values),
                "models": ", ".join(models) or None,
                "created_at": row.get("created_at"),
                **_usage_viewer_fields(usage),
                "judgement_winner": judgement.get("winner"),
                "judgement_king_score": judgement.get("king_score"),
                "judgement_challenger_score": judgement.get("challenger_score"),
                "judgement_model": judgement.get("model"),
                "judgement_rationale": judgement.get("rationale"),
                "judgement_error": judgement.get("error"),
                "judgement_attempts": judgement.get("attempts"),
                "judgement_duration_seconds": judgement.get("duration_seconds"),
                "judgement_created_at": judgement.get("created_at"),
            }
        )
        solution = row.get("solution_diff")
        if solution is not None:
            payloads.extend(
                _payload_chunks(
                    solution,
                    direction="solution",
                    event_id=f"{row.get('rollout_id')}:solution",
                    event_index=-1,
                    rollout=row,
                )
            )
        for ordinal, event in enumerate(event_values):
            if not isinstance(event, dict):
                continue
            event_index = event.get("index", ordinal)
            request_preview, request_chars, request_truncated = _preview(
                event.get("request")
            )
            response_preview, response_chars, response_truncated = _preview(
                event.get("response")
            )
            event_usage = _mapping(event.get("usage"))
            event_id = f"{row.get('rollout_id')}:{event_index}"
            events.append(
                {
                    "schema_version": row.get("schema_version"),
                    "event_id": event_id,
                    "rollout_id": row.get("rollout_id"),
                    "task_id": row.get("task_id"),
                    "task_owner_king_id": row.get("task_owner_king_id"),
                    "phase": row.get("phase"),
                    "role": row.get("role"),
                    "submission_id": row.get("submission_id"),
                    "challenge_id": row.get("challenge_id"),
                    "event_index": event_index,
                    "event_type": event.get("type"),
                    "source": event.get("source"),
                    "started_at": event.get("started_at"),
                    "finished_at": event.get("finished_at"),
                    "method": event.get("method"),
                    "path": event.get("path"),
                    "status_code": event.get("status_code"),
                    "latency_ms": event.get("latency_ms"),
                    "model_requested": event.get("model_requested"),
                    "model_effective": event.get("model_effective"),
                    "cost": event.get("cost"),
                    "prompt_tokens": event_usage.get("prompt_tokens"),
                    "completion_tokens": event_usage.get("completion_tokens"),
                    "total_tokens": event_usage.get("total_tokens"),
                    "cached_tokens": event_usage.get("cached_tokens"),
                    "cache_write_tokens": event_usage.get("cache_write_tokens"),
                    "reasoning_tokens": event_usage.get("reasoning_tokens"),
                    "request_preview": request_preview,
                    "request_chars": request_chars,
                    "request_truncated": request_truncated,
                    "response_preview": response_preview,
                    "response_chars": response_chars,
                    "response_truncated": response_truncated,
                }
            )
            for direction in ("request", "response"):
                value = event.get(direction)
                if value is None:
                    continue
                payloads.extend(
                    _payload_chunks(
                        value,
                        direction=direction,
                        event_id=event_id,
                        event_index=event_index,
                        rollout=row,
                    )
                )
    return tuple(rollouts), tuple(events), tuple(payloads)


def _payload_chunks(
    value: Any,
    *,
    direction: str,
    event_id: str,
    event_index: Any,
    rollout: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split a complete body into viewer-safe rows without losing a character."""
    text, encoding = _payload_text(value)
    chunks = [
        text[offset : offset + _PREVIEW_CHARS]
        for offset in range(0, len(text), _PREVIEW_CHARS)
    ] or [""]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return [
        {
            "schema_version": rollout.get("schema_version"),
            "payload_id": f"{event_id}:{direction}:{chunk_index}",
            "event_id": event_id,
            "rollout_id": rollout.get("rollout_id"),
            "task_id": rollout.get("task_id"),
            "task_owner_king_id": rollout.get("task_owner_king_id"),
            "phase": rollout.get("phase"),
            "role": rollout.get("role"),
            "submission_id": rollout.get("submission_id"),
            "challenge_id": rollout.get("challenge_id"),
            "event_index": event_index,
            "direction": direction,
            "chunk_index": chunk_index,
            "chunk_count": len(chunks),
            "content_offset": chunk_index * _PREVIEW_CHARS,
            "content": chunk,
            "total_chars": len(text),
            "content_sha256": digest,
            "encoding": encoding,
        }
        for chunk_index, chunk in enumerate(chunks)
    ]


def _usage_viewer_fields(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "usage_request_count": usage.get("request_count"),
        "usage_rejected_request_count": usage.get("rejected_request_count"),
        "usage_success_count": usage.get("success_count"),
        "usage_error_count": usage.get("error_count"),
        "usage_upstream_error_count": usage.get("upstream_error_count"),
        "usage_upstream_timeout_count": usage.get("upstream_timeout_count"),
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
        "usage_cached_tokens": usage.get("cached_tokens"),
        "usage_cache_write_tokens": usage.get("cache_write_tokens"),
        "usage_reasoning_tokens": usage.get("reasoning_tokens"),
        "usage_cost": usage.get("cost"),
        "usage_budget_exceeded_reason": usage.get("budget_exceeded_reason"),
    }


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if cleaned == value and cleaned:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:80] or 'king'}-{digest}"


async def _backfill(king_ids: Sequence[str] | None = None) -> int:
    """CLI implementation: publish selected kings or every already-retired king."""
    from huggingface_hub import CommitOperationAdd, HfApi

    from tau.db import DuelResolverDb
    from tau.workers.hf_archiver.config import HFArchiverConfig

    config = HFArchiverConfig.from_env()
    if not config.enabled:
        raise RuntimeError(
            "set TAU_HF_DATASET_REPO and HF_TOKEN (or TAU_HF_TOKEN) first"
        )
    api_kwargs: dict[str, Any] = {"token": config.token}
    if config.endpoint:
        api_kwargs["endpoint"] = config.endpoint
    db = DuelResolverDb()
    try:
        history = await db.king_history()
        next_king = {
            king_id: history[index + 1] if index + 1 < len(history) else None
            for index, king_id in enumerate(history)
        }
        selected = list(king_ids) if king_ids else history[:-1]
        publisher = HFDatasetPublisher(
            db,
            api=HfApi(**api_kwargs),
            operation_factory=CommitOperationAdd,
            config=config.dataset,
        )
        for king_id in selected:
            if king_id not in next_king:
                raise ValueError(f"unknown king id: {king_id}")
            await publisher.publish_retired_king(king_id, next_king[king_id])
            log.info("published retired king %s", king_id)
        return len(selected)
    finally:
        await db.aclose()


def main() -> None:
    """Backfill historical retired kings into the configured HF dataset repo."""
    parser = argparse.ArgumentParser(
        description="Export retired Tau king tasks and rollouts to Hugging Face"
    )
    parser.add_argument(
        "--king",
        action="append",
        dest="king_ids",
        help="specific historical king id (repeatable); default: all retired kings",
    )
    args = parser.parse_args()
    count = asyncio.run(_backfill(args.king_ids))
    print(f"published {count} retired king dataset shard(s)")


if __name__ == "__main__":
    main()
