from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pyarrow.parquet as pq

from tau.huggingface import (
    HFDatasetConfig,
    HFDatasetPublisher,
    KingDatasetSnapshot,
    SCHEMA_VERSION,
    _safe_component,
)

_LONG_CONTENT = "fix it " * 1_200


class _Source:
    def __init__(self) -> None:
        self.snapshot: KingDatasetSnapshot | None = None
        self.batch_sizes: list[int] = []

    async def export_king_tasks(self, king_id: str):  # noqa: ANN201
        self.snapshot = self._snapshot(king_id)
        return self.snapshot.tasks

    async def stream_king_rollouts(self, king_id: str, *, batch_size: int):  # noqa: ANN201
        self.batch_sizes.append(batch_size)
        snapshot = self.snapshot or self._snapshot(king_id)
        for start in range(0, len(snapshot.rollouts), batch_size):
            yield snapshot.rollouts[start : start + batch_size]

    def _snapshot(self, king_id: str) -> KingDatasetSnapshot:
        return KingDatasetSnapshot(
            king_id=king_id,
            tasks=(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": "t1",
                    "task_owner_king_id": king_id,
                    "problem_statement": "fix the parser",
                    "reference_patch": "+fixed",
                },
            ),
            rollouts=(
                {
                    "schema_version": SCHEMA_VERSION,
                    "rollout_id": "r1",
                    "task_id": "t1",
                    "task_owner_king_id": king_id,
                    "phase": "duel",
                    "role": "challenger",
                    "solution_diff": "+fixed",
                    "usage": {"total_tokens": 9},
                    "events": [
                        {
                            "index": 0,
                            "type": "llm_call",
                            "model_effective": "model-a",
                            "usage": {"total_tokens": 9},
                            "request": {"messages": [{"content": _LONG_CONTENT}]},
                            "response": {"content": "done"},
                        }
                    ],
                },
            ),
        )


class _Api:
    def __init__(self, remote_files: list[str] | None = None) -> None:
        self.created: dict | None = None
        self.commit: dict | None = None
        self.files: dict[str, bytes] = {}
        self.remote_files = remote_files or []

    def create_repo(self, **kwargs):  # noqa: ANN003, ANN201
        self.created = kwargs

    def create_commit(self, **kwargs):  # noqa: ANN003, ANN201
        self.commit = kwargs
        self.files = {
            operation["path_in_repo"]: Path(operation["path_or_fileobj"]).read_bytes()
            for operation in kwargs["operations"]
        }
        return {"ok": True}

    def list_repo_files(self, **kwargs):  # noqa: ANN003, ANN201
        return self.remote_files


class _NoEventSource:
    async def export_king_tasks(self, king_id: str):  # noqa: ANN201
        del king_id
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": "t1",
                "task_owner_king_id": "old-king",
            },
        )

    async def stream_king_rollouts(self, king_id: str, *, batch_size: int):  # noqa: ANN201
        del king_id, batch_size
        yield (
            {
                "schema_version": SCHEMA_VERSION,
                "rollout_id": "r1",
                "task_id": "t1",
                "task_owner_king_id": "old-king",
                "solution_diff": "+legacy",
                "events": [],
            },
        )


async def test_publisher_commits_parseable_normalized_shards(tmp_path: Path) -> None:
    api = _Api()
    source = _Source()
    publisher = HFDatasetPublisher(
        source,
        api=api,
        operation_factory=lambda **kwargs: kwargs,
        config=HFDatasetConfig(
            repo_id="org/tau-data", private=True, staging_dir=tmp_path
        ),
    )

    result = await publisher.publish_retired_king("king/one", "king-two")

    assert result == {"ok": True}
    assert source.batch_sizes == [25]
    assert api.created == {
        "repo_id": "org/tau-data",
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    assert api.commit is not None
    operations = api.files
    component = _safe_component("king/one")
    assert set(operations) == {
        "README.md",
        f"data/tasks/{component}.parquet",
        f"data/rollouts/{component}-00000.parquet",
        f"data/events/{component}-00000.parquet",
        f"data/payloads/{component}-00000.parquet",
        f"data/raw/tasks/{component}.jsonl.gz",
        f"data/raw/rollouts/{component}-00000.jsonl.gz",
        f"data/promotions/{component}.json",
    }
    task = pq.read_table(
        io.BytesIO(operations[f"data/tasks/{component}.parquet"])
    ).to_pylist()[0]
    rollout = pq.read_table(
        io.BytesIO(operations[f"data/rollouts/{component}-00000.parquet"])
    ).to_pylist()[0]
    event = pq.read_table(
        io.BytesIO(operations[f"data/events/{component}-00000.parquet"])
    ).to_pylist()[0]
    payloads = pq.read_table(
        io.BytesIO(operations[f"data/payloads/{component}-00000.parquet"])
    ).to_pylist()
    raw_rollout = json.loads(
        gzip.decompress(operations[f"data/raw/rollouts/{component}-00000.jsonl.gz"])
    )
    manifest = json.loads(operations[f"data/promotions/{component}.json"])
    assert task["task_id"] == rollout["task_id"] == "t1"
    assert rollout["event_count"] == 1
    assert rollout["models"] == "model-a"
    assert event["rollout_id"] == "r1"
    assert "fix it" in event["request_preview"]
    assert event["request_truncated"] is True
    request_chunks = sorted(
        (row for row in payloads if row["direction"] == "request"),
        key=lambda row: row["chunk_index"],
    )
    reconstructed = "".join(row["content"] for row in request_chunks)
    assert json.loads(reconstructed)["messages"][0]["content"] == _LONG_CONTENT
    assert all(row["chunk_count"] == len(request_chunks) for row in request_chunks)
    solution_chunks = sorted(
        (row for row in payloads if row["direction"] == "solution"),
        key=lambda row: row["chunk_index"],
    )
    assert "".join(row["content"] for row in solution_chunks) == "+fixed"
    assert raw_rollout["events"][0]["request"]["messages"][0]["content"] == (
        _LONG_CONTENT
    )
    assert manifest == {
        "schema_version": SCHEMA_VERSION,
        "retired_king_id": "king/one",
        "promoted_to": "king-two",
        "task_count": 1,
        "rollout_count": 1,
        "event_count": 1,
        "payload_chunk_count": len(payloads),
    }


async def test_publisher_omits_empty_event_shard_and_config(tmp_path: Path) -> None:
    api = _Api()
    publisher = HFDatasetPublisher(
        _NoEventSource(),
        api=api,
        operation_factory=lambda **kwargs: kwargs,
        config=HFDatasetConfig(repo_id="org/tau-data", staging_dir=tmp_path),
    )

    await publisher.publish_retired_king("old-king", "new-king")

    assert api.commit is not None
    operations = api.files
    assert not any(path.startswith("data/events/") for path in operations)
    card = operations["README.md"].decode("utf-8")
    assert "config_name: events" not in card
    assert 'load_dataset("org/tau-data", "events"' not in card


async def test_publisher_rotates_disk_shards_between_bounded_batches(
    tmp_path: Path,
) -> None:
    class _ShardedSource(_NoEventSource):
        async def stream_king_rollouts(self, king_id: str, *, batch_size: int):  # noqa: ANN201
            assert batch_size == 1
            for index in range(3):
                yield (
                    {
                        "schema_version": SCHEMA_VERSION,
                        "rollout_id": f"r{index}",
                        "task_id": "t1",
                        "task_owner_king_id": king_id,
                        "solution_diff": "x" * 100,
                        "events": [],
                    },
                )

    api = _Api()
    publisher = HFDatasetPublisher(
        _ShardedSource(),
        api=api,
        operation_factory=lambda **kwargs: kwargs,
        config=HFDatasetConfig(
            repo_id="org/tau-data",
            staging_dir=tmp_path,
            batch_size=1,
            shard_size_bytes=1,
        ),
    )

    await publisher.publish_retired_king("old-king", "new-king")

    rollout_shards = sorted(
        path for path in api.files if path.startswith("data/rollouts/")
    )
    payload_shards = sorted(
        path for path in api.files if path.startswith("data/payloads/")
    )
    assert len(rollout_shards) == len(payload_shards) == 3
    assert not (tmp_path / "old-king").exists()


def test_safe_component_is_stable_and_path_safe() -> None:
    first = _safe_component("../king/one")
    assert first == _safe_component("../king/one")
    assert "/" not in first
    assert ".." not in first
