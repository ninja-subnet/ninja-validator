from __future__ import annotations

import asyncio

from tau.db.duel_resolver import KingArchiveJob
from tau.workers.hf_archiver import HFArchiverConfig
from tau.workers.hf_archiver.main import run_hf_archiver


class _Db:
    def __init__(self, job: KingArchiveJob) -> None:
        self.job = job
        self.claimed = False
        self.completed: list[str] = []
        self.retried: list[tuple[str, str, float]] = []

    async def claim_king_archive(self, *, lease_seconds: float):  # noqa: ANN201
        assert lease_seconds > 0
        if self.claimed:
            return None
        self.claimed = True
        return self.job

    async def complete_king_archive(self, king_id: str) -> bool:
        self.completed.append(king_id)
        return True

    async def retry_king_archive(
        self, king_id: str, *, error: str, delay_seconds: float
    ) -> bool:
        self.retried.append((king_id, error, delay_seconds))
        return True


class _Publisher:
    def __init__(self, stop: asyncio.Event, *, error: Exception | None = None) -> None:
        self.stop = stop
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    async def publish_retired_king(
        self, king_id: str, promoted_to: str | None
    ) -> object:
        self.calls.append((king_id, promoted_to))
        self.stop.set()
        if self.error:
            raise self.error
        return object()


def test_config_builds_bounded_dataset_settings() -> None:
    config = HFArchiverConfig.from_env(
        {
            "TAU_HF_DATASET_REPO": "org/traces",
            "HF_TOKEN": "secret",
            "TAU_HF_EXPORT_BATCH_SIZE": "7",
            "TAU_HF_SHARD_SIZE_MB": "32",
        }
    )

    assert config.enabled
    assert config.dataset.batch_size == 7
    assert config.dataset.shard_size_bytes == 32 * 1024 * 1024
    assert config.retry_delay(1) == 60
    assert config.retry_delay(100) == 3600


async def test_worker_completes_successful_archive() -> None:
    stop = asyncio.Event()
    db = _Db(KingArchiveJob("old", "new", 1))
    publisher = _Publisher(stop)

    await run_hf_archiver(
        db=db, publisher=publisher, config=HFArchiverConfig(), stop=stop
    )

    assert publisher.calls == [("old", "new")]
    assert db.completed == ["old"]
    assert db.retried == []


async def test_worker_persists_retry_after_failure() -> None:
    stop = asyncio.Event()
    db = _Db(KingArchiveJob("old", "new", 2))
    publisher = _Publisher(stop, error=RuntimeError("upload failed"))

    await run_hf_archiver(
        db=db, publisher=publisher, config=HFArchiverConfig(), stop=stop
    )

    assert db.completed == []
    assert db.retried == [("old", "upload failed", 120.0)]
