"""Background worker for persistent retired-king Hugging Face archives."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Protocol

from tau.db import DuelResolverDb
from tau.huggingface import HFDatasetPublisher
from tau.utils.logging import configure_logging

from .config import HFArchiverConfig

log = logging.getLogger(__name__)


class ArchivePublisher(Protocol):
    async def publish_retired_king(
        self, king_id: str, promoted_to: str | None
    ) -> object: ...


async def run_hf_archiver(
    *,
    db: DuelResolverDb,
    publisher: ArchivePublisher,
    config: HFArchiverConfig,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        job = await db.claim_king_archive(lease_seconds=config.lease_seconds)
        if job is None:
            await _sleep_until_stop(stop, config.poll_seconds)
            continue
        try:
            await publisher.publish_retired_king(job.king_id, job.promoted_to)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            delay = config.retry_delay(job.attempt)
            await db.retry_king_archive(
                job.king_id, error=str(exc), delay_seconds=delay
            )
            log.exception(
                "retired king archive failed; retrying in %.0fs: %s",
                delay,
                job.king_id,
            )
        else:
            await db.complete_king_archive(job.king_id)
            log.info("archived retired king %s to Hugging Face", job.king_id)


async def _sleep_until_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def _serve(config: HFArchiverConfig) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    if not config.enabled:
        log.info("Hugging Face archiver disabled: dataset repo/token not configured")
        await stop.wait()
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api_kwargs: dict[str, object] = {"token": config.token}
    if config.endpoint:
        api_kwargs["endpoint"] = config.endpoint
    db = DuelResolverDb()
    try:
        publisher = HFDatasetPublisher(
            db,
            api=HfApi(**api_kwargs),
            operation_factory=CommitOperationAdd,
            config=config.dataset,
        )
        log.info(
            "Hugging Face archiver running: repo=%s batch=%d shard=%dMiB",
            config.repo_id,
            config.batch_size,
            config.shard_size_mb,
        )
        await run_hf_archiver(db=db, publisher=publisher, config=config, stop=stop)
    finally:
        await db.aclose()


def main() -> None:
    configure_logging()
    asyncio.run(_serve(HFArchiverConfig.from_env()))


if __name__ == "__main__":
    main()
