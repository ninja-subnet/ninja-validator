"""Configuration for the background Hugging Face archive worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tau.huggingface import HFDatasetConfig
from tau.utils.env import (
    env_bool_strict,
    env_float_strict,
    env_int_strict,
    env_str,
)


@dataclass(frozen=True, slots=True)
class HFArchiverConfig:
    repo_id: str | None = None
    token: str | None = None
    revision: str = "main"
    private: bool = True
    endpoint: str | None = None
    staging_dir: Path = Path("/var/lib/tau/hf-staging")
    batch_size: int = 25
    shard_size_mb: int = 512
    poll_seconds: float = 15.0
    lease_seconds: float = 6 * 60 * 60
    retry_base_seconds: float = 60.0
    retry_max_seconds: float = 60 * 60

    def __post_init__(self) -> None:
        if self.repo_id and not self.token:
            raise ValueError("TAU_HF_DATASET_REPO requires HF_TOKEN or TAU_HF_TOKEN")
        if not self.revision.strip():
            raise ValueError("TAU_HF_DATASET_REVISION must not be blank")
        if self.batch_size <= 0:
            raise ValueError("TAU_HF_EXPORT_BATCH_SIZE must be positive")
        if self.shard_size_mb <= 0:
            raise ValueError("TAU_HF_SHARD_SIZE_MB must be positive")
        if (
            min(
                self.poll_seconds,
                self.lease_seconds,
                self.retry_base_seconds,
                self.retry_max_seconds,
            )
            <= 0
        ):
            raise ValueError("Hugging Face archive timing values must be positive")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("TAU_HF_RETRY_MAX_SECONDS must be >= retry base")

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id and self.token)

    @property
    def dataset(self) -> HFDatasetConfig:
        if not self.repo_id:
            raise ValueError("Hugging Face archiver is disabled")
        return HFDatasetConfig(
            repo_id=self.repo_id,
            revision=self.revision,
            private=self.private,
            staging_dir=self.staging_dir,
            batch_size=self.batch_size,
            shard_size_bytes=self.shard_size_mb * 1024 * 1024,
        )

    def retry_delay(self, attempt: int) -> float:
        exponent = max(0, min(attempt - 1, 20))
        return min(self.retry_max_seconds, self.retry_base_seconds * (2**exponent))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> HFArchiverConfig:
        env = os.environ if environ is None else environ
        defaults = cls()
        repo = env_str(env, "TAU_HF_DATASET_REPO", "")
        token = env_str(env, "TAU_HF_TOKEN", env_str(env, "HF_TOKEN", ""))
        return cls(
            repo_id=repo or None,
            token=token or None,
            revision=env_str(env, "TAU_HF_DATASET_REVISION", defaults.revision),
            private=env_bool_strict(env, "TAU_HF_DATASET_PRIVATE", defaults.private),
            endpoint=env_str(env, "TAU_HF_ENDPOINT", "") or None,
            staging_dir=Path(
                env_str(env, "TAU_HF_STAGING_DIR", str(defaults.staging_dir))
            ),
            batch_size=env_int_strict(
                env, "TAU_HF_EXPORT_BATCH_SIZE", defaults.batch_size
            ),
            shard_size_mb=env_int_strict(
                env, "TAU_HF_SHARD_SIZE_MB", defaults.shard_size_mb
            ),
            poll_seconds=env_float_strict(
                env, "TAU_HF_POLL_SECONDS", defaults.poll_seconds
            ),
            lease_seconds=env_float_strict(
                env, "TAU_HF_LEASE_SECONDS", defaults.lease_seconds
            ),
            retry_base_seconds=env_float_strict(
                env, "TAU_HF_RETRY_BASE_SECONDS", defaults.retry_base_seconds
            ),
            retry_max_seconds=env_float_strict(
                env, "TAU_HF_RETRY_MAX_SECONDS", defaults.retry_max_seconds
            ),
        )
