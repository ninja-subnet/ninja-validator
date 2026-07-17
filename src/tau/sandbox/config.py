"""Tunable configuration for the sandbox executor.

Resource limits, timeouts, and hardening flags for the per-solve container, plus
the LLM model the proxy enforces. All knobs default to safe values and are
overridable via ``TAU_SANDBOX_*`` env vars, so a caller usually just calls
``SandboxConfig.from_env()``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tau.utils.env import env_bool, env_float, env_int, env_str


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    # --- image ---
    image_name: str = "tau-sandbox"
    no_cache: bool = False  # force a rebuild of the sandbox image

    # --- the model the proxy forces every agent request onto ---
    model: str = ""

    # --- container resource limits (untrusted miner code runs here) ---
    memory: str = "2g"  # docker mem_limit; memswap is pinned equal => no swap
    cpus: float = 1.0
    pids_limit: int = 256
    nofile_limit: int = 4096
    work_tmpfs_size: str = "1g"  # writable /work (read-only rootfs otherwise)
    tmp_tmpfs_size: str = "512m"  # writable /tmp

    # --- hardening ---
    drop_caps: bool = True
    no_new_privileges: bool = True
    read_only_rootfs: bool = True
    run_as_user: str | None = None  # e.g. "1000:1000"; None keeps the image default

    # --- timeouts (seconds) ---
    hard_timeout_seconds: int = 600  # absolute wall-clock cap on a solve
    first_token_timeout_seconds: int = 300  # kill if the model never responds
    container_ttl_seconds: int = 3600  # the `sleep` keeping the container alive
    # How long one upstream LLM call may take before the proxy times it out (the httpx
    # read timeout). A timeout is treated as a miner-unrelated infra fault (retry).
    proxy_request_timeout_seconds: float = 600.0
    smart_cache_routing: bool = True
    # Persist redacted request/response bodies for training/evaluation exports.
    rollout_capture_enabled: bool = True

    def memswap_limit(self) -> str:
        # Equal to memory => the container gets no swap.
        return self.memory

    def nano_cpus(self) -> int:
        return int(self.cpus * 1_000_000_000)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SandboxConfig:
        env = os.environ if environ is None else environ
        d = cls()
        model = env_str(env, "SOLVER_MODEL", "")
        if not model:
            raise OSError("SOLVER_MODEL not set")
        return cls(
            image_name=env_str(env, "TAU_SANDBOX_IMAGE_NAME", d.image_name),
            no_cache=env_bool(env, "TAU_SANDBOX_NO_CACHE", d.no_cache),
            model=model,
            memory=env_str(env, "TAU_SANDBOX_MEMORY", d.memory),
            cpus=env_float(env, "TAU_SANDBOX_CPUS", d.cpus),
            pids_limit=env_int(env, "TAU_SANDBOX_PIDS_LIMIT", d.pids_limit),
            nofile_limit=env_int(env, "TAU_SANDBOX_NOFILE_LIMIT", d.nofile_limit),
            work_tmpfs_size=env_str(env, "TAU_SANDBOX_WORK_TMPFS_SIZE", d.work_tmpfs_size),
            tmp_tmpfs_size=env_str(env, "TAU_SANDBOX_TMP_TMPFS_SIZE", d.tmp_tmpfs_size),
            drop_caps=env_bool(env, "TAU_SANDBOX_DROP_CAPS", d.drop_caps),
            no_new_privileges=env_bool(
                env, "TAU_SANDBOX_NO_NEW_PRIVILEGES", d.no_new_privileges
            ),
            read_only_rootfs=env_bool(env, "TAU_SANDBOX_READ_ONLY_ROOTFS", d.read_only_rootfs),
            run_as_user=(env.get("TAU_SANDBOX_USER") or None),
            hard_timeout_seconds=env_int(
                env, "TAU_SANDBOX_HARD_TIMEOUT_SECONDS", d.hard_timeout_seconds
            ),
            first_token_timeout_seconds=env_int(
                env, "TAU_SANDBOX_FIRST_TOKEN_TIMEOUT_SECONDS", d.first_token_timeout_seconds
            ),
            container_ttl_seconds=env_int(
                env, "TAU_SANDBOX_CONTAINER_TTL_SECONDS", d.container_ttl_seconds
            ),
            proxy_request_timeout_seconds=env_float(
                env, "TAU_PROXY_REQUEST_TIMEOUT_SECONDS", d.proxy_request_timeout_seconds
            ),
            smart_cache_routing=env_bool(
                env, "TAU_SOLVER_SMART_CACHE_ROUTING", d.smart_cache_routing
            ),
            rollout_capture_enabled=env_bool(
                env, "TAU_ROLLOUT_CAPTURE_ENABLED", d.rollout_capture_enabled
            ),
        )
