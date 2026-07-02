"""Tunable configuration for the weight-setter worker."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tau.bittensor.types import FINNEY, NETUID
from tau.utils.env import (
    env_bool_strict,
    env_float_strict,
    env_int_strict,
    env_required,
    env_str,
)
from tau.weights.compute import KING_EMISSION_SHARES


@dataclass(frozen=True, slots=True)
class WeightSetterConfig:
    # --- chain / wallet (used to build the live gateway) ---
    network: str = FINNEY
    netuid: int = NETUID
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str = "~/.bittensor/wallets"

    # --- distribution ---
    # How many recent kings share emission (the reigning king + prior reigns). Capped
    # by the share table, so values above its length pay no extra slots.
    window: int = len(KING_EMISSION_SHARES)
    # The uid unclaimed shares burn to (uid 0 by convention).
    burn_uid: int = 0
    # Emit 100% to the burn uid, ignoring king history. A deliberate, restart-toggled
    # burn switch (no synthetic "burn king" needed; empty kings already burn 100%).
    burn_mode: bool = False

    # --- cadence ---
    # Set this many blocks (or fewer) before each epoch boundary, so the vector lands
    # just before consensus reads it, capturing the king as late as possible while
    # leaving time for inclusion.
    set_margin: int = 12
    # Idle sleep between poll ticks (seconds); ~one block at 12s/block.
    poll_seconds: float = 12.0

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be >= 0")
        if self.burn_uid < 0:
            raise ValueError("burn_uid must be >= 0")
        if self.set_margin < 0:
            raise ValueError("set_margin must be >= 0")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WeightSetterConfig:
        """Build a config from the environment, failing closed on bad input."""
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            network=env_str(env, "BITTENSOR_NETWORK", d.network),
            netuid=env_int_strict(env, "NETUID", d.netuid),
            wallet_name=env_required(env, "BITTENSOR_WALLET_NAME"),
            wallet_hotkey=env_required(env, "BITTENSOR_WALLET_HOTKEY"),
            wallet_path=env_str(env, "BITTENSOR_WALLET_PATH", d.wallet_path),
            window=env_int_strict(env, "TAU_WEIGHT_WINDOW", d.window),
            burn_uid=env_int_strict(env, "TAU_WEIGHT_BURN_UID", d.burn_uid),
            burn_mode=env_bool_strict(env, "TAU_WEIGHT_BURN_MODE", d.burn_mode),
            set_margin=env_int_strict(env, "TAU_WEIGHT_SET_MARGIN", d.set_margin),
            poll_seconds=env_float_strict(
                env, "TAU_WEIGHT_POLL_SECONDS", d.poll_seconds
            ),
        )
