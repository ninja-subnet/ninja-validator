"""Shared duel-pool configuration.

How many tasks each duel pool should hold. Read by the **task-generator** (fills
each pool up to its target) and the **duel-resolver** (how many tasks a duel
draws from a pool), so the number lives here rather than inside either worker —
neither side owns it and the two cannot drift.

``PoolType.POOL_ONE`` is the monolith's "primary" pool, ``POOL_TWO`` its "retest"
pool. The targets are equal for now but kept as separate fields so they can
diverge without a schema or call-site change.

Importing this module touches nothing in the environment; call
:meth:`PoolTargets.from_env` explicitly to fold in ``TAU_POOL_*`` overrides.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tau.db.status import PoolType
from tau.utils.env import env_int


@dataclass(frozen=True, slots=True)
class PoolTargets:
    """Target task count per duel pool."""

    pool_one: int = 50
    pool_two: int = 50

    def __post_init__(self) -> None:
        if self.pool_one < 1:
            raise ValueError("pool_one target must be >= 1")
        if self.pool_two < 1:
            raise ValueError("pool_two target must be >= 1")

    def target(self, pool: PoolType) -> int:
        """Target task count for *pool*."""
        if pool is PoolType.POOL_ONE:
            return self.pool_one
        if pool is PoolType.POOL_TWO:
            return self.pool_two
        raise ValueError(f"unknown pool: {pool!r}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PoolTargets:
        """Build targets from ``TAU_POOL_*`` env vars, falling back to defaults.

        Pass *environ* to read from a mapping other than ``os.environ`` (tests).
        """
        env = os.environ if environ is None else environ
        d = cls()
        return cls(
            pool_one=env_int(env, "TAU_POOL_ONE_TARGET", d.pool_one),
            pool_two=env_int(env, "TAU_POOL_TWO_TARGET", d.pool_two),
        )
