"""Unit tests for the shared duel-pool configuration (no database needed)."""

from __future__ import annotations

import pytest

from tau.db.status import PoolType
from tau.pools import PoolTargets


def test_defaults_are_fifty_per_pool() -> None:
    targets = PoolTargets()
    assert targets.target(PoolType.POOL_ONE) == 50
    assert targets.target(PoolType.POOL_TWO) == 50


def test_from_env_overrides_each_pool_independently() -> None:
    targets = PoolTargets.from_env(
        {"TAU_POOL_ONE_TARGET": "7", "TAU_POOL_TWO_TARGET": "3"}
    )
    assert (targets.pool_one, targets.pool_two) == (7, 3)


def test_from_env_falls_back_to_defaults_for_missing_or_bad_values() -> None:
    targets = PoolTargets.from_env({"TAU_POOL_ONE_TARGET": "not-an-int"})
    assert (targets.pool_one, targets.pool_two) == (50, 50)


@pytest.mark.parametrize("kwargs", [{"pool_one": 0}, {"pool_two": -1}])
def test_non_positive_targets_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        PoolTargets(**kwargs)
