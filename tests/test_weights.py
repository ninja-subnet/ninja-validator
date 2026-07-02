"""Tests for the weight-setter's pure core and loop dispatch.

The distribution rule (`compute_weights`) and the cadence gate (`should_set`,
`blocks_until_next_epoch`) are pure; the loop `step`/`run` are driven by a fake chain
and a faked DB seam. None of this touches the bittensor SDK or a database.
"""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Sequence

import pytest

from tau.db.weight_setter import WeightSetterDb
from tau.weights.compute import (
    KING_EMISSION_SHARES,
    compute_weights,
    king_emission_shares,
)
from tau.weights.schedule import blocks_until_next_epoch, should_set
from tau.weights.types import (
    MetagraphView,
    PollState,
    RecentKing,
    SubnetParams,
    WeightPlan,
)
from tau.workers.weight_setter import WeightSetterConfig, step
from tau.workers.weight_setter.loop import StepResult, run

NETUID = 66
BURN_UID = 0


def _meta(hotkeys_by_uid: dict[int, str]) -> MetagraphView:
    return MetagraphView(
        uids=tuple(hotkeys_by_uid),
        uid_by_hotkey={hot: uid for uid, hot in hotkeys_by_uid.items()},
    )


def _king(king_id: str, hotkey: str) -> RecentKing:
    return RecentKing(king_id=king_id, hotkey=hotkey)


def _weight_of(plan: WeightPlan, uid: int) -> float:
    return plan.weights[plan.uids.index(uid)]


# -- king_emission_shares -----------------------------------------------------------


def test_shares_default_window_is_the_full_table() -> None:
    assert king_emission_shares(5) == KING_EMISSION_SHARES
    assert king_emission_shares(len(KING_EMISSION_SHARES)) == KING_EMISSION_SHARES


def test_shares_clip_to_window_and_table() -> None:
    assert king_emission_shares(0) == ()
    assert king_emission_shares(3) == (0.40, 0.15, 0.15)
    assert king_emission_shares(99) == KING_EMISSION_SHARES
    assert king_emission_shares(-1) == ()


# -- compute_weights ----------------------------------------------------------------


def test_full_window_distributes_40_15x4_with_no_burn() -> None:
    meta = _meta({10: "kA", 11: "kB", 12: "kC", 13: "kD", 14: "kE", 0: "burn"})
    kings = [
        _king("sA", "kA"),
        _king("sB", "kB"),
        _king("sC", "kC"),
        _king("sD", "kD"),
        _king("sE", "kE"),
    ]
    plan = compute_weights(kings, meta, window=5, burn_uid=BURN_UID)

    assert plan.submittable
    assert _weight_of(plan, 10) == pytest.approx(0.40)
    for uid in (11, 12, 13, 14):
        assert _weight_of(plan, uid) == pytest.approx(0.15)
    assert _weight_of(plan, 0) == pytest.approx(0.0)
    assert "burn=0.00" in plan.summary


def test_single_king_pays_40_and_burns_60() -> None:
    meta = _meta({10: "kA", 0: "burn"})
    plan = compute_weights([_king("sA", "kA")], meta, window=5, burn_uid=BURN_UID)

    assert plan.submittable
    assert _weight_of(plan, 10) == pytest.approx(0.40)
    assert _weight_of(plan, 0) == pytest.approx(0.60)
    assert "uid10=0.40" in plan.summary and "burn=0.60" in plan.summary


def test_no_kings_burns_everything() -> None:
    meta = _meta({10: "kA", 0: "burn"})
    plan = compute_weights([], meta, window=5, burn_uid=BURN_UID)

    assert plan.submittable
    assert _weight_of(plan, 0) == pytest.approx(1.0)
    assert _weight_of(plan, 10) == pytest.approx(0.0)
    assert "burn=1.00" in plan.summary


def test_deregistered_king_slot_burns() -> None:
    meta = _meta({10: "kA", 12: "kC", 0: "burn"})
    kings = [_king("sA", "kA"), _king("sB", "kB"), _king("sC", "kC")]
    plan = compute_weights(kings, meta, window=5, burn_uid=BURN_UID)

    assert _weight_of(plan, 10) == pytest.approx(0.40)
    assert _weight_of(plan, 12) == pytest.approx(0.15)
    assert _weight_of(plan, 0) == pytest.approx(0.45)  # kB's slot + slots 3,4 burn


def test_same_hotkey_in_two_slots_accumulates() -> None:
    meta = _meta({10: "kA", 11: "kB", 0: "burn"})
    kings = [_king("sA1", "kA"), _king("sB", "kB"), _king("sA2", "kA")]
    plan = compute_weights(kings, meta, window=5, burn_uid=BURN_UID)

    assert _weight_of(plan, 10) == pytest.approx(0.55)  # slots 0 + 2
    assert _weight_of(plan, 11) == pytest.approx(0.15)
    assert _weight_of(plan, 0) == pytest.approx(0.30)  # slots 3, 4 empty


def test_burn_owed_but_burn_uid_absent_is_unsubmittable() -> None:
    meta = _meta({10: "kA"})
    plan = compute_weights([_king("sA", "kA")], meta, window=5, burn_uid=BURN_UID)

    assert not plan.submittable
    assert plan.skip_reason is not None and "burn uid 0" in plan.skip_reason


def test_no_neurons_is_unsubmittable() -> None:
    plan = compute_weights([_king("sA", "kA")], _meta({}), window=5, burn_uid=BURN_UID)
    assert not plan.submittable
    assert plan.skip_reason == "no neurons in metagraph"


def test_full_burn_with_burn_uid_present_is_submittable() -> None:
    meta = _meta({0: "burn", 10: "x"})
    plan = compute_weights([], meta, window=5, burn_uid=BURN_UID)
    assert plan.submittable
    assert _weight_of(plan, 0) == pytest.approx(1.0)


def test_non_default_burn_uid_is_honored() -> None:
    meta = _meta({10: "kA", 5: "burn"})
    plan = compute_weights([_king("sA", "kA")], meta, window=5, burn_uid=5)
    assert _weight_of(plan, 5) == pytest.approx(0.60)


# -- epoch math ---------------------------------------------------------------------


def test_blocks_until_next_epoch_hits_zero_on_boundaries() -> None:
    tempo, netuid = 3, 0
    bus = [blocks_until_next_epoch(b, tempo, netuid) for b in range(8)]
    assert bus == [2, 1, 0, 3, 2, 1, 0, 3]


def test_degenerate_tempo_falls_back_to_per_block() -> None:
    assert blocks_until_next_epoch(100, 0, NETUID) == 0


# -- should_set ---------------------------------------------------------------------


def _decide(block: int, since: int, *, tempo: int = 100, rate: int = 100, margin: int = 10):
    return should_set(
        current_block=block,
        tempo=tempo,
        netuid=NETUID,
        blocks_since_last_update=since,
        weights_rate_limit=rate,
        set_margin=margin,
    )


def test_rate_off_blocks_counts_down_the_rate_limit() -> None:
    d = _decide(block=30, since=50, rate=100)
    assert not d.proceed and d.rate_off_blocks == 50


def test_epoch_blocks_reports_distance_to_boundary() -> None:
    # block 10: 23 blocks to boundary, off the rate limit but > margin 10.
    d = _decide(block=10, since=100, rate=100, margin=10)
    assert not d.proceed and d.rate_off_blocks == 0 and d.epoch_blocks == 23


def test_proceeds_within_margin_and_off_rate_limit() -> None:
    # block 30: 3 blocks to boundary (<= margin), and off the rate limit.
    d = _decide(block=30, since=100, rate=100, margin=10)
    assert d.proceed and d.next_try_block == 30


def test_next_try_combines_rate_limit_and_margin() -> None:
    # tempo 9 (period 10, boundaries at block == 9 mod 10). From block 5, rate clears in
    # 8 (at 13, just past boundary 8), so the next set is 3 (margin) before boundary 18.
    d = should_set(
        current_block=5,
        tempo=9,
        netuid=0,
        blocks_since_last_update=2,
        weights_rate_limit=10,
        set_margin=3,
    )
    assert not d.proceed
    assert d.rate_off_blocks == 8
    assert d.epoch_blocks == 3
    assert d.next_try_block == 15


# -- loop.step / run ----------------------------------------------------------------


class FakeChain:
    def __init__(
        self,
        *,
        params: SubnetParams,
        hotkeys: dict[int, str],
        poll: PollState,
        accept: bool = True,
    ) -> None:
        self._params = params
        self._hotkeys = hotkeys
        self._poll = poll
        self._accept = accept
        self.set_calls: list[tuple[list[int], list[float]]] = []
        self.params_calls = 0
        self.metagraph_reads = 0

    def params(self, netuid: int) -> SubnetParams:
        self.params_calls += 1
        return self._params

    def poll(self, netuid: int, uid: int) -> PollState:
        return self._poll

    def metagraph(self, netuid: int) -> MetagraphView:
        self.metagraph_reads += 1
        return MetagraphView(
            uids=tuple(self._hotkeys),
            uid_by_hotkey={h: u for u, h in self._hotkeys.items()},
        )

    def set_weights(
        self, netuid: int, uids: Sequence[int], weights: Sequence[float]
    ) -> bool:
        self.set_calls.append((list(uids), list(weights)))
        return self._accept


class FakeDb(WeightSetterDb):
    def __init__(self, *, kings: list[RecentKing]) -> None:
        self._kings = kings

    def recent_kings(self, window: int) -> list[RecentKing]:
        return self._kings[:window]


_PARAMS = SubnetParams(uid=7, tempo=100, weights_rate_limit=100)
_BASE_CONFIG = WeightSetterConfig(
    netuid=NETUID, window=5, set_margin=10, poll_seconds=1.0
)


def _config(**overrides: object) -> WeightSetterConfig:
    return dc.replace(_BASE_CONFIG, **overrides)


def _due_poll() -> PollState:
    # block 30: within margin of boundary; since 100 >= rate limit -> due.
    return PollState(current_block=30, blocks_since_last_update=100)


def test_step_waits_without_touching_metagraph_or_chain() -> None:
    chain = FakeChain(
        params=_PARAMS,
        hotkeys={10: "kA", 0: "burn"},
        poll=PollState(current_block=30, blocks_since_last_update=10),  # rate-limited
    )
    db = FakeDb(kings=[_king("sA", "kA")])

    result = step(chain, db, _config(), _PARAMS)

    assert result.action == "wait"
    assert result.epoch_blocks == 3  # block 30, tempo 100, netuid 66
    assert result.rate_off_blocks == 90  # rate limit 100, since 10
    assert result.next_try_block == 124  # rate clears at 124, already past a boundary
    assert chain.metagraph_reads == 0
    assert chain.set_calls == []


def test_step_sets_when_due() -> None:
    chain = FakeChain(
        params=_PARAMS, hotkeys={10: "kA", 11: "kB", 0: "burn"}, poll=_due_poll()
    )
    db = FakeDb(kings=[_king("sA", "kA"), _king("sB", "kB")])

    result = step(chain, db, _config(), _PARAMS)

    assert result.action == "set"
    assert result.epoch_blocks == 3
    assert len(chain.set_calls) == 1
    uids, weights = chain.set_calls[0]
    assert weights[uids.index(10)] == pytest.approx(0.40)
    assert weights[uids.index(11)] == pytest.approx(0.15)
    assert weights[uids.index(0)] == pytest.approx(0.45)


def test_step_burn_mode_ignores_kings_and_burns_all() -> None:
    chain = FakeChain(params=_PARAMS, hotkeys={10: "kA", 0: "burn"}, poll=_due_poll())
    db = FakeDb(kings=[_king("sA", "kA")])

    result = step(chain, db, _config(burn_mode=True), _PARAMS)

    assert result.action == "set"
    uids, weights = chain.set_calls[0]
    assert weights[uids.index(0)] == pytest.approx(1.0)
    assert weights[uids.index(10)] == pytest.approx(0.0)


def test_step_skips_unsubmittable_without_setting() -> None:
    chain = FakeChain(params=_PARAMS, hotkeys={10: "kA"}, poll=_due_poll())  # no burn uid
    db = FakeDb(kings=[_king("sA", "kA")])

    result = step(chain, db, _config(), _PARAMS)

    assert result.action == "skip"
    assert chain.set_calls == []


def test_step_failed_when_submission_rejected() -> None:
    chain = FakeChain(
        params=_PARAMS, hotkeys={10: "kA", 0: "burn"}, poll=_due_poll(), accept=False
    )
    db = FakeDb(kings=[_king("sA", "kA")])

    result = step(chain, db, _config(), _PARAMS)

    assert result.action == "failed"
    assert len(chain.set_calls) == 1


class _StopAfter:
    def __init__(self, n: int) -> None:
        self._left = n

    def __call__(self, _seconds: float) -> None:
        self._left -= 1
        if self._left <= 0:
            raise KeyboardInterrupt


def test_run_refreshes_params_after_each_set() -> None:
    chain = FakeChain(params=_PARAMS, hotkeys={10: "kA", 0: "burn"}, poll=_due_poll())
    db = FakeDb(kings=[_king("sA", "kA")])

    run(chain, db, _config(), sleep=_StopAfter(2))

    # startup params + one refresh per successful set (2 sets over 2 ticks).
    assert len(chain.set_calls) == 2
    assert chain.params_calls == 3


def test_run_fails_fast_when_not_registered() -> None:
    class Unregistered(FakeChain):
        def params(self, netuid: int) -> SubnetParams:
            raise RuntimeError("wallet hotkey is not registered")

    chain = Unregistered(params=_PARAMS, hotkeys={0: "burn"}, poll=_due_poll())
    with pytest.raises(RuntimeError, match="not registered"):
        run(chain, FakeDb(kings=[]), _config(), sleep=_StopAfter(1))


# -- config -------------------------------------------------------------------------


_WALLET_ENV = {"BITTENSOR_WALLET_NAME": "validator", "BITTENSOR_WALLET_HOTKEY": "vali-hot"}


def test_config_from_env_reads_overrides() -> None:
    config = WeightSetterConfig.from_env(
        {
            **_WALLET_ENV,
            "NETUID": "7",
            "TAU_WEIGHT_WINDOW": "3",
            "TAU_WEIGHT_SET_MARGIN": "4",
            "TAU_WEIGHT_BURN_MODE": "true",
            "TAU_WEIGHT_POLL_SECONDS": "6.0",
        }
    )
    assert config.netuid == 7
    assert config.window == 3
    assert config.set_margin == 4
    assert config.burn_mode is True
    assert config.poll_seconds == 6.0
    assert (config.wallet_name, config.wallet_hotkey) == ("validator", "vali-hot")


def test_config_requires_wallet_name_and_hotkey() -> None:
    with pytest.raises(ValueError):
        WeightSetterConfig.from_env({"BITTENSOR_WALLET_HOTKEY": "h"})
    with pytest.raises(ValueError):
        WeightSetterConfig.from_env({"BITTENSOR_WALLET_NAME": "n"})


def test_config_fails_closed_on_unparseable_env() -> None:
    with pytest.raises(ValueError):
        WeightSetterConfig.from_env({**_WALLET_ENV, "NETUID": "foo"})
    with pytest.raises(ValueError):
        WeightSetterConfig.from_env({**_WALLET_ENV, "TAU_WEIGHT_BURN_MODE": "ture"})
    with pytest.raises(ValueError):
        WeightSetterConfig.from_env({**_WALLET_ENV, "TAU_WEIGHT_POLL_SECONDS": "fast"})


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        WeightSetterConfig(window=-1)
    with pytest.raises(ValueError):
        WeightSetterConfig(set_margin=-1)
    with pytest.raises(ValueError):
        WeightSetterConfig(poll_seconds=0)


def test_human_renders_durations() -> None:
    from tau.workers.weight_setter.loop import _human

    assert _human(60) == "~60s"
    assert _human(1200) == "~20m"
    assert _human(12000) == "~3.3h"


def test_waiting_logs_once_per_next_try_target(caplog) -> None:
    import logging

    from tau.workers.weight_setter.loop import _TickLog

    ticklog = _TickLog()
    waiting = StepResult(
        "wait", "", epoch_blocks=32, rate_off_blocks=34, next_try_block=7463810,
        next_try_blocks=120,
    )
    with caplog.at_level(logging.INFO, logger="tau.workers.weight_setter.loop"):
        ticklog.report(waiting)
        ticklog.report(waiting)  # same next_try target -> not logged again

    lines = [m for m in caplog.messages if m.startswith("waiting:")]
    assert len(lines) == 1
    assert "rate-off ~34b" in lines[0]
    assert "epoch ~32b" in lines[0]
    assert "next try block 7463810" in lines[0]


def test_step_result_is_constructible() -> None:
    assert StepResult("wait", "because").next_try_block is None
