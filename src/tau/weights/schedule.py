"""When to set weights: once per epoch, a small margin before the boundary, and only
when the chain's rate limit allows."""

from __future__ import annotations

import dataclasses as dc


@dc.dataclass(frozen=True, slots=True)
class SetDecision:
    proceed: bool
    rate_off_blocks: int
    epoch_blocks: int
    next_try_block: int


def blocks_until_next_epoch(current_block: int, tempo: int, netuid: int) -> int:
    if tempo <= 0:
        return 0
    return tempo - (current_block + netuid + 1) % (tempo + 1)


def should_set(
    *,
    current_block: int,
    tempo: int,
    netuid: int,
    blocks_since_last_update: int,
    weights_rate_limit: int,
    set_margin: int,
) -> SetDecision:
    rate_off_blocks = max(0, weights_rate_limit - blocks_since_last_update)
    epoch_blocks = blocks_until_next_epoch(current_block, tempo, netuid)
    rate_clear = current_block + rate_off_blocks
    margin_wait = max(0, blocks_until_next_epoch(rate_clear, tempo, netuid) - set_margin)
    next_try_block = rate_clear + margin_wait
    proceed = rate_off_blocks == 0 and epoch_blocks <= set_margin
    return SetDecision(proceed, rate_off_blocks, epoch_blocks, next_try_block)
