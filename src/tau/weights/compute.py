"""The weight distribution rule: 40% to the reigning king, 15% to each of the next
four prior kings; unfilled or deregistered slots burn to the burn uid. Emission
follows the hotkey and shares accumulate per uid."""

from __future__ import annotations

from collections.abc import Sequence

from .types import MetagraphView, RecentKing, WeightPlan

KING_EMISSION_SHARES: tuple[float, ...] = (0.40, 0.15, 0.15, 0.15, 0.15)


def king_emission_shares(window: int) -> tuple[float, ...]:
    return KING_EMISSION_SHARES[: max(0, window)]


def compute_weights(
    kings: Sequence[RecentKing],
    meta: MetagraphView,
    *,
    window: int,
    burn_uid: int,
) -> WeightPlan:
    shares = king_emission_shares(window)
    weights_by_uid: dict[int, float] = {u: 0.0 for u in meta.uids}
    king_shares: dict[int, float] = {}
    burn = 0.0

    for slot, share in enumerate(shares):
        king = kings[slot] if slot < len(kings) else None
        uid = meta.uid_by_hotkey.get(king.hotkey) if king is not None else None
        if king is not None and uid is not None and uid in weights_by_uid:
            weights_by_uid[uid] += share
            king_shares[uid] = king_shares.get(uid, 0.0) + share
        else:
            burn += share

    if not meta.uids:
        return WeightPlan((), (), False, "no neurons in metagraph", "")
    if burn > 0 and burn_uid not in weights_by_uid:
        return WeightPlan((), (), False, f"burn uid {burn_uid} absent from metagraph", "")
    if burn > 0:
        weights_by_uid[burn_uid] += burn

    uids = tuple(meta.uids)
    weights = tuple(weights_by_uid[u] for u in uids)
    kings_str = ", ".join(f"uid{u}={s:.2f}" for u, s in king_shares.items())
    summary = f"kings=[{kings_str}] burn={burn:.2f}"
    return WeightPlan(uids, weights, True, None, summary)
