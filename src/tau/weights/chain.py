"""The chain seam: the worker's only view of the live network. The live
implementation is `tau.bittensor.weights.BittensorWeightChain`; tests pass a fake."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import MetagraphView, PollState, SubnetParams


@runtime_checkable
class WeightChain(Protocol):
    def params(self, netuid: int) -> SubnetParams:
        """Our uid + the subnet's tempo and rate limit. Raises if not registered."""
        ...

    def poll(self, netuid: int, uid: int) -> PollState:
        """The tip and blocks since our uid last set weights."""
        ...

    def metagraph(self, netuid: int) -> MetagraphView:
        ...

    def set_weights(
        self, netuid: int, uids: Sequence[int], weights: Sequence[float]
    ) -> bool:
        ...
