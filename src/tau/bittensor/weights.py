"""Live `WeightChain` backed by the bittensor SDK: the only weight-side file that
touches it. Callers get the plain contracts from `tau.weights.types`."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tau.weights.types import MetagraphView, PollState, SubnetParams

from .types import FINNEY

import bittensor as bt

logger = logging.getLogger(__name__)


class BittensorWeightChain:
    def __init__(
        self,
        *,
        network: str = FINNEY,
        wallet_name: str = "default",
        wallet_hotkey: str = "default",
        wallet_path: str = "~/.bittensor/wallets",
        subtensor: bt.Subtensor | None = None,
        wallet: bt.Wallet | None = None,
    ) -> None:
        self._subtensor = (
            subtensor if subtensor is not None else bt.Subtensor(network=network)
        )
        self._wallet = (
            wallet
            if wallet is not None
            else bt.Wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)
        )
        logger.info("Connected to %s as wallet %s/%s", network, wallet_name, wallet_hotkey)

    def params(self, netuid: int) -> SubnetParams:
        uid = self._subtensor.get_uid_for_hotkey_on_subnet(
            self._wallet.hotkey.ss58_address, netuid
        )
        if uid is None:
            raise RuntimeError(f"wallet hotkey is not registered on netuid {netuid}")
        tempo = self._subtensor.tempo(netuid)
        rate_limit = self._subtensor.weights_rate_limit(netuid)
        return SubnetParams(
            uid=int(uid),
            tempo=int(tempo) if tempo is not None else 0,
            weights_rate_limit=int(rate_limit) if rate_limit is not None else 0,
        )

    def poll(self, netuid: int, uid: int) -> PollState:
        block = int(self._subtensor.get_current_block())
        since = self._subtensor.blocks_since_last_update(netuid, uid, block=block)
        return PollState(
            current_block=block,
            blocks_since_last_update=int(since) if since is not None else 0,
        )

    def metagraph(self, netuid: int) -> MetagraphView:
        meta = self._subtensor.metagraph(netuid=netuid)
        count = meta.n.item()
        uids = tuple(int(meta.uids[i]) for i in range(count))
        uid_by_hotkey = {str(meta.hotkeys[i]): int(meta.uids[i]) for i in range(count)}
        return MetagraphView(uids=uids, uid_by_hotkey=uid_by_hotkey)

    def set_weights(
        self, netuid: int, uids: Sequence[int], weights: Sequence[float]
    ) -> bool:
        response = self._subtensor.set_weights(
            wallet=self._wallet,
            netuid=netuid,
            uids=list(uids),
            weights=list(weights),
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if not response.success:
            logger.warning(
                "set_weights rejected (error=%s message=%s)",
                response.error,
                response.message,
            )
        return bool(response.success)
