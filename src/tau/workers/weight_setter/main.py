"""Weight-setter entry point: wire the live chain + DB and run the loop."""

from __future__ import annotations

import logging

from tau.axiom import get_axiom
from tau.bittensor.weights import BittensorWeightChain
from tau.db.weight_setter import WeightSetterDb
from tau.utils.logging import configure_logging

from .config import WeightSetterConfig
from .loop import run

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = WeightSetterConfig.from_env()

    get_axiom().info(
        "weight-setter",
        "init_worker",
        network=config.network,
        netuid=config.netuid,
        window=config.window,
        burn_uid=config.burn_uid,
        burn_mode=config.burn_mode,
        set_margin=config.set_margin,
        poll_seconds=config.poll_seconds,
    )

    chain = BittensorWeightChain(
        network=config.network,
        wallet_name=config.wallet_name,
        wallet_hotkey=config.wallet_hotkey,
        wallet_path=config.wallet_path,
    )
    db = WeightSetterDb()
    log.info(
        "weight setter starting network=%s netuid=%d", config.network, config.netuid
    )

    try:
        run(chain, db, config)
    except Exception as ex:
        get_axiom().exception(
            "weight-setter",
            "unexpected_error",
            exception=str(ex),
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
