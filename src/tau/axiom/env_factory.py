import logging
import os

from .axiom_client import AxiomClient, AxiomClientNoop

log = logging.getLogger(__name__)


def create_axiom_client_from_env() -> AxiomClient | AxiomClientNoop:
    """Create an Axiom client from environment variables.

    Returns:
        AxiomClient if AXIOM_TOKEN and AXIOM_DATASET are set, otherwise AxiomClientNoop.
    """
    token = os.getenv("AXIOM_TOKEN")
    dataset = os.getenv("AXIOM_DATASET")
    environ = os.getenv("AXIOM_ENVIRON", "default")

    if token is None:
        log.debug("AXIOM_TOKEN not set; Axiom observability disabled")
        return AxiomClientNoop()

    if dataset is None:
        log.debug("AXIOM_DATASET not set; Axiom observability disabled")
        return AxiomClientNoop()

    try:
        client = AxiomClient(dataset=dataset, environ=environ, token=token)
        log.info("Axiom observability enabled (dataset=%s)", dataset)
        return client
    except Exception:  # noqa: BLE001 — fall back to a no-op if the client won't build
        log.warning("Axiom client init failed; observability disabled", exc_info=True)
        return AxiomClientNoop()
