"""Network plumbing so a sandbox can reach the proxy but not the internet.

Production path (the orchestrator itself runs in a container): a single, long-lived
``--internal`` bridge network that the orchestrator joins **once** (under a fixed
alias) and every sandbox attaches to. The sandbox reaches the proxy at
``http://<alias>:<port>/v1``; ``internal=True`` means no default gateway, so the
sandbox has no internet egress. Each solve still gets its own proxy port + auth
token, so sharing the L2 segment does not let one solve use another's proxy.

Crucially the network is shared, not per-solve: attaching the *orchestrator* to a
fresh internal network on every solve makes it multi-homed across many internal
networks at once under concurrency, which corrupts its own default route / embedded
DNS forwarding — the proxy's upstream calls then fail with "Name or service not
known". A single stable attachment ([compose external net] + [one internal net])
avoids that churn.

Dev path (orchestrator on the host): no network is created; the sandbox runs with a
``host.docker.internal`` host-gateway mapping and reaches the proxy on the host.
"""

from __future__ import annotations

import logging
import re
import socket
from pathlib import Path

import docker
from docker.errors import APIError, NotFound
from docker.models.containers import Container
from docker.models.networks import Network

log = logging.getLogger(__name__)

# A container's own id appears as the source of its /etc/hostname (etc.) bind mounts,
# under /var/lib/docker/containers/<id>/. Matching that path (not any bare 64-hex)
# avoids grabbing *another* container's id from a Docker host's global mount table.
_OWN_CONTAINER_ID = re.compile(r"/containers/([0-9a-f]{64})/")

# Stable DNS alias the orchestrator registers on the per-solve network; the sandbox
# addresses the proxy by this name.
PROXY_ALIAS = "tau-proxy-host"
# Hostname a host-gateway sandbox uses to reach the proxy on the host.
HOST_GATEWAY_HOST = "host.docker.internal"


def _in_container() -> bool:
    """Whether this process is running inside a container (not on a Docker host).

    Critical: on a Docker *host*, /proc/self/mountinfo lists every container's bind
    mounts, so the id parsing below would otherwise misidentify some other container
    as "self". The /.dockerenv marker (and the cgroup hint) are absent on the host.
    """
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    return "docker" in cgroup or "containerd" in cgroup or "/kubepods" in cgroup


def _container_id_from_proc() -> str | None:
    """This process's own container id from its /etc/hostname bind-mount source.

    Only meaningful inside a container (gated by the caller); covers a hostname
    overridden to a service name, where ``gethostname`` no longer matches an id.
    """
    try:
        text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = _OWN_CONTAINER_ID.search(text)
    return match.group(1) if match else None


def self_container(client: docker.DockerClient) -> Container | None:
    """The orchestrator's own container, or None if not running inside one.

    Returns None on a host (so the runner uses the host-gateway dev transport).
    Inside a container, resolves via the hostname (Docker's default is the short
    container id) then the id parsed from /proc.
    """
    if not _in_container():
        return None
    for ident in (socket.gethostname(), _container_id_from_proc()):
        if not ident:
            continue
        try:
            return client.containers.get(ident)
        except NotFound:
            continue
        except Exception:  # noqa: BLE001 — daemon/permission issue => treat as "not found"
            return None
    return None


# Single shared internal network for all sandboxes (see the module docstring for why
# it is shared rather than per-solve).
SHARED_NETWORK_NAME = "tau-sandbox-net"


def ensure_shared_network(client: docker.DockerClient, self_ctr: Container) -> Network:
    """Return the shared internet-less network, orchestrator attached under the alias.

    Idempotent and safe under concurrency: get-or-create the network, then attach the
    orchestrator (once) as ``PROXY_ALIAS``. Called at startup and defensively per solve;
    after the first attach it is a cheap membership check.
    """
    net = _get_or_create_network(client, SHARED_NETWORK_NAME)
    _ensure_connected(net, self_ctr)
    return net


def _get_or_create_network(client: docker.DockerClient, name: str) -> Network:
    try:
        return client.networks.get(name)
    except NotFound:
        pass
    try:
        return client.networks.create(name, driver="bridge", internal=True)
    except APIError:
        # Lost a create race with a concurrent solve — the winner's network exists now.
        return client.networks.get(name)


def _ensure_connected(net: Network, self_ctr: Container) -> None:
    net.reload()
    if self_ctr.id in (net.attrs.get("Containers") or {}):
        return
    try:
        net.connect(self_ctr, aliases=[PROXY_ALIAS])
    except APIError:
        # Already connected (won by a concurrent solve) — verify and move on.
        net.reload()
        if self_ctr.id not in (net.attrs.get("Containers") or {}):
            raise
