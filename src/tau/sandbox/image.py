"""Build (and cache) the sandbox image via the docker-py SDK.

The Dockerfile is embedded here as a string (not read from disk): the worker image
ships only ``src/`` + ``pyproject``, so the on-disk ``deploy/sandbox/Dockerfile`` is
NOT present at runtime, and the builder sends the Dockerfile as an in-memory build
context anyway. ``deploy/sandbox/Dockerfile`` is kept as a human-readable mirror.

The image is tagged by a content hash of that text, so a change forces a rebuild
while steady state is a cache hit. Built once at worker startup.
"""

from __future__ import annotations

import hashlib
import io
import logging

import docker
from docker.errors import ImageNotFound

from .config import SandboxConfig

log = logging.getLogger(__name__)

# The sandbox one miner agent runs in: Python + git + common agent runtimes. Kept in
# sync with deploy/sandbox/Dockerfile (that file is documentation; this is the source
# of truth at runtime). The agent's own bundle is bind-mounted in at run time, and it
# reaches the LLM only through the validator proxy (no internet on its network).
SANDBOX_DOCKERFILE = """\
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir mini-swe-agent httpx openai

WORKDIR /work

CMD ["bash"]
"""


def _dockerfile_text() -> str:
    return SANDBOX_DOCKERFILE


def image_tag(config: SandboxConfig) -> str:
    """Deterministic ``name:hash`` tag derived from the Dockerfile contents."""
    digest = hashlib.sha256(_dockerfile_text().encode("utf-8")).hexdigest()[:16]
    return f"{config.image_name}:{digest}"


def ensure_sandbox_image(client: docker.DockerClient, config: SandboxConfig) -> str:
    """Return the sandbox image tag, building it if absent (or if ``no_cache``).

    docker-py's high-level build streams JSON and does not always raise on a failed
    build, so the log stream is scanned for an ``error`` and surfaced.
    """
    tag = image_tag(config)
    if not config.no_cache:
        try:
            client.images.get(tag)
            log.debug("sandbox image %s already present", tag)
            return tag
        except ImageNotFound:
            pass

    log.info("building sandbox image %s", tag)
    dockerfile = _dockerfile_text().encode("utf-8")
    # A tiny in-memory build context containing just the Dockerfile.
    _, logs = client.images.build(
        fileobj=io.BytesIO(dockerfile),
        tag=tag,
        rm=True,
        nocache=config.no_cache,
        pull=False,
    )
    for chunk in logs:
        if isinstance(chunk, dict) and chunk.get("error"):
            raise RuntimeError(f"sandbox image build failed: {chunk['error']}")
    log.info("sandbox image %s ready", tag)
    return tag
