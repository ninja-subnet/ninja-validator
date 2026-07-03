"""Key-injecting LLM proxy in front of an OpenAI-compatible upstream.

Lets untrusted sandboxed agent code reach a model without ever seeing the upstream
key or the public internet: ``LLMProxy`` injects the real key, enforces the model /
provider / sampling params, caps spend with a ``SolveBudget``, and authenticates the
agent with a per-solve token. The upstream (OpenRouter, our own ``ninja`` backend, or
custom OpenAI-compatible endpoint URLs) is chosen by ``UpstreamTarget.from_env`` via
``LLM_PROVIDER``.

    from tau.proxy import LLMProxy, SolveBudget, UpstreamTarget

    with LLMProxy(UpstreamTarget.from_env(), bind_host="0.0.0.0", bind_port=0,
                  enforced_model="deepseek/deepseek-v4-flash",
                  solve_budget=SolveBudget.from_env()) as proxy:
        base_url = proxy.container_base_url(my_hostname)  # http://host:port/v1
        token = proxy.auth_token                          # the only secret the agent gets
"""

from __future__ import annotations

from .budget import (
    COST_LIMIT_EXIT_REASON,
    PROXY_ERROR_EXIT_REASON,
    REQUEST_LIMIT_EXIT_REASON,
    TOKEN_LIMIT_EXIT_REASON,
    ProxyRequestRecord,
    SolveBudget,
    SolveUsageSummary,
)
from .server import LLMProxy
from .target import UpstreamTarget

__all__ = [
    "LLMProxy",
    "UpstreamTarget",
    "SolveBudget",
    "SolveUsageSummary",
    "ProxyRequestRecord",
    "REQUEST_LIMIT_EXIT_REASON",
    "TOKEN_LIMIT_EXIT_REASON",
    "COST_LIMIT_EXIT_REASON",
    "PROXY_ERROR_EXIT_REASON",
]
