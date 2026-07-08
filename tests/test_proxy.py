"""Unit tests for the LLM proxy: key injection, model/sampling enforcement, auth,
and budget caps. Runs a real proxy on 127.0.0.1 with a fake upstream (no network).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx
import pytest

from tau.proxy import REQUEST_LIMIT_EXIT_REASON, LLMProxy, SolveBudget, UpstreamTarget
from tau.proxy.routing import (
    SMART_UPSTREAM_ROUTER,
    DisabledUpstreamStore,
    SmartUpstreamRouter,
)
from tau.proxy.upstream import HttpxUpstreamClient, UpstreamClient, UpstreamResponse
from tau.sandbox.config import SandboxConfig

_UPSTREAM = UpstreamTarget(name="test", base_url="http://upstream.invalid", api_key="UPSTREAM-KEY")
_BODY = {
    "model": "miner/model",
    "messages": [{"role": "user", "content": "hi"}],
    "top_k": 50,
    "seed": 999,
}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]


class FakeUpstream(UpstreamClient):
    """Captures what the proxy forwarded and returns a canned completion."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def fetch(self, *, command, url, headers, body, prepared_payload, request_path, start):  # noqa: ANN001, ANN002
        self.calls.append({"headers": dict(headers), "payload": prepared_payload, "url": url})
        payload = {
            "id": "gen-1",
            "model": "m",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.001},
        }
        return UpstreamResponse(
            body=json.dumps(payload).encode(),
            payload=payload,
            status=200,
            headers=httpx.Headers({"Content-Type": "application/json"}),
            first_token_latency_ms=None,
        )


@pytest.fixture(autouse=True)
def reset_smart_router() -> Iterator[None]:
    SMART_UPSTREAM_ROUTER.reset()
    yield
    SMART_UPSTREAM_ROUTER.reset()


def _running(proxy: LLMProxy, fake: FakeUpstream) -> Iterator[LLMProxy]:
    proxy._upstream_client = fake  # noqa: SLF001 — inject the fake transport
    proxy.start()
    try:
        yield proxy
    finally:
        proxy.stop()


@pytest.fixture
def proxy_with_model() -> Iterator[tuple[LLMProxy, FakeUpstream]]:
    fake = FakeUpstream()
    proxy = LLMProxy(_UPSTREAM, bind_host="127.0.0.1", bind_port=0, enforced_model="enforced/model")
    gen = _running(proxy, fake)
    yield next(gen), fake
    next(gen, None)


def _post(proxy: LLMProxy, token: str | None, body: dict) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.post(
        f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
        headers=headers, json=body, timeout=10,
    )


def test_injects_upstream_key_and_enforces_model(proxy_with_model) -> None:
    proxy, fake = proxy_with_model
    resp = _post(proxy, proxy.auth_token, {**_BODY, "tools": _TOOLS})
    assert resp.status_code == 200
    call = fake.calls[-1]
    # The proxy injected the upstream key, never exposing it to the caller.
    assert call["headers"]["Authorization"] == "Bearer UPSTREAM-KEY"
    # Forwarded to the configured upstream base URL.
    assert call["url"].startswith("http://upstream.invalid/v1/chat/completions")
    # The miner's model was overridden, and miner-controlled sampling stripped.
    assert call["payload"]["model"] == "enforced/model"
    assert "top_k" not in call["payload"]
    assert "seed" not in call["payload"]
    assert call["payload"]["temperature"] == 0.0
    assert call["payload"]["tools"] == _TOOLS


def test_enforces_validator_seed_over_miner_seed() -> None:
    fake = FakeUpstream()
    proxy = LLMProxy(
        _UPSTREAM,
        bind_host="127.0.0.1",
        bind_port=0,
        enforced_model="enforced/model",
        enforced_sampling_params={"temperature": 0.0, "top_p": 1.0, "seed": 12345},
    )
    gen = _running(proxy, fake)
    proxy = next(gen)
    try:
        resp = _post(proxy, proxy.auth_token, {**_BODY, "seed": 999})
        assert resp.status_code == 200
        assert fake.calls[-1]["payload"]["seed"] == 12345
    finally:
        next(gen, None)


def test_custom_upstream_from_env_accepts_multiple_base_urls() -> None:
    upstream = UpstreamTarget.from_env(
        {
            "LLM_PROVIDER": "custom",
            "LLM_UPSTREAM_BASE_URLS": (
                "http://10.0.0.5:8000/v1, http://10.0.0.5:8001/v1"
            ),
            "LLM_UPSTREAM_API_KEY": "LOCAL-KEY",
        }
    )
    assert upstream.base_url == "http://10.0.0.5:8000"
    assert upstream.base_urls == ("http://10.0.0.5:8000", "http://10.0.0.5:8001")
    assert upstream.endpoint_count == 2


def test_upstream_from_env_filters_disabled_base_urls(tmp_path) -> None:
    disabled_file = tmp_path / "disabled-upstreams.txt"
    disabled_file.write_text("http://10.0.0.5:8001/v1\n", encoding="utf-8")

    upstream = UpstreamTarget.from_env(
        {
            "LLM_PROVIDER": "custom",
            "LLM_UPSTREAM_BASE_URLS": (
                "http://10.0.0.5:8000/v1, http://10.0.0.5:8001/v1"
            ),
            "LLM_UPSTREAM_API_KEY": "LOCAL-KEY",
            "TAU_SOLVER_DISABLED_UPSTREAMS_FILE": str(disabled_file),
        }
    )

    assert upstream.base_urls == ("http://10.0.0.5:8000",)
    assert upstream.endpoint_count == 1


def test_upstream_from_env_errors_when_all_urls_are_disabled(tmp_path) -> None:
    disabled_file = tmp_path / "disabled-upstreams.txt"
    disabled_file.write_text("http://10.0.0.5:8000/v1\n", encoding="utf-8")

    with pytest.raises(OSError, match="all configured solver upstream endpoints"):
        UpstreamTarget.from_env(
            {
                "LLM_PROVIDER": "custom",
                "LLM_UPSTREAM_BASE_URLS": "http://10.0.0.5:8000/v1",
                "LLM_UPSTREAM_API_KEY": "LOCAL-KEY",
                "TAU_SOLVER_DISABLED_UPSTREAMS_FILE": str(disabled_file),
            }
        )


def test_proxy_round_robins_multiple_upstream_urls() -> None:
    fake = FakeUpstream()
    upstream = UpstreamTarget(
        name="test",
        base_url="http://10.0.0.5:8000/v1",
        base_urls=("http://10.0.0.5:8000/v1", "http://10.0.0.5:8001/v1"),
        api_key="UPSTREAM-KEY",
    )
    proxy = LLMProxy(
        upstream, bind_host="127.0.0.1", bind_port=0, enforced_model="enforced/model"
    )
    gen = _running(proxy, fake)
    proxy = next(gen)
    try:
        first = _post(proxy, proxy.auth_token, _BODY)
        second = _post(proxy, proxy.auth_token, _BODY)
        assert first.status_code == 200
        assert second.status_code == 200
        assert fake.calls[0]["url"].startswith(
            "http://10.0.0.5:8000/v1/chat/completions"
        )
        assert fake.calls[1]["url"].startswith(
            "http://10.0.0.5:8001/v1/chat/completions"
        )
    finally:
        next(gen, None)


def test_smart_cache_routing_sticks_to_one_endpoint_per_proxy() -> None:
    fake = FakeUpstream()
    upstream = UpstreamTarget(
        name="test",
        base_url="http://10.0.0.5:8000/v1",
        base_urls=("http://10.0.0.5:8000/v1", "http://10.0.0.5:8001/v1"),
        api_key="UPSTREAM-KEY",
    )
    proxy = LLMProxy(
        upstream,
        bind_host="127.0.0.1",
        bind_port=0,
        enforced_model="enforced/model",
        smart_cache_routing=True,
    )
    gen = _running(proxy, fake)
    proxy = next(gen)
    try:
        first = _post(proxy, proxy.auth_token, _BODY)
        second = _post(
            proxy,
            proxy.auth_token,
            {**_BODY, "messages": [{"role": "user", "content": "next"}]},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert fake.calls[0]["url"].startswith(
            "http://10.0.0.5:8000/v1/chat/completions"
        )
        assert fake.calls[1]["url"].startswith(
            "http://10.0.0.5:8000/v1/chat/completions"
        )
    finally:
        next(gen, None)


def test_smart_cache_routing_reuses_prompt_affinity_across_proxies() -> None:
    upstream = UpstreamTarget(
        name="test",
        base_url="http://10.0.0.5:8000/v1",
        base_urls=("http://10.0.0.5:8000/v1", "http://10.0.0.5:8001/v1"),
        api_key="UPSTREAM-KEY",
    )
    body = {
        "model": "miner/model",
        "messages": [
            {"role": "system", "content": "same agent prompt"},
            {"role": "user", "content": "same task"},
        ],
    }

    first_fake = FakeUpstream()
    first_proxy = LLMProxy(
        upstream,
        bind_host="127.0.0.1",
        bind_port=0,
        enforced_model="enforced/model",
        smart_cache_routing=True,
    )
    first_gen = _running(first_proxy, first_fake)
    first_proxy = next(first_gen)
    try:
        assert _post(first_proxy, first_proxy.auth_token, body).status_code == 200
    finally:
        next(first_gen, None)

    second_fake = FakeUpstream()
    second_proxy = LLMProxy(
        upstream,
        bind_host="127.0.0.1",
        bind_port=0,
        enforced_model="enforced/model",
        smart_cache_routing=True,
    )
    second_gen = _running(second_proxy, second_fake)
    second_proxy = next(second_gen)
    try:
        assert _post(second_proxy, second_proxy.auth_token, body).status_code == 200
        assert first_fake.calls[0]["url"].startswith(
            "http://10.0.0.5:8000/v1/chat/completions"
        )
        assert second_fake.calls[0]["url"].startswith(
            "http://10.0.0.5:8000/v1/chat/completions"
        )
    finally:
        next(second_gen, None)


def test_smart_cache_routing_skips_endpoint_on_infra_failure() -> None:
    upstream = UpstreamTarget(
        name="test",
        base_url="http://10.0.0.5:8000/v1",
        base_urls=("http://10.0.0.5:8000/v1", "http://10.0.0.5:8001/v1"),
        api_key="UPSTREAM-KEY",
    )
    body = {
        "model": "miner/model",
        "messages": [
            {"role": "system", "content": "same agent prompt"},
            {"role": "user", "content": "same task"},
        ],
    }

    failed_fake = _StatusUpstream(503)
    failed_proxy = LLMProxy(
        upstream,
        bind_host="127.0.0.1",
        bind_port=0,
        enforced_model="enforced/model",
        smart_cache_routing=True,
    )
    failed_gen = _running(failed_proxy, failed_fake)
    failed_proxy = next(failed_gen)
    try:
        assert _post(failed_proxy, failed_proxy.auth_token, body).status_code == 503
    finally:
        next(failed_gen, None)

    healthy_fake = FakeUpstream()
    healthy_proxy = LLMProxy(
        upstream,
        bind_host="127.0.0.1",
        bind_port=0,
        enforced_model="enforced/model",
        smart_cache_routing=True,
    )
    healthy_gen = _running(healthy_proxy, healthy_fake)
    healthy_proxy = next(healthy_gen)
    try:
        assert _post(healthy_proxy, healthy_proxy.auth_token, body).status_code == 200
        assert healthy_fake.calls[0]["url"].startswith(
            "http://10.0.0.5:8001/v1/chat/completions"
        )
    finally:
        next(healthy_gen, None)


def test_router_affinity_is_remembered_after_sticky_choice() -> None:
    router = SmartUpstreamRouter()
    urls = ("http://10.0.0.5:8000", "http://10.0.0.5:8001")

    first = router.acquire(urls, "same-prefix")
    router.release(first)
    second = router.acquire(urls, "same-prefix")
    router.release(second)

    assert first == "http://10.0.0.5:8000"
    assert second == "http://10.0.0.5:8001"

    router.remember_affinity("same-prefix", first)
    third = router.acquire(urls, "same-prefix")
    try:
        assert third == first
    finally:
        router.release(third)


def test_router_permanently_disables_endpoint_after_max_cooldown(tmp_path) -> None:
    store = DisabledUpstreamStore(tmp_path / "disabled-upstreams.txt")
    router = SmartUpstreamRouter(disabled_store=store)
    urls = ("http://10.0.0.5:8000", "http://10.0.0.5:8001")

    for _ in range(4):
        router.record_result(
            "http://10.0.0.5:8000",
            status_code=503,
            error="boom",
            base_urls=urls,
        )

    assert store.load() == {"http://10.0.0.5:8000"}
    selected = router.acquire(urls, "same-prefix")
    try:
        assert selected == "http://10.0.0.5:8001"
    finally:
        router.release(selected)


def test_router_keeps_last_enabled_endpoint_out_of_disabled_file(tmp_path) -> None:
    store = DisabledUpstreamStore(tmp_path / "disabled-upstreams.txt")
    router = SmartUpstreamRouter(disabled_store=store)
    urls = ("http://10.0.0.5:8000",)

    for _ in range(4):
        router.record_result(
            "http://10.0.0.5:8000",
            status_code=503,
            error="boom",
            base_urls=urls,
        )

    assert store.load() == set()


def test_sandbox_config_requires_solver_model_env() -> None:
    with pytest.raises(OSError, match="SOLVER_MODEL"):
        SandboxConfig.from_env({})


def test_sandbox_config_reads_solver_model_from_env() -> None:
    config = SandboxConfig.from_env({"SOLVER_MODEL": "provider/model"})
    assert config.model == "provider/model"
    assert config.smart_cache_routing is True


def test_sandbox_config_can_disable_smart_cache_routing() -> None:
    config = SandboxConfig.from_env(
        {"SOLVER_MODEL": "provider/model", "TAU_SOLVER_SMART_CACHE_ROUTING": "false"}
    )
    assert config.smart_cache_routing is False


def test_sandbox_config_defaults_task_timeout_to_600_seconds() -> None:
    config = SandboxConfig.from_env({"SOLVER_MODEL": "provider/model"})

    assert config.hard_timeout_seconds == 600


def test_rejects_request_without_auth(proxy_with_model) -> None:
    proxy, fake = proxy_with_model
    resp = _post(proxy, None, _BODY)
    assert resp.status_code == 401
    assert not fake.calls  # never reached the upstream


def test_budget_caps_request_count() -> None:
    fake = FakeUpstream()
    proxy = LLMProxy(
        _UPSTREAM, bind_host="127.0.0.1", bind_port=0,
        enforced_model="enforced/model", solve_budget=SolveBudget(max_requests=1),
    )
    gen = _running(proxy, fake)
    proxy = next(gen)
    try:
        first = _post(proxy, proxy.auth_token, _BODY)
        second = _post(proxy, proxy.auth_token, _BODY)
        assert first.status_code == 200
        assert second.status_code == 429
        assert proxy.budget_exceeded_reason == REQUEST_LIMIT_EXIT_REASON
        assert len(fake.calls) == 1  # the capped request never forwarded
    finally:
        next(gen, None)


# -- upstream infrastructure-error accounting --------------------------------
# The proxy separates *upstream* infra faults (our problem: unreachable/timeout/funds/
# rate/5xx/our-key) from the agent's own bad requests (4xx). Only the former should bump
# `upstream_error_count`, which the sandbox runner uses to mark a solve retryable.
class _StatusUpstream(UpstreamClient):
    def __init__(self, status: int) -> None:
        self.status = status

    def fetch(self, **kw) -> UpstreamResponse:  # noqa: ANN003
        payload = {"error": {"message": "boom"}} if self.status >= 400 else {"choices": []}
        return UpstreamResponse(
            body=json.dumps(payload).encode(), payload=payload, status=self.status,
            headers=httpx.Headers({"Content-Type": "application/json"}),
            first_token_latency_ms=None,
        )


class _RaisingUpstream(UpstreamClient):
    def fetch(self, **kw) -> UpstreamResponse:  # noqa: ANN003
        raise httpx.ConnectError("[Errno -2] Name or service not known")


def _snapshot_after_post(fake: UpstreamClient):  # noqa: ANN202
    proxy = LLMProxy(_UPSTREAM, bind_host="127.0.0.1", bind_port=0, enforced_model="enforced/model")
    gen = _running(proxy, fake)
    proxy = next(gen)
    try:
        resp = _post(proxy, proxy.auth_token, _BODY)
        return resp.status_code, proxy.usage_snapshot()
    finally:
        next(gen, None)


@pytest.mark.parametrize(
    ("status", "is_infra"),
    [(402, True), (429, True), (500, True), (503, True), (401, True),
     (400, False), (404, False), (422, False)],
)
def test_upstream_error_count_classifies_status(status: int, is_infra: bool) -> None:
    _, snap = _snapshot_after_post(_StatusUpstream(status))
    assert snap.error_count == 1  # every failure is still an error
    assert snap.upstream_error_count == (1 if is_infra else 0)  # ...but infra only when it is


def test_upstream_error_count_counts_transport_failure() -> None:
    client_status, snap = _snapshot_after_post(_RaisingUpstream())
    assert client_status == 502  # agent sees a proxy failure
    assert snap.upstream_error_count == 1  # unreachable/timeout counts as infra


def test_upstream_error_count_zero_on_success(proxy_with_model) -> None:
    proxy, _ = proxy_with_model
    _post(proxy, proxy.auth_token, _BODY)
    snap = proxy.usage_snapshot()
    assert snap.upstream_error_count == 0
    assert snap.upstream_timeout_count == 0
    assert snap.last_upstream_error is None


class _TimeoutUpstream(UpstreamClient):
    def fetch(self, **kw) -> UpstreamResponse:  # noqa: ANN003
        raise httpx.ReadTimeout("the read operation timed out")


def test_transport_timeout_is_counted_as_timeout() -> None:
    client_status, snap = _snapshot_after_post(_TimeoutUpstream())
    assert client_status == 502
    assert snap.upstream_error_count == 1
    assert snap.upstream_timeout_count == 1  # a timeout is flagged distinctly
    assert snap.last_upstream_error and "ReadTimeout" in snap.last_upstream_error


@pytest.mark.parametrize(
    ("status", "is_timeout"),
    [(504, True), (408, True), (402, False), (500, False)],
)
def test_status_timeout_classification(status: int, is_timeout: bool) -> None:
    _, snap = _snapshot_after_post(_StatusUpstream(status))
    assert snap.upstream_timeout_count == (1 if is_timeout else 0)
    assert snap.last_upstream_error is not None


def test_read_timeout_is_configurable() -> None:
    proxy = LLMProxy(_UPSTREAM, upstream_read_timeout_seconds=123.0)
    assert proxy._upstream_client._timeout.read == 123.0  # noqa: SLF001


class _SSEStreamResponse:
    status_code = 200
    headers = httpx.Headers({"Content-Type": "text/event-stream"})

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        return None

    def iter_lines(self) -> Iterator[str]:
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class _SSEClient:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.payload: dict | None = None

    def stream(self, method, url, headers, json):  # noqa: ANN001, ANN201
        self.payload = json
        return _SSEStreamResponse(self.chunks)


def test_streamed_chat_reassembly_preserves_tools_and_tool_calls() -> None:
    chunks = [
        {
            "id": "chatcmpl-1",
            "created": 123,
            "model": "enforced/model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "model": "enforced/model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "pwd\"}"}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "model": "enforced/model",
            "choices": [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    ]
    fake_client = _SSEClient(chunks)
    upstream = HttpxUpstreamClient()._fetch_streamed(  # noqa: SLF001
        client=fake_client,
        url="http://upstream.invalid/v1/chat/completions",
        headers={},
        payload={
            "model": "enforced/model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
        start=time.monotonic(),
    )

    assert fake_client.payload is not None
    assert fake_client.payload["tools"] == _TOOLS
    message = upstream.payload["choices"][0]["message"]
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
        }
    ]
    assert upstream.payload["choices"][0]["finish_reason"] == "tool_calls"
