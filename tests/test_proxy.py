"""Unit tests for the LLM proxy: key injection, model/sampling enforcement, auth,
and budget caps. Runs a real proxy on 127.0.0.1 with a fake upstream (no network).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from tau.proxy import REQUEST_LIMIT_EXIT_REASON, LLMProxy, SolveBudget, UpstreamTarget
from tau.proxy.upstream import UpstreamClient, UpstreamResponse

_UPSTREAM = UpstreamTarget(name="test", base_url="http://upstream.invalid", api_key="UPSTREAM-KEY")
_BODY = {"model": "miner/model", "messages": [{"role": "user", "content": "hi"}], "top_k": 50}


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
    resp = _post(proxy, proxy.auth_token, _BODY)
    assert resp.status_code == 200
    call = fake.calls[-1]
    # The proxy injected the upstream key, never exposing it to the caller.
    assert call["headers"]["Authorization"] == "Bearer UPSTREAM-KEY"
    # Forwarded to the configured upstream base URL.
    assert call["url"].startswith("http://upstream.invalid/v1/chat/completions")
    # The miner's model was overridden, and miner-controlled sampling stripped.
    assert call["payload"]["model"] == "enforced/model"
    assert "top_k" not in call["payload"]
    assert call["payload"]["temperature"] == 0.0


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
