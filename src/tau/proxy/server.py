"""``LLMProxy`` — a small key-injecting HTTP proxy in front of an OpenAI-compatible
upstream (OpenRouter, our own backend, or a custom endpoint; see ``target.py``).

Clean-room port of the legacy ``OpenRouterProxy``, made upstream-agnostic. It exists
so untrusted miner agent code in a sandbox can reach an LLM **without ever seeing the
upstream key or the internet**: the proxy injects the real key, enforces the model /
provider / sampling params, caps spend with a ``SolveBudget``, and authenticates the
agent with a per-solve token. Runs a threaded TCP server (and optionally a Unix-socket
server); ``bind_port=0`` lets the OS pick a free port.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socketserver
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import httpx

from .budget import (
    COST_LIMIT_EXIT_REASON,
    PROXY_ERROR_EXIT_REASON,
    REQUEST_LIMIT_EXIT_REASON,
    TOKEN_LIMIT_EXIT_REASON,
    ProxyRequestRecord,
    SolveBudget,
    SolveUsageSummary,
)
from .cache import CacheMissError
from .parsing import (
    estimate_prompt_tokens,
    extract_cache_write_tokens,
    extract_cached_tokens,
    extract_completion_tokens,
    extract_cost,
    extract_generation_id,
    extract_prompt_tokens,
    extract_reasoning_tokens,
    extract_request_model,
    extract_requested_max_output_tokens,
    extract_response_error,
    extract_response_model,
    extract_total_tokens,
    loads_json_bytes,
    request_payload_has_messages,
    set_requested_max_output_tokens,
)
from .rollout import build_llm_event, utc_now
from .target import UpstreamTarget
from .upstream import CachedUpstreamClient, HttpxUpstreamClient, UpstreamClient

log = logging.getLogger(__name__)

# Upstream statuses that indicate an infrastructure/provider fault (our problem, not the
# agent's): our key rejected (401/403), out of funds (402), request timeout (408), rate
# limited (429). 5xx is treated as infra separately. A 400/404/422 is the agent's own
# malformed request and is deliberately excluded.
_INFRA_UPSTREAM_STATUSES = frozenset({401, 402, 403, 408, 429})

_MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
_ALLOWED_METHODS = {"POST", "HEAD"}
_ALLOWED_PATHS = {"/v1/chat/completions", "/v1/messages"}
_VALIDATOR_SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 1.0}
_MINER_CONTROLLED_SAMPLING_PARAMS = {
    "top_k",
    "min_p",
    "top_a",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
}
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _is_upstream_infra_failure(request: ProxyRequestRecord) -> bool:
    """Whether a failed request reflects an upstream infrastructure fault, not the agent.

    True for a transport failure (no status but an error string — unreachable/timeout)
    or an infra-class status (``_INFRA_UPSTREAM_STATUSES`` or any 5xx). False for the
    agent's own bad requests (4xx like 400/404/422) and for successful responses.
    """
    if request.status_code is None:
        return request.error is not None  # transport error: connect/read timeout, DNS
    return request.status_code >= 500 or request.status_code in _INFRA_UPSTREAM_STATUSES


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _ReusableThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


@dataclass(slots=True)
class LLMProxy:
    upstream: UpstreamTarget
    solve_budget: SolveBudget | None = None
    bind_host: str | None = "127.0.0.1"
    bind_port: int = 0
    unix_socket_path: str | None = None
    enforced_model: str | None = None
    enforced_provider: dict[str, Any] | None = None
    enforced_sampling_params: dict[str, Any] | None = field(
        default_factory=lambda: dict(_VALIDATOR_SAMPLING_PARAMS)
    )
    require_auth: bool = True
    auth_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    rollout_event_sink: Callable[[dict[str, Any]], None] | None = None
    rollout_capture_bodies: bool = False
    cache_dir: Path | None = None
    cache_replay_only: bool = False
    # Upstream transport timeouts (seconds). read = how long one LLM call may take
    # (settable via TAU_PROXY_REQUEST_TIMEOUT_SECONDS); connect governs how fast an
    # unreachable upstream is detected. write/pool reuse the connect timeout.
    upstream_read_timeout_seconds: float = 600.0
    upstream_connect_timeout_seconds: float = 30.0
    _server: _ReusableThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _unix_server: _ReusableThreadingUnixServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _unix_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _upstream_index: int = field(default=0, init=False, repr=False)
    _usage: SolveUsageSummary = field(default_factory=SolveUsageSummary, init=False, repr=False)
    _upstream_client: UpstreamClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        timeout = httpx.Timeout(
            connect=self.upstream_connect_timeout_seconds,
            read=self.upstream_read_timeout_seconds,
            write=self.upstream_connect_timeout_seconds,
            pool=self.upstream_connect_timeout_seconds,
        )
        httpx_client = HttpxUpstreamClient(
            on_first_token=self._record_first_token, timeout=timeout
        )
        if self.cache_dir is not None:
            inner = None if self.cache_replay_only else httpx_client
            self._upstream_client = CachedUpstreamClient(self.cache_dir, inner=inner)
        else:
            self._upstream_client = httpx_client

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._server is not None or self._unix_server is not None:
            return

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                # Suppress BrokenPipeError from wfile.flush() after do_POST returns.
                try:
                    super().handle()
                except BrokenPipeError:
                    pass

            def do_GET(self) -> None:  # noqa: N802
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

            def address_string(self) -> str:
                # Unix sockets pass a (often empty) string as client_address.
                if isinstance(self.client_address, str):
                    return self.client_address or "unix"
                return super().address_string()

            def log_message(self, format: str, *args: object) -> None:
                log.debug("proxy %s - %s", self.address_string(), format % args)

            def _handle(self) -> None:
                try:
                    proxy._handle_request(self)
                except BrokenPipeError:
                    log.debug("Client disconnected (broken pipe)")
                except Exception:  # noqa: BLE001
                    log.exception("LLM proxy request failed")
                    try:
                        self.send_error(502, "Proxy request failed")
                    except BrokenPipeError:
                        pass

        if self.bind_host is not None:
            self._server = _ReusableThreadingHTTPServer((self.bind_host, self.bind_port), Handler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            log.debug(
                "LLM proxy listening on %s:%s -> %s (%d endpoint%s)",
                self.host,
                self.port,
                self.upstream.name,
                self.upstream.endpoint_count,
                "" if self.upstream.endpoint_count == 1 else "s",
            )

        if self.unix_socket_path:
            socket_dir = os.path.dirname(self.unix_socket_path)
            if socket_dir:
                os.makedirs(socket_dir, exist_ok=True)
            if os.path.exists(self.unix_socket_path):
                os.unlink(self.unix_socket_path)
            self._unix_server = _ReusableThreadingUnixServer(self.unix_socket_path, Handler)
            # World-writable so containers (which drop CAP_DAC_OVERRIDE) can connect.
            os.chmod(self.unix_socket_path, 0o777)
            self._unix_thread = threading.Thread(
                target=self._unix_server.serve_forever, daemon=True
            )
            self._unix_thread.start()
            log.debug("LLM proxy listening on unix socket %s", self.unix_socket_path)

    def stop(self) -> None:
        if self._server is None and self._unix_server is None:
            return
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._unix_server is not None:
            self._unix_server.shutdown()
            self._unix_server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._unix_thread is not None:
            self._unix_thread.join(timeout=5)
        self._server = None
        self._unix_server = None
        self._thread = None
        self._unix_thread = None
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            os.unlink(self.unix_socket_path)

    def __enter__(self) -> LLMProxy:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    @property
    def host(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy server is not running")
        return cast(str, self._server.server_address[0])

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("Proxy server is not running")
        return int(self._server.server_address[1])

    def container_base_url(self, host_name: str) -> str:
        """The base URL a container on the same network uses to reach this proxy."""
        return f"http://{host_name}:{self.port}/v1"

    def usage_snapshot(self) -> SolveUsageSummary:
        with self._lock:
            return self._usage.snapshot()

    @property
    def budget_exceeded_reason(self) -> str | None:
        with self._lock:
            return self._usage.budget_exceeded_reason

    # -- request handling -----------------------------------------------------
    def _handle_request(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.command == "HEAD":
            handler.send_response(200)
            handler.send_header("Content-Length", "0")
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.close_connection = True
            return
        if handler.command not in _ALLOWED_METHODS:
            self._reject_request(
                handler, reason=PROXY_ERROR_EXIT_REASON, status=405,
                error_type="proxy_policy_violation", message="Method not allowed",
                method=handler.command, path=handler.path, request_model=None,
            )
            return
        request_path = handler.path.split("?", 1)[0]
        if request_path not in _ALLOWED_PATHS:
            self._reject_request(
                handler, reason=PROXY_ERROR_EXIT_REASON, status=403,
                error_type="proxy_policy_violation", message="Endpoint not allowed",
                method=handler.command, path=handler.path, request_model=None,
            )
            return

        if self.require_auth:
            expected_auth = f"Bearer {self.auth_token}"
            auth_header = handler.headers.get("Authorization")
            api_key_header = handler.headers.get("x-api-key")
            if auth_header != expected_auth and api_key_header != self.auth_token:
                self._reject_request(
                    handler, reason=PROXY_ERROR_EXIT_REASON, status=401,
                    error_type="proxy_policy_violation", message="Unauthorized",
                    method=handler.command, path=handler.path, request_model=None,
                )
                return

        try:
            content_length = int(handler.headers.get("Content-Length", "0") or "0")
        except ValueError:
            content_length = -1
        if content_length < 0:
            self._reject_request(
                handler, reason=PROXY_ERROR_EXIT_REASON, status=400,
                error_type="proxy_policy_violation", message="Invalid Content-Length header",
                method=handler.command, path=handler.path, request_model=None,
            )
            return
        if content_length > _MAX_REQUEST_BODY_BYTES:
            self._reject_request(
                handler, reason=PROXY_ERROR_EXIT_REASON, status=413,
                error_type="proxy_policy_violation", message="Request body too large",
                method=handler.command, path=handler.path, request_model=None,
            )
            return
        body = handler.rfile.read(content_length) if content_length > 0 else None
        request_payload = loads_json_bytes(body)
        if content_length > 0 and not isinstance(request_payload, dict):
            self._reject_request(
                handler, reason=PROXY_ERROR_EXIT_REASON, status=400,
                error_type="proxy_policy_violation",
                message="Request body must be a JSON object",
                method=handler.command, path=handler.path, request_model=None,
            )
            return
        request_model = extract_request_model(request_payload)
        if (not request_model and not self.enforced_model) or not request_payload_has_messages(
            request_payload
        ):
            self._reject_request(
                handler, reason=PROXY_ERROR_EXIT_REASON, status=400,
                error_type="proxy_policy_violation",
                message="Request body must include model and messages",
                method=handler.command, path=handler.path, request_model=request_model,
            )
            return
        body, rejection_reason = self._prepare_request_body(
            body=body, request_payload=request_payload
        )
        if rejection_reason:
            self._reject_request(
                handler, reason=rejection_reason, status=429,
                error_type="budget_exceeded", message="Solve budget exceeded",
                method=handler.command, path=handler.path, request_model=request_model,
            )
            return
        prepared_payload = loads_json_bytes(body)
        if isinstance(prepared_payload, dict):
            request_model = extract_request_model(prepared_payload) or request_model

        started_at = utc_now()
        start = time.monotonic()

        try:
            upstream_base_url = self._next_upstream_base_url()
            upstream = self._upstream_client.fetch(
                command=handler.command,
                url=f"{upstream_base_url}{handler.path}",
                headers=self._build_upstream_headers(handler),
                body=body,
                prepared_payload=prepared_payload,
                request_path=request_path,
                start=start,
            )
        except CacheMissError:
            error_body = json.dumps(
                {
                    "error": {
                        "message": "No cached response available for this request",
                        "type": "proxy_cache_miss",
                        "code": PROXY_ERROR_EXIT_REASON,
                    },
                }
            ).encode("utf-8")
            with self._lock:
                self._usage.request_count -= 1  # undo the slot claimed in _prepare_request_body
                self._usage.rejected_request_count += 1
                self._usage.error_count += 1
                self._usage.requests.append(
                    ProxyRequestRecord(
                        method=handler.command, path=handler.path, status_code=503,
                        latency_ms=0, request_model=request_model, rejected=True,
                        error="proxy_cache_miss",
                    )
                )
            self._send_raw(handler, 503, error_body, content_type="application/json")
            return
        except httpx.HTTPError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            finished_at = utc_now()
            is_timeout = isinstance(exc, httpx.TimeoutException)
            request_record = ProxyRequestRecord(
                method=handler.command, path=handler.path, status_code=None,
                latency_ms=latency_ms, request_model=request_model,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._record_request(request_record, upstream_timeout=is_timeout)
            self._emit_rollout_llm_event(
                method=handler.command, path=handler.path,
                request_payload=prepared_payload if self.rollout_capture_bodies else None,
                response_payload={
                    "error": {"message": str(exc), "type": "upstream_transport_error"}
                },
                request_record=request_record, started_at=started_at, finished_at=finished_at,
            )
            raise

        latency_ms = int((time.monotonic() - start) * 1000)
        finished_at = utc_now()
        request_record = ProxyRequestRecord(
            method=handler.command,
            path=handler.path,
            status_code=upstream.status,
            latency_ms=latency_ms,
            request_model=request_model,
            response_model=extract_response_model(upstream.payload),
            generation_id=extract_generation_id(upstream.payload),
            first_token_latency_ms=upstream.first_token_latency_ms,
            prompt_tokens=extract_prompt_tokens(upstream.payload),
            completion_tokens=extract_completion_tokens(upstream.payload),
            total_tokens=extract_total_tokens(upstream.payload),
            cached_tokens=extract_cached_tokens(upstream.payload),
            cache_write_tokens=extract_cache_write_tokens(upstream.payload),
            reasoning_tokens=extract_reasoning_tokens(upstream.payload),
            cost=extract_cost(upstream.payload),
            error=extract_response_error(upstream.payload) if upstream.status >= 400 else None,
        )
        self._record_request(request_record)
        self._emit_rollout_llm_event(
            method=handler.command, path=handler.path,
            request_payload=prepared_payload if self.rollout_capture_bodies else None,
            response_payload=upstream.payload if self.rollout_capture_bodies else None,
            request_record=request_record, started_at=started_at, finished_at=finished_at,
        )

        handler.send_response(upstream.status)
        for key, value in upstream.headers.items():
            if key.lower() in _HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                continue
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(upstream.body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(upstream.body)
        handler.wfile.flush()
        handler.close_connection = True

    def _next_upstream_base_url(self) -> str:
        with self._lock:
            index = self._upstream_index % self.upstream.endpoint_count
            base_url = self.upstream.base_urls[index]
            self._upstream_index += 1
            return base_url

    @staticmethod
    def _send_raw(
        handler: BaseHTTPRequestHandler, status: int, body: bytes, *, content_type: str
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
        handler.close_connection = True

    def _emit_rollout_llm_event(
        self,
        *,
        method: str,
        path: str,
        request_payload: Any,
        response_payload: Any,
        request_record: ProxyRequestRecord,
        started_at: str,
        finished_at: str,
    ) -> None:
        if self.rollout_event_sink is None:
            return
        usage = {
            "prompt_tokens": request_record.prompt_tokens,
            "completion_tokens": request_record.completion_tokens,
            "total_tokens": request_record.total_tokens,
            "cached_tokens": request_record.cached_tokens,
            "cache_write_tokens": request_record.cache_write_tokens,
            "reasoning_tokens": request_record.reasoning_tokens,
        }
        secrets_tuple = tuple(
            value for value in (self.upstream.api_key, self.auth_token) if value
        )
        event = build_llm_event(
            method=method, path=path, request_payload=request_payload,
            response_payload=response_payload, status_code=request_record.status_code,
            latency_ms=request_record.latency_ms, request_model=request_record.request_model,
            response_model=request_record.response_model, usage=usage,
            cost=request_record.cost, started_at=started_at, finished_at=finished_at,
            secrets=secrets_tuple,
        )
        try:
            self.rollout_event_sink(event)
        except Exception:  # noqa: BLE001 — a faulty sink must not break the proxy
            log.exception("rollout LLM event sink failed")

    def _build_upstream_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        # Inject the upstream key here so it never reaches the sandbox; forward the
        # rest of the agent's headers minus hop-by-hop and the agent's own auth.
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.upstream.api_key}",
            "X-Title": "tau",
        }
        for key, value in handler.headers.items():
            lowered = key.lower()
            if lowered in _HOP_BY_HOP_HEADERS or lowered in {"authorization", "x-api-key"}:
                continue
            headers[key] = value
        return headers

    # -- budget enforcement ---------------------------------------------------
    def _prepare_request_body(
        self, *, body: bytes | None, request_payload: Any
    ) -> tuple[bytes | None, str | None]:
        if isinstance(request_payload, dict) and self.enforced_model:
            request_payload["model"] = self.enforced_model
            body = json.dumps(request_payload).encode("utf-8")
        if isinstance(request_payload, dict) and self.enforced_provider is not None:
            request_payload["provider"] = dict(self.enforced_provider)
            body = json.dumps(request_payload).encode("utf-8")
        if isinstance(request_payload, dict) and self.enforced_sampling_params is not None:
            for key in _MINER_CONTROLLED_SAMPLING_PARAMS:
                request_payload.pop(key, None)
            request_payload.update(self.enforced_sampling_params)
            body = json.dumps(request_payload).encode("utf-8")
        if not self.solve_budget or not self.solve_budget.enabled():
            with self._lock:
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._check_request_limit_locked()
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._usage.request_count += 1
            return body, None

        if not isinstance(request_payload, dict):
            with self._lock:
                self._check_pre_request_budget_locked()
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._check_request_limit_locked()
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._usage.request_count += 1
            return body, None

        estimated_prompt_tokens = estimate_prompt_tokens(request_payload)
        with self._lock:
            self._check_pre_request_budget_locked()
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._check_request_limit_locked()
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._check_estimated_request_budget_locked(
                estimated_prompt_tokens=estimated_prompt_tokens, request_payload=request_payload
            )
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._clamp_request_tokens_locked(
                request_payload=request_payload, estimated_prompt_tokens=estimated_prompt_tokens
            )
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._usage.request_count += 1

        return json.dumps(request_payload).encode("utf-8"), None

    def _check_pre_request_budget_locked(self) -> None:
        if self._usage.budget_exceeded_reason or not self.solve_budget:
            return
        budget = self.solve_budget
        if budget.max_cost is not None and self._usage.cost >= budget.max_cost:
            self._usage.budget_exceeded_reason = COST_LIMIT_EXIT_REASON
            return
        if budget.max_total_tokens is not None and self._usage.total_tokens >= budget.max_total_tokens:
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
            return
        if (
            budget.max_prompt_tokens is not None
            and self._usage.prompt_tokens >= budget.max_prompt_tokens
        ):
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
            return
        if (
            budget.max_completion_tokens is not None
            and self._usage.completion_tokens >= budget.max_completion_tokens
        ):
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON

    def _check_request_limit_locked(self) -> None:
        if self._usage.budget_exceeded_reason or not self.solve_budget:
            return
        if (
            self.solve_budget.max_requests is not None
            and self._usage.request_count >= self.solve_budget.max_requests
        ):
            self._usage.budget_exceeded_reason = REQUEST_LIMIT_EXIT_REASON

    def _check_estimated_request_budget_locked(
        self, *, estimated_prompt_tokens: int, request_payload: dict[str, Any]
    ) -> None:
        if not self.solve_budget or self._usage.budget_exceeded_reason:
            return
        if self.solve_budget.max_prompt_tokens is not None:
            remaining_prompt = max(
                0, self.solve_budget.max_prompt_tokens - self._usage.prompt_tokens
            )
            if estimated_prompt_tokens > remaining_prompt:
                self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
                return

        remaining_total = None
        if self.solve_budget.max_total_tokens is not None:
            remaining_total = max(0, self.solve_budget.max_total_tokens - self._usage.total_tokens)
            if estimated_prompt_tokens >= remaining_total:
                self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
                return

        requested_max_output = extract_requested_max_output_tokens(request_payload)
        if requested_max_output is None or remaining_total is None:
            return
        if estimated_prompt_tokens + requested_max_output > remaining_total:
            log.debug(
                "Estimated request would exceed total token budget "
                "prompt_estimate=%s requested_max_output=%s remaining_total=%s",
                estimated_prompt_tokens, requested_max_output, remaining_total,
            )

    def _clamp_request_tokens_locked(
        self, *, request_payload: dict[str, Any], estimated_prompt_tokens: int
    ) -> None:
        if not self.solve_budget:
            return
        limits: list[int] = []
        if self.solve_budget.max_tokens_per_request is not None:
            limits.append(self.solve_budget.max_tokens_per_request)
        if self.solve_budget.max_completion_tokens is not None:
            limits.append(
                max(0, self.solve_budget.max_completion_tokens - self._usage.completion_tokens)
            )
        if self.solve_budget.max_total_tokens is not None:
            remaining_total = max(0, self.solve_budget.max_total_tokens - self._usage.total_tokens)
            limits.append(max(0, remaining_total - estimated_prompt_tokens))
        average_cost_per_token = self._average_cost_per_token_locked()
        if (
            self.solve_budget.max_cost is not None
            and average_cost_per_token is not None
            and average_cost_per_token > 0
        ):
            remaining_cost = max(0.0, self.solve_budget.max_cost - self._usage.cost)
            estimated_prompt_cost = estimated_prompt_tokens * average_cost_per_token
            if estimated_prompt_cost >= remaining_cost:
                self._usage.budget_exceeded_reason = COST_LIMIT_EXIT_REASON
                return
            affordable_output = int((remaining_cost - estimated_prompt_cost) / average_cost_per_token)
            limits.append(max(0, affordable_output))
        if not limits:
            return
        allowed_max_tokens = min(limits)
        if allowed_max_tokens <= 0:
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
            return
        set_requested_max_output_tokens(request_payload, allowed_max_tokens)

    def _average_cost_per_token_locked(self) -> float | None:
        if self._usage.total_tokens <= 0 or self._usage.cost <= 0:
            return None
        return self._usage.cost / self._usage.total_tokens

    # -- accounting -----------------------------------------------------------
    def _reject_request(
        self,
        handler: BaseHTTPRequestHandler,
        *,
        reason: str,
        status: int,
        error_type: str,
        message: str,
        method: str,
        path: str,
        request_model: str | None,
    ) -> None:
        request_record = ProxyRequestRecord(
            method=method, path=path, status_code=status, latency_ms=0,
            request_model=request_model, rejected=True, error=reason,
        )
        with self._lock:
            self._usage.budget_exceeded_reason = reason
            self._usage.rejected_request_count += 1
            self._usage.requests.append(request_record)
        response_payload = {
            "error": {"message": message, "type": error_type, "code": reason},
        }
        now = utc_now()
        self._emit_rollout_llm_event(
            method=method, path=path, request_payload=None, response_payload=response_payload,
            request_record=request_record, started_at=now, finished_at=now,
        )
        self._send_raw(
            handler, status, json.dumps(response_payload).encode("utf-8"),
            content_type="application/json",
        )

    def _record_request(
        self, request: ProxyRequestRecord, *, upstream_timeout: bool = False
    ) -> None:
        with self._lock:
            self._usage.requests.append(request)
            if (
                request.status_code is not None
                and request.status_code < 400
                and request.error is None
            ):
                self._usage.success_count += 1
            else:
                self._usage.error_count += 1
                if _is_upstream_infra_failure(request):
                    self._usage.upstream_error_count += 1
                    self._usage.last_upstream_error = (
                        request.error or f"HTTP {request.status_code}"
                    )
                    if upstream_timeout or request.status_code in (408, 504):
                        self._usage.upstream_timeout_count += 1
            self._usage.prompt_tokens += int(request.prompt_tokens or 0)
            self._usage.completion_tokens += int(request.completion_tokens or 0)
            self._usage.total_tokens += int(request.total_tokens or 0)
            self._usage.cached_tokens += int(request.cached_tokens or 0)
            self._usage.cache_write_tokens += int(request.cache_write_tokens or 0)
            self._usage.reasoning_tokens += int(request.reasoning_tokens or 0)
            self._usage.cost += float(request.cost or 0.0)
            self._check_pre_request_budget_locked()

    def _record_first_token(self) -> None:
        with self._lock:
            self._usage.first_token_count += 1
