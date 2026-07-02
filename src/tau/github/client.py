"""HTTP access to the GitHub REST API."""

from __future__ import annotations

import json
import logging
import time
from types import TracebackType
from typing import Any

import httpx

from .errors import GitHubRequestError
from .tokens import GitHubTokenRotator

log = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "tau-task-generator",
}


def _rate_limit_cooldown(response: httpx.Response) -> float | None:
    """Seconds to wait before reusing a token, read from GitHub's own headers.

    Prefers ``Retry-After`` (secondary rate limits); otherwise, if the primary
    quota is exhausted (``x-ratelimit-remaining: 0``), waits until
    ``x-ratelimit-reset``. Returns None when the headers do not say, leaving the
    rotator to fall back to its default cooldown.
    """
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    if response.headers.get("x-ratelimit-remaining") == "0":
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            try:
                return max(0.0, float(reset) - time.time())
            except ValueError:
                pass
    return None


class GitHubClient:
    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        token_rotator: GitHubTokenRotator | None = None,
    ) -> None:
        self._http = http
        self._rotator = token_rotator

    @classmethod
    def create(
        cls,
        *,
        token: str | None = None,
        token_rotator: GitHubTokenRotator | None = None,
        timeout: float = 30.0,
    ) -> GitHubClient:
        """Build a client backed by a real ``httpx.AsyncClient`` against api.github.com."""
        headers = dict(_DEFAULT_HEADERS)
        if not token_rotator and token:
            headers["Authorization"] = f"Bearer {token}"
        http = httpx.AsyncClient(
            base_url=_GITHUB_API,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        return cls(http=http, token_rotator=token_rotator)

    async def aclose(self) -> None:
        log.debug("Closing HTTP client")
        await self._http.aclose()

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb  # exception info intentionally ignored
        await self.aclose()

    async def get_json(self, path: str, **params: Any) -> Any:
        """Return just the JSON payload for *path*."""
        _, payload = await self.get_with_response(path, **params)
        return payload

    async def get_with_response(
        self, path: str, **params: Any
    ) -> tuple[httpx.Response, Any]:
        """GET *path*; return ``(response, payload)`` or raise ``GitHubRequestError``."""
        used_token: str | None = None
        try:
            log.debug("GET %s params=%s", path, params or None)
            request_headers: dict[str, str] = {}
            if self._rotator:
                used_token = await self._rotator.get_token()
                if used_token:
                    request_headers["Authorization"] = f"Bearer {used_token}"
            response = await self._http.get(
                path, params=params or None, headers=request_headers or None
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # Cool the offending token down (never permanently) and let the caller
            # retry with a fresh token next time, rather than recursing here.
            if self._rotator and used_token:
                if status == 401:
                    self._rotator.mark_unauthorized(used_token)
                elif status in (403, 429):
                    self._rotator.mark_rate_limited(
                        used_token, cooldown_seconds=_rate_limit_cooldown(exc.response)
                    )
            raise GitHubRequestError(f"GET {path} failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise GitHubRequestError(f"GET {path} failed: {exc}") from exc

        try:
            return response, response.json()
        except json.JSONDecodeError as exc:
            raise GitHubRequestError(
                f"GET {path} returned invalid JSON: {exc}"
            ) from exc
