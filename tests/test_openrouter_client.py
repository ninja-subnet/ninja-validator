import unittest
from unittest.mock import patch

import httpx

from openrouter_client import complete_text


class _FakeResponse:
    def __init__(self, payload, *, status_code: int = 200, headers: dict[str, str] | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=httpx.Response(self.status_code, request=request, json=self._payload),
            )
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.request_json = None
        self.request_url = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers, json):
        self.request_url = url
        self.request_json = json
        return _FakeResponse(self.payload)


class OpenRouterClientTest(unittest.TestCase):
    def test_complete_text_passes_reasoning_config(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            text = complete_text(
                prompt="judge",
                model="deepseek/deepseek-v4-flash",
                timeout=10,
                openrouter_api_key="key",
                reasoning={"effort": "medium", "exclude": True},
            )

        self.assertEqual(text, "ok")
        self.assertEqual(client.request_json["reasoning"], {"effort": "medium", "exclude": True})

    def test_complete_text_passes_cache_control_config(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            text = complete_text(
                prompt="judge",
                model="anthropic/claude-sonnet-4.6",
                timeout=10,
                openrouter_api_key="key",
                cache_control={"type": "ephemeral"},
            )

        self.assertEqual(text, "ok")
        self.assertEqual(client.request_json["cache_control"], {"type": "ephemeral"})

    def test_complete_text_passes_provider_override(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            text = complete_text(
                prompt="judge",
                model="z-ai/glm-5.2",
                timeout=10,
                openrouter_api_key="key",
                provider={"only": ["z-ai/fp8"], "allow_fallbacks": False},
            )

        self.assertEqual(text, "ok")
        self.assertEqual(
            client.request_json["provider"],
            {"only": ["z-ai/fp8"], "allow_fallbacks": False},
        )

    def test_complete_text_passes_structured_content_blocks(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )
        content = [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "dynamic"},
        ]

        with patch("openrouter_client.httpx.Client", return_value=client):
            text = complete_text(
                prompt=content,
                model="anthropic/claude-sonnet-4.6",
                timeout=10,
                openrouter_api_key="key",
            )

        self.assertEqual(text, "ok")
        self.assertEqual(client.request_json["messages"][0]["content"], content)

    def test_reasoning_fallback_is_used_when_content_missing(self):
        client = _FakeClient(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None, "reasoning": "final answer"},
                    },
                ],
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            text = complete_text(
                prompt="judge",
                model="deepseek/deepseek-v4-flash",
                timeout=10,
                openrouter_api_key="key",
            )

        self.assertEqual(text, "final answer")

    def test_empty_content_error_includes_reasoning_metadata(self):
        client = _FakeClient(
            {
                "choices": [
                    {
                        "finish_reason": "error",
                        "native_finish_reason": "MALFORMED_FUNCTION_CALL",
                        "message": {"content": ""},
                    },
                ],
                "usage": {
                    "completion_tokens": 0,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "MALFORMED_FUNCTION_CALL"):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

    def test_no_choices_error_includes_openrouter_error_payload(self):
        client = _FakeClient(
            {
                "error": {
                    "code": 429,
                    "message": "rate limited by upstream provider",
                },
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "error_code=429"):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

        with self.assertRaisesRegex(RuntimeError, "rate limited by upstream provider"):
            with patch("openrouter_client.httpx.Client", return_value=client):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

    def test_complete_text_reads_base_url_from_env_at_call_time(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )

        with patch.dict(
            "openrouter_client.os.environ",
            {"OPENROUTER_BASE_URL": "https://example.test/custom"},
            clear=False,
        ):
            with patch("openrouter_client.httpx.Client", return_value=client):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

        self.assertEqual(
            client.request_url,
            "https://example.test/custom/v1/chat/completions",
        )

    def test_complete_text_retries_rate_limited_upstream_response(self):
        ok_response = _FakeResponse(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )
        rate_limited = _FakeResponse(
            {"error": {"code": 429, "message": "rate limited by upstream provider"}},
            status_code=429,
            headers={"retry-after": "2"},
        )
        call_count = {"value": 0}

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, headers, json):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    rate_limited.raise_for_status()
                return ok_response

        with patch("openrouter_client.httpx.Client", return_value=_Client()):
            with patch("openrouter_client.time.sleep") as sleep_mock:
                text = complete_text(
                    prompt="judge",
                    model="google/gemini-3.1-flash-lite",
                    timeout=10,
                    openrouter_api_key="key",
                    rate_limit_retries=3,
                )

        self.assertEqual(text, "ok")
        self.assertEqual(call_count["value"], 2)
        sleep_mock.assert_called_once_with(2.0)


if __name__ == "__main__":
    unittest.main()
