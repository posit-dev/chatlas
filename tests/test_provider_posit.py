import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
import requests
from chatlas._provider import ModelInfo
from chatlas._provider_posit import (
    OAUTH_SCOPE,
    ChatPosit,
    PositAnthropicProvider,
    PositAuth,
    PositCredentials,
    PositOpenAIProvider,
    _make_posit_error_hook,
    _make_posit_error_hook_async,
    _posit_error_message,
    list_models_posit,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any]):
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> dict[str, Any]:
        return self._json_data

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def test_get_token_returns_cached_token_when_not_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"
    cache_path.write_text(
        json.dumps(
            {
                "access_token": "cached-token",
                "refresh_token": "cached-refresh",
                "expires_at": time.time() + 3600,
            }
        )
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("should not make a network call for a fresh token")

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fail_if_called)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "cached-token"


def test_get_token_refreshes_expired_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"
    cache_path.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "old-refresh",
                "expires_at": time.time() - 10,
            }
        )
    )

    calls: list[dict[str, Any]] = []

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        calls.append({"url": url, "data": data})
        return _FakeResponse(
            200,
            {
                "access_token": "new-token",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "new-token"

    assert len(calls) == 1
    assert calls[0]["data"]["grant_type"] == "refresh_token"
    assert calls[0]["data"]["refresh_token"] == "old-refresh"
    assert calls[0]["data"]["scope"] == OAUTH_SCOPE

    cached = json.loads(cache_path.read_text())
    assert cached["access_token"] == "new-token"


def test_get_token_falls_back_to_device_flow_when_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"
    cache_path.write_text(
        json.dumps(
            {
                "access_token": "old-token",
                "refresh_token": "old-refresh",
                "expires_at": time.time() - 10,
            }
        )
    )

    device_response = _FakeResponse(
        200,
        {
            "device_code": "device-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://login.posit.cloud/device",
            "interval": 1,
            "expires_in": 60,
        },
    )
    poll_response = _FakeResponse(
        200,
        {
            "access_token": "fresh-token",
            "refresh_token": "fresh-refresh",
            "expires_in": 3600,
        },
    )

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        if data and data.get("grant_type") == "refresh_token":
            raise requests.HTTPError("refresh failed")
        if url.endswith("/device/authorize"):
            return device_response
        return poll_response

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)
    monkeypatch.setattr("chatlas._provider_posit.time.sleep", lambda _: None)
    monkeypatch.setattr("chatlas._provider_posit.is_interactive", lambda: False)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "fresh-token"


def test_get_token_runs_device_flow_when_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"

    device_response = _FakeResponse(
        200,
        {
            "device_code": "device-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://login.posit.cloud/device",
            "interval": 1,
            "expires_in": 60,
        },
    )
    poll_response = _FakeResponse(
        200,
        {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        },
    )
    poll_calls: list[dict[str, Any]] = []

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        if url.endswith("/device/authorize"):
            return device_response
        poll_calls.append({"url": url, "data": data})
        return poll_response

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)
    monkeypatch.setattr("chatlas._provider_posit.time.sleep", lambda _: None)
    monkeypatch.setattr("chatlas._provider_posit.is_interactive", lambda: False)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "new-token"
    assert len(poll_calls) == 1
    assert poll_calls[0]["data"]["scope"] == OAUTH_SCOPE

    assert cache_path.exists()
    assert (cache_path.stat().st_mode & 0o777) == 0o600
    cached = json.loads(cache_path.read_text())
    assert cached["access_token"] == "new-token"


def test_device_flow_handles_pending_and_slow_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "posit-credentials.json"

    device_response = _FakeResponse(
        200,
        {
            "device_code": "device-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://login.posit.cloud/device",
            "interval": 1,
            "expires_in": 60,
        },
    )
    responses = [
        device_response,
        _FakeResponse(400, {"error": "authorization_pending"}),
        _FakeResponse(400, {"error": "slow_down"}),
        _FakeResponse(200, {"access_token": "final-token", "expires_in": 3600}),
    ]

    def fake_post(url: str, data: Optional[dict[str, Any]] = None, **kwargs: Any):
        return responses.pop(0)

    sleep_calls: list[float] = []
    monkeypatch.setattr("chatlas._provider_posit.requests.post", fake_post)
    monkeypatch.setattr("chatlas._provider_posit.time.sleep", sleep_calls.append)
    monkeypatch.setattr("chatlas._provider_posit.is_interactive", lambda: False)

    creds = PositCredentials(cache_path=cache_path)
    assert creds.get_token() == "final-token"
    assert sleep_calls == [1, 1, 6]  # interval starts at 1s, +5s after slow_down


def test_posit_error_message_for_credits_depleted():
    assert _posit_error_message(402, None) == "Your Posit AI credits are depleted."


def test_posit_error_message_for_account_not_found():
    message = _posit_error_message(403, {"error_type": "prism_account_not_found"})
    assert message is not None
    assert "service agreement" in message


def test_posit_error_message_for_nested_account_not_found():
    message = _posit_error_message(
        403, {"error": {"error_type": "prism_account_not_found"}}
    )
    assert message is not None
    assert "service agreement" in message


def test_posit_error_message_returns_none_for_unrecognized_errors():
    assert _posit_error_message(400, {"error": "bad request"}) is None
    assert _posit_error_message(500, None) is None
    assert _posit_error_message(403, {"error_type": "something_else"}) is None


def test_posit_auth_sets_bearer_token_and_strips_api_key():
    auth = PositAuth(lambda: "test-token")
    request = httpx.Request("GET", "https://gateway.posit.ai/anthropic/v1/messages")
    request.headers["x-api-key"] = "not-used"

    flow = auth.sync_auth_flow(request)
    sent_request = next(flow)

    assert sent_request.headers["Authorization"] == "Bearer test-token"
    assert "x-api-key" not in sent_request.headers


def test_posit_auth_async_sets_bearer_token():
    async def run() -> httpx.Request:
        auth = PositAuth(lambda: "test-token")
        request = httpx.Request(
            "GET", "https://gateway.posit.ai/openai/v1/chat/completions"
        )
        flow = auth.async_auth_flow(request)
        return await flow.__anext__()

    sent_request = asyncio.run(run())
    assert sent_request.headers["Authorization"] == "Bearer test-token"


def test_anthropic_error_hook_propagates_through_client_send():
    from anthropic import Anthropic, AnthropicError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error_type": "prism_account_not_found"}, request=request
        )

    client = Anthropic(
        api_key="not-used",
        base_url="https://gateway.posit.ai/anthropic/v1",
        http_client=httpx.Client(
            auth=PositAuth(lambda: "test-token"),
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [_make_posit_error_hook(AnthropicError)]},
        ),
    )

    with pytest.raises(AnthropicError, match="service agreement"):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_anthropic_error_hook_propagates_through_async_client_send():
    from anthropic import AnthropicError, AsyncAnthropic

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error_type": "prism_account_not_found"}, request=request
        )

    client = AsyncAnthropic(
        api_key="not-used",
        base_url="https://gateway.posit.ai/anthropic/v1",
        http_client=httpx.AsyncClient(
            auth=PositAuth(lambda: "test-token"),
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [_make_posit_error_hook_async(AnthropicError)]},
        ),
    )

    with pytest.raises(AnthropicError, match="service agreement"):
        await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )


def test_openai_error_hook_propagates_through_client_send():
    from openai import OpenAI, OpenAIError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={}, request=request)

    client = OpenAI(
        api_key="not-used",
        base_url="https://gateway.posit.ai/openai/v1",
        http_client=httpx.Client(
            auth=PositAuth(lambda: "test-token"),
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [_make_posit_error_hook(OpenAIError)]},
        ),
    )

    with pytest.raises(OpenAIError, match="credits are depleted"):
        client.chat.completions.create(
            model="qwen3-8b", messages=[{"role": "user", "content": "hi"}]
        )


def test_unrecognized_gateway_error_passes_through_unchanged():
    from anthropic import Anthropic, APIStatusError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}}, request=request)

    from anthropic import AnthropicError

    client = Anthropic(
        api_key="not-used",
        base_url="https://gateway.posit.ai/anthropic/v1",
        http_client=httpx.Client(
            auth=PositAuth(lambda: "test-token"),
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [_make_posit_error_hook(AnthropicError)]},
        ),
    )

    with pytest.raises(APIStatusError):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )


def test_posit_anthropic_provider_uses_anthropic_flavor_base_url():
    provider = PositAnthropicProvider(
        base_url="https://gateway.posit.ai",
        model="claude-sonnet-4-6",
        credentials=lambda: "test-token",
    )
    # The anthropic SDK appends "/v1/messages" itself, so the base_url must
    # not also include "/v1" or requests 404 against the gateway.
    assert (
        str(provider._client.base_url).rstrip("/")
        == "https://gateway.posit.ai/anthropic"
    )


def test_posit_openai_provider_uses_openai_flavor_base_url():
    provider = PositOpenAIProvider(
        base_url="https://gateway.posit.ai",
        model="qwen3-8b",
        credentials=lambda: "test-token",
    )
    assert (
        str(provider._client.base_url).rstrip("/")
        == "https://gateway.posit.ai/openai/v1"
    )


def test_posit_provider_list_models_delegates_to_list_models_posit(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = PositAnthropicProvider(
        base_url="https://gateway.posit.ai",
        model="claude-sonnet-4-6",
        credentials=lambda: "test-token",
    )

    captured: dict[str, Any] = {}

    def fake_list_models_posit(base_url: str, credentials: Any) -> list[ModelInfo]:
        captured["base_url"] = base_url
        captured["credentials"] = credentials
        return [{"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"}]

    monkeypatch.setattr(
        "chatlas._provider_posit.list_models_posit", fake_list_models_posit
    )

    models = provider.list_models()

    assert models == [{"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"}]
    assert captured["base_url"] == "https://gateway.posit.ai"
    assert captured["credentials"]() == "test-token"


def test_list_models_posit_parses_response(monkeypatch: pytest.MonkeyPatch):
    response = _FakeResponse(
        200,
        {
            "chat": [
                {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
                {"id": "qwen3-8b", "display_name": "Qwen3 8B"},
            ]
        },
    )
    captured_headers: dict[str, str] = {}

    def fake_get(url: str, headers: Optional[dict[str, str]] = None, **kwargs: Any):
        captured_headers.update(headers or {})
        return response

    monkeypatch.setattr("chatlas._provider_posit.requests.get", fake_get)

    models = list_models_posit(
        base_url="https://gateway.posit.ai", credentials=lambda: "test-token"
    )

    assert models == [
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
        {"id": "qwen3-8b", "name": "Qwen3 8B"},
    ]
    assert captured_headers["Authorization"] == "Bearer test-token"


def test_list_models_posit_raises_friendly_error(monkeypatch: pytest.MonkeyPatch):
    response = _FakeResponse(402, {})
    monkeypatch.setattr(
        "chatlas._provider_posit.requests.get", lambda *a, **k: response
    )

    with pytest.raises(RuntimeError, match="credits are depleted"):
        list_models_posit(credentials=lambda: "test-token")


def test_chat_posit_dispatches_to_anthropic_flavor():
    chat = ChatPosit(model="claude-sonnet-4-6", credentials=lambda: "test-token")
    assert isinstance(chat.provider, PositAnthropicProvider)


def test_chat_posit_dispatches_to_openai_flavor():
    chat = ChatPosit(model="qwen3-8b", credentials=lambda: "test-token")
    assert isinstance(chat.provider, PositOpenAIProvider)


def test_chat_posit_defaults_to_claude_sonnet():
    chat = ChatPosit(credentials=lambda: "test-token")
    assert chat.provider.model == "claude-sonnet-4-6"


def test_chat_posit_custom_credentials_bypasses_device_flow(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("should not hit the network when credentials= is supplied")

    monkeypatch.setattr("chatlas._provider_posit.requests.post", fail_if_called)

    chat = ChatPosit(model="claude-sonnet-4-6", credentials=lambda: "test-token")
    assert isinstance(chat.provider, PositAnthropicProvider)
    assert chat.provider._credentials() == "test-token"


def test_chat_posit_importable_from_package_root():
    import chatlas

    assert chatlas.ChatPosit is ChatPosit
