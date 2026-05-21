from __future__ import annotations

import pytest

from chatlas._api_headers import ApiHeaders, resolve_api_headers


class TestResolveApiHeaders:
    def test_none_returns_none(self):
        assert resolve_api_headers(None) is None

    def test_dict_passed_through(self):
        headers = {"Authorization": "Bearer token123", "X-Org-Id": "org-456"}
        result = resolve_api_headers(headers)
        assert result == headers

    def test_callable_returning_dict(self):
        headers = {"Authorization": "Bearer refreshed"}
        result = resolve_api_headers(lambda: headers)
        assert result == headers

    def test_callable_called_each_time(self):
        call_count = 0

        def rotating_token() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"Authorization": f"Bearer token-{call_count}"}

        assert resolve_api_headers(rotating_token) == {
            "Authorization": "Bearer token-1"
        }
        assert resolve_api_headers(rotating_token) == {
            "Authorization": "Bearer token-2"
        }

    def test_callable_returning_wrong_type_raises(self):
        with pytest.raises(TypeError, match="dict"):
            resolve_api_headers(lambda: "not-a-dict")  # type: ignore

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="dict"):
            resolve_api_headers("not-a-dict")  # type: ignore


NO_PROXY: dict[str, str] = {"NO_PROXY": "*"}


class TestProviderApiHeaders:
    """Test that api_headers flow through to the provider correctly."""

    def test_openai_provider_stores_api_headers(self, monkeypatch):
        from chatlas._provider_openai_completions import OpenAICompletionsProvider

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        headers = {"Authorization": "Bearer dynamic-key"}
        provider = OpenAICompletionsProvider(
            api_key="dummy",
            model="gpt-4o",
            api_headers=headers,
        )
        assert provider._api_headers is not None
        assert provider._get_extra_headers() == headers

    def test_openai_provider_callable_api_headers(self, monkeypatch):
        from chatlas._provider_openai_completions import OpenAICompletionsProvider

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        provider = OpenAICompletionsProvider(
            api_key="dummy",
            model="gpt-4o",
            api_headers=lambda: {"Authorization": "Bearer dynamic-key"},
        )
        assert provider._get_extra_headers() == {
            "Authorization": "Bearer dynamic-key"
        }

    def test_openai_provider_no_api_headers(self, monkeypatch):
        from chatlas._provider_openai_completions import OpenAICompletionsProvider

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        provider = OpenAICompletionsProvider(
            api_key="real-key",
            model="gpt-4o",
        )
        assert provider._api_headers is None
        assert provider._get_extra_headers() is None


class TestChatFunctionApiHeaders:
    """Test that Chat* functions accept api_headers."""

    def test_chat_openai_completions_accepts_api_headers(self, monkeypatch):
        from chatlas import ChatOpenAICompletions

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        chat = ChatOpenAICompletions(
            model="gpt-4o",
            api_key="dummy",
            api_headers={"X-Custom": "value"},
        )
        assert chat.provider._api_headers is not None

    def test_chat_openai_accepts_api_headers(self, monkeypatch):
        from chatlas import ChatOpenAI

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        chat = ChatOpenAI(
            model="gpt-4o",
            api_key="dummy",
            api_headers={"Authorization": "Bearer my-token"},
        )
        assert chat.provider._api_headers is not None


class TestCallableApiKey:
    """Test that api_key accepts callables (passed through to the openai SDK)."""

    def test_callable_api_key_accepted(self, monkeypatch):
        from chatlas import ChatOpenAICompletions

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        chat = ChatOpenAICompletions(
            model="gpt-4o",
            api_key=lambda: "dynamic-key",
        )
        assert chat.provider._client.api_key is not None

    def test_callable_api_key_responses_api(self, monkeypatch):
        from chatlas import ChatOpenAI

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        chat = ChatOpenAI(
            model="gpt-4o",
            api_key=lambda: "dynamic-key",
        )
        assert chat.provider._client.api_key is not None

    def test_callable_api_key_async_client_wrapped(self, monkeypatch):
        import inspect

        from chatlas import ChatOpenAI

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        chat = ChatOpenAI(
            model="gpt-4o",
            api_key=lambda: "dynamic-key",
        )
        # The SDK stores the callable on _api_key_provider internally.
        # The async client's provider should be an async function (wrapped).
        async_provider = chat.provider._async_client._api_key_provider
        assert callable(async_provider)
        assert inspect.iscoroutinefunction(async_provider)

    def test_portkey_callable_api_key_rejected(self, monkeypatch):
        from chatlas import ChatPortkey

        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(k, raising=False)
            monkeypatch.delenv(k.lower(), raising=False)

        with pytest.raises(TypeError, match="callable"):
            ChatPortkey(
                model="gpt-4o",
                api_key=lambda: "dynamic-key",
            )
