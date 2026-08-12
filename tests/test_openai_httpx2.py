import asyncio
from typing import Any, cast, get_args, get_type_hints

import httpx
import httpx2
from chatlas import ChatAzureOpenAI, ChatOpenAI
from chatlas._provider_openai import OpenAIProvider
from chatlas._provider_openai_azure import OpenAIAzureProvider
from chatlas.types.openai import ChatAzureClientArgs, ChatClientArgs


def test_openai_client_args_use_native_httpx2_clients():
    expected = {
        type(None),
        httpx2.Client,
        httpx2.AsyncClient,
    }

    client_types = set(get_args(get_type_hints(ChatClientArgs)["http_client"]))
    azure_client_types = set(
        get_args(get_type_hints(ChatAzureClientArgs)["http_client"])
    )

    assert client_types == expected
    assert azure_client_types == expected


def test_openai_routes_legacy_sync_client():
    assert_openai_client_routed(httpx.Client(), is_async=False)


def test_openai_routes_legacy_async_client():
    assert_openai_client_routed(httpx.AsyncClient(), is_async=True)


def test_openai_routes_native_sync_client():
    assert_openai_client_routed(httpx2.Client(), is_async=False)


def test_openai_routes_native_async_client():
    assert_openai_client_routed(httpx2.AsyncClient(), is_async=True)


def test_azure_routes_native_async_client():
    client = httpx2.AsyncClient()
    chat = ChatAzureOpenAI(
        endpoint="https://example.openai.azure.com",
        deployment_id="test",
        api_version="2025-03-01-preview",
        kwargs={"http_client": client},
    )
    provider = cast(OpenAIAzureProvider, chat.provider)

    try:
        assert provider._async_client._client is client
    finally:
        provider._client.close()
        asyncio.run(provider._async_client.close())


def assert_openai_client_routed(
    client: httpx.Client | httpx.AsyncClient | httpx2.Client | httpx2.AsyncClient,
    *,
    is_async: bool,
) -> None:
    kwargs = cast(Any, {"http_client": client})
    chat = ChatOpenAI(kwargs=kwargs)
    provider = cast(OpenAIProvider, chat.provider)
    target = provider._async_client if is_async else provider._client

    try:
        assert target._client is client
    finally:
        provider._client.close()
        asyncio.run(provider._async_client.close())
