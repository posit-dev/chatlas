import sys
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from chatlas import Chat, ChatAnthropic, ChatOpenAI


def test_chat_close_delegates_to_provider():
    chat = ChatOpenAI(model="gpt-4o")
    with patch.object(chat.provider, "close") as mock_close:
        chat.close()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_chat_close_async_closes_mcp_and_provider():
    chat = ChatOpenAI(model="gpt-4o")
    chat._mcp_manager.close_sessions = AsyncMock()
    with patch.object(
        chat.provider, "close_async", new=AsyncMock()
    ) as mock_close:
        await chat.close_async()
        chat._mcp_manager.close_sessions.assert_awaited_once()
        mock_close.assert_awaited_once()


def test_chat_close_warns_with_open_mcp_sessions():
    chat = ChatOpenAI(model="gpt-4o")
    chat._mcp_manager._mcp_sessions["fake"] = MagicMock()
    with patch.object(chat.provider, "close"):
        with pytest.warns(UserWarning, match="MCP server sessions"):
            chat.close()
        # No warning once sessions are gone
        chat._mcp_manager._mcp_sessions.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chat.close()


def test_chat_context_manager():
    with patch.object(Chat, "close") as mock_close:
        with ChatOpenAI(model="gpt-4o"):
            pass
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_chat_async_context_manager():
    with patch.object(Chat, "close_async", new=AsyncMock()) as mock_close:
        async with ChatOpenAI(model="gpt-4o"):
            pass
        mock_close.assert_awaited_once()


def test_openai_provider_close():
    chat = ChatOpenAI(model="gpt-4o")
    provider = chat.provider
    chat.close()
    assert provider._client.is_closed
    # Idempotent
    chat.close()


@pytest.mark.asyncio
async def test_openai_provider_close_async():
    chat = ChatOpenAI(model="gpt-4o")
    provider = chat.provider
    await chat.close_async()
    assert provider._client.is_closed
    assert provider._async_client.is_closed


def test_anthropic_provider_close():
    chat = ChatAnthropic(model="claude-sonnet-4-5")
    provider = chat.provider
    chat.close()
    assert provider._client.is_closed


@pytest.mark.asyncio
async def test_anthropic_provider_close_async():
    chat = ChatAnthropic(model="claude-sonnet-4-5")
    provider = chat.provider
    await chat.close_async()
    assert provider._client.is_closed
    assert provider._async_client.is_closed


def _make_bedrock_converse_provider():
    from chatlas._provider_bedrock_converse import BedrockConverseProvider

    return BedrockConverseProvider(
        model="vendor/model:version",
        aws_profile=None,
        aws_region="us-east-1",
        base_url=None,
    )


def test_bedrock_converse_provider_close():
    provider = _make_bedrock_converse_provider()
    provider.close()
    assert provider._client.is_closed


@pytest.mark.asyncio
async def test_bedrock_converse_provider_close_async():
    provider = _make_bedrock_converse_provider()
    await provider.close_async()
    assert provider._client.is_closed
    assert provider._async_client.is_closed


def test_google_provider_close():
    from chatlas import ChatGoogle

    chat = ChatGoogle(model="gemini-2.5-flash")
    with patch.object(chat.provider._client, "close") as mock_close:
        chat.close()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_google_provider_close_async():
    from chatlas import ChatGoogle

    chat = ChatGoogle(model="gemini-2.5-flash")
    with patch.object(
        chat.provider._client.aio, "aclose", new=AsyncMock()
    ) as mock_close:
        await chat.close_async()
        mock_close.assert_awaited_once()


def _make_mock_snowflake_modules():
    session = MagicMock()
    builder = MagicMock()
    builder.configs.return_value = builder
    builder.create.return_value = session

    snowpark = MagicMock()
    snowpark.Session.builder = builder

    root = MagicMock()
    core = MagicMock()
    core.Root = root

    snowflake = MagicMock()
    snowflake.snowpark = snowpark
    snowflake.core = core

    return snowflake, session


def test_snowflake_provider_close():
    from chatlas import ChatSnowflake

    snowflake, session = _make_mock_snowflake_modules()
    with patch.dict(sys.modules, {"snowflake": snowflake, "snowflake.snowpark": snowflake.snowpark, "snowflake.core": snowflake.core}):
        chat = ChatSnowflake(model="llama3.1-70b", account="a", user="u", password="p")
    chat.close()
    session.close.assert_called_once()
    # Idempotent
    chat.close()
    assert session.close.call_count == 2


def test_snowflake_provider_close_ownership():
    from chatlas import ChatSnowflake

    snowflake, session = _make_mock_snowflake_modules()
    with patch.dict(sys.modules, {"snowflake": snowflake, "snowflake.snowpark": snowflake.snowpark, "snowflake.core": snowflake.core}):
        chat = ChatSnowflake(model="llama3.1-70b", session=session)
    chat.close()
    # A caller-supplied session is not closed.
    session.close.assert_not_called()


def test_databricks_provider_close_ownership():
    from chatlas import ChatDatabricks

    workspace_client = MagicMock()
    openai_client = MagicMock()
    openai_client.base_url = "https://example.com/serving-endpoints"
    openai_client._client.auth = None
    workspace_client.serving_endpoints.get_open_ai_client.return_value = openai_client

    chat = ChatDatabricks(
        model="databricks-claude-sonnet-4", workspace_client=workspace_client
    )
    chat.close()
    # The provider-created OpenAI client is closed...
    openai_client.close.assert_called_once()
    # ...but the caller-supplied WorkspaceClient is not touched.
    workspace_client.close.assert_not_called()
