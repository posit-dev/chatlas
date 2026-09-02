import asyncio
import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest
from chatlas import ChatDatabricks
from chatlas._content import ContentText, ContentThinking, ContentThinkingDelta
from chatlas._provider_databricks import DatabricksProvider
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_tools_async,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
    make_vcr_config,
)


def _databricks_provider() -> DatabricksProvider:
    """A DatabricksProvider that never touches the network."""
    sync_client = OpenAI(
        api_key="no-token",
        base_url="https://workspace.example.com/serving-endpoints",
        http_client=cast(Any, httpx.Client()),
    )
    workspace_client = SimpleNamespace(
        serving_endpoints=SimpleNamespace(get_open_ai_client=lambda: sync_client)
    )
    return DatabricksProvider(
        model="test", workspace_client=cast(Any, workspace_client)
    )


# Override VCR config to ignore host - Databricks host varies by environment
# but cassettes were recorded with a specific host
@pytest.fixture(scope="module")
def vcr_config():
    config = make_vcr_config()
    # Remove "host" from match_on since Databricks host varies by environment
    config["match_on"] = ["method", "scheme", "port", "path", "body"]
    return config


def chat_fun(**kwargs):
    return ChatDatabricks(model="databricks-claude-3-7-sonnet", **kwargs)


@pytest.mark.vcr
def test_databricks_simple_request():
    chat = chat_fun(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] == 26
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    assert turn.finish_reason == "success"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_databricks_simple_streaming_request():
    chat = chat_fun(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "success"


@pytest.mark.vcr
def test_databricks_respects_turns_interface():
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_databricks_empty_response():
    chat = chat_fun()
    chat.chat("Respond with only two blank lines")
    resp = chat.chat("What's 1+1? Just give me the number")
    assert "2" == str(resp).strip()


@pytest.mark.vcr
def test_databricks_tool_variations():
    assert_tools_simple(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_databricks_tool_variations_async():
    await assert_tools_async(chat_fun)


@pytest.mark.vcr
def test_databricks_data_extraction():
    assert_data_extraction(chat_fun)


@pytest.mark.vcr
def test_databricks_images():
    assert_images_inline(chat_fun)
    # Remote images don't seem to be supported yet
    # assert_images_remote(chat_fun)


# PDF doesn't seem to be supported yet
#
# def test_databricks_pdf():
#     chat_fun = ChatDatabricks
#     assert_pdf_local(chat_fun)


def test_connect_without_openai_key(monkeypatch):
    # Ensure OPENAI_API_KEY is not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # This should not raise an error
    chat = ChatDatabricks()
    assert chat is not None


def test_databricks_async_client_preserves_legacy_httpx_auth():
    auth = httpx.BasicAuth("user", "password")
    sync_client = OpenAI(
        api_key="no-token",
        base_url="https://workspace.example.com/serving-endpoints",
        http_client=cast(
            Any,
            httpx.Client(auth=auth),
        ),
    )
    workspace_client = SimpleNamespace(
        serving_endpoints=SimpleNamespace(
            get_open_ai_client=lambda: sync_client,
        )
    )

    provider = DatabricksProvider(
        model="test",
        workspace_client=cast(Any, workspace_client),
    )

    try:
        assert isinstance(provider._async_client._client, httpx.AsyncClient)
        assert provider._async_client._client.auth is auth
    finally:
        provider._client.close()
        asyncio.run(provider._async_client.close())


# GPT-OSS endpoints can return message.content / delta.content as a list of
# typed parts instead of a plain string (#392); these exercise the
# Databricks-only normalization layer, both non-streaming and streaming.


def test_response_as_turn_normalizes_typed_content_array():
    provider = _databricks_provider()
    try:
        message = Mock(
            reasoning=None,
            reasoning_content=None,
            content=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                },
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
            tool_calls=None,
        )
        completion = Mock(choices=[Mock(message=message, finish_reason="stop")])

        turn = provider._response_as_turn(completion, has_data_model=False)
        assert len(turn.contents) == 2
        assert isinstance(turn.contents[0], ContentThinking)
        assert turn.contents[0].thinking == "thinking"
        assert isinstance(turn.contents[1], ContentText)
        assert turn.contents[1].text == "Hello world"
    finally:
        provider._client.close()


def test_response_as_turn_normalizes_typed_content_array_text_only():
    provider = _databricks_provider()
    try:
        message = Mock(
            reasoning=None,
            reasoning_content=None,
            content=[{"type": "text", "text": "just an answer"}],
            tool_calls=None,
        )
        completion = Mock(choices=[Mock(message=message, finish_reason="stop")])

        turn = provider._response_as_turn(completion, has_data_model=False)
        assert len(turn.contents) == 1
        assert isinstance(turn.contents[0], ContentText)
        assert turn.contents[0].text == "just an answer"
    finally:
        provider._client.close()


def test_response_as_turn_preserves_plain_string_content():
    """Non-array content (the common case) is untouched by the normalization."""
    provider = _databricks_provider()
    try:
        message = Mock(
            reasoning=None,
            reasoning_content=None,
            content="a plain string reply",
            tool_calls=None,
        )
        completion = Mock(choices=[Mock(message=message, finish_reason="stop")])

        turn = provider._response_as_turn(completion, has_data_model=False)
        assert len(turn.contents) == 1
        assert isinstance(turn.contents[0], ContentText)
        assert turn.contents[0].text == "a plain string reply"
    finally:
        provider._client.close()


def test_stream_content_normalizes_typed_content_array():
    provider = _databricks_provider()
    try:
        delta = Mock(
            reasoning=None,
            reasoning_content=None,
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        )
        chunk = Mock(choices=[Mock(delta=delta)])

        result = provider.stream_content(chunk, None)
        assert result == [ContentText(text="Hello world")]
    finally:
        provider._client.close()


def test_stream_content_normalizes_typed_reasoning_array():
    provider = _databricks_provider()
    try:
        delta = Mock(
            reasoning=None,
            reasoning_content=None,
            content=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "hmm"}],
                }
            ],
        )
        chunk = Mock(choices=[Mock(delta=delta)])

        result = provider.stream_content(chunk, None)
        assert result == [ContentThinkingDelta(thinking="hmm")]
    finally:
        provider._client.close()


def test_stream_content_preserves_plain_string_content():
    """Non-array content (the common case) is untouched by the normalization."""
    provider = _databricks_provider()
    try:
        delta = Mock(reasoning=None, reasoning_content=None, content="partial text")
        chunk = Mock(choices=[Mock(delta=delta)])

        result = provider.stream_content(chunk, None)
        assert result == [ContentText(text="partial text")]
    finally:
        provider._client.close()


def _stream_chunk(content: Any) -> ChatCompletionChunk:
    """A chunk shaped like the ones a real stream carries, with choices[].index."""
    return ChatCompletionChunk.model_construct(
        id="c",
        model="gpt-oss",
        object="chat.completion.chunk",
        created=0,
        choices=[
            ChunkChoice.model_construct(
                index=0,
                delta=ChoiceDelta.model_construct(role="assistant", content=content),
                finish_reason=None,
            )
        ],
    )


def test_stream_merge_chunks_normalizes_typed_content_array():
    """The accumulated completion holds text, not the typed part array (#409)."""
    provider = _databricks_provider()
    try:
        chunks = [
            _stream_chunk(
                [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "thinking"}],
                    }
                ]
            ),
            _stream_chunk("Hello"),
            _stream_chunk(" world"),
        ]

        result = None
        streamed = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for chunk in chunks:
                result = provider.stream_merge_chunks(result, chunk)
                streamed.extend(provider.stream_content(chunk, result))

        delta = result["choices"][0]["delta"]
        assert delta["content"] == "Hello world"
        assert delta["reasoning"] == "thinking"
        assert [
            w
            for w in caught
            if "PydanticSerializationUnexpectedValue" in str(w.message)
        ] == []
        assert streamed == [
            ContentThinkingDelta(thinking="thinking"),
            ContentText(text="Hello"),
            ContentText(text=" world"),
        ]

        turn = provider.stream_turn(result, has_data_model=False)
        assert turn.text == "Hello world"
    finally:
        provider._client.close()


def test_stream_merge_chunks_preserves_plain_string_content():
    """Non-array content (the common case) is untouched by the normalization."""
    provider = _databricks_provider()
    try:
        result = None
        for chunk in [_stream_chunk("Hello"), _stream_chunk(" world")]:
            result = provider.stream_merge_chunks(result, chunk)

        assert result["choices"][0]["delta"]["content"] == "Hello world"
    finally:
        provider._client.close()
