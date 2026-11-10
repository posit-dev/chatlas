import os

import pytest
from chatlas import ChatCloudflare
from chatlas._content import ContentText
from chatlas._provider_cloudflare import CloudflareProvider
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from .conftest import assert_data_extraction, assert_turns_existing, assert_turns_system


def chat_fun(**kwargs):
    return ChatCloudflare(model="@cf/meta/llama-3.3-70b-instruct-fp8-fast", **kwargs)


@pytest.fixture(scope="module")
def live_cloudflare():
    # The VCR cassettes embed the account ID in the request path, so the
    # recorded interactions only replay for the account that recorded them;
    # gate the VCR tests on working credentials instead of replaying.
    try:
        chat_fun().chat("What is 1 + 1?")
    except Exception:
        pytest.skip("Cloudflare credentials aren't configured")


@pytest.mark.vcr
def test_cloudflare_simple_request(live_cloudflare):
    chat = chat_fun()
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] > 0  # input tokens
    assert turn.tokens[1] > 0  # output tokens
    assert turn.finish_reason == "success"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cloudflare_simple_streaming_request(live_cloudflare):
    chat = chat_fun()
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "success"


@pytest.mark.vcr
def test_cloudflare_respects_turns_interface(live_cloudflare):
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_cloudflare_data_extraction(live_cloudflare):
    assert_data_extraction(chat_fun)


def test_cloudflare_custom_model():
    chat = ChatCloudflare(
        account="test-account",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    )
    assert chat.provider.model == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def test_cloudflare_missing_account_id():
    original_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if "CLOUDFLARE_ACCOUNT_ID" in os.environ:
        del os.environ["CLOUDFLARE_ACCOUNT_ID"]

    try:
        with pytest.raises(ValueError, match="Cloudflare account ID is required"):
            ChatCloudflare()
    finally:
        if original_account_id is not None:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = original_account_id


def _numeric_chunk(content: object) -> ChatCompletionChunk:
    """A streamed chunk as the openai SDK delivers it: no validation, so a
    JSON number arrives as a raw `int`/`float`."""
    return ChatCompletionChunk.construct(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        choices=[
            ChunkChoice.construct(
                index=0,
                delta=ChoiceDelta.construct(content=content),
                finish_reason=None,
            )
        ],
    )


def test_cloudflare_stream_content_coerces_numeric_content():
    # Cloudflare streams numeric-only content as a JSON number (e.g.
    # `"content": 2`), which otherwise poisons the ContentText (even repr()
    # fails) and errors on display
    provider = CloudflareProvider(model="test")
    result = provider.stream_content(_numeric_chunk(2), None)
    assert result == [ContentText(text="2")]


def test_cloudflare_stream_merge_chunks_coerces_numeric_content():
    # Without coercion, merging str + int raises
    # "TypeError: can only concatenate str (not 'int') to str"
    provider = CloudflareProvider(model="test")
    merged = provider.stream_merge_chunks(None, _numeric_chunk("The answer is "))
    merged = provider.stream_merge_chunks(merged, _numeric_chunk(2))
    assert merged is not None
    assert merged["choices"][0]["delta"]["content"] == "The answer is 2"


def test_cloudflare_value_turn_coerces_numeric_content():
    # The non-streaming path fails ContentText validation on numeric content
    provider = CloudflareProvider(model="test")
    completion = ChatCompletion.construct(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        choices=[
            CompletionChoice.construct(
                index=0,
                message=ChatCompletionMessage.construct(
                    role="assistant", content=2
                ),
                finish_reason="stop",
            )
        ],
    )
    turn = provider.value_turn(completion, has_data_model=False)
    assert turn.contents == [ContentText(text="2")]


# Note: Tool calling and image tests are intentionally omitted
# since ellmer documentation indicates these don't work with Cloudflare
