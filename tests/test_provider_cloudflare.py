import os

import pytest
from chatlas import ChatCloudflare

from .conftest import assert_data_extraction, assert_turns_existing, assert_turns_system

api_key = os.getenv("CLOUDFLARE_API_KEY")
account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
if api_key is None or account_id is None:
    pytest.skip(
        "CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID are not set; skipping tests",
        allow_module_level=True,
    )


def test_cloudflare_simple_request():
    chat = ChatCloudflare(
        system_prompt="Be as terse as possible; no punctuation",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] > 0  # input tokens
    assert turn.tokens[1] > 0  # output tokens
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_cloudflare_simple_streaming_request():
    chat = ChatCloudflare(
        system_prompt="Be as terse as possible; no punctuation",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


def test_cloudflare_respects_turns_interface():
    chat_fun = ChatCloudflare
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_cloudflare_data_extraction():
    def chat_fun(**kwargs):
        return ChatCloudflare(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast", **kwargs
        )

    assert_data_extraction(chat_fun)


def test_cloudflare_custom_model():
    chat = ChatCloudflare(model="@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    assert chat.provider.model == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def test_cloudflare_base_url():
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    chat = ChatCloudflare()
    expected_base_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    )
    assert str(chat.provider._client.base_url) == expected_base_url


def test_cloudflare_provider_name():
    chat = ChatCloudflare()
    assert chat.provider.name == "Cloudflare"


def test_cloudflare_missing_account_id():
    """Test that missing account ID raises appropriate error."""
    # Temporarily remove account ID from environment
    original_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if "CLOUDFLARE_ACCOUNT_ID" in os.environ:
        del os.environ["CLOUDFLARE_ACCOUNT_ID"]

    try:
        with pytest.raises(ValueError, match="Cloudflare account ID is required"):
            ChatCloudflare()
    finally:
        # Restore original environment
        if original_account_id is not None:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = original_account_id


# Note: Tool calling and image tests are intentionally omitted
# since ellmer documentation indicates these don't work with Cloudflare
