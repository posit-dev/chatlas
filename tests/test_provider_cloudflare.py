import os

import pytest
from chatlas import ChatCloudflare

from ._test_providers import TestChatCloudflare
from .conftest import assert_data_extraction, assert_turns_existing, assert_turns_system


@pytest.mark.vcr
def test_cloudflare_simple_request():
    chat = TestChatCloudflare(
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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cloudflare_simple_streaming_request():
    chat = TestChatCloudflare(
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


@pytest.mark.vcr
def test_cloudflare_respects_turns_interface():
    assert_turns_system(TestChatCloudflare)
    assert_turns_existing(TestChatCloudflare)


@pytest.mark.vcr
def test_cloudflare_data_extraction():
    def chat_fun(**kwargs):
        return TestChatCloudflare(
            model="@cf/meta/llama-3.3-70b-instruct-fp8-fast", **kwargs
        )

    assert_data_extraction(chat_fun)


def test_cloudflare_custom_model():
    # This test doesn't use VCR, so use explicit dummy credentials
    chat = ChatCloudflare(
        api_key="test",
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


# Note: Tool calling and image tests are intentionally omitted
# since ellmer documentation indicates these don't work with Cloudflare
