import os

import pytest
from chatlas import ChatMistral

do_test = os.getenv("TEST_MISTRAL", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Mistral tests", allow_module_level=True)

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_tools_async,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def test_mistral_simple_request():
    chat = ChatMistral(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] > 0  # prompt tokens
    assert turn.tokens[1] > 0  # completion tokens
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_mistral_simple_streaming_request():
    chat = ChatMistral(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


def test_mistral_respects_turns_interface():
    chat_fun = ChatMistral
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_mistral_tool_variations():
    """Note: Tool calling may be unstable with Mistral."""
    chat_fun = ChatMistral
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)


@pytest.mark.asyncio
async def test_mistral_tool_variations_async():
    """Note: Tool calling may be unstable with Mistral."""
    await assert_tools_async(ChatMistral)


def test_data_extraction():
    assert_data_extraction(ChatMistral)


def test_mistral_images():
    """Note: Images require a model that supports vision."""
    chat_fun = lambda **kwargs: ChatMistral(model="pixtral-12b-latest", **kwargs)
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)
