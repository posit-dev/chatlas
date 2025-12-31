import os

import pytest

do_test = os.getenv("TEST_PORTKEY", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Portkey tests", allow_module_level=True)

# Skip if PORTKEY_API_KEY is not set or empty
api_key = os.getenv("PORTKEY_API_KEY", "")
if not api_key:
    pytest.skip("PORTKEY_API_KEY is not set; skipping tests", allow_module_level=True)

from chatlas import ChatPortkey

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def _chat_portkey_test(**kwargs):
    model = kwargs.pop("model", "@openai/gpt-4o-mini")
    system_prompt = kwargs.pop(
        "system_prompt", "Be as terse as possible; no punctuation"
    )
    return ChatPortkey(model=model, system_prompt=system_prompt, **kwargs)


def test_portkey_simple_request():
    chat = _chat_portkey_test()
    response = chat.chat("What is 1 + 1?")
    assert "2" in str(response)

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) >= 2  # input and output tokens
    assert all(token > 0 for token in turn.tokens[:2])


@pytest.mark.asyncio
async def test_portkey_simple_streaming_request():
    chat = _chat_portkey_test()
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)


def test_portkey_respects_turns_interface():
    def chat_fun(**kwargs):
        return _chat_portkey_test(**kwargs)

    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_portkey_tool_variations():
    def chat_fun(**kwargs):
        return _chat_portkey_test(**kwargs)

    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.asyncio
async def test_portkey_tool_variations_async():
    def chat_fun(**kwargs):
        return _chat_portkey_test(**kwargs)

    await assert_tools_async(chat_fun)


def test_portkey_data_extraction():
    def chat_fun(**kwargs):
        return _chat_portkey_test(**kwargs)

    assert_data_extraction(chat_fun)


def test_portkey_images():
    def chat_fun(**kwargs):
        return _chat_portkey_test(**kwargs)

    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)
