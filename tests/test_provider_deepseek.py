import pytest

from ._test_providers import TestChatDeepSeek
from .conftest import (
    assert_list_models,
    assert_tools_async,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)


@pytest.mark.vcr
def test_deepseek_simple_request():
    chat = TestChatDeepSeek(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] >= 10  # More lenient assertion for DeepSeek
    assert turn.finish_reason == "stop"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_deepseek_simple_streaming_request():
    chat = TestChatDeepSeek(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


@pytest.mark.vcr
def test_deepseek_respects_turns_interface():
    assert_turns_system(TestChatDeepSeek)
    assert_turns_existing(TestChatDeepSeek)


@pytest.mark.vcr
def test_deepseek_tool_variations():
    assert_tools_simple(TestChatDeepSeek)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_deepseek_tool_variations_async():
    await assert_tools_async(TestChatDeepSeek)


# Doesn't seem to support data extraction or images


@pytest.mark.vcr
def test_deepseek_list_models():
    assert_list_models(TestChatDeepSeek)
