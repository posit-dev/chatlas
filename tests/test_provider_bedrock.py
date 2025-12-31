import os

import pytest

do_test = os.getenv("TEST_BEDROCK", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Bedrock tests", allow_module_level=True)

from ._test_providers import TestChatBedrockAnthropic
from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_list_models,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)


@pytest.mark.vcr
def test_anthropic_simple_request():
    chat = TestChatBedrockAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = TestChatBedrockAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    foo = await chat.stream_async("What is 1 + 1?")
    async for x in foo:
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "end_turn"


@pytest.mark.vcr
def test_anthropic_respects_turns_interface():
    assert_turns_system(TestChatBedrockAnthropic)
    assert_turns_existing(TestChatBedrockAnthropic)


@pytest.mark.vcr
def test_anthropic_tool_variations():
    assert_tools_simple(TestChatBedrockAnthropic)
    assert_tools_parallel(TestChatBedrockAnthropic)
    assert_tools_sequential(TestChatBedrockAnthropic, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_tool_variations_async():
    await assert_tools_async(TestChatBedrockAnthropic)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(TestChatBedrockAnthropic)


@pytest.mark.vcr
def test_anthropic_images():
    assert_images_inline(TestChatBedrockAnthropic)
    assert_images_remote_error(
        TestChatBedrockAnthropic, message="URL sources are not supported"
    )


@pytest.mark.vcr
def test_anthropic_models():
    assert_list_models(TestChatBedrockAnthropic)
