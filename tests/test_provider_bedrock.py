import pytest
from chatlas import ChatBedrockAnthropic

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

try:
    chat = ChatBedrockAnthropic()
    chat.chat("What is 1 + 1?")
except Exception:
    pytest.skip("Bedrock credentials aren't configured", allow_module_level=True)


@pytest.mark.vcr
def test_anthropic_simple_request():
    chat = ChatBedrockAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = ChatBedrockAnthropic(
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
    assert_turns_system(ChatBedrockAnthropic)
    assert_turns_existing(ChatBedrockAnthropic)


@pytest.mark.vcr
def test_anthropic_tool_variations():
    assert_tools_simple(ChatBedrockAnthropic)
    assert_tools_parallel(ChatBedrockAnthropic)
    assert_tools_sequential(ChatBedrockAnthropic, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_tool_variations_async():
    await assert_tools_async(ChatBedrockAnthropic)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(ChatBedrockAnthropic)


@pytest.mark.vcr
def test_anthropic_images():
    assert_images_inline(ChatBedrockAnthropic)
    assert_images_remote_error(
        ChatBedrockAnthropic, message="URL sources are not supported"
    )


@pytest.mark.vcr
def test_anthropic_models():
    assert_list_models(ChatBedrockAnthropic)
