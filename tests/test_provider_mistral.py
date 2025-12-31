import pytest

from ._test_providers import TestChatMistral
from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_turns_existing,
    assert_turns_system,
)


@pytest.mark.vcr
def test_mistral_simple_request():
    chat = TestChatMistral(
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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistral_simple_streaming_request():
    chat = TestChatMistral(
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
def test_mistral_respects_turns_interface():
    assert_turns_system(TestChatMistral)
    assert_turns_existing(TestChatMistral)


# Tool calling is poorly supported
# def test_mistral_tool_variations():
#    chat_fun = TestChatMistral
#    assert_tools_simple(chat_fun)
#    assert_tools_simple_stream_content(chat_fun)

# Tool calling is poorly supported
# @pytest.mark.asyncio
# async def test_mistral_tool_variations_async():
#    await assert_tools_async(TestChatMistral)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(TestChatMistral)


@pytest.mark.vcr
def test_mistral_images():
    assert_images_inline(TestChatMistral)
    assert_images_remote(TestChatMistral)


@pytest.mark.vcr
def test_mistral_model_list():
    assert_list_models(TestChatMistral)
