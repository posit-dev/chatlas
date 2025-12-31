import pytest
from chatlas import ChatMistral

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


@pytest.mark.vcr
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


@pytest.mark.vcr
def test_mistral_respects_turns_interface():
    chat_fun = ChatMistral
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


# Tool calling is poorly supported
# def test_mistral_tool_variations():
#    chat_fun = ChatMistral
#    assert_tools_simple(chat_fun)
#    assert_tools_simple_stream_content(chat_fun)

# Tool calling is poorly supported
# @pytest.mark.asyncio
# async def test_mistral_tool_variations_async():
#    await assert_tools_async(ChatMistral)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(ChatMistral)


# TODO: Mistral image support seems buggy
#@pytest.mark.vcr
#def test_mistral_images():
#    def chat_fun(**kwargs):
#        return ChatMistral(model="pixtral-12b-latest", **kwargs)
#
#    assert_images_inline(ChatMistral)
#    assert_images_remote(ChatMistral)


@pytest.mark.vcr
def test_mistral_model_list():
    assert_list_models(ChatMistral)
