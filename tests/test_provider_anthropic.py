import pytest
from chatlas import ChatAnthropic

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_pdf_local,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
    retry_api_call,
)


def test_anthropic_simple_request():
    chat = ChatAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5)
    assert turn.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = ChatAnthropic(
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


def test_anthropic_respects_turns_interface():
    chat_fun = ChatAnthropic
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@retry_api_call
def test_anthropic_tool_variations():
    chat_fun = ChatAnthropic
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@retry_api_call
def test_anthropic_tool_variations_parallel():
    # For some reason, at the time of writing, Claude 3.7 doesn't
    # respond with multiple tools at once for this test (but it does)
    # answer the question correctly with sequential tools.
    def chat_fun(**kwargs):
        return ChatAnthropic(model="claude-3-5-sonnet-latest", **kwargs)

    assert_tools_parallel(chat_fun)


@pytest.mark.asyncio
@retry_api_call
async def test_anthropic_tool_variations_async():
    await assert_tools_async(ChatAnthropic)


def test_data_extraction():
    assert_data_extraction(ChatAnthropic)


@retry_api_call
def test_anthropic_images():
    chat_fun = ChatAnthropic

    assert_images_inline(chat_fun)
    assert_images_remote_error(chat_fun)


def test_anthropic_pdfs():
    chat_fun = ChatAnthropic
    assert_pdf_local(chat_fun)
