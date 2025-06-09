import pytest
from chatlas import ChatDatabricks

from .conftest import assert_turns_existing, assert_turns_system


def test_openai_simple_request():
    chat = ChatDatabricks(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 2
    assert turn.tokens[0] == 26
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_simple_streaming_request():
    chat = ChatDatabricks(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


def test_openai_respects_turns_interface():
    chat_fun = ChatDatabricks
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_anthropic_empty_response():
    chat = ChatDatabricks()
    chat.chat("Respond with only two blank lines")
    resp = chat.chat("What's 1+1? Just give me the number")
    assert "2" == str(resp).strip()


# Note: Databricks models cannot yet handle "continuing past the first tool
# call", which causes issues with how ellmer implements tool calling. Nor do
# they support parallel tool calls.
#
# See: https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html#limitations
# def test_openai_tool_variations():
#     chat_fun = ChatDatabricks
#     assert_tools_simple(chat_fun)
#     assert_tools_simple_stream_content(chat_fun)
#     assert_tools_parallel(chat_fun)
#     assert_tools_sequential(chat_fun, total_calls=6)


# @pytest.mark.asyncio
# async def test_openai_tool_variations_async():
#    await assert_tools_async(ChatDatabricks)

# I think this is only broken for Anthropic models, but I also
# don't know if I have access to non-Anthropic models on Databricks
# at this point for testing.
# def test_data_extraction():
#    assert_data_extraction(ChatDatabricks)


# Images don't seem to be supported yet
#
# def test_openai_images():
#     chat_fun = ChatDatabricks
#     assert_images_inline(chat_fun)
#     assert_images_remote(chat_fun)


# PDF doesn't seem to be supported yet
#
# def test_openai_pdf():
#     chat_fun = ChatDatabricks
#     assert_pdf_local(chat_fun)
