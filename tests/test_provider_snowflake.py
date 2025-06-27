import pytest
from chatlas import ChatSnowflake

from .conftest import (
    assert_data_extraction,
    assert_tools_async,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)

CONNECTION = "posit"
MODEL = "claude-3-7-sonnet"


@pytest.mark.filterwarnings("ignore")
def test_openai_simple_request():
    chat = ChatSnowflake(
        connection_name=CONNECTION,
        model=MODEL,
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert chat.provider.name == "Snowflake"

    # No token / finish_reason info available?
    # assert turn.tokens is not None
    # assert len(turn.tokens) == 2
    # assert turn.tokens[0] == 27
    # # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    # assert turn.finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore")
async def test_openai_simple_streaming_request():
    chat = ChatSnowflake(
        connection_name=CONNECTION,
        model=MODEL,
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None

    # No token / finish_reason info available
    # assert turn.finish_reason == "stop"

@pytest.mark.filterwarnings("ignore")
def test_respects_turns_interface():
    def chat_fun(**kwargs):
        return ChatSnowflake(connection_name=CONNECTION, model=MODEL, **kwargs)

    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)

@pytest.mark.filterwarnings("ignore")
def test_tool_variations():
    def chat_fun(**kwargs):
        return ChatSnowflake(connection_name=CONNECTION, model=MODEL, **kwargs)

    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    # Seems parallel tools are not supported by Snowflake?
    # It does get it right with sequential calls, though.
    # assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore")
async def test_tool_variations_async():
    def chat_fun(**kwargs):
        return ChatSnowflake(connection_name=CONNECTION, model=MODEL, **kwargs)

    await assert_tools_async(chat_fun)

@pytest.mark.filterwarnings("ignore")
def test_data_extraction():
    def chat_fun():
        return ChatSnowflake(connection_name=CONNECTION, model=MODEL)

    assert_data_extraction(chat_fun)


# def test_images():
#     def chat_fun(**kwargs):
#         return ChatSnowflake(connection_name=CONNECTION, **kwargs)
#
#     assert_images_inline(chat_fun)
#     assert_images_remote(chat_fun)
#
#
# def test_pdf():
#     def chat_fun(**kwargs):
#         return ChatSnowflake(connection_name=CONNECTION, **kwargs)
#
#     assert_pdf_local(chat_fun)
