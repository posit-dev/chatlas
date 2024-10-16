import pytest
from chatlas import ChatGoogle

from .conftest import (
    assert_images_inline,
    assert_images_remote_error,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
    retryassert,
)


@pytest.mark.filterwarnings("ignore:Defaulting to")
def test_google_simple_request():
    chat = ChatGoogle(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.last_turn()
    assert turn is not None
    assert turn.tokens == (17, 1)


@pytest.mark.filterwarnings("ignore:Defaulting to")
@pytest.mark.asyncio
async def test_google_simple_streaming_request():
    chat = ChatGoogle(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in chat.submit_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)


@pytest.mark.filterwarnings("ignore:Defaulting to")
def test_google_respects_turns_interface():
    chat_fun = ChatGoogle
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.filterwarnings("ignore:Defaulting to")
def test_google_tool_variations():
    chat_fun = ChatGoogle
    assert_tools_simple(chat_fun, stream=False)
    assert_tools_parallel(chat_fun, stream=False)

    # <10% of the time, it uses only 6 calls, suggesting that it's made a poor
    # choice. Running it twice (i.e. retrying 1) should reduce failure rate to <1%
    def run_sequentialassert():
        assert_tools_sequential(chat_fun, total_calls=8, stream=False)

    retryassert(run_sequentialassert)


@pytest.mark.filterwarnings("ignore:Defaulting to")
@pytest.mark.asyncio
async def test_google_tool_variations_async():
    await assert_tools_async(ChatGoogle, stream=False)


@pytest.mark.filterwarnings("ignore:Defaulting to")
def test_google_images():
    chat_fun = ChatGoogle
    assert_images_inline(chat_fun)
    assert_images_remote_error(chat_fun)
