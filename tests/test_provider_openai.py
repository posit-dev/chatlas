import httpx
import pytest
from chatlas import ChatOpenAI

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_pdf_local,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def test_openai_simple_request():
    chat = ChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 2
    assert turn.tokens[0] == 27
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_simple_streaming_request():
    chat = ChatOpenAI(
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
    chat_fun = ChatOpenAI
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_openai_tool_variations():
    chat_fun = ChatOpenAI
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    await assert_tools_async(ChatOpenAI)


def test_data_extraction():
    assert_data_extraction(ChatOpenAI)


def test_openai_images():
    chat_fun = ChatOpenAI
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = ChatOpenAI()

    pieces = []
    async for x in await chat.stream_async("Hi", kwargs={"logprobs": True}):
        pieces.append(x)

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.completion is not None
    assert turn.completion.choices[0].logprobs is not None
    logprobs = turn.completion.choices[0].logprobs.content
    assert logprobs is not None
    assert len(logprobs) == len(pieces)


def test_openai_pdf():
    chat_fun = ChatOpenAI
    assert_pdf_local(chat_fun)


def test_openai_custom_http_client():
    ChatOpenAI(kwargs={"http_client": httpx.AsyncClient()})
