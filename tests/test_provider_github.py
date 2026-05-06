import httpx
import pytest
from chatlas import ChatGithub

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def chat_fun(**kwargs):
    return ChatGithub(model="gpt-4.1", **kwargs)


@pytest.mark.vcr
def test_github_simple_request():
    chat = chat_fun(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] == 27
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    assert turn.finish_reason == "stop"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_github_simple_streaming_request():
    chat = chat_fun(
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
def test_github_respects_turns_interface():
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_github_tool_variations():
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_github_tool_variations_async():
    await assert_tools_async(chat_fun)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(chat_fun)


@pytest.mark.vcr
def test_github_images():
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_github_logprobs():
    chat = chat_fun()

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


# Doesn't seem to be supported
# def test_github_pdf():
#    assert_pdf_local(ChatGithub)


def test_github_custom_http_client():
    ChatGithub(kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_github_list_models():
    assert_list_models(ChatGithub)
