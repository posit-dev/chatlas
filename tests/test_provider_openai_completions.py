import httpx
import pytest
from chatlas import ChatOpenAICompletions

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_pdf_local,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


@pytest.mark.vcr
def test_openai_simple_request():
    chat = ChatOpenAICompletions(
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
async def test_openai_simple_streaming_request():
    chat = ChatOpenAICompletions(
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
def test_openai_respects_turns_interface():
    assert_turns_system(ChatOpenAICompletions)
    assert_turns_existing(ChatOpenAICompletions)


@pytest.mark.vcr
def test_openai_tool_variations():
    assert_tools_simple(ChatOpenAICompletions)
    assert_tools_simple_stream_content(ChatOpenAICompletions)
    assert_tools_parallel(ChatOpenAICompletions)
    assert_tools_sequential(ChatOpenAICompletions, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    await assert_tools_async(ChatOpenAICompletions)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(ChatOpenAICompletions)


@pytest.mark.vcr
def test_openai_images():
    assert_images_inline(ChatOpenAICompletions)
    assert_images_remote(ChatOpenAICompletions)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = ChatOpenAICompletions()
    chat.set_model_params(log_probs=True)

    pieces = []
    async for x in await chat.stream_async("Hi"):
        pieces.append(x)

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.completion is not None
    assert turn.completion.choices[0].logprobs is not None
    logprobs = turn.completion.choices[0].logprobs.content
    assert logprobs is not None
    assert len(logprobs) == len(pieces)


@pytest.mark.vcr
def test_openai_pdf():
    assert_pdf_local(ChatOpenAICompletions)


def test_openai_custom_http_client():
    ChatOpenAICompletions(kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_openai_list_models():
    assert_list_models(ChatOpenAICompletions)
