import httpx
import pytest
from chatlas import ChatOpenAI
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from ._test_providers import TestChatOpenAI
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
    chat = TestChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] == 27
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_simple_streaming_request():
    chat = TestChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None


@pytest.mark.vcr
def test_openai_respects_turns_interface():
    assert_turns_system(TestChatOpenAI)
    assert_turns_existing(TestChatOpenAI)


@pytest.mark.vcr
def test_openai_tool_variations():
    assert_tools_simple(TestChatOpenAI)
    assert_tools_simple_stream_content(TestChatOpenAI)
    assert_tools_parallel(TestChatOpenAI)
    assert_tools_sequential(TestChatOpenAI, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    await assert_tools_async(TestChatOpenAI)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(TestChatOpenAI)


@pytest.mark.vcr
def test_openai_images():
    assert_images_inline(TestChatOpenAI)
    assert_images_remote(TestChatOpenAI)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = TestChatOpenAI()
    chat.set_model_params(log_probs=True)

    pieces = []
    async for x in await chat.stream_async("Hi"):
        pieces.append(x)

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.completion is not None
    output = turn.completion.output[0]
    assert isinstance(output, ResponseOutputMessage)
    content = output.content[0]
    assert isinstance(content, ResponseOutputText)
    logprobs = content.logprobs
    assert logprobs is not None
    assert len(logprobs) == len(pieces)


@pytest.mark.vcr
def test_openai_pdf():
    assert_pdf_local(TestChatOpenAI)


def test_openai_custom_http_client():
    # This test doesn't use VCR, so use the real ChatOpenAI with explicit key
    ChatOpenAI(api_key="test", kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_openai_list_models():
    assert_list_models(TestChatOpenAI)


def test_openai_service_tier():
    # This test doesn't use VCR, so use the real ChatOpenAI with explicit key
    chat = ChatOpenAI(api_key="test", service_tier="flex")
    assert chat.kwargs_chat.get("service_tier") == "flex"


@pytest.mark.vcr
def test_openai_service_tier_affects_pricing():
    from chatlas._tokens import get_token_cost

    chat = TestChatOpenAI(service_tier="priority")
    chat.chat("What is 1+1?")

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert turn.cost is not None

    # Verify that cost was calculated using priority pricing
    tokens = turn.tokens
    priority_cost = get_token_cost("OpenAI", chat.provider.model, tokens, "priority")
    assert priority_cost is not None
    assert turn.cost == priority_cost

    # Verify priority pricing is more expensive than default
    default_cost = get_token_cost("OpenAI", chat.provider.model, tokens, "")
    assert default_cost is not None
    assert turn.cost > default_cost