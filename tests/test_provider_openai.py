import httpx
import pytest
from chatlas import ChatOpenAI
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

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
    chat = ChatOpenAI(
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
    chat = ChatOpenAI(
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
    chat_fun = ChatOpenAI
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_openai_tool_variations():
    chat_fun = ChatOpenAI
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    await assert_tools_async(ChatOpenAI)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(ChatOpenAI)


@pytest.mark.vcr
def test_openai_images():
    chat_fun = ChatOpenAI
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = ChatOpenAI()
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
    assert_pdf_local(ChatOpenAI)


def test_openai_custom_http_client():
    ChatOpenAI(kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_openai_list_models():
    assert_list_models(ChatOpenAI)


def test_openai_service_tier():
    chat = ChatOpenAI(service_tier="flex")
    assert chat.kwargs_chat.get("service_tier") == "flex"


@pytest.mark.vcr
def test_openai_service_tier_affects_pricing():
    from chatlas._tokens import get_token_cost

    chat = ChatOpenAI(service_tier="priority")
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