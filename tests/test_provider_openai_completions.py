import httpx
import pytest
from chatlas import ChatOpenAICompletions
from chatlas._content import ContentText, ContentThinking
from chatlas._provider_openai_completions import OpenAICompletionsProvider
from chatlas._turn import AssistantTurn

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
    assert turn.tokens[0] == 26
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


def test_stream_content_extracts_reasoning_content():
    provider = OpenAICompletionsProvider(model="test")

    class FakeDelta:
        def __init__(self, reasoning_content=None, content=None):
            self.reasoning_content = reasoning_content
            self.content = content

    class FakeChoice:
        def __init__(self, delta):
            self.delta = delta

    class FakeChunk:
        def __init__(self, choices):
            self.choices = choices

    chunk = FakeChunk([FakeChoice(FakeDelta(reasoning_content="think"))])
    result = provider.stream_content(chunk)
    assert isinstance(result, ContentThinking)
    assert result.thinking == "think"

    chunk = FakeChunk([FakeChoice(FakeDelta(content="hello"))])
    result = provider.stream_content(chunk)
    assert isinstance(result, ContentText)
    assert result.text == "hello"


def test_response_as_turn_extracts_reasoning_content():
    from unittest.mock import Mock

    completion = Mock()
    message = Mock()
    message.reasoning_content = "Let me think..."
    message.content = "The answer is 42."
    message.tool_calls = None
    completion.choices = [Mock(message=message, finish_reason="stop")]

    turn = OpenAICompletionsProvider._response_as_turn(completion, has_data_model=False)
    assert len(turn.contents) == 2
    assert isinstance(turn.contents[0], ContentThinking)
    assert turn.contents[0].thinking == "Let me think..."
    assert isinstance(turn.contents[1], ContentText)
    assert turn.contents[1].text == "The answer is 42."


def test_turns_as_inputs_drops_thinking_by_default():
    provider = OpenAICompletionsProvider(model="test")

    turn = AssistantTurn(
        [
            ContentThinking(thinking="Let me think..."),
            ContentText(text="The answer is 42."),
        ]
    )
    result = provider._turns_as_inputs([turn])
    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert "reasoning_content" not in msg
    assert msg["content"] == [{"type": "text", "text": "The answer is 42."}]


def test_turns_as_inputs_preserves_thinking_when_enabled():
    provider = OpenAICompletionsProvider(model="test", preserve_thinking=True)

    turn = AssistantTurn(
        [
            ContentThinking(thinking="Let me think..."),
            ContentText(text="The answer is 42."),
        ]
    )
    result = provider._turns_as_inputs([turn])
    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "Let me think..."
    assert msg["content"] == [{"type": "text", "text": "The answer is 42."}]
