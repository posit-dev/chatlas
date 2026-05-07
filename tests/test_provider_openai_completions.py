import httpx
import pytest
from chatlas import ChatOpenAICompletions
from chatlas._content import (
    ContentText,
    ContentThinking,
    ContentToolRequest,
    ContentToolResult,
)
from chatlas._provider_openai_completions import OpenAICompletionsProvider
from chatlas._turn import AssistantTurn, UserTurn
from openai.types.chat import ChatCompletion

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


def test_tool_results_ordering():
    """Ensure tool results precede user text in _turns_as_inputs."""
    req = ContentToolRequest(id="call_123", name="my_tool", arguments={})

    # Simulate a user turn containing both a tool result and new user text
    turn = UserTurn(
        [
            ContentToolResult(value="tool output", request=req),
            ContentText(text="Here is some extra user text"),
        ]
    )

    inputs = OpenAICompletionsProvider._turns_as_inputs([turn])

    # Must generate 2 distinct messages, and the tool message must come first
    assert len(inputs) == 2
    assert inputs[0]["role"] == "tool"
    assert inputs[0]["tool_call_id"] == "call_123"
    assert inputs[1]["role"] == "user"
    assert inputs[1]["content"][0]["text"] == "Here is some extra user text"


def test_reasoning_content_serialization():
    """Ensure ContentThinking is serialized to reasoning_content."""
    # Simulate an Assistant turn containing both thinking and final text
    turn = AssistantTurn(
        [ContentThinking(thinking="Let me think..."), ContentText(text="Final answer")]
    )

    inputs = OpenAICompletionsProvider._turns_as_inputs([turn])

    assert len(inputs) == 1
    assert inputs[0]["role"] == "assistant"
    assert inputs[0].get("reasoning_content") == "Let me think..."
    assert inputs[0]["content"][0]["text"] == "Final answer"


def test_reasoning_content_deserialization():
    """Ensure reasoning_content from OpenAI completion is parsed into ContentThinking."""
    # Simulate a raw dictionary returned from the API
    mock_data = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "Thinking process",
                },
            }
        ],
        "created": 123456,
        "model": "deepseek-chat",
        "object": "chat.completion",
    }

    mock_completion = ChatCompletion.model_validate(mock_data)

    # Parse using our updated method
    turn = OpenAICompletionsProvider._response_as_turn(
        mock_completion, has_data_model=False
    )

    # Verify ContentThinking is successfully parsed and precedes ContentText
    assert len(turn.contents) == 2
    assert isinstance(turn.contents[0], ContentThinking)
    assert turn.contents[0].thinking == "Thinking process"
    assert isinstance(turn.contents[1], ContentText)
    assert turn.contents[1].text == "Final answer"
