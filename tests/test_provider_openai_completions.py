import base64

import httpx
import pytest
from chatlas import ChatOpenAICompletions
from chatlas._content import (
    ContentDocument,
    ContentImageInline,
    ContentPDF,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentToolRequest,
    ContentUploaded,
)
from chatlas._provider_openai_completions import OpenAICompletionsProvider
from chatlas._provider_openai_completions import (
    normalize_finish_reason as completions_normalize_finish_reason,
)
from chatlas._turn import AssistantTurn, UserTurn
from openai.types.chat import ChatCompletion

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_pdf_local,
    assert_pdf_remote,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def test_normalize_finish_reason_maps_known_reasons():
    assert completions_normalize_finish_reason("stop") == "success"
    assert completions_normalize_finish_reason("length") == "max_tokens"
    assert completions_normalize_finish_reason("content_filter") == "content_filter"
    assert completions_normalize_finish_reason("tool_calls") == "tool_use"


def test_normalize_finish_reason_passes_through_unknown():
    assert completions_normalize_finish_reason("function_call") == "function_call"


def test_normalize_finish_reason_handles_none():
    assert completions_normalize_finish_reason(None) is None


@pytest.mark.vcr
def test_openai_simple_request():
    chat = ChatOpenAICompletions(
        model="gpt-5.4",
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] == 26
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    assert turn.finish_reason == "success"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_simple_streaming_request():
    chat = ChatOpenAICompletions(
        model="gpt-5.4",
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "success"


@pytest.mark.vcr
def test_openai_respects_turns_interface():
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_openai_tool_variations():
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    await assert_tools_async(chat_fun)


@pytest.mark.vcr
def test_data_extraction():
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    assert_data_extraction(chat_fun)


@pytest.mark.vcr
def test_openai_images():
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = ChatOpenAICompletions(model="gpt-5.4")
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
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    assert_pdf_local(chat_fun)


# No document counterpart here: OpenAI's own endpoint 400s on any non-PDF
# `file_data`, and the compatible backends that may accept one aren't
# recordable. `test_completions_document_passes_mime_type_through` covers the
# serialization instead.
@pytest.mark.vcr
def test_openai_completions_pdf_url():
    def chat_fun(**kwargs):
        return ChatOpenAICompletions(model="gpt-5.4", **kwargs)

    assert_pdf_remote(chat_fun)


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
    result = provider.stream_content(chunk, None)
    assert result == [ContentThinkingDelta(thinking="think")]

    chunk = FakeChunk([FakeChoice(FakeDelta(content="hello"))])
    result = provider.stream_content(chunk, None)
    assert result == [ContentText(text="hello")]


def test_response_as_turn_extracts_reasoning_content():
    from unittest.mock import Mock

    completion = Mock()
    message = Mock()
    message.reasoning = None
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


def test_stream_content_extracts_reasoning_field():
    """Ollama (e.g. qwen3) returns thinking in a `reasoning` field (#981)."""
    provider = OpenAICompletionsProvider(model="test")

    class FakeDelta:
        def __init__(self, reasoning=None, reasoning_content=None, content=None):
            self.reasoning = reasoning
            self.reasoning_content = reasoning_content
            self.content = content

    class FakeChoice:
        def __init__(self, delta):
            self.delta = delta

    class FakeChunk:
        def __init__(self, choices):
            self.choices = choices

    chunk = FakeChunk([FakeChoice(FakeDelta(reasoning="think"))])
    result = provider.stream_content(chunk, None)
    assert result == [ContentThinkingDelta(thinking="think")]


def test_response_as_turn_extracts_reasoning_field():
    """Ollama (e.g. qwen3) returns thinking in a `reasoning` field (#981)."""
    from unittest.mock import Mock

    completion = Mock()
    message = Mock()
    message.reasoning = "Let me think..."
    message.reasoning_content = None
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


def test_response_as_turn_treats_empty_content_as_none():
    """Databricks returns content: '' for tool-only turns (#932 in ellmer)."""
    from unittest.mock import Mock

    mock_func = Mock(arguments="{}")
    mock_func.name = "fn"

    completion = Mock()
    message = Mock()
    message.reasoning = None
    message.reasoning_content = None
    message.content = ""
    message.tool_calls = [Mock(type="function", id="call_1", function=mock_func)]
    completion.choices = [Mock(message=message, finish_reason="stop")]

    turn = OpenAICompletionsProvider._response_as_turn(completion, has_data_model=False)
    assert not any(isinstance(c, ContentText) for c in turn.contents)
    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentToolRequest)


def test_turns_as_inputs_drops_empty_content_text():
    """Empty ContentText (via model_construct) should be filtered during serialization."""
    provider = OpenAICompletionsProvider(model="test")

    # model_construct bypasses __init__, so text stays as ""
    turn = AssistantTurn([ContentText.model_construct(text="")])
    result = provider._turns_as_inputs([turn])
    assert len(result) == 1
    assert "content" not in result[0]

    turn = AssistantTurn(
        [ContentText.model_construct(text=""), ContentText(text="Hello")]
    )
    result = provider._turns_as_inputs([turn])
    assert result[0]["content"] == [{"type": "text", "text": "Hello"}]


def test_completions_uploaded_document_serializes():
    provider = OpenAICompletionsProvider(model="gpt-4o")
    turn = UserTurn(
        [ContentUploaded(id="file_1", mime_type="application/pdf", provider="openai")]
    )
    msgs = provider._turns_as_inputs([turn])
    part = msgs[-1]["content"][0]
    assert part["type"] == "file"
    assert part["file"]["file_id"] == "file_1"


def test_completions_uploaded_image_raises():
    provider = OpenAICompletionsProvider(model="gpt-4o")
    turn = UserTurn(
        [ContentUploaded(id="img_1", mime_type="image/png", provider="openai")]
    )
    with pytest.raises(ValueError, match="Chat Completions API"):
        provider._turns_as_inputs([turn])


def test_completions_uploaded_wrong_provider_raises():
    provider = OpenAICompletionsProvider(model="gpt-4o")
    turn = UserTurn(
        [ContentUploaded(id="x", mime_type="application/pdf", provider="anthropic")]
    )
    with pytest.raises(ValueError, match="uploaded to provider 'anthropic'"):
        provider._turns_as_inputs([turn])


def test_completions_pdf_downloads_bytes_when_only_url_set(monkeypatch):
    raw = b"%PDF-1.4"
    monkeypatch.setattr("chatlas._content_file.download_bytes", lambda url: raw)

    provider = OpenAICompletionsProvider(model="gpt-4o")
    turn = UserTurn([ContentPDF(filename="a.pdf", url="https://example.com/a.pdf")])
    msgs = provider._turns_as_inputs([turn])
    part = msgs[-1]["content"][0]
    assert part["file"]["file_data"] == (
        f"data:application/pdf;base64,{base64.b64encode(raw).decode('utf-8')}"
    )


@pytest.mark.parametrize(
    "mime_type",
    [
        "text/plain",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
)
def test_completions_document_passes_mime_type_through(mime_type):
    """The document's own MIME type goes out untouched.

    OpenAI's endpoint only accepts `application/pdf` here and 400s on the rest,
    but this provider backs a dozen OpenAI-compatible services with differing
    file support, so the decision belongs to whichever backend is configured.
    """
    provider = OpenAICompletionsProvider(model="gpt-4o")
    turn = UserTurn([ContentDocument(data=b"x", filename="a.dat", mime_type=mime_type)])
    msgs = provider._turns_as_inputs([turn])
    part = msgs[-1]["content"][0]
    assert part["type"] == "file"
    assert part["file"]["filename"] == "a.dat"
    assert part["file"]["file_data"] == (
        f"data:{mime_type};base64,{base64.b64encode(b'x').decode('utf-8')}"
    )


def test_completions_rejects_heic_images():
    provider = OpenAICompletionsProvider(model="gpt-4o")
    turn = UserTurn([ContentImageInline(image_content_type="image/heic", data="abcd")])
    with pytest.raises(ValueError, match="image/heic"):
        provider._turns_as_inputs([turn])


def test_turns_as_inputs_empty_text_with_tool_request():
    """Empty ContentText is stripped but tool requests are preserved."""
    provider = OpenAICompletionsProvider(model="test")

    turn = AssistantTurn(
        [
            ContentText.model_construct(text=""),
            ContentToolRequest(id="call_1", name="fn", arguments={}),
        ]
    )
    result = provider._turns_as_inputs([turn])
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "content" not in result[0]
    assert len(result[0]["tool_calls"]) == 1


def truncated_structured_completion() -> "ChatCompletion":
    """A structured-output response cut short by the token limit (gh-315)."""
    return ChatCompletion.construct(
        id="c1",
        object="chat.completion",
        created=0,
        model="gpt-4.1-nano",
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": '{"comments": [{"body": "trunc',
                },
            }
        ],
    )


def test_openai_completions_truncated_structured_output_errors_helpfully():
    with pytest.raises(ValueError, match="max_tokens"):
        OpenAICompletionsProvider._response_as_turn(
            truncated_structured_completion(), has_data_model=True
        )


def test_openai_completions_truncated_plain_text_still_returns_a_turn():
    turn = OpenAICompletionsProvider._response_as_turn(
        truncated_structured_completion(), has_data_model=False
    )

    assert turn.text == '{"comments": [{"body": "trunc'
    assert turn.finish_reason == "max_tokens"
