from typing import Literal, cast

import httpx
import pytest
from chatlas import (
    AssistantTurn,
    ChatAnthropic,
    UserTurn,
    content_image_file,
    tool_code_execution,
    tool_web_fetch,
    tool_web_search,
)
from chatlas._provider_anthropic import _ANTHROPIC_FINISH_REASON_MAP, AnthropicProvider
from chatlas._provider_anthropic import (
    normalize_finish_reason as anthropic_normalize_finish_reason,
)
from pydantic import BaseModel, Field

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_pdf_local,
    assert_tool_code_execution,
    assert_tool_web_fetch,
    assert_tool_web_search,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
    retry_api_call,
)


def test_normalize_finish_reason_maps_known_reasons():
    assert anthropic_normalize_finish_reason("end_turn") == "success"
    assert anthropic_normalize_finish_reason("max_tokens") == "max_tokens"
    assert anthropic_normalize_finish_reason("stop_sequence") == "stop_sequence"
    assert (
        anthropic_normalize_finish_reason("model_context_window_exceeded")
        == "context_window"
    )
    assert anthropic_normalize_finish_reason("refusal") == "content_filter"
    assert anthropic_normalize_finish_reason("tool_use") == "tool_use"


def test_normalize_finish_reason_maps_tool_use_explicitly():
    # tool_use must be an explicit mapping, not an incidental passthrough of an
    # unknown reason, so it isn't confused with a truly unrecognized reason.
    assert "tool_use" in _ANTHROPIC_FINISH_REASON_MAP


def test_normalize_finish_reason_passes_through_unknown():
    assert anthropic_normalize_finish_reason("some_new_reason") == "some_new_reason"


def test_normalize_finish_reason_handles_none():
    assert anthropic_normalize_finish_reason(None) is None


def chat_func(system_prompt: str = "", **kwargs):
    return ChatAnthropic(
        system_prompt=system_prompt,
        model="claude-haiku-4-5-20251001",
        **kwargs,
    )


@pytest.mark.vcr
def test_anthropic_simple_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)
    assert turn.finish_reason == "success"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    foo = await chat.stream_async("What is 1 + 1?")
    async for x in foo:
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "success"


@pytest.mark.vcr
def test_anthropic_respects_turns_interface():
    assert_turns_system(chat_func)
    assert_turns_existing(chat_func)


@pytest.mark.vcr
@retry_api_call
def test_anthropic_tool_variations():
    assert_tools_simple(chat_func)
    assert_tools_simple_stream_content(chat_func)
    assert_tools_sequential(chat_func, total_calls=6)


@pytest.mark.vcr
@retry_api_call
def test_anthropic_tool_variations_parallel():
    assert_tools_parallel(chat_func)


@pytest.mark.vcr
@pytest.mark.asyncio
@retry_api_call
async def test_anthropic_tool_variations_async():
    await assert_tools_async(chat_func)


@pytest.mark.vcr
def test_anthropic_web_fetch():
    def chat_fun(**kwargs):
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            kwargs={"default_headers": {"anthropic-beta": "web-fetch-2025-09-10"}},
            **kwargs,
        )

    assert_tool_web_fetch(chat_fun, tool_web_fetch())


@pytest.mark.vcr
def test_anthropic_web_search():
    assert_tool_web_search(chat_func, tool_web_search())


@pytest.mark.vcr
def test_anthropic_web_search_citations():
    """Test that citations from web search are preserved on the completion."""
    chat = chat_func()
    chat.register_tool(tool_web_search())
    chat.chat("When was ggplot2 1.0.0 released to CRAN? Answer in YYYY-MM-DD format.")

    # Get the turn and verify citations are on the completion
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.completion is not None

    # Find a text content block that should have citations
    text_blocks = [c for c in turn.completion.content if c.type == "text"]
    assert len(text_blocks) > 0

    # At least one text block should have citations from web search
    has_citations = any(getattr(block, "citations", None) for block in text_blocks)
    assert has_citations, "Expected citations on text blocks from web search"


@pytest.mark.vcr
def test_anthropic_code_execution():
    def chat_fun(**kwargs):
        return ChatAnthropic(
            kwargs={
                "default_headers": {"anthropic-beta": "code-execution-2025-05-22"}
            },
            **kwargs,
        )

    assert_tool_code_execution(chat_fun, tool_code_execution())


# N.B. no code execution *persistence* test for Anthropic: despite chatlas
# reusing the same `container` across turns, Claude's code execution tool
# does not persist Python REPL state (e.g. variables) across separate turns
# -- each execution starts a fresh interpreter. This was verified live and
# contradicts the tool's own SDK docstring ("REPL state persistence"), so
# it's treated the same as Google's documented per-turn-fresh-sandbox
# limitation rather than a bug in chatlas.


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(chat_func)


@pytest.mark.vcr
def test_stream_with_data_model():
    from chatlas._content import ContentJson

    chat = chat_func()

    class Person(BaseModel):
        name: str
        age: int

    chunks = list(chat.stream("John, age 15, won first prize", data_model=Person))
    result = "".join(chunks)
    person = Person.model_validate_json(result)
    assert person == Person(name="John", age=15)

    turn = chat.get_last_turn()
    assert turn is not None
    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentJson)
    assert turn.contents[0].value == {"name": "John", "age": 15}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_stream_async_with_data_model():
    from chatlas._content import ContentJson

    chat = chat_func()

    class Person(BaseModel):
        name: str
        age: int

    chunks = [
        chunk
        async for chunk in await chat.stream_async(
            "John, age 15, won first prize", data_model=Person
        )
    ]
    result = "".join(chunks)
    person = Person.model_validate_json(result)
    assert person == Person(name="John", age=15)

    turn = chat.get_last_turn()
    assert turn is not None
    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentJson)
    assert turn.contents[0].value == {"name": "John", "age": 15}


@pytest.mark.vcr
@retry_api_call
def test_anthropic_images():
    assert_images_inline(chat_func)
    assert_images_remote(chat_func)


@pytest.mark.vcr
def test_anthropic_pdfs():
    assert_pdf_local(chat_func)


@pytest.mark.vcr
def test_anthropic_empty_response():
    chat = chat_func()
    chat.chat("Respond with only two blank lines")
    resp = chat.chat("What's 1+1? Just give me the number")
    assert "2" == str(resp).strip()


@pytest.mark.vcr
def test_anthropic_image_tool(test_images_dir):
    def get_picture():
        "Returns an image"
        # Local copy of https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png
        # Using resize='none' to avoid platform-specific encoding differences
        return content_image_file(test_images_dir / "dice.png", resize="none")

    chat = chat_func()
    chat.register_tool(get_picture)

    res = chat.chat(
        "You have a tool called 'get_picture' available to you. "
        "When called, it returns an image. "
        "Tell me what you see in the image."
    )

    assert "dice" in res.get_content()


def test_anthropic_custom_http_client():
    ChatAnthropic(kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_anthropic_list_models():
    assert_list_models(chat_func)


def test_anthropic_removes_empty_assistant_turns():
    """Test that empty assistant turns are dropped to avoid API errors."""
    chat = chat_func()
    chat.set_turns(
        [
            UserTurn("Don't say anything"),
            AssistantTurn([]),
        ]
    )

    # Get the message params that would be sent to the API
    provider = cast(AnthropicProvider, chat.provider)
    turns_json = provider._as_message_params(chat.get_turns())

    # Should only have the user turn, not the empty assistant turn
    assert len(turns_json) == 1
    assert turns_json[0]["role"] == "user"
    assert turns_json[0]["content"][0]["text"] == "Don't say anything"  # type: ignore


@pytest.mark.vcr
def test_anthropic_nested_data_model_extraction():
    """
    Test that nested Pydantic models work for structured data extraction.

    This is a regression test for issue #100 where data extraction failed with
    nested models because $defs was placed inside the 'data' property instead
    of at the root of input_schema, breaking $ref JSON pointer references.

    See: https://github.com/posit-dev/chatlas/issues/100
    """

    # Models from issue #100
    class Classification(BaseModel):
        name: Literal[
            "Politics", "Sports", "Technology", "Entertainment", "Business", "Other"
        ] = Field(description="The category name")
        score: float = Field(
            description="The classification score for the category, ranging from 0.0 to 1.0."
        )

    class Classifications(BaseModel):
        """Array of classification results. The scores should sum to 1."""

        classifications: list[Classification]

    text = (
        "The new quantum computing breakthrough could revolutionize the tech industry."
    )

    chat = chat_func(system_prompt="You are a friendly but terse assistant.")
    data = chat.chat_structured(text, data_model=Classifications)

    # Verify we got a valid response with the nested structure
    assert isinstance(data, Classifications)
    assert len(data.classifications) > 0

    # Check that at least one classification is Technology (the obvious choice)
    categories = [c.name for c in data.classifications]
    assert "Technology" in categories, f"Expected 'Technology' in {categories}"

    # Verify scores are valid floats between 0 and 1
    for classification in data.classifications:
        assert 0.0 <= classification.score <= 1.0, (
            f"Score {classification.score} should be between 0 and 1"
        )


def test_anthropic_reasoning_int_budget():
    """An int `reasoning` maps to a fixed thinking budget (regression)."""
    chat = ChatAnthropic(reasoning=2048)
    assert chat.kwargs_chat == {"thinking": {"type": "enabled", "budget_tokens": 2048}}


def test_anthropic_reasoning_effort_string():
    """A string `reasoning` enables adaptive thinking via output_config (#997)."""
    chat = ChatAnthropic(reasoning="high")
    assert chat.kwargs_chat == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_anthropic_adaptive_effort_merges_with_structured_output():
    """When extracting data, adaptive effort merges into the native output_config."""

    class Person(BaseModel):
        name: str

    provider = AnthropicProvider(
        model="claude-sonnet-4-6", structured_output_mode="native"
    )
    args = provider._chat_perform_args(
        stream=False,
        turns=[],
        tools={},
        data_model=Person,
        kwargs={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
    )
    output_config = args["output_config"]
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"


def test_anthropic_code_execution_parses_request_and_response():
    """server_tool_use(code_execution) + code_execution_tool_result parse correctly."""
    from anthropic.types import Message
    from chatlas._content import (
        ContentToolRequestCodeExecution,
        ContentToolResponseCodeExecution,
    )
    from chatlas._provider_anthropic import AnthropicProvider

    chat = ChatAnthropic()
    provider = chat.provider
    assert isinstance(provider, AnthropicProvider)

    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "code_execution",
                    "input": {"code": "print(1 + 1)"},
                },
                {
                    "type": "code_execution_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": {
                        "type": "code_execution_result",
                        "stdout": "2\n",
                        "stderr": "",
                        "return_code": 0,
                        "content": [],
                    },
                },
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "container": {
                "id": "cntr_xyz",
                "expires_at": "2026-07-21T00:00:00Z",
            },
        }
    )

    turn = provider._as_turn(message, has_data_model=False)

    assert len(turn.contents) == 2
    request = turn.contents[0]
    assert isinstance(request, ContentToolRequestCodeExecution)
    assert request.code == "print(1 + 1)"

    response = turn.contents[1]
    assert isinstance(response, ContentToolResponseCodeExecution)
    assert response.output == "2\n"
    assert response.error is None
    assert response.container_id == "cntr_xyz"


def test_anthropic_code_execution_error_result():
    """An error result maps to `error`, not `output`."""
    from anthropic.types import Message
    from chatlas._content import ContentToolResponseCodeExecution
    from chatlas._provider_anthropic import AnthropicProvider

    chat = ChatAnthropic()
    provider = chat.provider
    assert isinstance(provider, AnthropicProvider)

    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {
                    "type": "code_execution_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": {
                        "type": "code_execution_tool_result_error",
                        "error_code": "execution_time_exceeded",
                    },
                },
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )

    turn = provider._as_turn(message, has_data_model=False)
    response = turn.contents[0]
    assert isinstance(response, ContentToolResponseCodeExecution)
    assert response.output is None
    assert response.error == "execution_time_exceeded"
    assert response.container_id is None


def test_anthropic_code_execution_tool_schema():
    from chatlas._provider_anthropic import AnthropicProvider

    schema = AnthropicProvider._anthropic_tool_schema(tool_code_execution())
    assert schema["name"] == "code_execution"
    assert schema["type"] == "code_execution_20250522"


def test_anthropic_code_execution_container_reuse():
    from chatlas._content import ContentToolResponseCodeExecution
    from chatlas._provider_anthropic import AnthropicProvider
    from chatlas._turn import AssistantTurn

    chat = ChatAnthropic()
    chat.register_tool(tool_code_execution())
    provider = chat.provider
    assert isinstance(provider, AnthropicProvider)

    turns = [
        AssistantTurn(
            [ContentToolResponseCodeExecution(output="1", container_id="cntr_xyz")]
        )
    ]
    kwargs = provider._chat_perform_args(
        stream=False,
        turns=turns,
        tools=chat._tools,  # type: ignore[reportPrivateUsage]
        data_model=None,
        kwargs=None,
    )
    assert kwargs["container"] == "cntr_xyz"
