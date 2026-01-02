from typing import Literal, cast

import httpx
import pytest
from chatlas import (
    AssistantTurn,
    ChatAnthropic,
    UserTurn,
    content_image_file,
    tool_web_fetch,
    tool_web_search,
)
from chatlas._provider_anthropic import AnthropicProvider
from pydantic import BaseModel, Field

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_pdf_local,
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
    assert turn.finish_reason == "end_turn"


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
    assert turn.finish_reason == "end_turn"


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
def test_data_extraction():
    assert_data_extraction(chat_func)


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

    text = "The new quantum computing breakthrough could revolutionize the tech industry."

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
