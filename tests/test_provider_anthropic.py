import base64

import httpx
import pytest
from typing import cast

from chatlas import ChatAnthropic, ContentToolResultImage, Turn
from chatlas._provider_anthropic import ProviderAnthropic

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
    retry_api_call,
)


def chat_func(system_prompt: str = "", **kwargs):
    return ChatAnthropic(
        system_prompt=system_prompt,
        model="claude-haiku-4-5-20251001",
        **kwargs,
    )


def test_anthropic_simple_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)
    assert turn.finish_reason == "end_turn"


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


def test_anthropic_respects_turns_interface():
    assert_turns_system(chat_func)
    assert_turns_existing(chat_func)


@retry_api_call
def test_anthropic_tool_variations():
    assert_tools_simple(chat_func)
    assert_tools_simple_stream_content(chat_func)
    assert_tools_sequential(chat_func, total_calls=6)


@retry_api_call
def test_anthropic_tool_variations_parallel():
    assert_tools_parallel(chat_func)


@pytest.mark.asyncio
@retry_api_call
async def test_anthropic_tool_variations_async():
    await assert_tools_async(chat_func)


def test_data_extraction():
    assert_data_extraction(chat_func)


@retry_api_call
def test_anthropic_images():
    assert_images_inline(chat_func)
    assert_images_remote(chat_func)


def test_anthropic_pdfs():
    assert_pdf_local(chat_func)


def test_anthropic_empty_response():
    chat = chat_func()
    chat.chat("Respond with only two blank lines")
    resp = chat.chat("What's 1+1? Just give me the number")
    assert "2" == str(resp).strip()


def test_anthropic_image_tool(test_images_dir):
    def get_picture():
        "Returns an image"
        # Local copy of https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png
        with open(test_images_dir / "dice.png", "rb") as image:
            bytez = image.read()
        return ContentToolResultImage(
            value=base64.b64encode(bytez).decode("utf-8"),
            mime_type="image/png",
        )

    chat = chat_func()
    chat.register_tool(get_picture)

    res = chat.chat(
        "You have a tool called 'get_picture' available to you. "
        "When called, it returns an image. "
        "Tell me what you see in the image."
    )

    assert "dice" in res.get_content()


def test_anthropic_custom_http_client():
    chat_func(kwargs={"http_client": httpx.AsyncClient()})


def test_anthropic_list_models():
    assert_list_models(chat_func)


def test_anthropic_removes_empty_assistant_turns():
    """Test that empty assistant turns are dropped to avoid API errors."""
    chat = chat_func()
    chat.set_turns(
        [
            Turn("user", "Don't say anything"),
            Turn("assistant", []),
        ]
    )

    # Get the message params that would be sent to the API
    provider = cast(ProviderAnthropic, chat.provider)
    turns_json = provider._as_message_params(chat.get_turns())

    # Should only have the user turn, not the empty assistant turn
    assert len(turns_json) == 1
    assert turns_json[0]["role"] == "user"
    assert turns_json[0]["content"][0]["text"] == "Don't say anything"