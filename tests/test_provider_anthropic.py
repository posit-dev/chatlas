import base64

import pytest

from chatlas import ChatAnthropic

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
    retryassert,
    retryassert_async,
)


def test_anthropic_simple_request():
    chat = ChatAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5)
    assert turn.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = ChatAnthropic(
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
    chat_fun = ChatAnthropic
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_anthropic_tool_variations():
    chat_fun = ChatAnthropic

    def run_simpleassert():
        assert_tools_simple(chat_fun)

    retryassert(run_simpleassert, retries=5)

    def run_parallelassert():
        # For some reason, at the time of writing, Claude 3.7 doesn't
        # respond with multiple tools at once for this test (but it does)
        # answer the question correctly with sequential tools.
        def chat_fun2(**kwargs):
            return ChatAnthropic(model="claude-3-5-sonnet-latest", **kwargs)

        assert_tools_parallel(chat_fun2)

    retryassert(run_parallelassert, retries=5)

    # Fails occassionally returning "" instead of Susan
    def run_sequentialassert():
        assert_tools_sequential(chat_fun, total_calls=6)

    retryassert(run_sequentialassert, retries=5)


@pytest.mark.asyncio
async def test_anthropic_tool_variations_async():
    async def run_asyncassert():
        await assert_tools_async(ChatAnthropic)

    await retryassert_async(run_asyncassert, retries=5)


def test_data_extraction():
    assert_data_extraction(ChatAnthropic)


def test_anthropic_images():
    chat_fun = ChatAnthropic

    def run_inlineassert():
        assert_images_inline(chat_fun)

    retryassert(run_inlineassert, retries=3)
    assert_images_remote_error(chat_fun)


def test_anthropic_image_tool(test_images_dir):
    from chatlas import ToolResult

    def get_picture():
        "Returns an image"
        # Local copy of https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png
        with open(test_images_dir / "dice.png", "rb") as image:
            bytez = image.read()
        res = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(bytez).decode("utf-8"),
                },
            }
        ]
        return ToolResult(res, format_as="as_is")

    chat = ChatAnthropic()
    chat.register_tool(get_picture)

    res = chat.chat(
        "You have a tool called 'get_picture' available to you. "
        "When called, it returns an image. "
        "Tell me what you see in the image."
    )

    assert "dice" in res.get_content()
