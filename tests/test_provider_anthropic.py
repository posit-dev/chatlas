import base64

import httpx
import pytest
from chatlas import ChatAnthropic, ContentToolResultImage

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
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


def test_anthropic_simple_request():
    chat = ChatAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)
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


@retry_api_call
def test_anthropic_tool_variations():
    chat_fun = ChatAnthropic
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@retry_api_call
def test_anthropic_tool_variations_parallel():
    # For some reason, at the time of writing, Claude 3.7 doesn't
    # respond with multiple tools at once for this test (but it does)
    # answer the question correctly with sequential tools.
    def chat_fun(**kwargs):
        return ChatAnthropic(model="claude-3-5-sonnet-latest", **kwargs)

    assert_tools_parallel(chat_fun)


@pytest.mark.asyncio
@retry_api_call
async def test_anthropic_tool_variations_async():
    await assert_tools_async(ChatAnthropic)


def test_data_extraction():
    assert_data_extraction(ChatAnthropic)


@retry_api_call
def test_anthropic_images():
    chat_fun = ChatAnthropic

    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


def test_anthropic_pdfs():
    chat_fun = ChatAnthropic
    assert_pdf_local(chat_fun)


def test_anthropic_empty_response():
    chat = ChatAnthropic()
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

    chat = ChatAnthropic()
    chat.register_tool(get_picture)

    res = chat.chat(
        "You have a tool called 'get_picture' available to you. "
        "When called, it returns an image. "
        "Tell me what you see in the image."
    )

    assert "dice" in res.get_content()


def test_anthropic_custom_http_client():
    ChatAnthropic(kwargs={"http_client": httpx.AsyncClient()})


def test_anthropic_server_tools_basic():
    # Test that server tools can be passed through
    server_tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3
        }
    ]
    chat = ChatAnthropic(server_tools=server_tools)
    assert chat.provider._server_tools == server_tools


def test_anthropic_server_tools_mixed_with_custom_tools():
    # Test that server tools work alongside custom tools
    server_tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search"
        }
    ]
    
    def my_tool(x: int) -> int:
        """Double a number"""
        return x * 2
    
    chat = ChatAnthropic(server_tools=server_tools)
    chat.register_tool(my_tool)
    
    # Verify both server tools and custom tools are present
    assert chat.provider._server_tools == server_tools
    assert len(chat.get_tools()) == 1
    assert chat.get_tools()[0].name == "my_tool"


def test_anthropic_server_tools_in_api_args():
    # Test that server tools are included in API arguments
    server_tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 2
        },
        {
            "type": "bash_20250124",
            "name": "bash"
        }
    ]
    
    chat = ChatAnthropic(server_tools=server_tools)
    
    from chatlas._turn import user_turn
    turn = user_turn("test")
    
    # Get the args that would be sent to the API
    args = chat.provider._chat_perform_args(
        stream=False,
        turns=[turn],
        tools={},  # No custom tools
        data_model=None,
        kwargs=None
    )
    
    # Check that server tools are in the tools list
    assert len(args["tools"]) == 2
    assert args["tools"][0]["type"] == "web_search_20250305"
    assert args["tools"][0]["name"] == "web_search"
    assert args["tools"][0]["max_uses"] == 2
    assert args["tools"][1]["type"] == "bash_20250124"
    assert args["tools"][1]["name"] == "bash"


def test_anthropic_server_tools_disabled_by_default():
    # Test that no server tools are included by default
    chat = ChatAnthropic()
    assert chat.provider._server_tools == []


def test_anthropic_server_tools_empty_list():
    # Test that empty list works
    chat = ChatAnthropic(server_tools=[])
    assert chat.provider._server_tools == []
