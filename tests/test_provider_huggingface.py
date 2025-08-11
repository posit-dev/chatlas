import os
import pytest
from chatlas import ChatHuggingFace

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def test_huggingface_simple_request():
    chat = ChatHuggingFace(
        system_prompt="Be as terse as possible; no punctuation",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] > 0  # input tokens
    assert turn.tokens[1] > 0  # output tokens
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_huggingface_simple_streaming_request():
    chat = ChatHuggingFace(
        system_prompt="Be as terse as possible; no punctuation",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


def test_huggingface_respects_turns_interface():
    chat_fun = ChatHuggingFace
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_huggingface_tools():
    chat_fun = lambda **kwargs: ChatHuggingFace(
        model="meta-llama/Llama-3.1-8B-Instruct", **kwargs
    )
    assert_tools_simple(chat_fun)
    assert_tools_sequential(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_simple_stream_content(chat_fun)


@pytest.mark.asyncio
async def test_huggingface_tools_async():
    chat_fun = lambda **kwargs: ChatHuggingFace(
        model="meta-llama/Llama-3.1-8B-Instruct", **kwargs
    )
    await assert_tools_async(chat_fun)


def test_huggingface_data_extraction():
    chat_fun = lambda **kwargs: ChatHuggingFace(
        model="meta-llama/Llama-3.1-8B-Instruct", **kwargs
    )
    assert_data_extraction(chat_fun)


def test_huggingface_images():
    # Use a vision model that supports images
    chat_fun = lambda **kwargs: ChatHuggingFace(
        model="Qwen/Qwen2.5-VL-7B-Instruct", **kwargs
    )
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


def test_huggingface_api_key_from_env():
    # Test that API key is read from environment
    original_key = os.environ.get("HUGGINGFACE_API_KEY")
    test_key = "test_key_123"
    
    try:
        os.environ["HUGGINGFACE_API_KEY"] = test_key
        chat = ChatHuggingFace()
        assert chat.provider._client.api_key == test_key
    finally:
        if original_key is not None:
            os.environ["HUGGINGFACE_API_KEY"] = original_key
        elif "HUGGINGFACE_API_KEY" in os.environ:
            del os.environ["HUGGINGFACE_API_KEY"]


def test_huggingface_custom_model():
    chat = ChatHuggingFace(model="microsoft/DialoGPT-medium")
    assert chat.provider.model == "microsoft/DialoGPT-medium"


def test_huggingface_base_url():
    chat = ChatHuggingFace()
    assert "huggingface.co" in str(chat.provider._client.base_url)


def test_huggingface_provider_name():
    chat = ChatHuggingFace()
    assert chat.provider.name == "HuggingFace"