import os

import pytest

from chatlas import ChatHuggingFace

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_tools_async,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)

# I think we would need to pay Hugging Face to actually run these tests?
api_key = os.getenv("HUGGINGFACE_API_KEY")
if api_key is None:
    pytest.skip(
        "HUGGINGFACE_API_KEY is not set; skipping tests", allow_module_level=True
    )


@pytest.mark.vcr
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


@pytest.mark.vcr
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


@pytest.mark.vcr
def test_huggingface_respects_turns_interface():
    chat_fun = ChatHuggingFace
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_huggingface_tools():
    def chat_fun(**kwargs):
        return ChatHuggingFace(model="meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    assert_tools_simple(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_huggingface_tools_async():
    def chat_fun(**kwargs):
        return ChatHuggingFace(model="meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    await assert_tools_async(chat_fun)


@pytest.mark.vcr
def test_huggingface_data_extraction():
    def chat_fun(**kwargs):
        return ChatHuggingFace(model="meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    assert_data_extraction(chat_fun)


@pytest.mark.vcr
def test_huggingface_images():
    # Use a vision model that supports images
    def chat_fun(**kwargs):
        return ChatHuggingFace(model="Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)

    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.vcr
def test_huggingface_model_list():
    assert_list_models(ChatHuggingFace)


def test_huggingface_custom_model():
    chat = ChatHuggingFace(model="microsoft/DialoGPT-medium")
    assert chat.provider.model == "microsoft/DialoGPT-medium"


def test_huggingface_provider_name():
    chat = ChatHuggingFace()
    assert chat.provider.name == "HuggingFace"
