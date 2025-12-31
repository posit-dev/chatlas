import pytest
from chatlas import ChatHuggingFace

from ._test_providers import TestChatHuggingFace
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


@pytest.mark.vcr
def test_huggingface_simple_request():
    chat = TestChatHuggingFace(
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
    chat = TestChatHuggingFace(
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
    assert_turns_system(TestChatHuggingFace)
    assert_turns_existing(TestChatHuggingFace)


@pytest.mark.vcr
def test_huggingface_tools():
    def chat_fun(**kwargs):
        return TestChatHuggingFace(model="meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    assert_tools_simple(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_huggingface_tools_async():
    def chat_fun(**kwargs):
        return TestChatHuggingFace(model="meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    await assert_tools_async(chat_fun)


@pytest.mark.vcr
def test_huggingface_data_extraction():
    def chat_fun(**kwargs):
        return TestChatHuggingFace(model="meta-llama/Llama-3.1-8B-Instruct", **kwargs)

    assert_data_extraction(chat_fun)


@pytest.mark.vcr
def test_huggingface_images():
    # Use a vision model that supports images
    def chat_fun(**kwargs):
        return TestChatHuggingFace(model="Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)

    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.vcr
def test_huggingface_model_list():
    assert_list_models(TestChatHuggingFace)


def test_huggingface_custom_model():
    # This test doesn't use VCR, so use explicit dummy key
    chat = ChatHuggingFace(api_key="test", model="microsoft/DialoGPT-medium")
    assert chat.provider.model == "microsoft/DialoGPT-medium"


def test_huggingface_provider_name():
    # This test doesn't use VCR, so use explicit dummy key
    chat = ChatHuggingFace(api_key="test")
    assert chat.provider.name == "HuggingFace"
