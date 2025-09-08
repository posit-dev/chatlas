import os

import pytest

do_test = os.getenv("TEST_VLLM", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping vLLM tests", allow_module_level=True)

from chatlas import ChatVllm

from .conftest import (
    assert_tools_async,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)


def test_vllm_simple_request():
    # This test assumes you have a vLLM server running locally
    # Skip if TEST_VLLM_BASE_URL is not set
    base_url = os.getenv("TEST_VLLM_BASE_URL")
    if base_url is None:
        pytest.skip("TEST_VLLM_BASE_URL is not set; skipping vLLM tests")
    
    model = os.getenv("TEST_VLLM_MODEL", "llama3")
    
    chat = ChatVllm(
        base_url=base_url,
        model=model,
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] >= 10  # More lenient assertion for vLLM
    assert turn.finish_reason == "stop"


@pytest.mark.asyncio
async def test_vllm_simple_streaming_request():
    base_url = os.getenv("TEST_VLLM_BASE_URL")
    if base_url is None:
        pytest.skip("TEST_VLLM_BASE_URL is not set; skipping vLLM tests")
    
    model = os.getenv("TEST_VLLM_MODEL", "llama3")
    
    chat = ChatVllm(
        base_url=base_url,
        model=model,
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


def test_vllm_respects_turns_interface():
    base_url = os.getenv("TEST_VLLM_BASE_URL")
    if base_url is None:
        pytest.skip("TEST_VLLM_BASE_URL is not set; skipping vLLM tests")
    
    model = os.getenv("TEST_VLLM_MODEL", "llama3")
    
    def chat_fun(**kwargs):
        return ChatVllm(base_url=base_url, model=model, **kwargs)
    
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_vllm_tool_variations():
    base_url = os.getenv("TEST_VLLM_BASE_URL")
    if base_url is None:
        pytest.skip("TEST_VLLM_BASE_URL is not set; skipping vLLM tests")
    
    model = os.getenv("TEST_VLLM_MODEL", "llama3")
    
    def chat_fun(**kwargs):
        return ChatVllm(base_url=base_url, model=model, **kwargs)
    
    assert_tools_simple(chat_fun)


@pytest.mark.asyncio
async def test_vllm_tool_variations_async():
    base_url = os.getenv("TEST_VLLM_BASE_URL")
    if base_url is None:
        pytest.skip("TEST_VLLM_BASE_URL is not set; skipping vLLM tests")
    
    model = os.getenv("TEST_VLLM_MODEL", "llama3")
    
    def chat_fun(**kwargs):
        return ChatVllm(base_url=base_url, model=model, **kwargs)
    
    await assert_tools_async(chat_fun)


# Note: vLLM support for data extraction and images depends on the specific model
# and configuration, so we skip those tests for now