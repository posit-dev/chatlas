import os

import pytest
import requests
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from chatlas import ChatGoogle, ChatVertex

from ._test_providers import TestChatGoogle
from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_list_models,
    assert_pdf_local,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)

do_test = os.getenv("TEST_GOOGLE", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Google tests", allow_module_level=True)


def chat_func(vertex: bool = False, **kwargs):
    # For Vertex, use ChatVertex directly (not commonly used in VCR tests)
    # For Google, use TestChatGoogle with fallback credentials
    if vertex:
        chat = ChatVertex(**kwargs)
    else:
        chat = TestChatGoogle(**kwargs)
    chat.set_model_params(temperature=0)
    return chat


# https://github.com/googleapis/python-genai/issues/336
def _is_retryable_error(exception: BaseException) -> bool:
    """
    Checks if the exception is a retryable error based on the criteria.
    """
    if isinstance(exception, APIError):
        return exception.code in [429, 502, 503, 504]
    if isinstance(exception, requests.exceptions.ConnectionError):
        return True
    return False


retry_gemini_call = retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(min=1, max=500),
    stop=stop_after_attempt(5),
    reraise=True,
)


@pytest.mark.vcr
@retry_gemini_call
def test_google_simple_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert turn.tokens[0] == 18  # input tokens
    # Output tokens can vary (1-29), so just check it's positive
    assert turn.tokens[1] > 0
    assert turn.finish_reason == "STOP"
    assert chat.provider.name == "Google/Gemini"


# Something recently changed with Vertex auth, I don't have time to debug it right now
# def test_vertex_simple_request():
#    chat = chat_func(
#        vertex=True,
#        system_prompt="Be as terse as possible; no punctuation",
#    )
#    chat.chat("What is 1 + 1?")
#    turn = chat.get_last_turn()
#    assert turn is not None
#    assert turn.tokens == (16, 2)
#    assert turn.finish_reason == "STOP"
#    assert chat.provider.name == "Google/Vertex"


def test_name_setting():
    # This test doesn't use VCR, so use explicit dummy key for Google
    chat = ChatGoogle(
        api_key="test-dummy-key",
        system_prompt="Be as terse as possible; no punctuation",
    )
    assert chat.provider.name == "Google/Gemini"

    # Skip Vertex test - it requires actual GCP auth which can't be easily mocked
    # chat = ChatVertex(system_prompt="...")
    # assert chat.provider.name == "Google/Vertex"


# Google streaming tests don't work with VCR due to SDK-specific response handling
# This test requires live API access
@pytest.mark.asyncio
@retry_gemini_call
async def test_google_simple_streaming_request():
    # Skip if no real Google API key or if streaming tests are disabled
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY required for streaming test (VCR incompatible)")
    if os.getenv("TEST_GOOGLE_STREAMING", "true").lower() == "false":
        pytest.skip("Google streaming tests disabled (VCR incompatible)")
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation. Do not spell out numbers.",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "STOP"


@pytest.mark.vcr
@retry_gemini_call
def test_google_respects_turns_interface():
    assert_turns_system(chat_func)
    assert_turns_existing(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_tools_simple():
    assert_tools_simple(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_tools_simple_stream_content():
    assert_tools_simple_stream_content(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_tools_parallel():
    assert_tools_parallel(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_tools_sequential():
    assert_tools_sequential(
        chat_func,
        total_calls=6,
    )


# TODO: this test runs fine in isolation, but fails for some reason when run with the other tests
# Seems google isn't handling async 100% correctly
# @pytest.mark.asyncio
# async def test_google_tool_variations_async():
#     await assert_tools_async(ChatGoogle, stream=False)


@pytest.mark.vcr
@retry_gemini_call
def test_data_extraction():
    assert_data_extraction(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_images_inline():
    assert_images_inline(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_images_remote_error():
    assert_images_remote_error(chat_func)


@pytest.mark.vcr
@retry_gemini_call
def test_google_pdfs():
    assert_pdf_local(chat_func)


@pytest.mark.vcr
def test_google_list_models():
    assert_list_models(TestChatGoogle)
