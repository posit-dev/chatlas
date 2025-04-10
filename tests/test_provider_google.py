import os

import pytest
import requests
from chatlas import ChatGoogle
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
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


def test_google_simple_request():
    chat = ChatGoogle(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (16, 2)
    assert turn.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_google_simple_streaming_request():
    chat = ChatGoogle(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "STOP"


def test_google_respects_turns_interface():
    chat_fun = ChatGoogle
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


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
    wait=wait_exponential(min=1, max=100),
    stop=stop_after_attempt(10),
    reraise=True,
)


@retry_gemini_call
def test_tools_simple():
    assert_tools_simple(ChatGoogle)


@retry_gemini_call
def test_tools_simple_stream_content():
    assert_tools_simple_stream_content(ChatGoogle)


@retry_gemini_call
def test_tools_parallel():
    assert_tools_parallel(ChatGoogle)


@retry_gemini_call
def test_tools_sequential():
    assert_tools_sequential(
        ChatGoogle,
        total_calls=6,
    )


# TODO: this test runs fine in isolation, but fails for some reason when run with the other tests
# Seems google isn't handling async 100% correctly
# @pytest.mark.asyncio
# async def test_google_tool_variations_async():
#     await assert_tools_async(ChatGoogle, stream=False)


@retry_gemini_call
def test_data_extraction():
    assert_data_extraction(ChatGoogle)


@retry_gemini_call
def test_images_inline():
    assert_images_inline(ChatGoogle)


@retry_gemini_call
def test_images_remote_error():
    assert_images_remote_error(ChatGoogle)


@retry_gemini_call
def test_google_pdfs():
    assert_pdf_local(ChatGoogle)
