import os

import pytest
import requests
from chatlas import ChatGoogle, ChatVertex
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, wait_exponential

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


def chat_func(vertex: bool = False, **kwargs):
    chat = ChatGoogle(**kwargs) if not vertex else ChatVertex(**kwargs)
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
    reraise=True,
)


@retry_gemini_call
def test_google_simple_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (18, 1)
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


@retry_gemini_call
def test_name_setting():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    assert chat.provider.name == "Google/Gemini"

    chat = chat_func(
        vertex=True,
        system_prompt="Be as terse as possible; no punctuation",
    )
    assert chat.provider.name == "Google/Vertex"


@pytest.mark.asyncio
@retry_gemini_call
async def test_google_simple_streaming_request():
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


@retry_gemini_call
def test_google_respects_turns_interface():
    assert_turns_system(chat_func)
    assert_turns_existing(chat_func)


@retry_gemini_call
def test_tools_simple():
    assert_tools_simple(chat_func)


@retry_gemini_call
def test_tools_simple_stream_content():
    assert_tools_simple_stream_content(chat_func)


@retry_gemini_call
def test_tools_parallel():
    assert_tools_parallel(chat_func, total_calls=6)


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


@retry_gemini_call
def test_data_extraction():
    assert_data_extraction(chat_func)


@retry_gemini_call
def test_images_inline():
    assert_images_inline(chat_func)


@retry_gemini_call
def test_images_remote_error():
    assert_images_remote_error(chat_func)


@retry_gemini_call
def test_google_pdfs():
    assert_pdf_local(chat_func)
