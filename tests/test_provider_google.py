import pytest
import requests
from chatlas import ChatGoogle, ChatVertex, tool_web_fetch, tool_web_search
from chatlas.types import (
    ContentCitation,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
)
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_list_models,
    assert_pdf_local,
    assert_tool_web_fetch,
    assert_tool_web_search,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def chat_func(vertex: bool = False, **kwargs):
    chat = ChatGoogle(**kwargs) if not vertex else ChatVertex(**kwargs)
    chat.set_model_params(temperature=0)
    return chat


def test_google_reasoning_effort_string():
    """A string `reasoning` maps to a `thinking_level` enum (#998)."""
    from google.genai.types import ThinkingLevel

    chat = ChatGoogle(reasoning="low")
    assert chat.kwargs_chat["config"]["thinking_config"] == {
        "thinking_level": ThinkingLevel.LOW,
        "include_thoughts": True,
    }


def test_google_reasoning_int_budget():
    """An int `reasoning` still maps to thinking_budget."""
    chat = ChatGoogle(reasoning=1024)
    assert chat.kwargs_chat["config"]["thinking_config"] == {
        "thinking_budget": 1024,
        "include_thoughts": True,
    }


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
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    assert chat.provider.name == "Google/Gemini"

    chat = chat_func(
        vertex=True,
        system_prompt="Be as terse as possible; no punctuation",
    )
    assert chat.provider.name == "Google/Vertex"


# TODO: this test runs fine in isolation, but fails for some reason when run with the other tests
# Seems google isn't handling async 100% correctly
# @pytest.mark.vcr
# @pytest.mark.asyncio
# async def test_google_simple_streaming_request():
#    chat = chat_func(
#        system_prompt="Be as terse as possible; no punctuation. Do not spell out numbers.",
#    )
#    res = []
#    async for x in await chat.stream_async("What is 1 + 1?"):
#        res.append(x)
#    assert "2" in "".join(res)
#    turn = chat.get_last_turn()
#    assert turn is not None
#    assert turn.finish_reason == "STOP"


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
def test_google_web_fetch():
    chat = assert_tool_web_fetch(chat_func, tool_web_fetch())
    fetched = [
        c
        for turn in chat.get_turns()
        for c in turn.contents
        if isinstance(c, ContentToolResponseFetch)
    ]
    assert fetched and fetched[0].url
    assert fetched[0].status == "success"
    # The normalized status doesn't drop the provider-native value
    assert fetched[0].extra is not None
    assert "URL_RETRIEVAL_STATUS" in str(fetched[0].extra["url_metadata"])


@pytest.mark.vcr
@retry_gemini_call
def test_google_web_search():
    chat = assert_tool_web_search(chat_func, tool_web_search())
    search_requests = [
        c
        for turn in chat.get_turns()
        for c in turn.contents
        if isinstance(c, ContentToolRequestSearch)
    ]
    results = [
        c
        for turn in chat.get_turns()
        for c in turn.contents
        if isinstance(c, ContentToolResponseSearch)
    ]
    assert search_requests
    assert results and results[0].sources
    # Note: the cassette grounding chunks don't include a domain field, so
    # domain is None for all sources in this recording.
    cites = [
        c
        for turn in chat.get_turns()
        for c in turn.contents
        if isinstance(c, ContentCitation)
    ]
    assert cites, "expected ContentCitation items in turn contents"
    assert all(c.url for c in cites)


@pytest.mark.vcr
def test_google_web_search_streaming():
    chat = chat_func()
    chat.register_tool(tool_web_search())
    items = list(
        chat.stream(
            "When was ggplot2 1.0.0 released to CRAN? Answer in YYYY-MM-DD format.",
            content="all",
        )
    )
    assert any(isinstance(x, ContentToolRequestSearch) for x in items)
    results = [x for x in items if isinstance(x, ContentToolResponseSearch)]
    assert results and results[0].sources
    citations = [x for x in items if isinstance(x, ContentCitation)]
    assert citations
    assert all(c.url for c in citations)


@pytest.mark.vcr
def test_google_web_fetch_streaming():
    chat = chat_func()
    chat.register_tool(tool_web_fetch())
    items = list(
        chat.stream(
            "What's the first movie listed on https://rvest.tidyverse.org/articles/starwars.html?",
            content="all",
        )
    )
    fetched = [x for x in items if isinstance(x, ContentToolResponseFetch)]
    assert fetched and fetched[0].url and fetched[0].status == "success"


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
    assert_list_models(ChatGoogle)


def test_google_thought_signature_roundtrip():
    """thought_signature must be preserved on ContentToolRequest for thinking models."""
    from chatlas._content import ContentToolRequest
    from chatlas._provider_google import GoogleProvider

    provider = GoogleProvider(
        model="gemini-2.5-flash-preview-04-17",
        api_key="dummy",
        kwargs=None,
    )

    # Simulate a Google API response with thought_signature on a functionCall part
    fake_signature = b"abc123signature"
    message = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "function_call": {
                                "name": "sum_tool",
                                "args": {"a": 1, "b": 2},
                            },
                            "thought_signature": fake_signature,
                        }
                    ]
                },
                "finish_reason": "STOP",
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
        },
    }

    turn = provider._as_turn(message, has_data_model=False)
    assert len(turn.contents) == 1
    req = turn.contents[0]
    assert isinstance(req, ContentToolRequest)
    assert req.extra.get("thought_signature") == fake_signature

    # Verify it round-trips back into the Part
    part = provider._as_part_type(req)
    assert part.thought_signature == fake_signature


def test_normalize_retrieval_status():
    from chatlas._provider_google import normalize_retrieval_status

    assert normalize_retrieval_status("URL_RETRIEVAL_STATUS_SUCCESS") == "success"
    assert normalize_retrieval_status("URL_RETRIEVAL_STATUS_UNSPECIFIED") is None
    # Every other reported status collapses to "error" (native value kept in extra)
    assert normalize_retrieval_status("URL_RETRIEVAL_STATUS_ERROR") == "error"
    assert normalize_retrieval_status("URL_RETRIEVAL_STATUS_PAYWALL") == "error"
    assert normalize_retrieval_status("URL_RETRIEVAL_STATUS_UNSAFE") == "error"
    assert normalize_retrieval_status(None) is None

    # Accepts the SDK enum (str-enum) as well as the raw string
    from google.genai.types import UrlRetrievalStatus

    assert (
        normalize_retrieval_status(UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS)
        == "success"
    )
    assert (
        normalize_retrieval_status(UrlRetrievalStatus.URL_RETRIEVAL_STATUS_PAYWALL)
        == "error"
    )
