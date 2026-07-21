import pytest
import requests
from chatlas import ChatGoogle, ChatVertex, tool_web_fetch, tool_web_search
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
    assert_tool_web_fetch(chat_func, tool_web_fetch())


@pytest.mark.vcr
@retry_gemini_call
def test_google_web_search():
    assert_tool_web_search(chat_func, tool_web_search())


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


def test_google_batch_supported_for_gemini_not_vertex():
    """Batch is only supported on the Gemini Developer API, not Vertex."""
    from chatlas._provider_google import GoogleProvider

    gemini = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)
    assert gemini.has_batch_support() is True

    vertex = GoogleProvider(
        model="gemini-3.5-flash",
        api_key="dummy",
        name="Google/Vertex",
        kwargs=None,
    )
    assert vertex.has_batch_support() is False


def test_google_batch_submit_reuses_chat_perform_args():
    """batch_submit() builds one InlinedRequest per conversation via _chat_perform_args."""
    from unittest.mock import MagicMock

    from chatlas._provider_google import GoogleProvider
    from chatlas._turn import Turn, user_turn
    from google.genai import types

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)

    fake_batch = types.BatchJob(
        name="batches/123", state=types.JobState.JOB_STATE_PENDING
    )
    mock_create = MagicMock(return_value=fake_batch)
    provider._client.batches.create = mock_create

    conversations: list[list[Turn]] = [
        [user_turn("What's the capital of France?")],
        [user_turn("What's the capital of Germany?")],
    ]
    result = provider.batch_submit(conversations)

    assert result == fake_batch.model_dump()
    assert mock_create.call_count == 1
    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "gemini-3.5-flash"
    requests = kwargs["src"]
    assert len(requests) == 2
    assert all(isinstance(r, types.InlinedRequest) for r in requests)


def test_google_batch_status_mapping():
    from chatlas._provider_google import GoogleProvider

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)

    working = provider.batch_status(
        {
            "name": "batches/123",
            "state": "JOB_STATE_RUNNING",
            "completion_stats": {
                "successful_count": 1,
                "failed_count": 0,
                "incomplete_count": 1,
            },
        }
    )
    assert working.working is True
    assert working.n_succeeded == 1
    assert working.n_failed == 0
    assert working.n_processing == 1

    done = provider.batch_status(
        {
            "name": "batches/123",
            "state": "JOB_STATE_SUCCEEDED",
            "completion_stats": {
                "successful_count": 2,
                "failed_count": 0,
                "incomplete_count": 0,
            },
        }
    )
    assert done.working is False
    assert done.n_succeeded == 2

    no_stats = provider.batch_status(
        {"name": "batches/123", "state": "JOB_STATE_PENDING"}
    )
    assert no_stats.working is True
    assert no_stats.n_processing == 0
    assert no_stats.n_succeeded == 0
    assert no_stats.n_failed == 0


def test_google_batch_retrieve_and_result_turn():
    from chatlas._provider_google import GoogleProvider

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)

    batch = {
        "name": "batches/123",
        "state": "JOB_STATE_SUCCEEDED",
        "dest": {
            "inlined_responses": [
                {
                    "response": {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "Paris"}]},
                                "finish_reason": "STOP",
                            }
                        ]
                    }
                },
                {"error": {"code": 500, "message": "internal error"}},
            ]
        },
    }

    results = provider.batch_retrieve(batch)
    assert len(results) == 2

    turn = provider.batch_result_turn(results[0], has_data_model=False)
    assert turn is not None
    assert turn.text == "Paris"

    with pytest.warns(UserWarning, match="Batch request failed"):
        failed_turn = provider.batch_result_turn(results[1], has_data_model=False)
    assert failed_turn is None
