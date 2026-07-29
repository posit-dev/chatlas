from typing import cast

import pytest
import requests
from chatlas import ChatGoogle, ChatVertex, tool_web_fetch, tool_web_search
from chatlas._provider_google import (
    GoogleProvider,
    google_grounding_citations,
)
from chatlas._provider_google import (
    normalize_finish_reason as google_normalize_finish_reason,
)
from chatlas.types import (
    Content,
    ContentCitation,
    ContentText,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    WebSource,
)
from google.genai.errors import APIError
from google.genai.types import GroundingMetadataDict
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


def test_normalize_finish_reason_maps_known_reasons():
    assert google_normalize_finish_reason("STOP") == "success"
    assert google_normalize_finish_reason("MAX_TOKENS") == "max_tokens"
    assert google_normalize_finish_reason("SAFETY") == "content_filter"
    assert google_normalize_finish_reason("RECITATION") == "content_filter"
    assert google_normalize_finish_reason("BLOCKLIST") == "content_filter"
    assert google_normalize_finish_reason("PROHIBITED_CONTENT") == "content_filter"
    assert google_normalize_finish_reason("SPII") == "content_filter"


def test_normalize_finish_reason_passes_through_unknown():
    assert google_normalize_finish_reason("OTHER") == "OTHER"
    assert google_normalize_finish_reason("NEW_REASON") == "NEW_REASON"


def test_normalize_finish_reason_handles_none():
    assert google_normalize_finish_reason(None) is None


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
    assert turn.finish_reason == "success"
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
#    assert turn.finish_reason == "success"
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
#    assert turn.finish_reason == "success"


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


def _contents_of_type(chat, cls: type):
    return [c for turn in chat.get_turns() for c in turn.contents if isinstance(c, cls)]


@pytest.mark.vcr
@retry_gemini_call
def test_google_web_fetch():
    chat = assert_tool_web_fetch(chat_func, tool_web_fetch())
    fetched = _contents_of_type(chat, ContentToolResponseFetch)
    assert fetched and fetched[0].url
    assert fetched[0].status == "success"
    # The normalized status doesn't drop the provider-native value
    assert fetched[0].extra is not None
    assert "URL_RETRIEVAL_STATUS" in str(fetched[0].extra["url_metadata"])


@pytest.mark.vcr
@retry_gemini_call
def test_google_web_search():
    chat = assert_tool_web_search(
        lambda: chat_func(model="gemini-2.5-flash"), tool_web_search()
    )
    # Gemini issued two queries for this prompt; each is its own request, so a
    # renderer showing "searched for ..." doesn't silently drop the second.
    queries = [c.query for c in _contents_of_type(chat, ContentToolRequestSearch)]
    assert "ggplot2 1.0.0 CRAN release date" in queries
    assert "ggplot2 release history" in queries
    results = _contents_of_type(chat, ContentToolResponseSearch)
    assert results and results[0].sources
    cites = _contents_of_type(chat, ContentCitation)
    assert cites, "expected ContentCitation items in turn contents"
    assert all(c.source and c.source.url for c in cites)

    # grounded_span: citations carry the answer-side span they ground, and
    # that span is drawn from the turn's own text.
    grounded_citations_found = False
    for turn in chat.get_turns():
        contents = turn.contents
        turn_cites = [c for c in contents if isinstance(c, ContentCitation)]
        if not turn_cites:
            continue
        answer = "".join(c.text for c in contents if isinstance(c, ContentText))
        for c in turn_cites:
            assert c.grounded_span is not None
            assert c.grounded_span in answer
            grounded_citations_found = True
        # Ordering fix: citations follow the grounded text they annotate.
        first_text = next(
            i for i, c in enumerate(contents) if isinstance(c, ContentText)
        )
        for i, c in enumerate(contents):
            if isinstance(c, ContentCitation):
                assert i > first_text
    assert grounded_citations_found


@pytest.mark.vcr
def test_google_web_search_streaming():
    chat = chat_func(model="gemini-2.5-flash")
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
    assert all(c.source and c.source.url for c in citations)


def _grounding_chunk(
    text: str, url: str, title: str, query: str, index: int | None = 0
):
    """One streamed chunk carrying text plus grounding metadata for that text."""
    from google.genai.types import (
        Candidate,
        GenerateContentResponse,
        GroundingChunk,
        GroundingChunkWeb,
        GroundingMetadata,
        GroundingSupport,
        Part,
        Segment,
    )
    from google.genai.types import (
        Content as GoogleContent,
    )

    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=GoogleContent(parts=[Part(text=text)], role="model"),
                index=index,
                grounding_metadata=GroundingMetadata(
                    web_search_queries=[query],
                    grounding_chunks=[
                        GroundingChunk(web=GroundingChunkWeb(uri=url, title=title))
                    ],
                    grounding_supports=[
                        GroundingSupport(
                            segment=Segment(text=text), grounding_chunk_indices=[0]
                        )
                    ],
                ),
            )
        ]
    )


def _plain_chunk(text: str, index: int | None = 0):
    from google.genai.types import (
        Candidate,
        GenerateContentResponse,
        Part,
    )
    from google.genai.types import (
        Content as GoogleContent,
    )

    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=GoogleContent(parts=[Part(text=text)], role="model"),
                index=index,
            )
        ]
    )


def test_google_grounding_metadata_matches_streamed_content():
    """Streaming and final-turn citations agree for Gemini's real response shape.

    Verified live against gemini-2.5-flash and gemini-3-flash-preview: grounding
    metadata arrives in exactly one chunk (the last), every candidate carries
    `index=0` so the chunks merge into a single candidate, and segment offsets
    are absolute into the whole answer.
    """
    provider = GoogleProvider(
        model="gemini-2.5-flash", api_key="dummy", name="Google/Gemini", kwargs=None
    )
    chunks = [
        _plain_chunk("ggplot2 1.0.0 "),
        _grounding_chunk("was released on 2014-05-21.", "https://a.com", "A", "q1"),
    ]

    streamed: list[Content] = []
    completion = None
    for chunk in chunks:
        completion = provider.stream_merge_chunks(completion, chunk)
        streamed.extend(provider.stream_content(chunk, completion))

    assert completion is not None
    turn = provider.stream_turn(completion, has_data_model=False)

    streamed_cites = [c for c in streamed if isinstance(c, ContentCitation)]
    turn_cites = [c for c in turn.contents if isinstance(c, ContentCitation)]
    assert len(streamed_cites) == 1
    assert [
        (c.source.url, c.grounded_span)
        for c in streamed_cites
        if isinstance(c.source, WebSource)
    ] == [
        (c.source.url, c.grounded_span)
        for c in turn_cites
        if isinstance(c.source, WebSource)
    ]


def test_google_late_grounding_metadata_not_dropped():
    """Metadata on a later candidate must survive into the final turn.

    Gemini has only ever been observed sending one metadata-bearing chunk, but
    when candidates arrive without an `index` they don't merge, and collapsing
    them first-wins silently discarded everything after the first.
    """
    provider = GoogleProvider(
        model="gemini-2.5-flash", api_key="dummy", name="Google/Gemini", kwargs=None
    )
    chunks = [
        _grounding_chunk("alpha", "https://a.com", "A", "q1", index=None),
        _grounding_chunk("beta", "https://b.com", "B", "q2", index=None),
    ]

    completion = None
    for chunk in chunks:
        completion = provider.stream_merge_chunks(completion, chunk)

    assert completion is not None
    turn = provider.stream_turn(completion, has_data_model=False)
    cited_urls = [
        c.source.url
        for c in turn.contents
        if isinstance(c, ContentCitation) and isinstance(c.source, WebSource)
    ]
    assert cited_urls == ["https://a.com", "https://b.com"]

    queries = [
        c.query for c in turn.contents if isinstance(c, ContentToolRequestSearch)
    ]
    assert queries == ["q1", "q2"]


def test_google_grounding_citations_without_a_web_source():
    """A chunk with no URL still grounds text; a bogus index grounds nothing."""
    grounding_metadata: GroundingMetadataDict = {
        "grounding_chunks": [
            {"retrieved_context": {"title": "internal doc"}},
            {"web": {"uri": "https://a.com", "title": "A"}},
        ],
        "grounding_supports": [
            {"segment": {"text": "sourceless span"}, "grounding_chunk_indices": [0]},
            {"segment": {"text": "linked span"}, "grounding_chunk_indices": [1]},
            {"segment": {"text": "out of range"}, "grounding_chunk_indices": [7]},
        ],
    }

    citations = google_grounding_citations(grounding_metadata)

    assert [(c.grounded_span, c.source) for c in citations] == [
        ("sourceless span", None),
        ("linked span", WebSource(url="https://a.com", title="A")),
    ]


@pytest.mark.vcr
def test_google_web_fetch_streaming():
    chat = chat_func(model="gemini-2.5-flash")
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

    assert result == fake_batch.model_dump(mode="json")
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


def test_google_batch_retrieve_handles_thought_signature_bytes():
    """Regression test for a real crash: reasoning-capable Gemini models attach
    a non-UTF-8 thought_signature byte blob to every response part, and a
    plain model_dump() leaves those as raw bytes, which crashes
    BatchState.model_dump_json() once persisted to the batch state file
    (reproduced against a live gemini-3.6-flash batch response)."""
    from unittest.mock import MagicMock

    from chatlas._batch_job import BatchState
    from chatlas._provider_google import GoogleProvider
    from google.genai import types

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)

    non_utf8_signature = bytes([0x12, 0xE9, 0x03, 0x0A, 0xE6, 0x03, 0x01])
    completed_batch = types.BatchJob(
        name="batches/123",
        state=types.JobState.JOB_STATE_SUCCEEDED,
        dest=types.BatchJobDestination(
            inlined_responses=[
                types.InlinedResponse(
                    response=types.GenerateContentResponse(
                        candidates=[
                            types.Candidate(
                                content=types.Content(
                                    parts=[
                                        types.Part(
                                            text="42",
                                            thought_signature=non_utf8_signature,
                                        )
                                    ]
                                ),
                                finish_reason=types.FinishReason.STOP,
                            )
                        ]
                    )
                )
            ]
        ),
    )

    # batch_poll() and batch_retrieve() both go through model_dump(mode="json"),
    # so their output must already be JSON-safe -- exercise that boundary here
    # rather than passing the raw pydantic-object dump directly.
    provider._client.batches.get = MagicMock(return_value=completed_batch)
    polled = provider.batch_poll({"name": "batches/123"})

    # This is the exact call that crashed in production: persisting the batch
    # dict into the on-disk job state.
    BatchState(
        version=1,
        stage="retrieving",
        batch=polled,
        results=[],
        started_at=0,
        hash={"provider": "x", "model": "x", "prompts": "x", "user_turns": "x"},
    ).model_dump_json()

    results = provider.batch_retrieve(polled)
    assert len(results) == 1

    turn = provider.batch_result_turn(results[0], has_data_model=False)
    assert turn is not None
    assert turn.text == "42"


@pytest.mark.vcr
@retry_gemini_call
def test_google_mixed_tools_end_to_end():
    """Custom + built-in tools can be combined on Gemini 3+ (#1054)."""

    def double(x: float) -> float:
        """Double a number."""
        return x * 2

    chat = chat_func()
    chat.register_tool(double)
    chat.register_tool(tool_web_search())

    response = chat.chat("What is double 21?")
    assert "42" in str(response)


def _double_tool():
    from chatlas._tools import Tool

    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    return Tool.from_func(double)


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.5-flash",
        # `list_models()` surfaces IDs prefixed with "models/" (#1054)
        "models/gemini-3.5-flash",
    ],
)
def test_google_mixed_tools_sets_tool_config_on_gemini_3_plus(model: str):
    """Mixing custom + built-in tools requires an explicit opt-in on Gemini 3+ (#1054)."""
    from chatlas._provider_google import GoogleProvider
    from chatlas._turn import user_turn

    provider = GoogleProvider(model=model, api_key="dummy", kwargs=None)
    tools = {"double": _double_tool(), "web_search": tool_web_search()}

    kwargs = provider._chat_perform_args(turns=[user_turn("hi")], tools=tools)

    tool_config = kwargs["config"].tool_config
    assert tool_config is not None
    assert tool_config.include_server_side_tool_invocations is True


def test_google_mixed_tools_skips_tool_config_on_vertex():
    """Vertex AI rejects `include_server_side_tool_invocations` outright, so it must
    never be set there, regardless of whether tools are mixed."""
    from chatlas._provider_google import GoogleProvider
    from chatlas._turn import user_turn

    provider = GoogleProvider(
        model="gemini-3.5-flash",
        api_key="dummy",
        name="Google/Vertex",
        kwargs=None,
    )
    tools = {"double": _double_tool(), "web_search": tool_web_search()}

    kwargs = provider._chat_perform_args(turns=[user_turn("hi")], tools=tools)

    assert kwargs["config"].tool_config is None


def test_google_mixed_tools_skips_tool_config_on_older_model():
    """Older Gemini models don't support mixing at all; setting the flag there
    replaces a clear "pick one" API error with a confusing one, so leave it unset."""
    from chatlas._provider_google import GoogleProvider
    from chatlas._turn import user_turn

    provider = GoogleProvider(model="gemini-2.5-flash", api_key="dummy", kwargs=None)
    tools = {"double": _double_tool(), "web_search": tool_web_search()}

    kwargs = provider._chat_perform_args(turns=[user_turn("hi")], tools=tools)

    assert kwargs["config"].tool_config is None


def test_google_mixed_tools_preserves_existing_tool_config():
    """Setting the opt-in flag must not clobber a user-supplied `tool_config`."""
    from chatlas._provider_google import GoogleProvider
    from chatlas._turn import user_turn
    from google.genai.types import (
        FunctionCallingConfig,
        FunctionCallingConfigMode,
        GenerateContentConfig,
        ToolConfig,
    )

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)
    tools = {"double": _double_tool(), "web_search": tool_web_search()}

    existing_config = GenerateContentConfig(
        tool_config=ToolConfig(
            function_calling_config=FunctionCallingConfig(
                mode=FunctionCallingConfigMode.ANY
            )
        )
    )

    kwargs = provider._chat_perform_args(
        turns=[user_turn("hi")],
        tools=tools,
        kwargs={"config": existing_config},
    )

    tool_config = kwargs["config"].tool_config
    assert tool_config is not None
    assert tool_config.include_server_side_tool_invocations is True
    assert tool_config.function_calling_config.mode == FunctionCallingConfigMode.ANY


def test_google_tool_config_not_set_when_tools_not_mixed():
    """The opt-in is only needed when both custom and built-in tools are present."""
    from chatlas._provider_google import GoogleProvider
    from chatlas._turn import user_turn

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)

    only_custom = {"double": _double_tool()}
    kwargs = provider._chat_perform_args(turns=[user_turn("hi")], tools=only_custom)
    assert kwargs["config"].tool_config is None

    only_builtin = {"web_search": tool_web_search()}
    kwargs = provider._chat_perform_args(turns=[user_turn("hi")], tools=only_builtin)
    assert kwargs["config"].tool_config is None


def truncated_structured_completion() -> dict:
    """A structured-output response cut short by the token limit (gh-315)."""
    return {
        "candidates": [
            {
                "index": 0,
                "finish_reason": "MAX_TOKENS",
                "content": {
                    "role": "model",
                    "parts": [{"text": '{"comments": [{"body": "trunc'}],
                },
            }
        ]
    }


def test_google_truncated_structured_output_errors_helpfully():
    provider = cast(GoogleProvider, chat_func().provider)

    with pytest.raises(ValueError, match="max_tokens"):
        provider._as_turn(truncated_structured_completion(), has_data_model=True)


def test_google_truncated_plain_text_still_returns_a_turn():
    provider = cast(GoogleProvider, chat_func().provider)

    turn = provider._as_turn(truncated_structured_completion(), has_data_model=False)

    assert turn.text == '{"comments": [{"body": "trunc'
    assert turn.finish_reason == "max_tokens"
