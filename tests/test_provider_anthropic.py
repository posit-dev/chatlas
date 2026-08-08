from typing import Literal, cast

import httpx
import pytest
from anthropic.types import (
    CitationCharLocation,
    CitationsDelta,
    CitationsWebSearchResultLocation,
    DocumentBlock,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageStartEvent,
    TextBlock,
    TextDelta,
    Usage,
    WebFetchBlock,
    WebFetchToolResultBlock,
)
from chatlas import (
    AssistantTurn,
    ChatAnthropic,
    UserTurn,
    content_image_file,
    tool_web_fetch,
    tool_web_search,
)
from chatlas._content import (
    ContentDocument,
    ContentImageInline,
    ContentPDF,
    ContentUploaded,
)
from chatlas._provider_anthropic import (
    _ANTHROPIC_FINISH_REASON_MAP,
    AnthropicProvider,
    anthropic_fetch_result,
    serving_model,
)
from chatlas._provider_anthropic import (
    normalize_finish_reason as anthropic_normalize_finish_reason,
)
from chatlas.types import (
    ContentCitation,
    ContentText,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
)
from pydantic import BaseModel, Field

from .conftest import (
    assert_data_extraction,
    assert_document_local,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_pdf_local,
    assert_pdf_remote,
    assert_tool_web_fetch,
    assert_tool_web_search,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
    retry_api_call,
)


def test_normalize_finish_reason_maps_known_reasons():
    assert anthropic_normalize_finish_reason("end_turn") == "success"
    assert anthropic_normalize_finish_reason("max_tokens") == "max_tokens"
    assert anthropic_normalize_finish_reason("stop_sequence") == "stop_sequence"
    assert (
        anthropic_normalize_finish_reason("model_context_window_exceeded")
        == "context_window"
    )
    assert anthropic_normalize_finish_reason("refusal") == "content_filter"
    assert anthropic_normalize_finish_reason("tool_use") == "tool_use"


def test_normalize_finish_reason_maps_tool_use_explicitly():
    # tool_use must be an explicit mapping, not an incidental passthrough of an
    # unknown reason, so it isn't confused with a truly unrecognized reason.
    assert "tool_use" in _ANTHROPIC_FINISH_REASON_MAP


def test_normalize_finish_reason_passes_through_unknown():
    assert anthropic_normalize_finish_reason("some_new_reason") == "some_new_reason"


def test_normalize_finish_reason_handles_none():
    assert anthropic_normalize_finish_reason(None) is None


def chat_func(system_prompt: str = "", **kwargs):
    return ChatAnthropic(
        system_prompt=system_prompt,
        model="claude-haiku-4-5-20251001",
        **kwargs,
    )


@pytest.mark.vcr
def test_anthropic_simple_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)
    assert turn.finish_reason == "success"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = chat_func(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    foo = await chat.stream_async("What is 1 + 1?")
    async for x in foo:
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "success"


@pytest.mark.vcr
def test_anthropic_respects_turns_interface():
    assert_turns_system(chat_func)
    assert_turns_existing(chat_func)


@pytest.mark.vcr
@retry_api_call
def test_anthropic_tool_variations():
    assert_tools_simple(chat_func)
    assert_tools_simple_stream_content(chat_func)
    assert_tools_sequential(chat_func, total_calls=6)


@pytest.mark.vcr
@retry_api_call
def test_anthropic_tool_variations_parallel():
    assert_tools_parallel(chat_func)


@pytest.mark.vcr
@pytest.mark.asyncio
@retry_api_call
async def test_anthropic_tool_variations_async():
    await assert_tools_async(chat_func)


@pytest.mark.vcr
def test_anthropic_web_fetch():
    def chat_fun(**kwargs):
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            kwargs={"default_headers": {"anthropic-beta": "web-fetch-2025-09-10"}},
            **kwargs,
        )

    chat = assert_tool_web_fetch(chat_fun, tool_web_fetch())
    fetched = [
        c
        for turn in chat.get_turns()
        for c in turn.contents
        if isinstance(c, ContentToolResponseFetch)
    ]
    assert fetched and fetched[0].url
    assert fetched[0].status == "success"


@pytest.mark.vcr
def test_anthropic_web_search():
    chat = assert_tool_web_search(chat_func, tool_web_search())
    results = [
        c
        for turn in chat.get_turns()
        for c in turn.contents
        if isinstance(c, ContentToolResponseSearch)
    ]
    assert results and results[0].sources
    assert all(s.url for s in results[0].sources)
    assert any(s.title for s in results[0].sources)


@pytest.mark.vcr
def test_anthropic_web_search_streaming():
    chat = chat_func()
    chat.register_tool(tool_web_search())
    items = list(
        chat.stream(
            "When was ggplot2 1.0.0 released to CRAN? Answer in YYYY-MM-DD format.",
            content="all",
        )
    )
    answer = "".join(x for x in items if isinstance(x, str))
    cites = [x for x in items if isinstance(x, ContentCitation)]
    results = [x for x in items if isinstance(x, ContentToolResponseSearch)]
    reqs = [x for x in items if isinstance(x, ContentToolRequestSearch)]
    assert results and results[0].sources
    assert reqs and reqs[0].query
    assert cites and all(c.source and c.source.url for c in cites)
    assert any(c.cited_quote for c in cites)  # Anthropic supplies source-side quotes
    for c in cites:
        assert c.grounded_span is not None
        assert c.grounded_span in answer  # answer-side span
        assert c.extra is not None  # raw payload retained
    # interleaved: at least one citation is not the very last item
    cite_idx = [i for i, x in enumerate(items) if isinstance(x, ContentCitation)]
    assert cite_idx and min(cite_idx) < len(items) - 1


@pytest.mark.vcr
def test_anthropic_web_search_citations():
    """Test that citations from web search are ContentCitation items in the turn."""
    chat = chat_func()
    chat.register_tool(tool_web_search())
    chat.chat("When was ggplot2 1.0.0 released to CRAN? Answer in YYYY-MM-DD format.")

    turn = chat.get_last_turn()
    assert turn is not None

    answer = "".join(c.text for c in turn.contents if isinstance(c, ContentText))
    cites = [c for c in turn.contents if isinstance(c, ContentCitation)]
    assert cites, "expected ContentCitation items in turn contents"
    assert all(c.source and c.source.url for c in cites)
    for c in cites:
        assert c.grounded_span is not None
        assert c.grounded_span in answer  # answer-side span
        assert c.extra is not None  # raw payload retained


def test_anthropic_web_fetch_citation_uses_document_index_for_source():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )
    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            web_fetch_result("https://first.example", "First document"),
            web_fetch_result("https://second.example", "Second document"),
            TextBlock(
                type="text",
                text="Grounded answer",
                citations=[
                    CitationCharLocation(
                        type="char_location",
                        cited_text="second source evidence",
                        document_index=1,
                        document_title="Second document",
                        start_char_index=0,
                        end_char_index=22,
                    )
                ],
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    turn = provider._as_turn(completion)
    citation = next(c for c in turn.contents if isinstance(c, ContentCitation))

    assert citation.source is not None
    assert citation.source.url == "https://second.example"
    assert citation.source.title == "Second document"


def test_anthropic_web_fetch_citation_stream_uses_document_index_for_source():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )
    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            web_fetch_result("https://first.example", "First document"),
            web_fetch_result("https://second.example", "Second document"),
            TextBlock(
                type="text",
                text="Grounded answer",
                citations=[
                    CitationCharLocation(
                        type="char_location",
                        cited_text="second source evidence",
                        document_index=1,
                        document_title="Second document",
                        start_char_index=0,
                        end_char_index=22,
                    )
                ],
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    contents = provider.stream_content(
        RawContentBlockStopEvent(type="content_block_stop", index=2),
        completion,
    )
    citation = next(c for c in contents if isinstance(c, ContentCitation))

    assert citation.source is not None
    assert citation.source.url == "https://second.example"
    assert citation.source.title == "Second document"


def test_anthropic_citation_with_url_keeps_its_own_source():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )
    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            TextBlock(
                type="text",
                text="Grounded answer",
                citations=[
                    CitationsWebSearchResultLocation(
                        type="web_search_result_location",
                        cited_text="search evidence",
                        encrypted_index="x",
                        title="Search result",
                        url="https://search.example",
                    )
                ],
            )
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    turn = provider._as_turn(completion)
    citation = next(c for c in turn.contents if isinstance(c, ContentCitation))

    assert citation.source is not None
    assert citation.source.url == "https://search.example"
    assert citation.source.title == "Search result"


def test_anthropic_web_fetch_citation_with_invalid_document_index_has_no_source():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )
    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            web_fetch_result("https://first.example", "First document"),
            TextBlock(
                type="text",
                text="Grounded answer",
                citations=[
                    CitationCharLocation(
                        type="char_location",
                        cited_text="missing source evidence",
                        document_index=1,
                        document_title="Missing document",
                        start_char_index=0,
                        end_char_index=23,
                    )
                ],
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    turn = provider._as_turn(completion)
    citation = next(c for c in turn.contents if isinstance(c, ContentCitation))

    assert citation.source is None


def test_anthropic_web_fetch_citation_resolves_across_turns():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )

    turn_1_fetch = web_fetch_result("https://a.example", "First document")
    prior_turns = [
        UserTurn("Look at https://a.example"),
        AssistantTurn(
            [
                anthropic_fetch_result(turn_1_fetch),
            ],
            finish_reason="stop",
        ),
    ]

    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            web_fetch_result("https://b.example", "Second document"),
            TextBlock(
                type="text",
                text="Comparing both pages",
                citations=[
                    CitationCharLocation(
                        type="char_location",
                        cited_text="claim about the first page",
                        document_index=0,
                        document_title="First document",
                        start_char_index=0,
                        end_char_index=27,
                    ),
                    CitationCharLocation(
                        type="char_location",
                        cited_text="claim about the second page",
                        document_index=1,
                        document_title="Second document",
                        start_char_index=28,
                        end_char_index=56,
                    ),
                ],
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    turn = provider._as_turn(completion, turns=prior_turns)
    citations = [c for c in turn.contents if isinstance(c, ContentCitation)]

    assert citations[0].source is not None
    assert citations[0].source.url == "https://a.example"
    assert citations[1].source is not None
    assert citations[1].source.url == "https://b.example"


def test_anthropic_web_fetch_citation_accounts_for_attached_document():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )

    prior_turns = [
        UserTurn(
            [
                ContentPDF(data=b"%PDF-1.4 fake", filename="report.pdf"),
                "Compare this report with https://site.example",
            ]
        ),
    ]

    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            web_fetch_result("https://site.example", "Site page"),
            TextBlock(
                type="text",
                text="The report and the site disagree",
                citations=[
                    CitationCharLocation(
                        type="char_location",
                        cited_text="claim from the PDF",
                        document_index=0,
                        document_title="report.pdf",
                        start_char_index=0,
                        end_char_index=19,
                    ),
                    CitationCharLocation(
                        type="char_location",
                        cited_text="claim from the site",
                        document_index=1,
                        document_title="Site page",
                        start_char_index=20,
                        end_char_index=40,
                    ),
                ],
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    turn = provider._as_turn(completion, turns=prior_turns)
    citations = [c for c in turn.contents if isinstance(c, ContentCitation)]

    # The PDF has no URL to attribute -- must stay unresolved, not
    # borrow the fetched page's URL.
    assert citations[0].source is None
    # The fetch keeps its own, correctly-offset URL.
    assert citations[1].source is not None
    assert citations[1].source.url == "https://site.example"


def test_anthropic_web_fetch_citation_uses_pdf_url_when_present():
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )

    prior_turns = [
        UserTurn(
            [
                ContentPDF(
                    url="https://docs.example/report.pdf", filename="report.pdf"
                ),
                "Summarize this report.",
            ]
        ),
    ]

    completion = Message(
        id="msg",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[
            TextBlock(
                type="text",
                text="The report says X",
                citations=[
                    CitationCharLocation(
                        type="char_location",
                        cited_text="claim from the PDF",
                        document_index=0,
                        document_title="report.pdf",
                        start_char_index=0,
                        end_char_index=18,
                    ),
                ],
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    turn = provider._as_turn(completion, turns=prior_turns)
    citation = next(c for c in turn.contents if isinstance(c, ContentCitation))

    assert citation.source is not None
    assert citation.source.url == "https://docs.example/report.pdf"


def web_fetch_result(url: str, title: str) -> WebFetchToolResultBlock:
    return WebFetchToolResultBlock(
        type="web_fetch_tool_result",
        tool_use_id=f"fetch-{url}",
        content=WebFetchBlock(
            type="web_fetch_result",
            url=url,
            content=DocumentBlock(
                type="document",
                title=title,
                source={
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "",
                },
            ),
        ),
    )


def test_anthropic_concurrent_streams_dont_share_state():
    """Two streams on one provider must not cross-contaminate citations.

    `Chat.__deepcopy__` shares a single provider across forked chats (e.g. every
    `eval_inspect` sample), and those forks stream concurrently, so anything
    `stream_content` needs across chunks has to live with the stream -- not on
    the provider.
    """
    provider = AnthropicProvider(
        model="claude-sonnet-4-5", api_key="dummy", kwargs=None
    )

    def events(text: str, url: str):
        return [
            RawMessageStartEvent(
                type="message_start",
                message=Message(
                    id="msg",
                    type="message",
                    role="assistant",
                    model="claude-sonnet-4-5",
                    content=[],
                    stop_reason=None,
                    stop_sequence=None,
                    usage=Usage(input_tokens=1, output_tokens=1),
                ),
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                # Anthropic sends `citations: []` on citation-bearing blocks
                content_block=TextBlock(type="text", text="", citations=[]),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text=text),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=CitationsDelta(
                    type="citations_delta",
                    citation=CitationsWebSearchResultLocation(
                        type="web_search_result_location",
                        cited_text=f"quote from {url}",
                        encrypted_index="x",
                        title="t",
                        url=url,
                    ),
                ),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
        ]

    stream_a = events("alpha answer", "https://a.com")
    stream_b = events("beta answer", "https://b.com")

    # Interleave the two streams chunk-for-chunk, as concurrent samples would.
    completions: dict[str, object] = {"a": None, "b": None}
    cites: dict[str, list[ContentCitation]] = {"a": [], "b": []}
    for chunk_a, chunk_b in zip(stream_a, stream_b):
        for key, chunk in (("a", chunk_a), ("b", chunk_b)):
            completions[key] = provider.stream_merge_chunks(completions[key], chunk)
            cites[key].extend(
                c
                for c in provider.stream_content(chunk, completions[key])
                if isinstance(c, ContentCitation)
            )

    assert len(cites["a"]) == 1
    assert len(cites["b"]) == 1
    cite_a, cite_b = cites["a"][0], cites["b"][0]
    assert cite_a.source and cite_a.source.url == "https://a.com"
    assert cite_a.grounded_span == "alpha answer"
    assert cite_b.source and cite_b.source.url == "https://b.com"
    assert cite_b.grounded_span == "beta answer"


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(chat_func)


@pytest.mark.vcr
def test_stream_with_data_model():
    from chatlas._content import ContentJson

    chat = chat_func()

    class Person(BaseModel):
        name: str
        age: int

    chunks = list(chat.stream("John, age 15, won first prize", data_model=Person))
    result = "".join(chunks)
    person = Person.model_validate_json(result)
    assert person == Person(name="John", age=15)

    turn = chat.get_last_turn()
    assert turn is not None
    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentJson)
    assert turn.contents[0].value == {"name": "John", "age": 15}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_stream_async_with_data_model():
    from chatlas._content import ContentJson

    chat = chat_func()

    class Person(BaseModel):
        name: str
        age: int

    chunks = [
        chunk
        async for chunk in await chat.stream_async(
            "John, age 15, won first prize", data_model=Person
        )
    ]
    result = "".join(chunks)
    person = Person.model_validate_json(result)
    assert person == Person(name="John", age=15)

    turn = chat.get_last_turn()
    assert turn is not None
    assert len(turn.contents) == 1
    assert isinstance(turn.contents[0], ContentJson)
    assert turn.contents[0].value == {"name": "John", "age": 15}


@pytest.mark.vcr
@retry_api_call
def test_anthropic_images():
    assert_images_inline(chat_func)
    assert_images_remote(chat_func)


@pytest.mark.vcr
def test_anthropic_pdfs():
    assert_pdf_local(chat_func)


@pytest.mark.vcr
def test_anthropic_pdf_url():
    assert_pdf_remote(chat_func)


@pytest.mark.vcr
def test_anthropic_document():
    assert_document_local(chat_func)


def test_anthropic_uploaded_document_block():
    c = ContentUploaded(id="file_1", mime_type="application/pdf", provider="anthropic")
    block = AnthropicProvider._as_content_block(c)
    assert block["type"] == "document"
    assert block["source"] == {"type": "file", "file_id": "file_1"}


def test_anthropic_uploaded_image_block():
    c = ContentUploaded(id="img_1", mime_type="image/png", provider="anthropic")
    block = AnthropicProvider._as_content_block(c)
    assert block["type"] == "image"
    assert block["source"] == {"type": "file", "file_id": "img_1"}


def test_anthropic_uploaded_cross_provider_raises():
    c = ContentUploaded(id="file_1", mime_type="application/pdf", provider="openai")
    with pytest.raises(ValueError, match="uploaded to provider 'openai'"):
        AnthropicProvider._as_content_block(c)


def test_anthropic_uploaded_triggers_beta_header():
    provider = AnthropicProvider(model="claude-sonnet-4-6")
    turn = UserTurn(
        [
            ContentUploaded(
                id="file_1", mime_type="application/pdf", provider="anthropic"
            )
        ]
    )
    args = provider._chat_perform_args(
        stream=False, turns=[turn], tools={}, data_model=None
    )
    assert args["extra_headers"]["anthropic-beta"] == "files-api-2025-04-14"


def test_anthropic_no_uploaded_omits_beta_header():
    provider = AnthropicProvider(model="claude-sonnet-4-6")
    turn = UserTurn(["hello"])
    args = provider._chat_perform_args(
        stream=False, turns=[turn], tools={}, data_model=None
    )
    assert "anthropic-beta" not in (args.get("extra_headers") or {})


def test_anthropic_token_count_args_keeps_beta_header():
    provider = AnthropicProvider(model="claude-sonnet-4-6")
    turn = UserTurn(
        [
            ContentUploaded(
                id="file_1", mime_type="application/pdf", provider="anthropic"
            )
        ]
    )
    args = provider._token_count_args(
        [turn],
        tools={},
        data_model=None,
    )
    assert args["extra_headers"]["anthropic-beta"] == "files-api-2025-04-14"


def test_anthropic_pdf_with_url_uses_url_source_without_downloading():
    c = ContentPDF(filename="a.pdf", url="https://example.com/a.pdf")
    block = AnthropicProvider._as_content_block(c)
    assert block["type"] == "document"
    assert block["source"] == {"type": "url", "url": "https://example.com/a.pdf"}


def test_anthropic_pdf_with_data_uses_base64_source():
    c = ContentPDF(data=b"%PDF-1.4", filename="a.pdf")
    block = AnthropicProvider._as_content_block(c)
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "application/pdf"


def test_anthropic_document_coerces_text_mime_type_to_plain_text():
    c = ContentDocument(data=b"a,b\n1,2\n", filename="data.csv", mime_type="text/csv")
    block = AnthropicProvider._as_content_block(c)
    assert block["type"] == "document"
    assert block["source"] == {
        "type": "text",
        "media_type": "text/plain",
        "data": "a,b\n1,2\n",
    }
    # The only surviving hint of what the document actually is.
    assert block["title"] == "data.csv"


def test_anthropic_document_rejects_binary_office_types():
    c = ContentDocument(
        data=b"PK\x03\x04",
        filename="report.docx",
        mime_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
    )
    with pytest.raises(ValueError, match="doesn't support"):
        AnthropicProvider._as_content_block(c)


def test_anthropic_document_rejects_undecodable_bytes_with_clear_error():
    c = ContentDocument(
        data=b"\xff\xfe\x00", filename="weird.txt", mime_type="text/plain"
    )
    with pytest.raises(ValueError, match="UTF-8"):
        AnthropicProvider._as_content_block(c)


def test_anthropic_rejects_heic_images():
    c = ContentImageInline(image_content_type="image/heic", data="abcd")
    with pytest.raises(ValueError, match="image/heic"):
        AnthropicProvider._as_content_block(c)


@pytest.mark.parametrize(
    "content_type", ["image/png", "image/jpeg", "image/webp", "image/gif"]
)
def test_anthropic_image_preserves_media_type(content_type):
    c = ContentImageInline(image_content_type=content_type, data="abcd")
    block = AnthropicProvider._as_content_block(c)
    assert block["source"] == {
        "type": "base64",
        "media_type": content_type,
        "data": "abcd",
    }


@pytest.mark.vcr
def test_anthropic_empty_response():
    chat = chat_func()
    chat.chat("Respond with only two blank lines")
    resp = chat.chat("What's 1+1? Just give me the number")
    assert "2" == str(resp).strip()


@pytest.mark.vcr
def test_anthropic_image_tool(test_images_dir):
    def get_picture():
        "Returns an image"
        # Local copy of https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png
        # Using resize='none' to avoid platform-specific encoding differences
        return content_image_file(test_images_dir / "dice.png", resize="none")

    chat = chat_func()
    chat.register_tool(get_picture)

    res = chat.chat(
        "You have a tool called 'get_picture' available to you. "
        "When called, it returns an image. "
        "Tell me what you see in the image."
    )

    assert "dice" in res.get_content()


def test_anthropic_custom_http_client():
    ChatAnthropic(kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_anthropic_list_models():
    assert_list_models(chat_func)


def test_anthropic_removes_empty_assistant_turns():
    """Test that empty assistant turns are dropped to avoid API errors."""
    chat = chat_func()
    chat.set_turns(
        [
            UserTurn("Don't say anything"),
            AssistantTurn([]),
        ]
    )

    # Get the message params that would be sent to the API
    provider = cast(AnthropicProvider, chat.provider)
    turns_json = provider._as_message_params(chat.get_turns())

    # Should only have the user turn, not the empty assistant turn
    assert len(turns_json) == 1
    assert turns_json[0]["role"] == "user"
    assert turns_json[0]["content"][0]["text"] == "Don't say anything"  # type: ignore


@pytest.mark.vcr
def test_anthropic_nested_data_model_extraction():
    """
    Test that nested Pydantic models work for structured data extraction.

    This is a regression test for issue #100 where data extraction failed with
    nested models because $defs was placed inside the 'data' property instead
    of at the root of input_schema, breaking $ref JSON pointer references.

    See: https://github.com/posit-dev/chatlas/issues/100
    """

    # Models from issue #100
    class Classification(BaseModel):
        name: Literal[
            "Politics", "Sports", "Technology", "Entertainment", "Business", "Other"
        ] = Field(description="The category name")
        score: float = Field(
            description="The classification score for the category, ranging from 0.0 to 1.0."
        )

    class Classifications(BaseModel):
        """Array of classification results. The scores should sum to 1."""

        classifications: list[Classification]

    text = (
        "The new quantum computing breakthrough could revolutionize the tech industry."
    )

    chat = chat_func(system_prompt="You are a friendly but terse assistant.")
    data = chat.chat_structured(text, data_model=Classifications)

    # Verify we got a valid response with the nested structure
    assert isinstance(data, Classifications)
    assert len(data.classifications) > 0

    # Check that at least one classification is Technology (the obvious choice)
    categories = [c.name for c in data.classifications]
    assert "Technology" in categories, f"Expected 'Technology' in {categories}"

    # Verify scores are valid floats between 0 and 1
    for classification in data.classifications:
        assert 0.0 <= classification.score <= 1.0, (
            f"Score {classification.score} should be between 0 and 1"
        )


def test_anthropic_reasoning_int_budget():
    """An int `reasoning` maps to a fixed thinking budget (regression)."""
    chat = ChatAnthropic(reasoning=2048)
    assert chat.kwargs_chat == {"thinking": {"type": "enabled", "budget_tokens": 2048}}


def test_anthropic_reasoning_effort_string():
    """A string `reasoning` enables adaptive thinking via output_config (#997)."""
    chat = ChatAnthropic(reasoning="high")
    assert chat.kwargs_chat == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_anthropic_adaptive_effort_merges_with_structured_output():
    """When extracting data, adaptive effort merges into the native output_config."""

    class Person(BaseModel):
        name: str

    provider = AnthropicProvider(
        model="claude-sonnet-4-6", structured_output_mode="native"
    )
    args = provider._chat_perform_args(
        stream=False,
        turns=[],
        tools={},
        data_model=Person,
        kwargs={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
    )
    output_config = args["output_config"]
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"


@pytest.mark.vcr
def test_anthropic_token_count_complete_exceeds_new():
    chat = ChatAnthropic(
        model="claude-sonnet-4-6", system_prompt="You are a terse assistant."
    )
    chat.set_turns(
        [
            UserTurn("an earlier question with some length to it"),
            AssistantTurn("an earlier answer", tokens=(10, 5, 0)),
        ]
    )
    new_only = chat.token_count("and one more question", include="new")
    complete = chat.token_count("and one more question", include="complete")

    assert new_only > 0
    assert complete > new_only


def truncated_structured_message() -> "Message":
    """A structured-output response cut short by the `max_tokens` limit (gh-315)."""
    return Message.construct(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-haiku-4-5-20251001",
        stop_reason="max_tokens",
        stop_sequence=None,
        content=[TextBlock(type="text", text='{"comments": [{"body": "trunc')],
        usage=None,
    )


def test_anthropic_truncated_structured_output_errors_helpfully():
    provider = cast(AnthropicProvider, chat_func().provider)

    with pytest.raises(ValueError, match="max_tokens"):
        provider._as_turn(truncated_structured_message(), has_data_model=True)


def test_anthropic_truncated_plain_text_still_returns_a_turn():
    # Without a data model there's nothing to parse, so the partial text is
    # still worth handing back -- `Chat` warns about it instead.
    provider = cast(AnthropicProvider, chat_func().provider)

    turn = provider._as_turn(truncated_structured_message(), has_data_model=False)

    assert turn.text == '{"comments": [{"body": "trunc'
    assert turn.finish_reason == "max_tokens"


def fallback_message(
    *, requested_model: str = "claude-fable-5", served_model: str = "claude-opus-4-8"
) -> "Message":
    """A response where a server-side refusal fallback swapped the serving model.

    https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
    """
    return Message.construct(
        id="msg_1",
        type="message",
        role="assistant",
        model=requested_model,
        stop_reason="end_turn",
        stop_sequence=None,
        content=[
            TextBlock.construct(
                type="fallback",
                from_={"model": requested_model},
                to={"model": served_model},
            ),
            TextBlock(type="text", text="ok"),
        ],
        usage=Usage(input_tokens=1000, output_tokens=50),
    )


def test_anthropic_fallback_block_excluded_from_contents():
    provider = cast(AnthropicProvider, chat_func().provider)

    turn = provider._as_turn(fallback_message(), has_data_model=False)

    assert turn.text == "ok"
    assert len(turn.contents) == 1


def test_serving_model_prefers_last_fallback_blocks_to_model():
    assert serving_model(fallback_message()) == "claude-opus-4-8"

    no_fallback = Message.construct(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-fable-5",
        stop_reason="end_turn",
        stop_sequence=None,
        content=[TextBlock(type="text", text="hi")],
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    assert serving_model(no_fallback) == "claude-fable-5"


def test_anthropic_value_cost_prices_fallback_at_serving_models_rate():
    provider = AnthropicProvider(model="claude-fable-5")

    completion = fallback_message()
    tokens = provider.value_tokens(completion)
    cost = provider.value_cost(completion, tokens)

    # opus-4-8 rates ($5/$25 per 1M), not fable-5's ($10/$50).
    assert cost == pytest.approx((1000 * 5 + 50 * 25) / 1e6)
