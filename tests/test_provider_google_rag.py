import pytest

from chatlas import ChatGoogle
from chatlas._content import ContentCitation, ContentJson
from chatlas.types import DocumentSource

from .conftest import assert_citations_grounded
from .test_provider_anthropic_rag import KeywordStore


def chat_func(**kwargs):
    return ChatGoogle(**kwargs)


# Both tests below are xfail (not skipped): the request/response cycle works
# and is recorded correctly, but chatlas's Google streaming turn assembly has
# a real, pre-existing bug that this is the first end-to-end test to exercise.
#
# Root cause (verified independently of RAG, e.g. via plain
# `ChatGoogle().stream("...", data_model=SomeModel)`): Gemini's streamed
# `content.parts` dicts, unlike `candidates`, carry no `"index"` key, so
# `merge_lists`/`merge_dicts` (chatlas/_merge.py) append successive text-delta
# parts as separate list entries instead of concatenating their text. Then
# `GoogleProvider._as_turn` (chatlas/_provider_google.py) JSON-decodes each
# part's `text` independently when `has_data_model=True`, so any structured
# response that spans more than one SSE chunk -- which is the normal case,
# not a RAG-specific one -- raises `orjson.JSONDecodeError` on the first,
# incomplete fragment. This affects any `has_data_model=True` streamed Google
# turn, not just the RAG hand-rolled tier's segments schema.
_GOOGLE_MERGE_BUG_REASON = (
    "chatlas bug: GoogleProvider streaming turn assembly does not concatenate "
    "multi-chunk structured-output text (Gemini parts lack an 'index' key for "
    "merge_lists to match on), so has_data_model=True + streaming raises "
    "orjson.JSONDecodeError on real API responses. See test file comment."
)


@pytest.mark.vcr
@pytest.mark.xfail(reason=_GOOGLE_MERGE_BUG_REASON, strict=True)
def test_google_rag_tool_mode_citations():
    chat = chat_func()
    chat.rag.register_store(KeywordStore(), top_k=2)
    chat.chat("How does Flurbo stream responses?", echo="none")

    turn = chat.get_last_turn(role="assistant")
    assert not any(isinstance(c, ContentJson) for c in turn.contents)
    citations = [c for c in turn.contents if isinstance(c, ContentCitation)]
    assert citations
    assert any(
        isinstance(c.source, DocumentSource)
        and c.source.id == "kb://flurbo/streaming"
        for c in citations
    )
    assert_citations_grounded(chat)


@pytest.mark.vcr
@pytest.mark.xfail(reason=_GOOGLE_MERGE_BUG_REASON, strict=True)
def test_google_rag_streaming_yields_prose_not_json():
    chat = chat_func()
    chat.rag.register_store(KeywordStore(), top_k=2)
    chunks = list(chat.stream("How does Flurbo stream responses?", content="all"))
    text = "".join(c for c in chunks if isinstance(c, str))
    assert '"segments"' not in text
    assert any(isinstance(c, ContentCitation) for c in chunks)
