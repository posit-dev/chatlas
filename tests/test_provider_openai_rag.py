import pytest

from chatlas import ChatOpenAI
from chatlas._content import ContentCitation, ContentJson
from chatlas.types import DocumentSource

from .conftest import assert_citations_grounded
from .test_provider_anthropic_rag import KeywordStore


def chat_func(**kwargs):
    return ChatOpenAI(**kwargs)


@pytest.mark.vcr
def test_openai_rag_tool_mode_citations():
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
def test_openai_rag_streaming_yields_prose_not_json():
    chat = chat_func()
    chat.rag.register_store(KeywordStore(), top_k=2)
    chunks = list(chat.stream("How does Flurbo stream responses?", content="all"))
    text = "".join(c for c in chunks if isinstance(c, str))
    assert '"segments"' not in text
    assert any(isinstance(c, ContentCitation) for c in chunks)
