import pytest

from chatlas import ChatAnthropic
from chatlas._content import ContentCitation
from chatlas.types import DocumentSource

from .conftest import assert_citations_grounded

# Invented facts force retrieval (the model can't know them) and make
# citation grounding unambiguous.
FLURBO_CHUNKS = [
    {
        "text": "The Flurbo framework was created in 2019 by Ada Quist. "
        "Its default port is 7113.",
        "origin": "kb://flurbo/intro",
        "context": "Flurbo > Introduction",
    },
    {
        "text": "Flurbo streams responses via the flb.stream() generator, "
        "which yields FlurboChunk objects.",
        "origin": "kb://flurbo/streaming",
        "context": "Flurbo > Streaming",
    },
]


class DictChunk:
    def __init__(self, d: dict):
        self.text = d["text"]
        self.origin = d["origin"]
        self.context = d["context"]


class KeywordStore:
    """Tiny deterministic store: rank by naive keyword overlap."""

    def __init__(self, chunks=FLURBO_CHUNKS):
        self._chunks = [DictChunk(c) for c in chunks]

    def retrieve(self, text: str, top_k: int):
        words = set(text.lower().split())
        ranked = sorted(
            self._chunks,
            key=lambda c: len(words & set(c.text.lower().split())),
            reverse=True,
        )
        return ranked[:top_k]


def chat_func(**kwargs):
    return ChatAnthropic(**kwargs)


@pytest.mark.vcr
def test_anthropic_rag_tool_mode_citations():
    chat = chat_func()
    chat.rag.register_store(KeywordStore(), top_k=2)
    chat.chat("How does Flurbo stream responses?", echo="none")

    turn = chat.get_last_turn(role="assistant")
    assert turn is not None
    citations = [c for c in turn.contents if isinstance(c, ContentCitation)]
    assert citations, "expected at least one citation"
    assert any(
        isinstance(c.source, DocumentSource)
        and c.source.id == "kb://flurbo/streaming"
        for c in citations
    )
    assert all(c.cited_quote for c in citations)
    assert_citations_grounded(chat)


@pytest.mark.vcr
def test_anthropic_rag_streaming_interleaves_citations():
    chat = chat_func()
    chat.rag.register_store(KeywordStore(), top_k=2)
    chunks = list(chat.stream("How does Flurbo stream responses?", content="all"))
    citations = [c for c in chunks if isinstance(c, ContentCitation)]
    assert citations
    assert isinstance(citations[0].source, DocumentSource)
