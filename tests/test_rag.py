from dataclasses import dataclass
from typing import Any, Optional, Sequence

import orjson
import pytest
from chatlas import Chat
from chatlas._content import ContentCitation, ContentJson, ContentText
from chatlas._provider import Provider
from chatlas._rag import ChunkLike, RetrievalStore, SegmentedAnswer, normalize_chunk
from chatlas._turn import AssistantTurn
from pydantic import BaseModel


@dataclass
class FakeChunk:
    """Mirrors raghilda.chunk.Chunk's relevant attributes."""

    text: str
    origin: Optional[str] = None
    context: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None


class FakeStore:
    """Mirrors raghilda's BaseStore.retrieve(text, top_k) signature."""

    def __init__(self, chunks: Sequence[FakeChunk]):
        self.chunks = list(chunks)
        self.queries: list[str] = []

    def retrieve(self, text: str, top_k: int) -> Sequence[FakeChunk]:
        self.queries.append(text)
        return self.chunks[:top_k]


def test_raghilda_shaped_store_satisfies_protocol():
    assert isinstance(FakeStore([]), RetrievalStore)
    assert isinstance(FakeChunk(text="t"), ChunkLike)


def test_object_without_retrieve_fails_protocol():
    assert not isinstance(object(), RetrievalStore)


def test_normalize_chunk_maps_fields():
    chunk = FakeChunk(
        text="body", origin="kb://d1", context="Guide > Setup", attributes={"k": 1}
    )
    sr = normalize_chunk(chunk, id="c1")
    assert (sr.id, sr.text, sr.source, sr.title) == ("c1", "body", "kb://d1", "Guide > Setup")
    assert sr.extra == {"k": 1}


def test_normalize_chunk_minimal():
    class Bare:
        text = "body"

    sr = normalize_chunk(Bare(), id="c2")
    assert (sr.source, sr.title, sr.extra) == (None, None, {})


def test_provider_capability_defaults():
    from chatlas import ChatAnthropic, ChatGroq, ChatOpenAI

    openai = ChatOpenAI(api_key="fake").provider
    anthropic = ChatAnthropic(api_key="fake").provider
    groq = ChatGroq(api_key="fake").provider

    assert not openai.supports_native_search_results()
    assert anthropic.supports_native_search_results()
    assert not groq.supports_native_search_results()

    assert openai.supports_tools_with_data_model()
    assert anthropic.supports_tools_with_data_model()
    assert not groq.supports_tools_with_data_model()  # Completions family drops tools


def make_chat(**kwargs):
    from chatlas import ChatOpenAI

    return ChatOpenAI(api_key="fake", **kwargs)


def test_rag_accessor_is_cached_manager():
    from chatlas._rag import RagManager

    chat = make_chat()
    assert isinstance(chat.rag, RagManager)
    assert chat.rag is chat.rag


def test_register_store_creates_tool():
    chat = make_chat()
    chat.rag.register_store(FakeStore([FakeChunk(text="t")]))
    tools = chat.get_tools()
    assert any(t.name == "search_documents" for t in tools)


def test_register_second_store_needs_distinct_name():
    import pytest

    chat = make_chat()
    chat.rag.register_store(FakeStore([]))
    with pytest.raises(ValueError, match="name"):
        chat.rag.register_store(FakeStore([]))
    chat.rag.register_store(FakeStore([]), name="search_runbooks")
    assert {t.name for t in chat.get_tools()} >= {"search_documents", "search_runbooks"}


def test_unregister_store_removes_tool():
    chat = make_chat()
    chat.rag.register_store(FakeStore([]))
    chat.rag.unregister_store("search_documents")
    assert all(t.name != "search_documents" for t in chat.get_tools())
    assert not chat.rag.uses_segments_schema()


def test_tool_mode_rejected_when_provider_cannot_combine():
    import pytest
    from chatlas import ChatGroq

    chat = ChatGroq(api_key="fake")
    with pytest.raises(ValueError, match="response schema"):
        chat.rag.register_store(FakeStore([]))


def test_retrieval_tool_returns_tool_search_results():
    from chatlas import ToolSearchResults
    from chatlas._content import ContentToolResult

    chat = make_chat()
    store = FakeStore([FakeChunk(text="alpha", origin="kb://a"), FakeChunk(text="beta")])
    chat.rag.register_store(store, top_k=1)
    tool = next(t for t in chat.get_tools() if t.name == "search_documents")

    result = tool.func(query="anything")
    assert isinstance(result, ContentToolResult)
    assert isinstance(result.value, ToolSearchResults)
    assert [r.id for r in result.value.results] == ["c1"]
    assert store.queries == ["anything"]
    assert chat.rag.chunks["c1"].source == "kb://a"


def test_retrieval_tool_description_is_well_formed():
    chat = make_chat()
    chat.rag.register_store(FakeStore([]))
    tool = next(t for t in chat.get_tools() if t.name == "search_documents")

    description = tool.schema["function"]["description"]
    assert "results.Search" not in description
    assert "Ground your answer in the returned results." in description
    assert "Search the document store." in description
    assert "\n\nSearch the document store." in description
    assert "\n            Parameters" not in description
    assert "Parameters\n----------" in description


def test_register_store_description_customizes_tool():
    chat = make_chat()
    chat.rag.register_store(
        FakeStore([]),
        description="Search the internal runbook collection.",
    )

    tool = next(t for t in chat.get_tools() if t.name == "search_documents")
    description = tool.schema["function"]["description"]
    assert description.startswith("Search the internal runbook collection.")
    assert "Parameters\n----------" in description


def test_chunk_ids_unique_across_calls():
    chat = make_chat()
    store = FakeStore([FakeChunk(text="alpha")])
    chat.rag.register_store(store)
    tool = next(t for t in chat.get_tools() if t.name == "search_documents")
    tool.func(query="q1")
    tool.func(query="q2")
    assert set(chat.rag.chunks) == {"c1", "c2"}


SEGMENTS_JSON = (
    '{"segments": ['
    '{"text": "Flurbo streams via flb.stream(). ", "chunk_ids": ["c2"]}, '
    '{"text": "Its default port is 7113.", "chunk_ids": ["c1"]}'
    "]}"
)


def registry():
    from chatlas.types import SearchResult

    return {
        "c1": SearchResult(id="c1", text="port is 7113", source="kb://intro"),
        "c2": SearchResult(id="c2", text="flb.stream() yields", source="kb://stream"),
    }


def decode_all(deltas):
    from chatlas._rag import SegmentsDecoder

    dec = SegmentsDecoder(registry())
    out = []
    for d in deltas:
        out.extend(dec.feed(d))
    out.extend(dec.finish())
    return out


def flatten(contents):
    from chatlas._content import ContentCitation, ContentText

    text = "".join(c.text for c in contents if isinstance(c, ContentText))
    kinds = ["cite" if isinstance(c, ContentCitation) else "text" for c in contents]
    cites = [c for c in contents if isinstance(c, ContentCitation)]
    return text, kinds, cites


@pytest.mark.parametrize("split", [1, 3, 7, len(SEGMENTS_JSON)])
def test_decoder_text_identical_for_any_split(split):
    deltas = [SEGMENTS_JSON[i : i + split] for i in range(0, len(SEGMENTS_JSON), split)]
    text, _, cites = flatten(decode_all(deltas))
    assert text == "Flurbo streams via flb.stream(). Its default port is 7113."
    assert [c.extra["chunk_id"] for c in cites] == ["c2", "c1"]


def test_decoder_interleaves_citations_in_segment_order():
    text, kinds, cites = flatten(decode_all([SEGMENTS_JSON]))
    first_cite = kinds.index("cite")
    assert "text" in kinds[:first_cite]
    assert cites[0].grounded_span == "Flurbo streams via flb.stream(). "
    assert cites[0].source.id == "kb://stream"
    assert cites[0].cited_quote == "flb.stream() yields"


def test_decoder_never_emits_truncated_chunk_id():
    # split mid-way through the "c2" id: no citation may be emitted for "c"
    idx = SEGMENTS_JSON.index('"c2"') + 2
    _, _, cites = flatten(decode_all([SEGMENTS_JSON[:idx], SEGMENTS_JSON[idx:]]))
    assert [c.extra["chunk_id"] for c in cites] == ["c2", "c1"]


def test_decoder_drops_unknown_ids():
    bad = SEGMENTS_JSON.replace('"c1"', '"c99"')
    _, _, cites = flatten(decode_all([bad]))
    assert [c.extra["chunk_id"] for c in cites] == ["c2"]


def test_decode_segments_json_one_shot():
    from chatlas._rag import decode_segments_json

    contents = decode_segments_json(SEGMENTS_JSON, registry())
    text, _, cites = flatten(contents)
    assert text.endswith("7113.")
    assert len(cites) == 2


def test_decode_segments_json_malformed_falls_back_to_text():
    from chatlas._content import ContentText
    from chatlas._rag import decode_segments_json

    (only,) = decode_segments_json("not json at all", registry())
    assert isinstance(only, ContentText)
    assert only.text == "not json at all"


def chunked(text: str, size: int) -> list[str]:
    """Split `text` into fixed-size pieces, mirroring streamed deltas."""
    return [text[i : i + size] for i in range(0, len(text), size)]


class RagFakeProvider(Provider):
    """Streams pre-canned text deltas and records the `data_model` it's called
    with, so tests can assert the hand-rolled RAG tier injects `SegmentedAnswer`
    (and only when the caller hasn't already supplied their own `data_model`)."""

    def __init__(
        self,
        deltas: Sequence[str],
        native_search_results: bool = False,
    ):
        super().__init__(name="rag-fake", model="fake-model")
        self._deltas = list(deltas)
        self._native_search_results = native_search_results
        self.seen_data_model: Optional[type[BaseModel]] = None

    def list_models(self):
        return []

    def chat_perform(self, *, stream, turns, tools, data_model, kwargs):
        self.seen_data_model = data_model
        if not stream:
            return "".join(self._deltas)
        return iter(self._deltas)

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):
        self.seen_data_model = data_model
        if not stream:
            return "".join(self._deltas)

        async def _gen():
            for d in self._deltas:
                yield d

        return _gen()

    def stream_content(self, chunk, completion):
        return [ContentText.model_construct(text=chunk)] if chunk else []

    def stream_merge_chunks(self, completion, chunk):
        return (completion or "") + chunk

    def stream_turn(self, completion, has_data_model):
        if has_data_model:
            return AssistantTurn([ContentJson(value=orjson.loads(completion))])
        return AssistantTurn([ContentText.model_construct(text=completion)])

    def value_turn(self, completion, has_data_model):
        if has_data_model:
            return AssistantTurn([ContentJson(value=orjson.loads(completion))])
        return AssistantTurn([ContentText.model_construct(text=completion)])

    def value_tokens(self, completion):
        return None

    def value_cost(self, completion, tokens=None):
        return None

    def token_count(self, *args, **kwargs):
        return 0

    async def token_count_async(self, *args, **kwargs):
        return 0

    def translate_model_params(self, *args, **kwargs):
        return {}

    def supported_model_params(self):
        return set()

    def supports_native_search_results(self) -> bool:
        return self._native_search_results


def make_rag_fake_chat(
    deltas: Sequence[str], native_search_results: bool = False
) -> Chat:
    provider = RagFakeProvider(deltas, native_search_results=native_search_results)
    return Chat(provider=provider)


def seed_registry(chat: Chat) -> None:
    """Register a store (so `uses_segments_schema()` is True) and pre-register
    c1/c2 chunks matching `registry()`'s texts/sources, via the public API."""
    chat.rag.register_store(FakeStore([]))
    chat.rag.register_chunks(
        [
            FakeChunk(text="port is 7113", origin="kb://intro"),
            FakeChunk(text="flb.stream() yields", origin="kb://stream"),
        ]
    )


def test_handrolled_stream_yields_prose_and_citations():
    chat = make_rag_fake_chat(deltas=chunked(SEGMENTS_JSON, 5))
    seed_registry(chat)

    out = list(chat.stream("q", content="all"))

    text = "".join(c for c in out if isinstance(c, str))
    assert "{" not in text and "segments" not in text
    assert "flb.stream()" in text
    cites = [c for c in out if isinstance(c, ContentCitation)]
    assert [c.extra["chunk_id"] for c in cites] == ["c2", "c1"]
    assert chat.provider.seen_data_model is SegmentedAnswer


def test_handrolled_final_turn_has_text_and_citations_not_json():
    chat = make_rag_fake_chat(deltas=chunked(SEGMENTS_JSON, 5))
    seed_registry(chat)
    list(chat.stream("q"))
    turn = chat.get_last_turn(role="assistant")
    assert turn is not None
    assert not any(isinstance(c, ContentJson) for c in turn.contents)
    texts = [c for c in turn.contents if isinstance(c, ContentText)]
    assert "".join(t.text for t in texts).endswith("7113.")
    assert sum(isinstance(c, ContentCitation) for c in turn.contents) == 2


def test_user_data_model_wins_over_rag_schema():
    class Person(BaseModel):
        name: str

    chat = make_rag_fake_chat(deltas=['{"name": "Ada"}'])
    seed_registry(chat)
    chat.chat_structured("q", data_model=Person)
    assert chat.provider.seen_data_model is Person


def test_native_tier_gets_no_schema():
    chat = make_rag_fake_chat(deltas=["plain text"], native_search_results=True)
    seed_registry(chat)
    list(chat.stream("q"))
    assert chat.provider.seen_data_model is None
