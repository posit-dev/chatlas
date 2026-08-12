"""The `Provider` streaming contract.

`stream_content()` gets the merged-so-far `completion` precisely so a provider
never has to keep per-stream state on `self`: `Chat.__deepcopy__` shares one
provider instance across forked chats, and those forks stream concurrently.
"""

from typing import Optional

import pytest
from chatlas import Chat
from chatlas._content import (
    Content,
    ContentText,
    ContentThinkingDelta,
    ContentToolRequestSearch,
)
from chatlas._provider import Provider
from chatlas._turn import AssistantTurn


class Chunk:
    def __init__(self, text: str = "", emit_request: bool = False):
        self.text = text
        self.emit_request = emit_request


class AccumulatingProvider(Provider):
    """Derives its output from `completion` rather than from `self`.

    `stream_merge_chunks` accumulates each chunk's text onto the completion, and
    `stream_content` reads that accumulation back out. Any provider that needs
    cross-chunk context (Anthropic's citations, say) has this shape.
    """

    def __init__(self, chunks: list[Chunk]):
        super().__init__(name="fake", model="fake-model")
        self._chunks = chunks
        self.seen_completions: list[Optional[dict]] = []
        self.seen_turns: list[list[tuple[str, str]]] = []

    def stream_merge_chunks(self, completion, chunk) -> dict:
        merged = dict(completion or {"text": ""})
        merged["text"] += chunk.text
        return merged

    def stream_content(self, chunk, completion, turns=()) -> list[Content]:
        self.seen_completions.append(completion)
        self.seen_turns.append([(turn.role, turn.text) for turn in turns])
        out: list[Content] = []
        if chunk.text:
            out.append(ContentText.model_construct(text=chunk.text))
        if chunk.emit_request:
            # Only correct if `completion` is this stream's accumulation
            out.append(ContentToolRequestSearch(query=(completion or {})["text"]))
        return out

    def chat_perform(self, *, stream, turns, tools, data_model, kwargs):
        if stream:
            return iter(self._chunks)
        raise NotImplementedError

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):
        if stream:

            async def gen():
                for c in self._chunks:
                    yield c

            return gen()
        raise NotImplementedError

    def stream_turn(self, completion, has_data_model, turns=()):
        self.seen_turns.append([(turn.role, turn.text) for turn in turns])
        return AssistantTurn(
            contents=[ContentText.model_construct(text=completion["text"])],
            tokens=None,
            completion=None,
        )

    def list_models(self):
        return []

    def value_turn(self, completion, has_data_model, turns=()):
        raise NotImplementedError

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


def test_interleaved_streams_on_one_provider_stay_independent():
    provider = AccumulatingProvider([])

    stream_a = [Chunk("alpha "), Chunk("answer"), Chunk(emit_request=True)]
    stream_b = [Chunk("beta "), Chunk("answer"), Chunk(emit_request=True)]

    completions: dict[str, Optional[dict]] = {"a": None, "b": None}
    requests: dict[str, list[str]] = {"a": [], "b": []}
    for chunk_a, chunk_b in zip(stream_a, stream_b):
        for key, chunk in (("a", chunk_a), ("b", chunk_b)):
            completions[key] = provider.stream_merge_chunks(completions[key], chunk)
            requests[key].extend(
                c.query
                for c in provider.stream_content(chunk, completions[key])
                if isinstance(c, ContentToolRequestSearch)
            )

    assert requests["a"] == ["alpha answer"]
    assert requests["b"] == ["beta answer"]


def test_stream_content_sees_completion_including_current_chunk():
    """`stream_merge_chunks` runs first, so `completion` already has this chunk."""
    provider = AccumulatingProvider([Chunk("a"), Chunk("b"), Chunk("c")])
    chat = Chat(provider=provider)

    assert "".join(chat.stream("hi")) == "abc"
    assert [c["text"] for c in provider.seen_completions if c] == ["a", "ab", "abc"]


def test_stream_content_receives_exact_request_turns():
    provider = AccumulatingProvider([Chunk("a"), Chunk("b")])
    chat = Chat(provider=provider)

    assert "".join(chat.stream("hi")) == "ab"
    assert provider.seen_turns == [[("user", "hi")]] * 3


def test_multiple_contents_from_one_chunk_are_all_processed():
    provider = AccumulatingProvider([])
    chat = Chat(provider=provider)
    # Bypass chat_perform so this test owns the chunk -> content mapping
    provider.stream_content = lambda chunk, completion, turns=(): [  # type: ignore[method-assign]
        ContentThinkingDelta(thinking="why"),
        ContentText.model_construct(text="what"),
    ]
    provider.chat_perform = lambda **kwargs: iter([Chunk("what")])  # type: ignore[method-assign]

    out = list(chat.stream("hi", content="all"))

    assert [x for x in out if isinstance(x, str)] == ["what"]
    thinking = [x for x in out if isinstance(x, ContentThinkingDelta)]
    assert [t.thinking for t in thinking] == ["why", ""]


@pytest.mark.asyncio
async def test_stream_content_sees_completion_including_current_chunk_async():
    provider = AccumulatingProvider([Chunk("a"), Chunk("b")])
    chat = Chat(provider=provider)

    out = [x async for x in await chat.stream_async("hi")]

    assert "".join(out) == "ab"
    assert [c["text"] for c in provider.seen_completions if c] == ["a", "ab"]


@pytest.mark.asyncio
async def test_stream_content_receives_exact_request_turns_async():
    provider = AccumulatingProvider([Chunk("a"), Chunk("b")])
    chat = Chat(provider=provider)

    out = [x async for x in await chat.stream_async("hi")]

    assert "".join(out) == "ab"
    assert provider.seen_turns == [[("user", "hi")]] * 3
