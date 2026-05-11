"""Tests for streaming thinking with ContentThinkingDelta."""

from collections.abc import Sequence
from typing import Optional

import pytest
from chatlas import Chat
from chatlas._content import Content, ContentText, ContentThinkingDelta
from chatlas._provider import Provider
from chatlas._turn import AssistantTurn


class FakeChunk:
    """A fake chunk that carries a Content object."""

    def __init__(self, content: Optional[Content]):
        self.content = content


class FakeProvider(Provider):
    """Minimal provider that yields a predetermined sequence of content chunks."""

    def __init__(self, chunks: Sequence[Optional[Content]]):
        super().__init__(name="fake", model="fake-model")
        self._chunks = chunks

    def list_models(self):
        return []

    def chat_perform(self, *, stream, turns, tools, data_model, kwargs):
        if stream:
            return iter([FakeChunk(c) for c in self._chunks])
        raise NotImplementedError

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):
        if stream:

            async def _gen():
                for c in self._chunks:
                    yield FakeChunk(c)

            return _gen()
        raise NotImplementedError

    def stream_content(self, chunk) -> Optional[Content]:
        return chunk.content

    def stream_merge_chunks(self, completion, chunk):
        return completion or {}

    def stream_turn(self, completion, has_data_model):
        return AssistantTurn(
            contents=[ContentText.model_construct(text="response")],
            tokens=None,
            completion=None,
        )

    def value_turn(self, completion, has_data_model):
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


def _make_chat(chunks: Sequence[Optional[Content]]) -> Chat:
    provider = FakeProvider(chunks)
    return Chat(provider=provider)


class FinalizingAsyncProvider(FakeProvider):
    """Fake async provider that records when its stream is finalized."""

    def __init__(self, chunks: Sequence[Optional[Content]]):
        super().__init__(chunks)
        self.finalized = False

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):
        if stream:

            async def _gen():
                try:
                    for c in self._chunks:
                        yield FakeChunk(c)
                finally:
                    self.finalized = True

            return _gen()
        raise NotImplementedError


class TestStreamThinkingText:
    """Tests for content='text' mode — thinking is suppressed."""

    def test_thinking_suppressed(self):
        chunks = [
            ContentThinkingDelta(thinking="step 1 "),
            ContentThinkingDelta(thinking="step 2"),
            ContentText.model_construct(text="Hello world"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        combined = "".join(result)
        assert combined == "Hello world"

    def test_thinking_only(self):
        chunks = [
            ContentThinkingDelta(thinking="reasoning here"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        assert result == []

    def test_text_only(self):
        chunks = [
            ContentText.model_construct(text="Just text"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        combined = "".join(result)
        assert combined == "Just text"


class TestStreamThinkingAll:
    """Tests for content='all' mode — ContentThinkingDelta objects yielded with phase."""

    def test_thinking_then_text(self):
        chunks = [
            ContentThinkingDelta(thinking="step 1 "),
            ContentThinkingDelta(thinking="step 2"),
            ContentText.model_construct(text="Hello"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test", content="all"))

        thinking_chunks = [x for x in result if isinstance(x, ContentThinkingDelta)]
        assert len(thinking_chunks) == 3
        assert thinking_chunks[0].thinking == "step 1 "
        assert thinking_chunks[0].phase == "start"
        assert thinking_chunks[1].thinking == "step 2"
        assert thinking_chunks[1].phase == "body"
        assert thinking_chunks[2].thinking == ""
        assert thinking_chunks[2].phase == "end"

    def test_phase_sequence(self):
        chunks = [
            ContentThinkingDelta(thinking="thought"),
            ContentText.model_construct(text="answer"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test", content="all"))

        assert isinstance(result[0], ContentThinkingDelta)
        assert result[0].phase == "start"
        assert result[0].thinking == "thought"
        assert isinstance(result[1], ContentThinkingDelta)
        assert result[1].phase == "end"
        assert result[1].thinking == ""
        assert result[2] == "answer"

    def test_thinking_only(self):
        chunks = [
            ContentThinkingDelta(thinking="reasoning"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test", content="all"))

        thinking_chunks = [x for x in result if isinstance(x, ContentThinkingDelta)]
        assert len(thinking_chunks) == 2
        assert thinking_chunks[0].phase == "start"
        assert thinking_chunks[1].phase == "end"

    def test_str_on_delta_has_no_tags(self):
        chunks = [
            ContentThinkingDelta(thinking="thought"),
            ContentText.model_construct(text="answer"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test", content="all"))

        thinking_chunks = [x for x in result if isinstance(x, ContentThinkingDelta)]
        for chunk in thinking_chunks:
            assert "<thinking>" not in str(chunk)
            assert "</thinking>" not in str(chunk)


@pytest.mark.asyncio
class TestStreamThinkingAsync:
    """Tests for async streaming with thinking."""

    async def test_thinking_suppressed_text_async(self):
        chunks = [
            ContentThinkingDelta(thinking="async thought "),
            ContentThinkingDelta(thinking="more"),
            ContentText.model_construct(text="response"),
        ]
        chat = _make_chat(chunks)
        result = [chunk async for chunk in await chat.stream_async("test")]
        combined = "".join(result)
        assert combined == "response"

    async def test_thinking_only_async(self):
        chunks = [
            ContentThinkingDelta(thinking="reasoning"),
        ]
        chat = _make_chat(chunks)
        result = [chunk async for chunk in await chat.stream_async("test")]
        assert result == []

    async def test_content_all_async(self):
        chunks = [
            ContentThinkingDelta(thinking="thought"),
            ContentText.model_construct(text="answer"),
        ]
        chat = _make_chat(chunks)
        result = [
            chunk
            async for chunk in await chat.stream_async("test", content="all")
        ]

        thinking_chunks = [x for x in result if isinstance(x, ContentThinkingDelta)]
        assert len(thinking_chunks) == 2
        assert thinking_chunks[0].thinking == "thought"
        assert thinking_chunks[0].phase == "start"
        assert thinking_chunks[1].phase == "end"
        assert "answer" in [x for x in result if isinstance(x, str)]

    async def test_stream_async_close_finalizes_inner_generator_immediately(self):
        provider = FinalizingAsyncProvider(
            [ContentText.model_construct(text="a"), ContentText.model_construct(text="b")]
        )
        chat = Chat(provider=provider)

        gen = await chat.stream_async("test")
        assert await anext(gen) == "a"

        await gen.aclose()

        assert provider.finalized is True
