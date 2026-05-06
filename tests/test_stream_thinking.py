"""Tests for streaming thinking tag boundary emission."""

from collections.abc import Sequence
from typing import Optional

import pytest
from chatlas import Chat
from chatlas._content import Content, ContentText, ContentThinking
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


class TestStreamThinkingText:
    """Tests for content='text' mode — tags should be yielded as string chunks."""

    def test_thinking_then_text(self):
        """Streaming thinking → text produces proper tag boundaries."""
        chunks = [
            ContentThinking(thinking="step 1 "),
            ContentThinking(thinking="step 2"),
            ContentText.model_construct(text="Hello world"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        combined = "".join(result)
        assert combined == "<thinking>\nstep 1 step 2\n</thinking>\n\nHello world"

    def test_thinking_only(self):
        """If stream ends during thinking, close tag is still emitted."""
        chunks = [
            ContentThinking(thinking="reasoning here"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        combined = "".join(result)
        assert combined == "<thinking>\nreasoning here\n</thinking>\n\n"

    def test_text_only(self):
        """No thinking chunks means no tags emitted."""
        chunks = [
            ContentText.model_construct(text="Just text"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        combined = "".join(result)
        assert combined == "Just text"

    def test_tag_chunks_are_separate(self):
        """Opening and closing tags are yielded as separate chunks."""
        chunks = [
            ContentThinking(thinking="thought"),
            ContentText.model_construct(text="answer"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test"))
        assert result[0] == "<thinking>\n"
        assert result[1] == "thought"
        assert result[2] == "\n</thinking>\n\n"
        assert result[3] == "answer"


class TestStreamThinkingAll:
    """Tests for content='all' mode — ContentThinking objects yielded, no tag strings."""

    def test_thinking_then_text(self):
        """content='all' yields ContentThinking objects, not tag strings."""
        chunks = [
            ContentThinking(thinking="step 1 "),
            ContentThinking(thinking="step 2"),
            ContentText.model_construct(text="Hello"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test", content="all"))

        thinking_chunks = [x for x in result if isinstance(x, ContentThinking)]
        text_chunks = [x for x in result if isinstance(x, str)]

        assert len(thinking_chunks) == 2
        assert thinking_chunks[0].thinking == "step 1 "
        assert thinking_chunks[1].thinking == "step 2"
        assert text_chunks == ["Hello"]

    def test_no_tag_strings_yielded(self):
        """content='all' mode should NOT yield tag boundary strings."""
        chunks = [
            ContentThinking(thinking="thought"),
            ContentText.model_construct(text="answer"),
        ]
        chat = _make_chat(chunks)
        result = list(chat.stream("test", content="all"))

        str_chunks = [x for x in result if isinstance(x, str)]
        for s in str_chunks:
            assert "<thinking>" not in s
            assert "</thinking>" not in s


@pytest.mark.asyncio
class TestStreamThinkingAsync:
    """Tests for async streaming with thinking boundaries."""

    async def test_thinking_then_text_async(self):
        """Async streaming thinking → text produces proper tag boundaries."""
        chunks = [
            ContentThinking(thinking="async thought "),
            ContentThinking(thinking="more"),
            ContentText.model_construct(text="response"),
        ]
        chat = _make_chat(chunks)
        result = [chunk async for chunk in await chat.stream_async("test")]
        combined = "".join(result)
        assert combined == "<thinking>\nasync thought more\n</thinking>\n\nresponse"

    async def test_thinking_only_async(self):
        """Async: close tag emitted even if stream ends during thinking."""
        chunks = [
            ContentThinking(thinking="reasoning"),
        ]
        chat = _make_chat(chunks)
        result = [chunk async for chunk in await chat.stream_async("test")]
        combined = "".join(result)
        assert combined == "<thinking>\nreasoning\n</thinking>\n\n"

    async def test_content_all_async(self):
        """Async content='all' yields ContentThinking objects, no tag strings."""
        chunks = [
            ContentThinking(thinking="thought"),
            ContentText.model_construct(text="answer"),
        ]
        chat = _make_chat(chunks)
        result = [chunk async for chunk in await chat.stream_async("test", content="all")]

        thinking_chunks = [x for x in result if isinstance(x, ContentThinking)]
        str_chunks = [x for x in result if isinstance(x, str)]

        assert len(thinking_chunks) == 1
        assert thinking_chunks[0].thinking == "thought"
        assert str_chunks == ["answer"]
        for s in str_chunks:
            assert "<thinking>" not in s
