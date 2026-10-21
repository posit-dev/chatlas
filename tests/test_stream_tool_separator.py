import pytest
from chatlas import Chat
from chatlas._content import ContentText, ContentToolRequest
from chatlas._turn import AssistantTurn

from .test_provider_stream_contract import AccumulatingProvider, Chunk


class TwoRoundProvider(AccumulatingProvider):
    """Answers "Checking." with a tool request, then "Done." after the tool ran."""

    def __init__(self, first_round_text: str = "Checking."):
        super().__init__([])
        self.first_round_text = first_round_text
        self.round = 0

    def _next(self):
        self.round += 1
        text = self.first_round_text if self.round == 1 else "Done."
        return iter([Chunk(text)])

    def chat_perform(self, *, stream, turns, tools, data_model, kwargs):
        return self._next()

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):
        it = self._next()

        async def gen():
            for c in it:
                yield c

        return gen()

    def stream_turn(self, completion, has_data_model, turns=()):
        contents = []
        if completion["text"]:
            contents.append(ContentText.model_construct(text=completion["text"]))
        if self.round == 1:
            contents.append(ContentToolRequest(id="c1", name="get_time", arguments={}))
        return AssistantTurn(contents=contents, tokens=None, completion=None)


def _chat(first_round_text: str = "Checking.") -> Chat:
    chat = Chat(provider=TwoRoundProvider(first_round_text))

    def get_time() -> str:
        """Get the time."""
        return "12:00"

    chat.register_tool(get_time)
    return chat


def test_stream_text_separates_text_around_a_tool_call():
    out = "".join(str(x) for x in _chat().stream("hi", content="text"))
    assert out == "Checking.\n\nDone."


@pytest.mark.asyncio
async def test_stream_async_text_separates_text_around_a_tool_call():
    stream = await _chat().stream_async("hi", content="text")
    out = "".join([str(x) async for x in stream])
    assert out == "Checking.\n\nDone."


def test_stream_text_does_not_start_with_a_separator():
    out = "".join(str(x) for x in _chat("").stream("hi", content="text"))
    assert out == "Done."


@pytest.mark.asyncio
async def test_stream_async_text_does_not_start_with_a_separator():
    stream = await _chat("").stream_async("hi", content="text")
    out = "".join([str(x) async for x in stream])
    assert out == "Done."


def test_chat_text_separates_text_around_a_tool_call():
    assert str(_chat().chat("hi", echo="none")) == "Checking.\n\nDone."
    assert str(_chat("").chat("hi", echo="none")) == "Done."


@pytest.mark.asyncio
async def test_chat_async_text_separates_text_around_a_tool_call():
    response = await _chat().chat_async("hi", echo="none")
    assert await response.get_content() == "Checking.\n\nDone."
    response = await _chat("").chat_async("hi", echo="none")
    assert await response.get_content() == "Done."
