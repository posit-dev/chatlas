import warnings

import pytest

from chatlas import ChatOpenAI, StreamController


def test_stream_controller_initial_state():
    ctrl = StreamController()
    assert ctrl.cancelled is False
    assert ctrl.reason is None


def test_stream_controller_cancel():
    ctrl = StreamController()
    ctrl.cancel()
    assert ctrl.cancelled is True
    assert ctrl.reason == "cancelled"


def test_stream_controller_cancel_with_reason():
    ctrl = StreamController()
    ctrl.cancel(reason="timeout")
    assert ctrl.cancelled is True
    assert ctrl.reason == "timeout"


def test_stream_controller_reset():
    ctrl = StreamController()
    ctrl.cancel(reason="timeout")
    ctrl.reset()
    assert ctrl.cancelled is False
    assert ctrl.reason is None


def test_stream_controller_ensure_ready_warns_and_resets():
    ctrl = StreamController()
    ctrl.cancel(reason="stale")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctrl._ensure_ready()
        assert len(w) == 1
        assert "already cancelled" in str(w[0].message)
    assert ctrl.cancelled is False
    assert ctrl.reason is None


def test_stream_controller_ensure_ready_noop_when_not_cancelled():
    ctrl = StreamController()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctrl._ensure_ready()
        assert len(w) == 0
    assert ctrl.cancelled is False


@pytest.mark.vcr
def test_stream_cancel_after_chunks():
    chat = ChatOpenAI()
    ctrl = StreamController()

    chunks = []
    for chunk in chat.stream(
        """
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """,
        controller=ctrl,
    ):
        chunks.append(chunk)
        if len(chunks) >= 3:
            ctrl.cancel()

    turns = chat.get_turns()
    assert len(turns) == 2
    assert turns[1].is_partial
    assert turns[1].partial_reason == "cancelled"
    assert len(turns[1].text) > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_stream_cancel_after_chunks_async():
    chat = ChatOpenAI()
    ctrl = StreamController()

    chunks = []
    async for chunk in await chat.stream_async(
        """
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """,
        controller=ctrl,
    ):
        chunks.append(chunk)
        if len(chunks) >= 3:
            ctrl.cancel()

    turns = chat.get_turns()
    assert len(turns) == 2
    assert turns[1].is_partial
    assert turns[1].partial_reason == "cancelled"
    assert len(turns[1].text) > 0
