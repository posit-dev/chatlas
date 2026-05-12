from pathlib import Path

from chatlas import content_pdf_file
from chatlas._content import ContentPDF, ContentThinking, ContentToolRequest
from chatlas._turn import AssistantTurn, Turn, UserTurn


def test_can_create_pdf_from_local_file():
    apples = Path(__file__).parent / "apples.pdf"
    obj = content_pdf_file(apples)
    assert isinstance(obj, ContentPDF)
    assert obj.filename == "apples.pdf"
    assert isinstance(obj.data, bytes)


def test_pdf_bytes_round_trip():
    raw = b"\x00\x01\x02\xd6\x05\x06"
    pdf = ContentPDF(data=raw, filename="test.pdf")
    turn = UserTurn([pdf])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].data == raw


def test_tool_request_extra_bytes_round_trip():
    sig = b"\xab\xcd\xef\x00\x01\x02"
    request = ContentToolRequest(
        id="call_1",
        name="query",
        arguments={"q": "SELECT 1"},
        extra={"thought_signature": sig},
    )
    turn = AssistantTurn([request])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].extra["thought_signature"] == sig


def test_thinking_extra_bytes_round_trip():
    sig = b"\xd6\x01\x02\x03\x04\x05"
    thinking = ContentThinking(thinking="reasoning...", extra={"thought_signature": sig})
    turn = AssistantTurn([thinking])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].extra["thought_signature"] == sig


def test_tool_request_extra_without_bytes():
    request = ContentToolRequest(
        id="call_1",
        name="query",
        arguments={"q": "SELECT 1"},
        extra={"key": "value", "num": 42},
    )
    turn = AssistantTurn([request])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].extra == {"key": "value", "num": 42}
