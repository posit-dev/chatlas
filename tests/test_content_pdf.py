import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from chatlas import content_pdf_file, content_pdf_url
from chatlas._content import ContentPDF, ContentThinking, ContentToolRequest
from chatlas._content_file import ensure_bytes
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


def test_content_pdf_url_does_not_download_eagerly():
    with patch("chatlas._content_file.download_bytes") as mock_download:
        obj = content_pdf_url("https://example.com/apples.pdf")

    mock_download.assert_not_called()
    assert obj.data is None
    assert obj.url == "https://example.com/apples.pdf"


def test_content_pdf_data_url_still_decodes_inline():
    raw = b"%PDF-1.4 fake"
    b64 = base64.b64encode(raw).decode("ascii")
    obj = content_pdf_url(f"data:application/pdf;base64,{b64}")
    assert obj.data == raw
    assert obj.url is None


def test_content_pdf_requires_data_or_url():
    with pytest.raises(ValueError):
        ContentPDF(filename="test.pdf")


def test_content_pdf_url_only_round_trip():
    pdf = ContentPDF(filename="test.pdf", url="https://example.com/test.pdf")
    turn = UserTurn([pdf])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].data is None
    assert restored.contents[0].url == "https://example.com/test.pdf"


def test_content_pdf_url_lazily_downloads_and_caches_bytes():
    raw = b"downloaded-bytes"
    with patch(
        "chatlas._content_file.download_bytes", return_value=raw
    ) as mock_download:
        obj = content_pdf_url("https://example.com/apples.pdf")
        result = ensure_bytes(obj, "PDF")

    mock_download.assert_called_once_with("https://example.com/apples.pdf")
    assert result == raw
    assert obj.data == raw

    # Cached on the object now, so a second call doesn't re-download.
    with patch("chatlas._content_file.download_bytes") as mock_download2:
        assert ensure_bytes(obj, "PDF") == raw
    mock_download2.assert_not_called()


def test_ensure_bytes_wraps_download_failures():
    with patch("chatlas._content_file.download_bytes", side_effect=OSError("boom")):
        obj = content_pdf_url("https://example.com/apples.pdf")
        with pytest.raises(ValueError, match="https://example.com/apples.pdf"):
            ensure_bytes(obj, "PDF")


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
    thinking = ContentThinking(
        thinking="reasoning...", extra={"thought_signature": sig}
    )
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
