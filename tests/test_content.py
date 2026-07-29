import pytest
from chatlas import ChatOpenAI
from chatlas._content import ContentDocument, ContentUploaded, create_content


def test_invalid_inputs_give_useful_errors():
    chat = ChatOpenAI()

    with pytest.raises(TypeError):
        chat.chat(question="Are unicorns real?")  # type: ignore

    with pytest.raises(ValueError):
        chat.chat(True)  # type: ignore


def test_content_uploaded_roundtrip():
    c = ContentUploaded(id="file_123", mime_type="application/pdf", provider="openai")
    assert c.content_type == "uploaded"
    assert str(c) == "<uploaded file id=file_123 mime_type=application/pdf>"

    dumped = c.model_dump()
    restored = create_content(dumped)
    assert isinstance(restored, ContentUploaded)
    assert restored.id == "file_123"
    assert restored.provider == "openai"


def test_content_document_roundtrip():
    c = ContentDocument(data=b"hello", filename="a.txt", mime_type="text/plain")
    assert c.content_type == "document"

    dumped = c.model_dump(mode="json")
    restored = create_content(dumped)
    assert isinstance(restored, ContentDocument)
    assert restored.data == b"hello"
    assert restored.filename == "a.txt"
    assert restored.mime_type == "text/plain"
