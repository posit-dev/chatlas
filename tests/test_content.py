import pytest
from chatlas import ChatOpenAI
from chatlas._content import ContentUploaded, create_content


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
