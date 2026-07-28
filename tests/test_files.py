import pytest
from chatlas import ChatOllama
from chatlas._files import FileManager


def test_files_accessor_type(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    chat = ChatOllama(model="llama3.1")
    assert isinstance(chat.files, FileManager)


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    chat = ChatOllama(model="llama3.1")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.upload("some.pdf")
