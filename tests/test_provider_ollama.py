import pytest

from chatlas import ChatOllama


@pytest.fixture
def mock_has_ollama(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)


def test_ollama_reasoning_effort(mock_has_ollama):
    """reasoning_effort is passed through as a request body field."""
    chat = ChatOllama(model="qwen3:4b", reasoning_effort="none")
    assert chat.kwargs_chat == {"reasoning_effort": "none"}


def test_ollama_no_reasoning_effort_by_default(mock_has_ollama):
    chat = ChatOllama(model="llama3.2")
    assert chat.kwargs_chat == {}


def test_ollama_options_passed_as_extra_body(mock_has_ollama):
    """Ollama options (e.g. num_ctx) are sent via extra_body."""
    chat = ChatOllama(model="llama3.2", options={"num_ctx": 8192})
    assert chat.kwargs_chat == {"extra_body": {"num_ctx": 8192}}


def test_ollama_no_model_lists_available(monkeypatch, mock_has_ollama):
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info",
        lambda base_url: [{"id": "llama3.2"}, {"id": "qwen3:4b"}],
    )

    with pytest.raises(ValueError, match="Available models: llama3.2, qwen3:4b"):
        ChatOllama()
