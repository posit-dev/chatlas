import pytest

from chatlas import ChatOllama
from chatlas._provider_ollama import is_local_ollama

MODELS = [{"id": "llama3.2"}, {"id": "qwen3:4b"}]


def mock_models(monkeypatch, models=MODELS):
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info",
        lambda base_url: models,
    )


def mock_no_connection(monkeypatch):
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info",
        lambda base_url: None,
    )


def test_ollama_reasoning_effort(monkeypatch):
    """reasoning_effort is passed through as a request body field."""
    mock_models(monkeypatch)

    chat = ChatOllama(model="qwen3:4b", reasoning_effort="none")
    assert chat.kwargs_chat == {"reasoning_effort": "none"}


def test_ollama_no_reasoning_effort_by_default(monkeypatch):
    mock_models(monkeypatch)

    chat = ChatOllama(model="llama3.2")
    assert chat.kwargs_chat == {}


def test_ollama_options_passed_as_extra_body(monkeypatch):
    """Ollama options (e.g. num_ctx) are sent via extra_body."""
    mock_models(monkeypatch)

    chat = ChatOllama(model="llama3.2", options={"num_ctx": 8192})
    assert chat.kwargs_chat == {"extra_body": {"num_ctx": 8192}}


def test_ollama_model_latest_tag_normalized(monkeypatch):
    """A model supplied with an explicit `:latest` tag still validates."""
    mock_models(monkeypatch)

    chat = ChatOllama(model="llama3.2:latest")
    assert chat.provider.model == "llama3.2:latest"


def test_ollama_no_model_lists_available(monkeypatch):
    mock_models(monkeypatch)

    with pytest.raises(ValueError, match="Available models: llama3.2, qwen3:4b"):
        ChatOllama()


def test_ollama_unknown_model_local(monkeypatch):
    mock_models(monkeypatch)

    with pytest.raises(ValueError, match="not installed locally.*ollama pull"):
        ChatOllama(model="not-a-model")


def test_ollama_unknown_model_remote(monkeypatch):
    mock_models(monkeypatch)

    with pytest.raises(ValueError, match="not available on http://ollama:11434"):
        ChatOllama(model="not-a-model", base_url="http://ollama:11434")


def test_ollama_connection_failure_local(monkeypatch):
    mock_no_connection(monkeypatch)

    with pytest.raises(RuntimeError, match="Can't find locally running ollama"):
        ChatOllama(model="llama3.2")


def test_ollama_connection_failure_remote(monkeypatch):
    mock_no_connection(monkeypatch)

    with pytest.raises(
        RuntimeError, match="Can't connect to ollama at http://ollama:11434"
    ):
        ChatOllama(model="llama3.2", base_url="http://ollama:11434")


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("https://ollama.example.com", False),
        ("http://192.168.1.10:11434", False),
    ],
)
def test_is_local_ollama(base_url, expected):
    assert is_local_ollama(base_url) is expected
