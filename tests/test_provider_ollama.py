import pytest

from chatlas import ChatOllama
from chatlas._provider_ollama import is_local_ollama

_MODELS = [
    {"id": "qwen3:4b", "created_at": "x", "size": 1},
    {"id": "llama3.2", "created_at": "x", "size": 1},
]


def test_ollama_reasoning_effort(monkeypatch):
    """reasoning_effort is passed through as a request body field."""
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info", lambda base_url: _MODELS
    )

    chat = ChatOllama(model="qwen3:4b", reasoning_effort="none")
    assert chat.kwargs_chat == {"reasoning_effort": "none"}


def test_ollama_no_reasoning_effort_by_default(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info", lambda base_url: _MODELS
    )

    chat = ChatOllama(model="llama3.2")
    assert chat.kwargs_chat == {}


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("http://remote-host.example.com:11434", False),
        ("https://ollama.internal.example.org", False),
    ],
)
def test_is_local_ollama(base_url, expected):
    assert is_local_ollama(base_url) == expected


def test_ollama_local_connection_failure(monkeypatch):
    """A local endpoint that can't be reached keeps the local-install guidance."""
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: False)

    with pytest.raises(RuntimeError, match="Can't find locally running ollama."):
        ChatOllama(model="llama3.2", base_url="http://localhost:11434")


def test_ollama_remote_connection_failure(monkeypatch):
    """A remote endpoint that can't be reached names the endpoint, not 'locally'."""
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: False)

    with pytest.raises(
        RuntimeError,
        match=r"Can't connect to ollama at http://remote-host\.example\.com:11434\.",
    ):
        ChatOllama(model="llama3.2", base_url="http://remote-host.example.com:11434")


def test_ollama_model_none_lists_available_models(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info", lambda base_url: _MODELS
    )

    with pytest.raises(
        ValueError, match="Must specify model. Available models: qwen3:4b, llama3.2"
    ):
        ChatOllama(model=None, base_url="http://localhost:11434")


def test_ollama_unavailable_model_local_guidance(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info", lambda base_url: _MODELS
    )

    with pytest.raises(ValueError, match="not installed locally"):
        ChatOllama(model="not-a-real-model", base_url="http://localhost:11434")


def test_ollama_unavailable_model_remote_guidance(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info", lambda base_url: _MODELS
    )

    with pytest.raises(
        ValueError,
        match=r"not available on http://remote-host\.example\.com:11434",
    ):
        ChatOllama(
            model="not-a-real-model",
            base_url="http://remote-host.example.com:11434",
        )


def test_ollama_model_with_explicit_latest_tag_accepted(monkeypatch):
    """ollama_model_info() strips ':latest', so a model requested with the
    tag users copy straight from `ollama list` must still match."""
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info", lambda base_url: _MODELS
    )

    ChatOllama(model="llama3.2:latest", base_url="http://localhost:11434")
