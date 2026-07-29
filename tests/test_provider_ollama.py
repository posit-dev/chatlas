from chatlas import ChatOllama


def test_ollama_reasoning_effort(monkeypatch):
    """reasoning_effort is passed through as a request body field."""
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)

    chat = ChatOllama(model="qwen3:4b", reasoning_effort="none")
    assert chat.kwargs_chat == {"reasoning_effort": "none"}


def test_ollama_no_reasoning_effort_by_default(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)

    chat = ChatOllama(model="llama3.2")
    assert chat.kwargs_chat == {}
