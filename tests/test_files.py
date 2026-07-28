import pytest
from chatlas import ChatAzureOpenAI, ChatBedrockAnthropic, ChatOllama, ChatPosit
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


def test_azure_openai_raises():
    # OpenAIAzureProvider subclasses OpenAIProvider (which has real file
    # management), but Azure OpenAI has no Files API.
    chat = ChatAzureOpenAI(
        endpoint="https://example.openai.azure.com/",
        deployment_id="gpt-5-nano",
        api_version="2025-03-01-preview",
    )
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.upload("some.pdf")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.list()


def test_bedrock_anthropic_raises():
    # AnthropicBedrockProvider subclasses AnthropicProvider (which has real
    # file management), but Bedrock's client has no `.beta.files`.
    chat = ChatBedrockAnthropic(
        aws_secret_key="fake",
        aws_access_key="fake",
        aws_region="us-east-1",
    )
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.upload("some.pdf")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.list()


def test_posit_anthropic_raises():
    # PositAnthropicProvider subclasses AnthropicProvider (which has real
    # file management), but Posit's gateway proxy doesn't support the beta
    # Files API.
    chat = ChatPosit(model="claude-sonnet-4-6", credentials=lambda: "test-token")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.upload("some.pdf")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.list()
