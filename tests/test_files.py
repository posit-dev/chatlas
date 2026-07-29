import pytest
from chatlas import ChatAzureOpenAI, ChatBedrockAnthropic, ChatOllama, ChatPosit
from chatlas._files import FileManager, maybe_write
from chatlas._provider_openai import OpenAIProvider


def fake_download(provider: OpenAIProvider, data: bytes) -> None:
    from unittest.mock import MagicMock

    content = MagicMock()
    content.read.return_value = data
    provider._client.files.content = MagicMock(return_value=content)


# Whether a provider lets you download a given file depends on the file's
# purpose, not on who uploaded it: OpenAI refuses purpose="user_data" (what
# upload() uses) but allows purpose="batch"/"batch_output" (what batch_chat()
# creates), and Google only serves model-GENERATED files. So exercise chatlas's
# own write-through logic here rather than a real upload/download round trip.
def test_download_returns_bytes_and_writes_to_path(tmp_path):
    provider = OpenAIProvider(model="gpt-4o", api_key="dummy", kwargs=None)
    fake_download(provider, b"%PDF-1.4 payload")

    dest = tmp_path / "out.pdf"
    data = provider.file_download("file-abc", dest)

    assert data == b"%PDF-1.4 payload"
    assert dest.read_bytes() == b"%PDF-1.4 payload"


def test_download_without_path_writes_nothing(tmp_path):
    provider = OpenAIProvider(model="gpt-4o", api_key="dummy", kwargs=None)
    fake_download(provider, b"bytes")

    assert provider.file_download("file-abc") == b"bytes"
    assert list(tmp_path.iterdir()) == []


def test_maybe_write_roundtrip(tmp_path):
    dest = tmp_path / "nested.bin"
    assert maybe_write(b"abc", dest) == b"abc"
    assert dest.read_bytes() == b"abc"
    assert maybe_write(b"abc", None) == b"abc"


def test_files_accessor_type(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    chat = ChatOllama(model="llama3.1")
    assert isinstance(chat.files, FileManager)


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    chat = ChatOllama(model="llama3.1")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.upload("some.pdf")


# Each of these subclasses a provider that *does* implement file management, so
# each has to opt back out: Azure OpenAI has no Files API, Bedrock's Anthropic
# client has no `.beta.files`, and Posit's Anthropic gateway proxy doesn't
# support the beta Files API.
def azure_chat():
    return ChatAzureOpenAI(
        endpoint="https://example.openai.azure.com/",
        deployment_id="gpt-5-nano",
        api_version="2025-03-01-preview",
    )


def bedrock_chat():
    return ChatBedrockAnthropic(
        aws_secret_key="fake", aws_access_key="fake", aws_region="us-east-1"
    )


def posit_chat():
    return ChatPosit(model="claude-sonnet-4-6", credentials=lambda: "test-token")


@pytest.mark.parametrize("make_chat", [azure_chat, bedrock_chat, posit_chat])
@pytest.mark.parametrize(
    "call",
    [
        lambda f: f.upload("some.pdf"),
        lambda f: f.list(),
        lambda f: f.get("id"),
        lambda f: f.download("id"),
        lambda f: f.delete("id"),
    ],
    ids=["upload", "list", "get", "download", "delete"],
)
def test_opted_out_provider_raises(make_chat, call):
    # Every file method must opt out, not just the couple we happened to spot
    # check -- a partially-disabled provider would reach a client that can't
    # serve the request.
    with pytest.raises(NotImplementedError, match="file management"):
        call(make_chat().files)


def test_no_file_management_covers_every_file_method():
    # Unlike the hand-listed cases above, this picks up any `file_*` method
    # added to Provider later -- which is the point of the decorator: a new one
    # can't silently reach an opted-out provider's client.
    from chatlas._provider import Provider
    from chatlas._provider_anthropic import AnthropicBedrockProvider
    from chatlas._provider_openai_azure import OpenAIAzureProvider
    from chatlas._provider_posit import PositAnthropicProvider

    names = [n for n in vars(Provider) if n.startswith("file_")]
    assert len(names) == 10, "expected 5 file methods x sync/async"

    for cls in (OpenAIAzureProvider, AnthropicBedrockProvider, PositAnthropicProvider):
        for name in names:
            assert getattr(cls, name) is getattr(Provider, name), (
                f"{cls.__name__}.{name} is not opted out of file management"
            )


@pytest.mark.parametrize("make_chat", [azure_chat, bedrock_chat, posit_chat])
@pytest.mark.parametrize(
    "call",
    [
        lambda f: f.upload_async("some.pdf"),
        lambda f: f.list_async(),
        lambda f: f.get_async("id"),
        lambda f: f.download_async("id"),
        lambda f: f.delete_async("id"),
    ],
    ids=["upload", "list", "get", "download", "delete"],
)
@pytest.mark.asyncio
async def test_opted_out_provider_raises_async(make_chat, call):
    with pytest.raises(NotImplementedError, match="file management"):
        await call(make_chat().files)
