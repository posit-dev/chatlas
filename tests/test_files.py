from types import SimpleNamespace

import pytest
from chatlas import (
    ChatAzureOpenAI,
    ChatBedrock,
    ChatBedrockAnthropic,
    ChatOllama,
    ChatPosit,
)
from chatlas._content import ContentUploaded
from chatlas._files import FileManager, FileMetadata, maybe_write
from chatlas._provider_openai import OpenAIProvider, openai_meta


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


def test_to_content_uses_known_mime_type():
    meta = FileMetadata(
        id="file-abc",
        filename="report.pdf",
        mime_type="application/pdf",
        provider="anthropic",
    )
    content = meta.to_content()
    assert isinstance(content, ContentUploaded)
    assert content.id == "file-abc"
    assert content.mime_type == "application/pdf"
    assert content.provider == "anthropic"
    # extra holds plain, serializable normalized fields -- not the provider's
    # own (often unserializable) file object from FileMetadata.extra
    assert content.extra == {"filename": "report.pdf", "size_bytes": None}
    content.model_dump_json()


def test_to_content_guesses_mime_type_from_filename():
    meta = FileMetadata(id="file-abc", filename="report.pdf", provider="openai")
    assert meta.to_content().mime_type == "application/pdf"


def test_to_content_falls_back_to_octet_stream():
    meta = FileMetadata(id="file-abc", filename="data.unknownext", provider="openai")
    assert meta.to_content().mime_type == "application/octet-stream"

    meta_no_name = FileMetadata(id="file-abc", provider="openai")
    assert meta_no_name.to_content().mime_type == "application/octet-stream"


def test_openai_meta_to_content_roundtrip():
    # OpenAI's Files API doesn't report a MIME type, so file_get() yields
    # mime_type=None; to_content() must still produce a valid ContentUploaded
    # (previously this meant hand-building one from files.get() failed).
    fake_file = SimpleNamespace(
        id="file-abc",
        filename="quarterly-report.pdf",
        bytes=1024,
        created_at=1_700_000_000,
    )
    meta = openai_meta(fake_file)  # type: ignore[arg-type]
    assert meta.mime_type is None

    content = meta.to_content()
    assert content.mime_type == "application/pdf"
    assert content.provider == "openai"


def test_chat_inputs_accept_file_metadata_directly():
    # FileMetadata is coerced to ContentUploaded at the turn boundary, so
    # chat.files.get() output can be passed straight to .chat() and friends.
    from chatlas._turn import UserTurn, user_turn

    meta = FileMetadata(
        id="file-abc",
        filename="report.pdf",
        mime_type="application/pdf",
        provider="anthropic",
    )
    turn = user_turn("Summarize this.", meta)
    assert turn.contents[1] == meta.to_content()

    # Same coercion when constructing a UserTurn directly (a bare FileMetadata
    # isn't accepted, matching how a bare Content object isn't)
    assert UserTurn([meta]).contents[0] == meta.to_content()


def test_files_accessor_type(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info",
        lambda base_url: [{"id": "llama3.1", "created_at": "x", "size": 1}],
    )
    chat = ChatOllama(model="llama3.1")
    assert isinstance(chat.files, FileManager)


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr("chatlas._provider_ollama.has_ollama", lambda base_url: True)
    monkeypatch.setattr(
        "chatlas._provider_ollama.ollama_model_info",
        lambda base_url: [{"id": "llama3.1", "created_at": "x", "size": 1}],
    )
    chat = ChatOllama(model="llama3.1")
    with pytest.raises(NotImplementedError, match="file management"):
        chat.files.upload("some.pdf")


# Each of these subclasses a provider that *does* implement file management, so
# each has to opt back out: Azure OpenAI has no Files API, Bedrock's Anthropic
# client has no `.beta.files`, Posit's Anthropic gateway proxy doesn't support
# the beta Files API, and bedrock-mantle's OpenAI Responses path 404s on
# GET /openai/v1/files (files are only served from mantle's other path, /v1).
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


def bedrock_mantle_responses_chat():
    return ChatBedrock(
        model="openai.gpt-5.6-sol", api="responses", aws_region="us-east-1"
    )


def posit_chat():
    return ChatPosit(model="claude-sonnet-4-6", credentials=lambda: "test-token")


@pytest.mark.parametrize(
    "make_chat", [azure_chat, bedrock_chat, bedrock_mantle_responses_chat, posit_chat]
)
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
    from chatlas._provider_bedrock import BedrockResponsesProvider
    from chatlas._provider_openai_azure import OpenAIAzureProvider
    from chatlas._provider_posit import PositAnthropicProvider

    names = [n for n in vars(Provider) if n.startswith("file_")]
    assert len(names) == 10, "expected 5 file methods x sync/async"

    for cls in (
        OpenAIAzureProvider,
        AnthropicBedrockProvider,
        BedrockResponsesProvider,
        PositAnthropicProvider,
    ):
        for name in names:
            assert getattr(cls, name) is getattr(Provider, name), (
                f"{cls.__name__}.{name} is not opted out of file management"
            )


@pytest.mark.parametrize(
    "make_chat", [azure_chat, bedrock_chat, bedrock_mantle_responses_chat, posit_chat]
)
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
