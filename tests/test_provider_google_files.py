from pathlib import Path

import pytest
from chatlas import ChatGoogle
from chatlas._content import ContentUploaded
from chatlas._provider_google import GoogleProvider

from .conftest import VCR_MATCH_ON_WITHOUT_BODY, make_vcr_config

# A ContentUploaded id must be the *full* File API URI. The live API rejects a
# bare `files/abc` with "Unsupported file URI type ... must be a File API (e.g.
# https://generativelanguage.googleapis.com...)", so don't use the short form
# here -- it would pass this unit test while being unusable against the API.
FILE_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc"


def test_google_uploaded_part():
    prov = GoogleProvider(model="gemini-2.0-flash", api_key="dummy", kwargs=None)
    c = ContentUploaded(id=FILE_URI, mime_type="application/pdf", provider="google")
    part = prov._as_part_type(c)
    assert part.file_data is not None
    assert part.file_data.file_uri == FILE_URI
    assert part.file_data.mime_type == "application/pdf"


def test_google_uploaded_part_wrong_provider_raises():
    prov = GoogleProvider(model="gemini-2.0-flash", api_key="dummy", kwargs=None)
    c = ContentUploaded(id="file_123", mime_type="application/pdf", provider="openai")
    with pytest.raises(ValueError, match="uploaded to provider 'openai'"):
        prov._as_part_type(c)


def google_file(state: str, *, uri: str | None = FILE_URI, error: object = None):
    """A `types.File` in `state`, as the Files API would return it."""
    from google.genai import types

    return types.File(
        name="files/abc",
        uri=uri,
        mime_type="application/pdf",
        size_bytes=123,
        state=types.FileState(state),
        error=types.FileStatus(message="boom") if error else None,
    )


# Gemini processes large media (video/audio) asynchronously, and referencing a
# file before it goes ACTIVE fails with "The File ... is not in an ACTIVE state
# and usage is not allowed", so upload() must wait it out.
def test_google_upload_waits_for_active(monkeypatch):
    from unittest.mock import MagicMock

    prov = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)
    monkeypatch.setattr("chatlas._provider_google.time.sleep", lambda _: None)
    prov._client.files.upload = MagicMock(return_value=google_file("PROCESSING"))
    get = MagicMock(
        side_effect=[google_file("PROCESSING"), google_file("ACTIVE")],
    )
    prov._client.files.get = get

    f = prov.file_upload("some.pdf")

    assert isinstance(f, ContentUploaded)
    assert f.id == FILE_URI
    assert f.extra["state"] == "ACTIVE"
    assert get.call_count == 2
    assert get.call_args.kwargs == {"name": "files/abc"}


def test_google_upload_raises_on_failed_state(monkeypatch):
    from unittest.mock import MagicMock

    prov = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)
    monkeypatch.setattr("chatlas._provider_google.time.sleep", lambda _: None)
    prov._client.files.upload = MagicMock(return_value=google_file("PROCESSING"))
    prov._client.files.get = MagicMock(
        return_value=google_file("FAILED", uri=None, error=True)
    )

    with pytest.raises(ValueError, match="failed to process"):
        prov.file_upload("some.pdf")


@pytest.mark.asyncio
async def test_google_upload_async_waits_for_active(monkeypatch):
    from unittest.mock import AsyncMock

    prov = GoogleProvider(model="gemini-3.5-flash", api_key="dummy", kwargs=None)

    async def no_sleep(_):
        return None

    monkeypatch.setattr("chatlas._provider_google.asyncio.sleep", no_sleep)
    prov._client.aio.files.upload = AsyncMock(
        return_value=google_file("PROCESSING"),
    )
    get = AsyncMock(side_effect=[google_file("PROCESSING"), google_file("ACTIVE")])
    prov._client.aio.files.get = get

    f = await prov.file_upload_async("some.pdf")

    assert f.id == FILE_URI
    assert f.extra["state"] == "ACTIVE"
    assert get.await_count == 2


# A FAILED file has no uri, but listing must still work -- one bad file in the
# account shouldn't poison the whole call. `files/abc` is a valid id for
# management ops (get/delete), which is all you can do with a failed file.
def test_google_meta_falls_back_to_name_without_uri():
    from chatlas._provider_google import google_meta

    meta = google_meta(google_file("FAILED", uri=None))
    assert meta.id == "files/abc"


def test_google_files_api_raises_on_vertex():
    prov = GoogleProvider(
        model="gemini-2.0-flash",
        api_key="dummy",
        name="Google/Vertex",
        kwargs={"vertexai": True},
    )
    with pytest.raises(NotImplementedError, match="Vertex"):
        prov.file_list()


# The upload request body is multipart with a randomly generated boundary, so
# it can never match byte-for-byte on replay. Also disable vcrpy's post-data
# parameter filtering: it assumes a form-encoded/UTF-8 body and crashes on the
# raw binary PDF bytes in the multipart body (the API key is already stripped
# via filter_headers, so nothing is lost).
@pytest.fixture(scope="module")
def vcr_config():
    config = make_vcr_config(match_on=VCR_MATCH_ON_WITHOUT_BODY)
    config["filter_post_data_parameters"] = []
    return config


@pytest.mark.vcr
def test_google_file_lifecycle():
    pdf = Path(__file__).parent / "apples.pdf"
    chat = ChatGoogle()
    f = chat.files.upload(str(pdf))
    assert isinstance(f, ContentUploaded)
    assert f.provider == "google"
    got = chat.files.get(f.id)
    assert got.id == f.id
    assert any(m.id == f.id for m in chat.files.list())
    chat.files.delete(f.id)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_google_file_lifecycle_async():
    pdf = Path(__file__).parent / "apples.pdf"
    chat = ChatGoogle()
    f = await chat.files.upload_async(str(pdf))
    assert isinstance(f, ContentUploaded)
    assert f.provider == "google"
    got = await chat.files.get_async(f.id)
    assert got.id == f.id
    assert any(m.id == f.id for m in await chat.files.list_async())
    await chat.files.delete_async(f.id)
