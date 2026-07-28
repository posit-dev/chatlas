from pathlib import Path

import pytest
from chatlas import ChatGoogle
from chatlas._content import ContentUploaded
from chatlas._provider_google import GoogleProvider

from .conftest import VCR_MATCH_ON_WITHOUT_BODY, make_vcr_config


def test_google_uploaded_part():
    prov = GoogleProvider(model="gemini-2.0-flash", api_key="dummy", kwargs=None)
    c = ContentUploaded(id="files/abc", mime_type="application/pdf", provider="google")
    part = prov._as_part_type(c)
    assert part.file_data.file_uri == "files/abc"
    assert part.file_data.mime_type == "application/pdf"


def test_google_uploaded_part_wrong_provider_raises():
    prov = GoogleProvider(model="gemini-2.0-flash", api_key="dummy", kwargs=None)
    c = ContentUploaded(id="file_123", mime_type="application/pdf", provider="openai")
    with pytest.raises(ValueError, match="uploaded to provider 'openai'"):
        prov._as_part_type(c)


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
