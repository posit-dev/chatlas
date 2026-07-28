from pathlib import Path

import pytest
from chatlas import ChatOpenAI
from chatlas._content import ContentUploaded

from .conftest import VCR_MATCH_ON_WITHOUT_BODY, make_vcr_config


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
def test_openai_file_lifecycle():
    pdf = Path(__file__).parent / "apples.pdf"
    chat = ChatOpenAI()
    f = chat.files.upload(str(pdf))
    assert isinstance(f, ContentUploaded)
    assert f.provider == "openai"
    got = chat.files.get(f.id)
    assert got.id == f.id
    assert any(m.id == f.id for m in chat.files.list())
    chat.files.delete(f.id)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_file_lifecycle_async():
    pdf = Path(__file__).parent / "apples.pdf"
    chat = ChatOpenAI()
    f = await chat.files.upload_async(str(pdf))
    assert isinstance(f, ContentUploaded)
    assert f.provider == "openai"
    got = await chat.files.get_async(f.id)
    assert got.id == f.id
    assert any(m.id == f.id for m in await chat.files.list_async())
    await chat.files.delete_async(f.id)
