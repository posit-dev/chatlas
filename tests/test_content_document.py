import base64

import pytest
from chatlas import content_document_file, content_document_url
from chatlas._content import ContentDocument
from chatlas._turn import Turn, UserTurn


def test_infers_mime_type_for_binary_office_extensions(tmp_path):
    """These are OpenAI-only, but must not fall through to the text/plain default."""
    cases = {
        "a.rtf": "application/rtf",
        "a.doc": "application/msword",
        "a.odt": "application/vnd.oasis.opendocument.text",
        "a.xls": "application/vnd.ms-excel",
    }
    for name, mime in cases.items():
        path = tmp_path / name
        path.write_bytes(b"\xd0\xcf\x11\xe0")
        obj = content_document_file(path)
        assert obj.mime_type == mime, name


def test_content_document_url_keeps_url_without_downloading():
    obj = content_document_url("https://example.com/data/q3.csv")
    assert isinstance(obj, ContentDocument)
    assert obj.url == "https://example.com/data/q3.csv"
    assert obj.data is None
    assert obj.filename == "q3.csv"
    assert obj.mime_type == "text/csv"


def test_content_document_url_honors_explicit_mime_type():
    obj = content_document_url("https://example.com/export", mime_type="text/markdown")
    assert obj.mime_type == "text/markdown"
    assert obj.data is None


def test_content_document_url_falls_back_to_a_filename():
    obj = content_document_url("https://example.com/")
    assert obj.filename
    assert obj.mime_type == "text/plain"


def test_content_document_url_inlines_data_urls():
    obj = content_document_url("data:text/csv;base64,YSxiCjEsMgo=")
    assert obj.data == b"a,b\n1,2\n"
    assert obj.mime_type == "text/csv"
    assert obj.url is None


def test_data_url_filename_comes_from_the_mime_type_not_the_payload():
    """A `data:` URL has no path, so the base64 payload must not leak into the name."""
    obj = content_document_url("data:text/csv;base64,YSxiCjEsMgo=")
    assert "base64" not in obj.filename
    assert "YSxiCjEsMgo" not in obj.filename
    assert obj.filename.endswith(".csv")


def test_generated_filenames_stay_distinct():
    """Filenames are what a model uses to tell multi-document prompts apart."""
    data_url = "data:text/csv;base64,YSxiCjEsMgo="
    names = [
        content_document_url(data_url).filename,
        content_document_url(data_url).filename,
        content_document_url("https://example.com/").filename,
        content_document_url("https://example.com/").filename,
    ]
    assert len(set(names)) == len(names)


def test_content_document_url_rejects_pdfs():
    with pytest.raises(ValueError, match="content_pdf_url"):
        content_document_url("https://example.com/report.pdf")


def test_can_create_document_from_local_txt_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello world")

    obj = content_document_file(path)
    assert isinstance(obj, ContentDocument)
    assert obj.filename == "notes.txt"
    assert obj.mime_type == "text/plain"
    assert obj.data == b"hello world"


def test_infers_mime_type_from_common_extensions(tmp_path):
    cases = {
        "a.md": "text/markdown",
        "a.csv": "text/csv",
        "a.json": "application/json",
        "a.html": "text/html",
        "a.xml": "text/xml",
        "a.docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        "a.xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }
    for name, mime in cases.items():
        path = tmp_path / name
        path.write_bytes(b"x")
        obj = content_document_file(path)
        assert obj.mime_type == mime, name


def test_unknown_extension_defaults_to_text_plain(tmp_path):
    # code files with extensions Python's mimetypes module doesn't know about
    for name in ["script.rs", "script.r", "script.go", "config.yaml"]:
        path = tmp_path / name
        path.write_text("some source code")
        obj = content_document_file(path)
        assert obj.mime_type == "text/plain", name


def test_explicit_mime_type_overrides_guess(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x00\x01")
    obj = content_document_file(path, mime_type="application/octet-stream")
    assert obj.mime_type == "application/octet-stream"


def test_rejects_pdf_files(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="content_pdf_file"):
        content_document_file(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        content_document_file(tmp_path / "nope.txt")


def test_document_bytes_round_trip():
    doc = ContentDocument(data=b"hello", filename="a.txt", mime_type="text/plain")
    turn = UserTurn([doc])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].data == b"hello"


def test_content_document_requires_data_or_url():
    with pytest.raises(ValueError):
        ContentDocument(filename="a.txt", mime_type="text/plain")


def test_content_document_rejects_pdf_mime_type():
    with pytest.raises(ValueError, match="content_pdf_file"):
        ContentDocument(data=b"x", filename="a.pdf", mime_type="application/pdf")


def test_content_document_url_only_is_valid():
    obj = ContentDocument(
        filename="notes.txt", mime_type="text/plain", url="https://example.com/n.txt"
    )
    assert obj.data is None
    assert obj.url == "https://example.com/n.txt"


def test_content_document_str_representation():
    doc = ContentDocument(data=b"hello", filename="a.txt", mime_type="text/plain")
    s = str(doc)
    assert "a.txt" in s
    assert "text/plain" in s


def test_content_document_data_serializes_as_base64():
    doc = ContentDocument(data=b"hello", filename="a.txt", mime_type="text/plain")
    dumped = doc.model_dump(mode="json")
    assert dumped["data"] == base64.b64encode(b"hello").decode("ascii")
