from __future__ import annotations

from pathlib import Path
from typing import Literal

from ._content import DOCX_MIME_TYPE, XLSX_MIME_TYPE, ContentDocument

__all__ = ("content_document_file",)


# A small, curated set of extensions the major providers document as
# first-class document input: plain text, Markdown, CSV, and (OpenAI only)
# docx/xlsx. Anything else is assumed to be source code or another text
# format and defaults to "text/plain" -- Python's `mimetypes` module doesn't
# know about most code file extensions (`.r`, `.rs`, `.go`, `.yaml`, ...), and
# guessing wrong there is worse than a generic text default.
DOCUMENT_MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "text/xml",
    ".docx": DOCX_MIME_TYPE,
    ".xlsx": XLSX_MIME_TYPE,
}


def content_document_file(
    path: str | Path,
    mime_type: Literal["auto"] | str = "auto",
) -> ContentDocument:
    """
    Prepare a local text/data file for input to a chat.

    Use this for plain text, Markdown, CSV, code, and other document files
    that aren't PDFs (use [](`~chatlas.content_pdf_file`) for those). Not all
    providers accept every document type -- e.g. `.docx`/`.xlsx` only work
    with `ChatOpenAI()`, and Anthropic only accepts documents it can treat as
    plain text -- so check the documentation for the provider you are using.

    This embeds the file's bytes in every request that includes it. For a
    large document, or one referenced across many turns, upload it once with
    `chat.files.upload()` instead and pass the resulting
    [](`~chatlas.types.ContentUploaded`) -- at the cost of working on fewer
    providers and tying the chat to the one that hosts the file.

    Parameters
    ----------
    path
        A path to a local file.
    mime_type
        The file's MIME type. If `"auto"`, it's guessed from the file's
        extension, falling back to `"text/plain"` for unrecognized
        extensions (e.g. most source code files).

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If `path` points to a PDF file (use `content_pdf_file()` instead).
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        raise ValueError(
            "content_document_file() doesn't support PDF files. Use "
            "content_pdf_file() instead, which unlocks PDF-specific handling "
            "(page-image understanding, citations, and URL passthrough)."
        )

    resolved_mime_type = mime_type if mime_type != "auto" else guess_mime_type(path)

    with open(path, "rb") as f:
        data = f.read()

    return ContentDocument(data=data, filename=path.name, mime_type=resolved_mime_type)


def guess_mime_type(path: Path) -> str:
    return DOCUMENT_MIME_TYPES.get(path.suffix.lower(), "text/plain")
