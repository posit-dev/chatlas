from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from ._content import DOCX_MIME_TYPE, XLSX_MIME_TYPE, ContentDocument
from ._content_pdf import parse_data_url

__all__ = ("content_document_file", "content_document_url")


# A small, curated set of extensions the major providers document as
# first-class document input: plain text, Markdown, CSV, and (OpenAI only)
# the binary office formats. Anything else is assumed to be source code or
# another text format and defaults to "text/plain" -- Python's `mimetypes`
# module doesn't know about most code file extensions (`.r`, `.rs`, `.go`,
# `.yaml`, ...), and guessing wrong there is worse than a generic text default.
# Binary formats must be listed explicitly, since that text default would
# otherwise mislabel them and ship raw bytes as text.
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
    ".rtf": "application/rtf",
    ".doc": "application/msword",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".xls": "application/vnd.ms-excel",
}

PDF_REDIRECT_MESSAGE = (
    "{fn}() doesn't support PDF files. Use {pdf_fn}() instead, which unlocks "
    "PDF-specific handling (page-image understanding, citations, and URL "
    "passthrough)."
)


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
            PDF_REDIRECT_MESSAGE.format(
                fn="content_document_file", pdf_fn="content_pdf_file"
            )
        )

    resolved_mime_type = mime_type if mime_type != "auto" else guess_mime_type(path.name)

    with open(path, "rb") as f:
        data = f.read()

    return ContentDocument(data=data, filename=path.name, mime_type=resolved_mime_type)


def content_document_url(
    url: str,
    mime_type: Literal["auto"] | str = "auto",
) -> ContentDocument:
    """
    Prepare a remote text/data file for input to a chat.

    Use this for plain text, Markdown, CSV, code, and other non-PDF documents
    hosted at a URL (use [](`~chatlas.content_pdf_url`) for PDFs). Not all
    providers accept every document type, so check the documentation for the
    provider you are using.

    Parameters
    ----------
    url
        A URL to a remote file, or a `data:` URL carrying the document inline.
    mime_type
        The document's MIME type. If `"auto"`, it's guessed from the URL's
        file extension, falling back to `"text/plain"`. `data:` URLs always
        use the type declared in the URL itself.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Raises
    ------
    ValueError
        If the URL points to a PDF (use `content_pdf_url()` instead), or is a
        malformed `data:` URL.
    """

    if url.startswith("data:"):
        content_type, base64_data = parse_data_url(url)
        return ContentDocument(
            data=base64.b64decode(base64_data),
            filename=filename_from_url(url) or "document",
            mime_type=content_type,
        )

    name = filename_from_url(url)

    if name.lower().endswith(".pdf"):
        raise ValueError(
            PDF_REDIRECT_MESSAGE.format(
                fn="content_document_url", pdf_fn="content_pdf_url"
            )
        )

    return ContentDocument(
        url=url,
        filename=name or "document",
        mime_type=mime_type if mime_type != "auto" else guess_mime_type(name),
    )


def guess_mime_type(name: str) -> str:
    return DOCUMENT_MIME_TYPES.get(Path(name).suffix.lower(), "text/plain")


def filename_from_url(url: str) -> str:
    """The last path segment of `url`, which may legitimately be empty."""
    return Path(urlparse(url).path).name
